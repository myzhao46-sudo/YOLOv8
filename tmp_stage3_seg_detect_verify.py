#!/usr/bin/env python
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent
LOCAL_ULTRA_ROOT = ROOT / "ultralytics"
if LOCAL_ULTRA_ROOT.exists():
    sys.path.insert(0, str(LOCAL_ULTRA_ROOT))

from ultralytics.models.yolo.detect import IncrementalDistillTrainer
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.torch_utils import intersect_dicts


def section(title: str) -> None:
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)


def layers_from_model(model: Any) -> list[Any]:
    inner = getattr(model, "model", None)
    if inner is None:
        return []
    try:
        return list(inner)
    except Exception:
        return []


def param_count(module: Any) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def shape_of(x: Any) -> Any:
    return tuple(x.shape) if isinstance(x, torch.Tensor) else None


def model22_key_summary(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    c = Counter()
    for k in state_dict.keys():
        if not k.startswith("model.22."):
            continue
        parts = k.split(".")
        if len(parts) >= 3:
            c[parts[2]] += 1
        else:
            c["<other>"] += 1
    return dict(sorted(c.items()))


def inspect_model(label: str, model: Any) -> dict[str, Any]:
    layers = layers_from_model(model)
    head = layers[22] if len(layers) > 22 else (layers[-1] if layers else None)
    sd = model.state_dict()
    names = getattr(model, "names", None)
    pe = getattr(model, "pe", None)
    out = {
        "label": label,
        "model_type": type(model).__name__,
        "task": getattr(model, "task", None),
        "yaml_file": model.yaml.get("yaml_file") if isinstance(getattr(model, "yaml", None), dict) else None,
        "yaml_scale": model.yaml.get("scale") if isinstance(getattr(model, "yaml", None), dict) else None,
        "nc": getattr(model, "nc", None),
        "names": names,
        "total_params": param_count(model),
        "layer_count": len(layers),
        "layer22_type": type(head).__name__ if head is not None else None,
        "layer22_submodules": sorted(list(head._modules.keys())) if head is not None else [],
        "layer22_params": param_count(head) if head is not None else 0,
        "model22_key_summary": model22_key_summary(sd),
        "has_cv2": hasattr(head, "cv2") if head is not None else False,
        "has_cv3": hasattr(head, "cv3") if head is not None else False,
        "has_cv4": hasattr(head, "cv4") if head is not None else False,
        "has_cv5": hasattr(head, "cv5") if head is not None else False,
        "has_proto": hasattr(head, "proto") if head is not None else False,
        "has_savpe": hasattr(head, "savpe") if head is not None else False,
        "has_reprta": hasattr(head, "reprta") if head is not None else False,
        "has_lrpc": hasattr(head, "lrpc") if head is not None else False,
        "has_one2one": hasattr(head, "one2one") if head is not None else False,
        "has_pe": isinstance(pe, torch.Tensor),
        "pe_shape": shape_of(pe),
    }
    return out


def prefix3(k: str) -> str:
    p = k.split(".")
    if len(p) >= 4:
        return ".".join(p[:4])
    if len(p) >= 3:
        return ".".join(p[:3])
    if len(p) >= 2:
        return ".".join(p[:2])
    return p[0]


def layer_idx_from_key(k: str) -> int | None:
    m = re.match(r"^model\.(\d+)\.", k)
    return int(m.group(1)) if m else None


def transfer_report(source: Any, target: Any) -> dict[str, Any]:
    s = source.state_dict()
    t = target.state_dict()
    matched = intersect_dicts(s, t)
    matched_keys = set(matched.keys())

    missed_target = [k for k in t.keys() if k not in matched_keys]
    source_only = [k for k in s.keys() if k not in t]
    target_only = [k for k in t.keys() if k not in s]
    shape_mismatch = [k for k in s.keys() if k in t and tuple(s[k].shape) != tuple(t[k].shape)]

    by_prefix = Counter(prefix3(k) for k in missed_target)
    missed_model22 = [k for k in missed_target if k.startswith("model.22.")]
    missed_cv3 = [k for k in missed_target if k.startswith("model.22.cv3.")]
    missed_cv4 = [k for k in missed_target if k.startswith("model.22.cv4.")]
    missed_savpe = [k for k in missed_target if k.startswith("model.22.savpe.")]
    missed_backbone_neck = [k for k in missed_target if (layer_idx_from_key(k) is not None and layer_idx_from_key(k) <= 21)]

    status = "HEAD_RISK" if len(missed_model22) > 0 else "OK"
    if len(missed_backbone_neck) > 0:
        status = "FAIL"

    return {
        "transferred": len(matched_keys),
        "target_total": len(t),
        "missed_total": len(missed_target),
        "missed_keys": missed_target,
        "missed_by_prefix": dict(sorted(by_prefix.items(), key=lambda x: (-x[1], x[0]))),
        "missed_model22_count": len(missed_model22),
        "missed_cv3_count": len(missed_cv3),
        "missed_cv4_count": len(missed_cv4),
        "missed_savpe_count": len(missed_savpe),
        "missed_backbone_neck_count": len(missed_backbone_neck),
        "missed_backbone_neck_keys": missed_backbone_neck,
        "shape_mismatch_examples": shape_mismatch[:20],
        "target_only_examples": target_only[:20],
        "source_only_examples": source_only[:20],
        "status": status,
    }


def freeze_and_grad_check(trainer: IncrementalDistillTrainer, model: Any) -> dict[str, Any]:
    layers = layers_from_model(model)
    per_layer_trainable = []
    for i, layer in enumerate(layers):
        per_layer_trainable.append(
            {
                "layer": i,
                "type": type(layer).__name__,
                "trainable_params": int(sum(p.numel() for p in layer.parameters() if p.requires_grad)),
            }
        )

    l0_21_all_zero = all(x["trainable_params"] == 0 for x in per_layer_trainable[:22]) if len(per_layer_trainable) >= 22 else False
    l22_trainable_gt0 = per_layer_trainable[22]["trainable_params"] > 0 if len(per_layer_trainable) > 22 else False

    trainer.model = model.to(trainer.device)
    trainer.set_model_attributes()

    opt = trainer.build_optimizer(
        model=trainer.model,
        name=trainer.args.optimizer,
        lr=trainer.args.lr0,
        momentum=trainer.args.momentum,
        decay=trainer.args.weight_decay,
        iterations=100,
    )

    frozen_in_opt = 0
    total_opt_params = 0
    for g in opt.param_groups:
        for p in g["params"]:
            total_opt_params += 1
            if not p.requires_grad:
                frozen_in_opt += 1

    # One-batch backward
    loader = trainer.get_dataloader(trainer.data["train"], batch_size=1, rank=-1, mode="train")
    batch = next(iter(loader))
    batch = trainer.preprocess_batch(batch)

    trainer.model.train()
    opt.zero_grad(set_to_none=True)
    loss, _ = trainer.model(batch)
    loss = loss.sum() if hasattr(loss, "sum") else loss
    loss.backward()

    grad_layer = []
    for i, layer in enumerate(layers):
        has_grad = False
        nonzero_grad = False
        for p in layer.parameters():
            if p.grad is not None:
                has_grad = True
                if torch.is_tensor(p.grad) and torch.any(p.grad != 0):
                    nonzero_grad = True
        grad_layer.append({"layer": i, "has_grad": has_grad, "nonzero_grad": nonzero_grad})

    l0_21_grad_clean = True
    for g in grad_layer[:22]:
        if g["has_grad"] and g["nonzero_grad"]:
            l0_21_grad_clean = False
            break
    l22_grad_exists = grad_layer[22]["has_grad"] if len(grad_layer) > 22 else False

    status = "OK"
    if not l0_21_all_zero or not l0_21_grad_clean:
        status = "FAIL"

    return {
        "per_layer_trainable": per_layer_trainable,
        "layer_0_21_trainable_all_zero": l0_21_all_zero,
        "layer22_trainable_gt0": l22_trainable_gt0,
        "optimizer_total_params": total_opt_params,
        "optimizer_contains_frozen_params_count": frozen_in_opt,
        "grad_by_layer": grad_layer,
        "layer_0_21_grad_clean": l0_21_grad_clean,
        "layer22_grad_exists": l22_grad_exists,
        "status": status,
    }


def parse_eval_output(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {"raw_tail": "\n".join(text.splitlines()[-40:])}
    patterns = {
        "ap50": r"AP50 \(manual\):\s*([0-9.]+)",
        "precision": r"Precision:\s*([0-9.]+)",
        "recall": r"Recall:\s*([0-9.]+)",
        "tp_fp_fn": r"TP / FP / FN:\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d+)",
    }
    for k, p in patterns.items():
        m = re.search(p, text)
        if not m:
            out[k] = None
            continue
        if k == "tp_fp_fn":
            out["tp"] = int(m.group(1))
            out["fp"] = int(m.group(2))
            out["fn"] = int(m.group(3))
        else:
            out[k] = float(m.group(1))
    return out


def run_eval(pyexe: Path, model_path: Path, note: str) -> dict[str, Any]:
    cmd = [
        str(pyexe),
        "eval_oiltank_simple.py",
        "--model",
        str(model_path.resolve()),
        "--dataset-root",
        str((ROOT / "ultralytics" / "datasets" / "tank_extratest").resolve()),
        "--split",
        "val",
        "--target-class",
        "tank",
        "--gt-class-id",
        "2",
        "--imgsz",
        "640",
        "--conf",
        "0.1",
        "--iou",
        "0.5",
        "--device",
        "cpu",
        "--batch",
        "8",
        "--skip-official-val",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    info = {
        "returncode": p.returncode,
        "stdout": p.stdout,
        "stderr": p.stderr,
        "note": note,
    }
    if p.returncode == 0:
        info.update(parse_eval_output(p.stdout))
    return info


def main() -> None:
    pyexe = Path(sys.executable)
    best_pt = (ROOT / "best.pt").resolve()

    section("ENV")
    print(json.dumps({"python": str(pyexe), "torch": torch.__version__}, ensure_ascii=False, indent=2))

    section("1) 原始 best.pt inspect")
    source_model, _ = load_checkpoint(str(best_pt), device="cpu", inplace=True, fuse=False)
    source_info = inspect_model("best.pt", source_model)
    print(json.dumps(source_info, ensure_ascii=False, indent=2))

    section("2) 构建当前 student 并 inspect")
    overrides = {
        "task": "detect",
        "mode": "train",
        "model": str(best_pt),
        "student_arch": "yoloe",
        "data": "ultralytics/cfg/datasets/ship_oiltank_joint.yaml",
        "epochs": 1,
        "batch": 2,
        "imgsz": 640,
        "workers": 0,
        "device": "cpu",
        "project": "runs/detect",
        "name": "tmp_stage3_verify",
        "exist_ok": True,
        "val": False,
        "plots": False,
        "save": False,
        "enable_distillation": False,
        "enable_replay": False,
        "enable_slicing": False,
        "freeze": 22,
        "yoloe_prompt_mode": "fixed_text",
        "yoloe_allow_seg_init": True,
        "yoloe_zero_embedding_fallback": True,
    }
    trainer = IncrementalDistillTrainer(cfg=DEFAULT_CFG, overrides=overrides)
    student = trainer.get_model(cfg=source_model.yaml, weights=source_model, verbose=False)
    student_info = inspect_model("student", student)
    pe = getattr(student, "pe", None)
    student_info["set_classes_names"] = getattr(student, "names", None)
    student_info["set_classes_pe_shape"] = shape_of(pe)
    student_info["zero_embedding_fallback_likely"] = bool(
        isinstance(pe, torch.Tensor) and pe.numel() > 0 and torch.count_nonzero(pe).item() == 0
    )
    print(json.dumps(student_info, ensure_ascii=False, indent=2))

    section("3) 权重迁移统计")
    tr = transfer_report(source_model, student)
    print(json.dumps(tr, ensure_ascii=False, indent=2))

    section("4) freeze / optimizer / grad 检查")
    fr = freeze_and_grad_check(trainer, student)
    print(json.dumps(fr, ensure_ascii=False, indent=2))

    section("5) no-train eval 对比（若可执行）")
    eval_rows = []
    src_eval = run_eval(pyexe, best_pt, note="source best.pt direct eval")
    eval_rows.append(
        {
            "model": str(best_pt),
            "task": source_info.get("task"),
            "layer22": source_info.get("layer22_type"),
            "trained": "no",
            "nc": source_info.get("nc"),
            "names": source_info.get("names"),
            "embedding_shape": source_info.get("pe_shape"),
            "AP50": src_eval.get("ap50"),
            "P": src_eval.get("precision"),
            "R": src_eval.get("recall"),
            "FP": src_eval.get("fp"),
            "FN": src_eval.get("fn"),
            "note": src_eval.get("note"),
            "returncode": src_eval.get("returncode"),
        }
    )

    student_tmp = Path(tempfile.gettempdir()) / "tmp_stage3_student.pt"
    torch.save({"model": student}, student_tmp)
    stu_eval = run_eval(pyexe, student_tmp, note="seg->detect student built, no train")
    eval_rows.append(
        {
            "model": str(student_tmp),
            "task": student_info.get("task"),
            "layer22": student_info.get("layer22_type"),
            "trained": "no",
            "nc": student_info.get("nc"),
            "names": student_info.get("names"),
            "embedding_shape": student_info.get("pe_shape"),
            "AP50": stu_eval.get("ap50"),
            "P": stu_eval.get("precision"),
            "R": stu_eval.get("recall"),
            "FP": stu_eval.get("fp"),
            "FN": stu_eval.get("fn"),
            "note": stu_eval.get("note"),
            "returncode": stu_eval.get("returncode"),
        }
    )

    print(json.dumps({"eval_rows": eval_rows}, ensure_ascii=False, indent=2))
    if src_eval.get("returncode", 1) != 0:
        print("\n[SOURCE_EVAL_STDERR]\n" + src_eval.get("stderr", ""))
    if stu_eval.get("returncode", 1) != 0:
        print("\n[STUDENT_EVAL_STDERR]\n" + stu_eval.get("stderr", ""))

    section("SUMMARY FLAG")
    summary = {
        "transfer_status": tr["status"],
        "freeze_status": fr["status"],
        "head_risk": tr["status"] == "HEAD_RISK",
        "backbone_neck_missed_count": tr["missed_backbone_neck_count"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
