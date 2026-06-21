from __future__ import annotations

import sys
sys.path.insert(0, r"C:/Users/DOCTOR/Documents/GitHub/YOLOv8/ultralytics")
sys.path.insert(1, r"C:/Users/DOCTOR/Documents/GitHub/YOLOv8")

import argparse
import hashlib
import json
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from tools.eval.eval_teacher_ship_external import (
    collect_ground_truth,
    collect_predictions,
    evaluate_predictions,
    labels_dir_from_images_dir,
    list_images,
    obj_type,
    raw_yoloeseg_fused_forward,
    shape_of,
    to_plain_names,
)


REPO_ROOT = Path(r"C:/Users/DOCTOR/Documents/GitHub/YOLOv8")
PACKAGE_ROOT = REPO_ROOT / "ultralytics"
DEFAULT_WEIGHTS = PACKAGE_ROOT / "best.pt"
SHIP_ROOT = PACKAGE_ROOT / "datasets" / "ship_extratest_no_overlap"
SUMMARY_PATH = REPO_ROOT / "runs" / "probe" / "yoloeseg_global4_ship_preserve_summary.json"
EXPANDED_WEIGHTS = (
    REPO_ROOT
    / "runs"
    / "probe"
    / "yoloeseg_global4_expanded_init"
    / "weights"
    / "global4_expanded_bestpt.pt"
)

NAMES3 = ["ship", "harbor", "tank"]
NAMES4 = ["ship", "harbor", "tank", "bridge"]
MODALITIES = {
    "RGB": "extratest_rgb",
    "SAR": "extratest_sar",
    "IR": "extratest_ir",
}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def log(message: object = "") -> None:
    print(message, flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plain(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return str(value)


def get_head(model):
    return model.model.model[-1]


def task_state(model) -> dict:
    inner = model.model
    args = getattr(inner, "args", {})
    args_task = args.get("task") if isinstance(args, dict) else getattr(args, "task", None)
    return {
        "wrapper.task": getattr(model, "task", None),
        "inner.task": getattr(inner, "task", None),
        "inner.args.task": args_task,
    }


def count_params(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def set_classes(model, names: list[str]) -> torch.Tensor:
    embeddings = model.get_text_pe(names)
    model.set_classes(names, embeddings)
    # YOLOE.set_classes can intentionally no-op when names already match. Ensure the freshly generated PE is recorded.
    model.model.set_classes(names, embeddings)
    return embeddings


def score_channels(model, device: str, imgsz: int) -> list[int]:
    inner = model.model.eval().to(device)
    dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)
    with torch.no_grad():
        _, channels = raw_yoloeseg_fused_forward(inner, dummy)
    return [int(channels)]


def probe_model(model, embeddings: torch.Tensor, device: str, imgsz: int) -> dict:
    inner = model.model
    head = get_head(model)
    probe = {
        "wrapper_type": obj_type(model),
        "model_class": obj_type(inner),
        "task": task_state(model),
        "last_layer_type": obj_type(head),
        "last_layer.nc": getattr(head, "nc", None),
        "last_layer.nl": getattr(head, "nl", None),
        "last_layer.no": getattr(head, "no", None),
        "last_layer.reg_max": getattr(head, "reg_max", None),
        "last_layer.embed": getattr(head, "embed", None),
        "last_layer.is_fused": getattr(head, "is_fused", None),
        "names": to_plain_names(getattr(inner, "names", None)),
        "pe_shape": shape_of(embeddings),
        "score_channels_seen": score_channels(model, device, imgsz),
        "total_params": count_params(inner),
        "head_params": count_params(head),
    }
    log(json.dumps(probe, ensure_ascii=False, indent=2))
    return probe


def inspect_head(model) -> dict:
    head = get_head(model)
    keys = ["nc", "nl", "no", "reg_max", "nm", "npr", "embed", "is_fused", "end2end"]
    result = {
        "type": obj_type(head),
        "attrs": {key: plain(getattr(head, key, None)) for key in keys},
        "modules": {},
        "state_dict": [],
    }
    log("\n[FUSED HEAD ANALYSIS]")
    log(f"head type: {obj_type(head)}")
    log(f"head attrs: {json.dumps(result['attrs'], ensure_ascii=False)}")
    for name in ["cv2", "cv3", "cv4", "cv5", "proto", "reprta", "savpe", "lrpc", "one2one"]:
        exists = hasattr(head, name)
        value = getattr(head, name, None)
        entry = {"exists": exists, "type": obj_type(value) if exists else None}
        if name in {"cv2", "cv3", "cv4", "cv5"} and value is not None:
            entry["layers"] = [repr(layer) for layer in value]
            for index, layer in enumerate(value):
                log(f"{name}[{index}]: {layer}")
        else:
            entry["repr"] = repr(value) if exists else None
            log(f"{name}: exists={exists}, type={entry['type']}")
        result["modules"][name] = entry

    for key, tensor in model.model.state_dict().items():
        if key.startswith("model.22."):
            row = {"key": key, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            result["state_dict"].append(row)
            log(f"{key}: {row['shape']} {row['dtype']}")

    result["diagnosis"] = (
        "For a fused YOLOESegment, cv3[i][-1] is the fixed vocabulary Conv2d and cv4[i] is fused to an "
        "identity pass-through. set_classes changes nc/names/pe but does not resize cv3[i][-1]."
    )
    return result


def check_ship_data() -> tuple[dict, dict[str, list[Path]]]:
    report = {"root": str(SHIP_ROOT), "modalities": {}, "class_histogram": {}, "bad_rows": 0, "label_rows": 0}
    image_groups: dict[str, list[Path]] = {}
    for modality, split in MODALITIES.items():
        images_dir = SHIP_ROOT / "images" / split
        labels_dir = SHIP_ROOT / "labels" / split
        images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMG_EXTS)
        image_groups[modality] = images
        missing_labels = 0
        rows = 0
        bad_rows = 0
        histogram: dict[int, int] = {}
        for image in images:
            label = labels_dir / f"{image.stem}.txt"
            if not label.exists():
                missing_labels += 1
                continue
            for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                parts = line.split()
                if not parts:
                    continue
                if len(parts) != 5:
                    bad_rows += 1
                    continue
                try:
                    cls = int(float(parts[0]))
                    [float(value) for value in parts[1:]]
                except ValueError:
                    bad_rows += 1
                    continue
                rows += 1
                histogram[cls] = histogram.get(cls, 0) + 1
        report["modalities"][modality] = {
            "images_dir": str(images_dir),
            "labels_dir": str(labels_dir),
            "image_count": len(images),
            "missing_label_files": missing_labels,
            "label_rows": rows,
            "bad_rows": bad_rows,
            "class_histogram": histogram,
        }
        report["label_rows"] += rows
        report["bad_rows"] += bad_rows
        for cls, count in histogram.items():
            report["class_histogram"][cls] = report["class_histogram"].get(cls, 0) + count
    report["only_class_0"] = set(report["class_histogram"]) <= {0}
    if report["bad_rows"] or not report["only_class_0"]:
        raise RuntimeError(f"External ship labels failed validation: {report}")
    log("\n[SHIP DATA CHECK]")
    log(json.dumps(report, ensure_ascii=False, indent=2))
    return report, image_groups


def evaluate_ship(model, image_groups: dict[str, list[Path]], args) -> dict:
    eval_args = SimpleNamespace(
        device=args.device,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        pred_iou=args.pred_iou,
        match_iou=args.match_iou,
        max_det=args.max_det,
        target_class=0,
        strict_label=True,
    )
    report = {}
    all_predictions = []
    all_ground_truth = {}
    all_score_channels: set[int] = set()
    all_gt_stats = {"image_count": 0, "missing_label_files": 0, "bad_label_rows": 0, "non_target_rows": 0}
    for modality, images in image_groups.items():
        labels_dir = labels_dir_from_images_dir(images[0].parent)
        predictions, channels, nms_boxes = collect_predictions(model.model, images, eval_args)
        ground_truth, gt_stats = collect_ground_truth(images, labels_dir, eval_args)
        metrics = evaluate_predictions(predictions, ground_truth, args.match_iou)
        report[modality] = {
            **metrics,
            **gt_stats,
            "nms_boxes_all_classes": nms_boxes,
            "score_channels_seen": sorted(channels),
        }
        all_predictions.extend(predictions)
        all_ground_truth.update(ground_truth)
        all_score_channels.update(channels)
        for key in all_gt_stats:
            all_gt_stats[key] += gt_stats[key]
        log(f"{modality}: AP50={metrics['ap50']:.6f} P={metrics['precision']:.6f} R={metrics['recall']:.6f}")
    all_metrics = evaluate_predictions(all_predictions, all_ground_truth, args.match_iou)
    report["ALL"] = {**all_metrics, **all_gt_stats, "score_channels_seen": sorted(all_score_channels)}
    log(f"ALL: AP50={all_metrics['ap50']:.6f} P={all_metrics['precision']:.6f} R={all_metrics['recall']:.6f}")
    return report


def expand_fused_score_branch(model) -> dict:
    head = get_head(model)
    if not getattr(head, "is_fused", False):
        raise RuntimeError("Expected a fused YOLOESegment head.")
    if head.__class__.__name__ != "YOLOESegment":
        raise TypeError(f"Expected YOLOESegment, got {obj_type(head)}")

    before = {key: tensor.detach().cpu().clone() for key, tensor in model.model.state_dict().items()}
    changes = []
    for index, branch in enumerate(head.cv3):
        old = branch[-1]
        if not isinstance(old, nn.Conv2d) or old.out_channels != 3 or old.kernel_size != (1, 1):
            raise TypeError(f"cv3[{index}][-1] is not the expected fused 3-class Conv2d: {old}")
        new = nn.Conv2d(
            old.in_channels,
            4,
            old.kernel_size,
            old.stride,
            old.padding,
            old.dilation,
            old.groups,
            old.bias is not None,
            old.padding_mode,
            device=old.weight.device,
            dtype=old.weight.dtype,
        ).requires_grad_(old.weight.requires_grad)
        with torch.no_grad():
            new.weight[:3].copy_(old.weight)
            new.weight[3].copy_(old.weight.mean(dim=0))
            if old.bias is not None:
                new.bias[:3].copy_(old.bias)
                new.bias[3].copy_(old.bias.mean())
        branch[-1] = new
        changes.extend(
            [
                {
                    "module": f"model.22.cv3.{index}.2",
                    "key": f"model.22.cv3.{index}.2.weight",
                    "old_shape": list(old.weight.shape),
                    "new_shape": list(new.weight.shape),
                },
                {
                    "module": f"model.22.cv3.{index}.2",
                    "key": f"model.22.cv3.{index}.2.bias",
                    "old_shape": list(old.bias.shape),
                    "new_shape": list(new.bias.shape),
                },
            ]
        )
    head.nc = 4
    head.no = head.reg_max * 4 + 4

    after = model.model.state_dict()
    modified_keys = []
    old_rows_exact = True
    unrelated_exact = True
    for key, old_tensor in before.items():
        new_tensor = after[key].detach().cpu()
        if old_tensor.shape != new_tensor.shape:
            modified_keys.append(key)
            old_rows_exact &= torch.equal(old_tensor, new_tensor[:3])
        elif not torch.equal(old_tensor, new_tensor):
            unrelated_exact = False
            modified_keys.append(key)
    expected = sorted(change["key"] for change in changes)
    actual = sorted(modified_keys)
    return {
        "strategy": "mean_of_old_class_rows",
        "modified_modules": sorted({change["module"] for change in changes}),
        "modified_state_dict_keys": actual,
        "expected_modified_state_dict_keys": expected,
        "shape_changes": changes,
        "old_classes_0_1_2_copied_exactly": bool(old_rows_exact),
        "all_unrelated_parameters_unchanged": bool(unrelated_exact),
        "only_expected_keys_modified": actual == expected,
    }


def save_expanded_checkpoint(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S1 global4 fused YOLOESegment ship-preservation probe.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--expanded-weights", default=str(EXPANDED_WEIGHTS))
    parser.add_argument("--summary", default=str(SUMMARY_PATH))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--pred-iou", type=float, default=0.7)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if str(args.device).isdigit():
        args.device = f"cuda:{args.device}"
    weights = Path(args.weights).resolve()
    expanded_weights = Path(args.expanded_weights).resolve()
    summary_path = Path(args.summary).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    from ultralytics.models.yolo.model import YOLOE
    import ultralytics

    source_hash_before = sha256(weights)
    summary = {
        "time": datetime.now().isoformat(),
        "git_commit": None,
        "ultralytics_imported_from": str(Path(ultralytics.__file__).resolve()),
        "weights": str(weights),
        "expanded_weights": str(expanded_weights),
        "settings": vars(args),
        "native_val_called": False,
        "training_called": False,
        "masks_ignored": True,
        "source_sha256_before": source_hash_before,
    }
    try:
        import subprocess

        summary["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        summary["data_check"], image_groups = check_ship_data()

        log("\n[BASELINE NAMES3 PROBE]")
        baseline = YOLOE(str(weights), task="segment")
        baseline_pe = set_classes(baseline, NAMES3)
        summary["baseline_probe"] = probe_model(baseline, baseline_pe, args.device, args.imgsz)
        summary["head_analysis"] = inspect_head(baseline)
        log("\n[BASELINE NAMES3 SHIP EVAL]")
        summary["baseline_eval"] = evaluate_ship(baseline, image_groups, args)

        log("\n[GLOBAL4 SET_CLASSES-ONLY PROBE]")
        global4 = YOLOE(str(weights), task="segment")
        global4_pe = set_classes(global4, NAMES4)
        summary["global4_set_only_probe"] = probe_model(global4, global4_pe, args.device, args.imgsz)
        summary["global4_set_only_probe"]["names_pe_changed_but_scores_still_3"] = (
            summary["global4_set_only_probe"]["last_layer.nc"] == 4
            and summary["global4_set_only_probe"]["score_channels_seen"] == [3]
        )
        log("\n[GLOBAL4 SET_CLASSES-ONLY SHIP EVAL]")
        summary["global4_set_only_eval"] = evaluate_ship(global4, image_groups, args)

        log("\n[GLOBAL4 FUSED SCORE EXPANSION]")
        summary["expansion"] = expand_fused_score_branch(global4)
        summary["expansion"]["source_checkpoint_modified"] = False
        expanded_probe_memory = probe_model(global4, global4_pe, args.device, args.imgsz)
        summary["expanded_probe_in_memory"] = expanded_probe_memory
        expansion_ok = (
            expanded_probe_memory["model_class"].endswith("YOLOESegModel")
            and expanded_probe_memory["last_layer_type"].endswith("YOLOESegment")
            and expanded_probe_memory["last_layer.nc"] == 4
            and expanded_probe_memory["score_channels_seen"] == [4]
            and summary["expansion"]["old_classes_0_1_2_copied_exactly"]
            and summary["expansion"]["all_unrelated_parameters_unchanged"]
            and summary["expansion"]["only_expected_keys_modified"]
        )
        summary["expansion"]["success_in_memory"] = expansion_ok
        if not expansion_ok:
            raise RuntimeError("Global4 expansion failed structural verification; expanded checkpoint was not saved.")

        save_expanded_checkpoint(global4, expanded_weights)
        summary["expanded_checkpoint_saved"] = expanded_weights.exists()
        summary["expanded_checkpoint_sha256"] = sha256(expanded_weights)

        log("\n[EXPANDED CHECKPOINT RELOAD PROBE]")
        expanded = YOLOE(str(expanded_weights), task="segment")
        expanded_pe = set_classes(expanded, NAMES4)
        summary["expanded_probe"] = probe_model(expanded, expanded_pe, args.device, args.imgsz)
        if summary["expanded_probe"]["score_channels_seen"] != [4]:
            raise RuntimeError("Reloaded expanded checkpoint does not emit four score channels.")
        log("\n[EXPANDED CHECKPOINT SHIP EVAL]")
        summary["expanded_eval"] = evaluate_ship(expanded, image_groups, args)

        baseline_ap = summary["baseline_eval"]["ALL"]["ap50"]
        expanded_ap = summary["expanded_eval"]["ALL"]["ap50"]
        summary["conclusion"] = {
            "global4_expansion_success": True,
            "baseline_all_ship_ap50": baseline_ap,
            "global4_set_only_all_ship_ap50": summary["global4_set_only_eval"]["ALL"]["ap50"],
            "expanded_all_ship_ap50": expanded_ap,
            "expanded_minus_baseline_ap50": expanded_ap - baseline_ap,
            "ship_preserved": expanded_ap > 0 and expanded_ap >= baseline_ap - 0.02,
            "ready_for_next_box_only_trainer_step": expanded_ap > 0 and expanded_ap >= baseline_ap - 0.02,
        }
    except Exception as error:
        summary["error"] = repr(error)
        summary["traceback"] = traceback.format_exc()
        log(summary["traceback"])
    finally:
        summary["source_sha256_after"] = sha256(weights)
        summary["source_bestpt_untouched"] = summary["source_sha256_before"] == summary["source_sha256_after"]
        if "expansion" in summary:
            summary["expansion"]["source_checkpoint_modified"] = not summary["source_bestpt_untouched"]
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"\n[SUMMARY] {summary_path}")
    return 1 if "error" in summary else 0


if __name__ == "__main__":
    raise SystemExit(main())
