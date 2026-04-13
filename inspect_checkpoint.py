#!/usr/bin/env python
"""
Read-only inspector for a Ultralytics checkpoint (.pt), focused on model identity.

It tries two independent paths:
1) raw torch.load(...)
2) ultralytics.YOLO(...)

No files are modified.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

# ---------------------------------------------------------------------------
# Hard-coded checkpoint path (as requested, ready to run on your machine)
# ---------------------------------------------------------------------------
CKPT_PATH = Path(r"C:\Users\DOCTOR\Documents\GitHub\YOLOv8\ultralytics\ultralytics\pretrain\best.pt")

# Ensure local project ultralytics package is preferred over pip-installed one.
THIS_DIR = Path(__file__).resolve().parent
LOCAL_ULTRALYTICS_ROOT = THIS_DIR / "ultralytics"
if LOCAL_ULTRALYTICS_ROOT.exists():
    sys.path.insert(0, str(LOCAL_ULTRALYTICS_ROOT))


def torch_runtime_ok() -> bool:
    return bool(
        torch is not None
        and hasattr(torch, "load")
        and hasattr(torch, "nn")
        and hasattr(torch.nn, "Module")
    )


def is_torch_module(obj: Any) -> bool:
    if not torch_runtime_ok():
        return False
    return isinstance(obj, torch.nn.Module)


def print_section(title: str) -> None:
    bar = "=" * 96
    print(f"\n{bar}\n{title}\n{bar}")


def safe_repr(value: Any, max_len: int = 240) -> str:
    try:
        text = repr(value)
    except Exception as exc:
        text = f"<repr failed: {exc}>"
    text = text.replace("\n", " ")
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return text


def class_name(obj: Any) -> str:
    if obj is None:
        return "None"
    cls = obj.__class__
    return f"{cls.__module__}.{cls.__name__}"


def count_params(module: Any) -> tuple[int | None, int | None]:
    if not is_torch_module(module):
        return None, None
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def compact_names(names: Any, max_items: int = 20) -> str:
    if names is None:
        return "None"
    if isinstance(names, dict):
        items = sorted(names.items(), key=lambda x: x[0])
        suffix = "" if len(items) <= max_items else f" ... (total {len(items)})"
        return str(items[:max_items]) + suffix
    if isinstance(names, (list, tuple)):
        suffix = "" if len(names) <= max_items else f" ... (total {len(names)})"
        return str(list(names[:max_items])) + suffix
    return safe_repr(names)


def extract_names_nc(*candidates: Any) -> tuple[Any, Any]:
    names = None
    nc = None
    for obj in candidates:
        if obj is None:
            continue
        if isinstance(obj, dict):
            if names is None and "names" in obj:
                names = obj.get("names")
            if nc is None and "nc" in obj:
                nc = obj.get("nc")
        else:
            if names is None and hasattr(obj, "names"):
                names = getattr(obj, "names")
            if nc is None and hasattr(obj, "nc"):
                nc = getattr(obj, "nc")
    return names, nc


def inspect_model_object(model_obj: Any, tag: str = "model") -> dict[str, Any]:
    info: dict[str, Any] = {}
    print_section(f"Model Object Summary ({tag})")
    print(f"- object class: {class_name(model_obj)}")

    total, trainable = count_params(model_obj)
    if total is not None:
        print(f"- params total/trainable: {total:,} / {trainable:,}")
        info["params_total"] = total
        info["params_trainable"] = trainable

    names, nc = extract_names_nc(model_obj)
    print(f"- names: {compact_names(names)}")
    print(f"- nc: {safe_repr(nc)}")
    info["names"] = names
    info["nc"] = nc

    inner = getattr(model_obj, "model", None)
    info["has_inner_model_attr"] = inner is not None
    if inner is not None:
        print(f"- has `.model`: yes ({class_name(inner)})")
        try:
            layers = list(inner)
        except Exception:
            layers = []
        if layers:
            last_n = min(8, len(layers))
            print(f"- last {last_n} layers (index: class):")
            for i, m in enumerate(layers[-last_n:], start=len(layers) - last_n):
                print(f"  [{i}] {class_name(m)}")
            head = layers[-1]
            info["head_class"] = class_name(head)
            print(f"- inferred head (last layer): {info['head_class']}")
            for attr in ("nm", "npr", "nc", "reg_max", "embed"):
                if hasattr(head, attr):
                    print(f"  head.{attr} = {safe_repr(getattr(head, attr))}")
            print(
                f"  head has proto/mask clues: proto={hasattr(head, 'proto')}, cv5={hasattr(head, 'cv5')}, nm={hasattr(head, 'nm')}"
            )
            print(
                f"  head has text/prompt clues: reprta={hasattr(head, 'reprta')}, savpe={hasattr(head, 'savpe')}, lrpc={hasattr(head, 'lrpc')}"
            )
            info["head_has_proto"] = hasattr(head, "proto")
            info["head_has_cv5"] = hasattr(head, "cv5")
            info["head_has_nm"] = hasattr(head, "nm")
            info["head_has_lrpc"] = hasattr(head, "lrpc")

    if is_torch_module(model_obj):
        all_cls = {m.__class__.__name__ for m in model_obj.modules()}
        lower_names = [n.lower() for n in all_cls]
        seg_hint = any(("segment" in n) or ("proto" in n) or ("mask" in n) for n in lower_names)
        text_hint = any(
            any(k in n for k in ("yoloe", "prompt", "text", "clip", "contrastive", "reprta", "savpe", "lrpc"))
            for n in lower_names
        )
        pose_hint = any((n == "pose") or n.endswith("pose") or ("posemodel" in n) for n in lower_names)
        cls_hint = any(("classify" in n) or ("classification" in n) for n in lower_names)
        obb_hint = any((n == "obb") or ("obb" in n) for n in lower_names)
        info["module_class_names"] = sorted(all_cls)
        info["seg_hint"] = seg_hint
        info["text_hint"] = text_hint
        info["pose_hint"] = pose_hint
        info["classify_hint"] = cls_hint
        info["obb_hint"] = obb_hint
        print(f"- module keyword hints: seg={seg_hint}, text/prompt={text_hint}, pose={pose_hint}, classify={cls_hint}, obb={obb_hint}")

    return info


def torch_load_checkpoint(path: Path) -> tuple[Any, dict[str, Any]]:
    info: dict[str, Any] = {"ok": False}
    print_section("A) torch.load(...) Inspection")
    print(f"- checkpoint path: {path}")
    print(f"- exists: {path.exists()}")
    if not path.exists():
        return None, info
    if not torch_runtime_ok():
        info["error"] = "torch runtime unavailable (missing torch.load / torch.nn.Module)"
        print(f"- skip torch.load: {info['error']}")
        return None, info

    ckpt = None
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        info["ok"] = True
        info["load_mode"] = "weights_only=False"
    except TypeError:
        try:
            ckpt = torch.load(path, map_location="cpu")
            info["ok"] = True
            info["load_mode"] = "legacy torch.load(...)"
        except Exception as exc:
            info["error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"

    if not info["ok"]:
        print(f"- load failed: {info.get('error', 'unknown error')}")
        print("- traceback:")
        print(traceback.format_exc())
        return None, info

    print(f"- load success via: {info['load_mode']}")
    print(f"- checkpoint object type: {class_name(ckpt)}")
    info["ckpt_type"] = class_name(ckpt)

    if isinstance(ckpt, dict):
        keys = list(ckpt.keys())
        info["keys"] = keys
        print(f"- top-level keys ({len(keys)}): {keys}")
        for key in ("model", "ema", "train_args", "args", "yaml", "cfg", "epoch", "best_fitness"):
            print(f"  has `{key}`: {key in ckpt}")

        for key in ("train_args", "args", "yaml", "cfg"):
            if key in ckpt:
                print(f"- {key}: {safe_repr(ckpt[key])}")

        raw_model = ckpt.get("model", None)
        raw_ema = ckpt.get("ema", None)
        info["has_model"] = raw_model is not None
        info["has_ema"] = raw_ema is not None
        print(f"- `model` class: {class_name(raw_model)}")
        print(f"- `ema` class: {class_name(raw_ema)}")

        names, nc = extract_names_nc(ckpt, raw_ema, raw_model)
        info["names"] = names
        info["nc"] = nc
        print(f"- names (from ckpt/model/ema): {compact_names(names)}")
        print(f"- nc (from ckpt/model/ema): {safe_repr(nc)}")

        target = raw_ema if is_torch_module(raw_ema) else raw_model
        if target is not None:
            info["model_inspect"] = inspect_model_object(target, tag="ckpt.ema-or-model")
    else:
        print("- checkpoint is not a dict, skipping key-level inspection.")

    return ckpt, info


def ultralytics_load_checkpoint(path: Path) -> tuple[Any, dict[str, Any]]:
    info: dict[str, Any] = {"ok": False}
    print_section("B) ultralytics.YOLO(...) Inspection")
    print(f"- checkpoint path: {path}")
    if not path.exists():
        print("- checkpoint file not found, skip YOLO load.")
        return None, info

    try:
        from ultralytics import YOLO
    except Exception as exc:
        info["error"] = f"Import ultralytics failed: {type(exc).__name__}: {exc}"
        print(f"- {info['error']}")
        print("- traceback:")
        print(traceback.format_exc())
        return None, info

    try:
        wrapper = YOLO(str(path))
        info["ok"] = True
    except Exception as exc:
        info["error"] = f"YOLO(...) load failed: {type(exc).__name__}: {exc}"
        print(f"- {info['error']}")
        print("- traceback:")
        print(traceback.format_exc())
        return None, info

    print(f"- wrapper class: {class_name(wrapper)}")
    info["wrapper_class"] = class_name(wrapper)

    task = getattr(wrapper, "task", None)
    print(f"- wrapper.task: {safe_repr(task)}")
    info["task"] = task

    ckpt_path = getattr(wrapper, "ckpt_path", None)
    print(f"- wrapper.ckpt_path: {safe_repr(ckpt_path)}")

    wrapper_ckpt = getattr(wrapper, "ckpt", None)
    print(f"- wrapper.ckpt type: {class_name(wrapper_ckpt)}")
    if isinstance(wrapper_ckpt, dict):
        keys = list(wrapper_ckpt.keys())
        print(f"- wrapper.ckpt keys ({len(keys)}): {keys}")
        info["wrapper_ckpt_keys"] = keys

    names = None
    try:
        names = wrapper.names
    except Exception:
        pass
    print(f"- wrapper.names: {compact_names(names)}")
    info["names"] = names

    inner = getattr(wrapper, "model", None)
    print(f"- wrapper.model class: {class_name(inner)}")
    info["inner_class"] = class_name(inner)

    if inner is not None:
        info["inner_model_inspect"] = inspect_model_object(inner, tag="wrapper.model")
        inner_names, inner_nc = extract_names_nc(inner)
        if names is None and inner_names is not None:
            info["names"] = inner_names
        info["nc"] = inner_nc

    return wrapper, info


def infer_final_summary(path: Path, torch_info: dict[str, Any], yolo_info: dict[str, Any]) -> None:
    print_section("C) Final Human Summary")

    hints: list[str] = []
    task_votes: list[str] = []
    family_votes: list[str] = []
    evidence_ready = bool(torch_info.get("ok")) or bool(yolo_info.get("ok"))

    if not evidence_ready:
        stem = path.stem.lower()
        coarse_task = "unknown"
        if "-seg" in stem or "segment" in stem:
            coarse_task = "segment"
        elif "-pose" in stem or "pose" in stem:
            coarse_task = "pose"
        elif "class" in stem:
            coarse_task = "classify"
        elif "obb" in stem:
            coarse_task = "obb"
        elif "detect" in stem or "yolo" in stem:
            coarse_task = "detect"

        coarse_family = "unknown"
        if "yoloe" in stem:
            coarse_family = "YOLOE (filename hint only)"
        elif "yolo" in stem:
            coarse_family = "Ultralytics YOLO family (filename hint only)"

        print("- Inferred task: unknown (runtime evidence unavailable)")
        print("- Inferred family: unknown (runtime evidence unavailable)")
        print("- Class names: None")
        print("- nc: None")
        print("\n[人话判断]")
        print("- 当前 Python 环境缺少可用 PyTorch（torch.load / torch.save 不可用），所以无法做可靠模型身份识别。")
        print("- 你需要切到训练环境后再跑，脚本会给出完整判断。")
        if coarse_task != "unknown" or coarse_family != "unknown":
            print(f"- 仅从文件名弱线索看：task≈{coarse_task}，family≈{coarse_family}。")
        return

    # Gather textual material for heuristic matching
    text_blob_parts = [str(path).lower()]
    for d in (torch_info, yolo_info):
        for k in ("wrapper_class", "inner_class", "task", "head_class"):
            if k in d and d[k] is not None:
                text_blob_parts.append(str(d[k]).lower())
        for k in ("keys", "wrapper_ckpt_keys"):
            if k in d and isinstance(d[k], list):
                text_blob_parts.append(" ".join(map(str, d[k])).lower())
        for k in ("model_inspect", "inner_model_inspect"):
            v = d.get(k)
            if isinstance(v, dict):
                text_blob_parts.append(" ".join(v.get("module_class_names", [])).lower())
                if "head_class" in v:
                    text_blob_parts.append(str(v["head_class"]).lower())

    blob = " ".join(text_blob_parts)

    # Task inference
    explicit_task = yolo_info.get("task")
    if explicit_task:
        task_votes.append(str(explicit_task).lower())
    if any(k in blob for k in ("segment", "proto", "mask", "yoloesegment")):
        task_votes.append("segment")
    if "pose" in blob:
        task_votes.append("pose")
    if "classify" in blob or "classification" in blob:
        task_votes.append("classify")
    if "obb" in blob:
        task_votes.append("obb")
    if "detect" in blob:
        task_votes.append("detect")

    inferred_task = "unknown"
    for cand in ("segment", "pose", "classify", "obb", "detect"):
        if cand in task_votes:
            inferred_task = cand
            break

    # Family inference
    is_yoloe = any(k in blob for k in ("yoloe", "yoloedetect", "yoloesegment", "reprta", "savpe", "lrpc"))
    is_world = "yoloworld" in blob or "worlddetect" in blob
    if is_yoloe:
        inferred_family = "YOLOE"
        family_votes.append("YOLOE clues found (class names / prompt-text modules)")
    elif is_world:
        inferred_family = "YOLO-World"
        family_votes.append("YOLO-World clues found")
    elif "ultralytics" in blob or "detectionmodel" in blob or "segmentationmodel" in blob:
        inferred_family = "Ultralytics YOLO family (non-YOLOE)"
        family_votes.append("Ultralytics model classes found, but no strong YOLOE clues")
    else:
        inferred_family = "Unknown / custom"
        family_votes.append("No strong Ultralytics family marker")

    # Names / class count
    names = yolo_info.get("names", None)
    if names is None:
        names = torch_info.get("names", None)
    nc = yolo_info.get("nc", None)
    if nc is None:
        nc = torch_info.get("nc", None)

    print(f"- Inferred task: {inferred_task}")
    print(f"- Inferred family: {inferred_family}")
    print(f"- Class names: {compact_names(names)}")
    print(f"- nc: {safe_repr(nc)}")

    # Plain-language advice
    print("\n[人话判断]")
    if inferred_task == "segment":
        print("- 这个权重大概率是分割模型（seg）路线，不是纯 detect 头。")
    elif inferred_task == "detect":
        print("- 这个权重大概率是检测模型（detect）路线。")
    else:
        print("- 任务类型没有 100% 锁死，但从线索看更接近上面的推断。")

    if inferred_family == "YOLOE":
        print("- 这个权重更像 YOLOE 体系（有 text/prompt 相关结构线索）。")
    elif inferred_family == "Ultralytics YOLO family (non-YOLOE)":
        print("- 这个权重看起来是常规 Ultralytics YOLO 家族，不是明显 YOLOE。")
    else:
        print("- 模型家族线索不够强，可能有自定义包装。")

    has_lrpc = False
    for key in ("model_inspect", "inner_model_inspect"):
        for src in (torch_info, yolo_info):
            if isinstance(src.get(key), dict) and src[key].get("head_has_lrpc"):
                has_lrpc = True
    if has_lrpc:
        print("- 检测到 lrpc 线索：可能是 YOLOE prompt-free 变体，蒸馏兼容性需要额外确认。")

    if inferred_task == "segment":
        print("- 如果你的蒸馏主线是 detect student，这个 seg 权重通常不能“无脑直连”到 detect 训练头。")
        print("- 常见做法是：用它做 backbone/neck 初始化，或先对齐到 detect 头配置再进蒸馏。")
    elif inferred_task == "detect":
        print("- 对 detect-distill 管线更友好，可优先考虑作为 teacher 或 student 初始化来源。")

    if family_votes:
        print("\n[证据摘要]")
        for item in family_votes:
            print(f"- {item}")


def main() -> None:
    print_section("Checkpoint Identity Inspector (Read-Only)")
    print(f"- python: {sys.version.split()[0]}")
    torch_ver = getattr(torch, "__version__", "<unavailable>")
    print(f"- torch: {torch_ver}")
    if not torch_runtime_ok():
        print("- torch runtime check: FAILED (this environment cannot run torch.load safely)")
    print(f"- target checkpoint: {CKPT_PATH}")

    ckpt_obj, torch_info = torch_load_checkpoint(CKPT_PATH)
    _wrapper, yolo_info = ultralytics_load_checkpoint(CKPT_PATH)

    infer_final_summary(CKPT_PATH, torch_info=torch_info, yolo_info=yolo_info)

    print_section("Done")
    print("- This script is read-only. No file modifications were performed.")
    _ = ckpt_obj  # keep variable for possible debugging without lint complaint


if __name__ == "__main__":
    main()
