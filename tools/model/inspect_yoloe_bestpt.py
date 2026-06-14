# tools/model/inspect_yoloe_bestpt.py
# -*- coding: utf-8 -*-

"""
Direct-run YOLOE best.pt inspector.

Purpose:
1. Inspect upstream best.pt structure.
2. Check whether it is YOLOESegModel / YOLOESegment.
3. Check task / names / params / layer table.
4. Check set_classes(["ship", "harbor", "tank", "bridge"]).
5. Check get_text_pe / prompt-like tensors.

Run:
Just click "Run Python File" in VS Code / PyCharm.

This script is read-only:
- It does not train.
- It does not val.
- It does not modify best.pt on disk.
- set_classes only changes the in-memory loaded model object.
"""

from pathlib import Path
import sys
import json
import inspect
import traceback
from datetime import datetime

import torch


# ======================================================================================
# 你主要只需要改这里
# ======================================================================================

REPO_ROOT = Path(r"C:\Users\DOCTOR\Documents\GitHub\YOLOv8")

# best.pt 位置：
# C:\Users\DOCTOR\Documents\GitHub\YOLOv8\ultralytics\best.pt
WEIGHTS_PATH = Path(r"C:\Users\DOCTOR\Documents\GitHub\YOLOv8\ultralytics\best.pt")

GLOBAL4_CLASS_NAMES = ["ship", "harbor", "tank", "bridge"]

SAVE_DIR = REPO_ROOT / "runs" / "model_inspect"
SAVE_JSON = SAVE_DIR / "inspect_yoloe_bestpt.json"
SAVE_TXT = SAVE_DIR / "inspect_yoloe_bestpt.txt"

PAUSE_AT_END = False  # 如果你双击运行窗口一闪而过，可以改成 True


# ======================================================================================
# 基础工具函数
# ======================================================================================

def log(msg=""):
    print(msg, flush=True)


def setup_local_ultralytics_import(repo_root: Path):
    """
    你的仓库结构大概率是：

      YOLOv8/
        ultralytics/
          best.pt
          ultralytics/
            __init__.py
            nn/
            models/
            ...

    所以真正应该加入 sys.path 的优先路径是：

      C:/Users/DOCTOR/Documents/GitHub/YOLOv8/ultralytics

    而不是只加：

      C:/Users/DOCTOR/Documents/GitHub/YOLOv8
    """
    repo_root = repo_root.resolve()
    package_root = repo_root / "ultralytics"

    # 先移除已有重复路径，避免顺序混乱
    for p in [str(package_root), str(repo_root)]:
        while p in sys.path:
            sys.path.remove(p)

    # package_root 必须排在 repo_root 前面
    sys.path.insert(0, str(package_root))
    sys.path.insert(1, str(repo_root))

    log("[PYTHON PATH CHECK]")
    log(f"  repo_root:              {repo_root}")
    log(f"  package_root:           {package_root}")
    log(f"  expected __init__.py:   {package_root / 'ultralytics' / '__init__.py'}")
    log(f"  package __init__ exists:{(package_root / 'ultralytics' / '__init__.py').exists()}")
    log("  first sys.path entries:")
    for x in sys.path[:5]:
        log(f"    {x}")


def resolve_weights_path(path: Path) -> Path:
    """
    支持两种写法：
    1. WEIGHTS_PATH = .../best.pt
    2. WEIGHTS_PATH = .../ultralytics
       如果是文件夹，就自动寻找 .../ultralytics/best.pt
    """
    path = path.resolve()

    if path.is_file():
        return path

    if path.is_dir():
        candidate = path / "best.pt"
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Cannot find best.pt.\n"
        f"Current WEIGHTS_PATH = {path}\n"
        f"Please check whether best.pt exists."
    )


def safe_getattr(obj, name, default=None):
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def obj_type(obj):
    if obj is None:
        return "None"
    return f"{obj.__class__.__module__}.{obj.__class__.__name__}"


def safe_signature(fn):
    try:
        return str(inspect.signature(fn))
    except Exception:
        return "signature unavailable"


def has_callable(obj, name):
    value = safe_getattr(obj, name, None)
    return callable(value)


def to_plain_names(names):
    """
    Normalize model.names into a printable dict.
    Ultralytics may store names as dict/list/tuple.
    """
    if names is None:
        return None

    if isinstance(names, dict):
        out = {}
        for k, v in names.items():
            try:
                kk = int(k)
            except Exception:
                kk = str(k)
            out[kk] = str(v)
        return out

    if isinstance(names, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names)}

    return str(names)


def count_params(module):
    if module is None or not hasattr(module, "parameters"):
        return 0, 0

    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def shape_of(x):
    if x is None:
        return None

    if torch.is_tensor(x):
        return list(x.shape)

    if isinstance(x, torch.nn.Parameter):
        return list(x.shape)

    if isinstance(x, (list, tuple)):
        return [shape_of(v) for v in x]

    if isinstance(x, dict):
        return {str(k): shape_of(v) for k, v in x.items()}

    if hasattr(x, "shape"):
        try:
            return list(x.shape)
        except Exception:
            return str(x.shape)

    return None


def summarize_value(x):
    if torch.is_tensor(x):
        return {
            "type": "Tensor",
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "device": str(x.device),
        }

    if isinstance(x, torch.nn.Parameter):
        return {
            "type": "Parameter",
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "device": str(x.device),
        }

    if isinstance(x, (list, tuple)):
        return {
            "type": type(x).__name__,
            "len": len(x),
            "shape": shape_of(x),
        }

    if isinstance(x, dict):
        return {
            "type": "dict",
            "keys": list(map(str, list(x.keys())[:30])),
            "shape": shape_of(x),
        }

    if hasattr(x, "shape"):
        return {
            "type": type(x).__name__,
            "shape": shape_of(x),
        }

    return {
        "type": obj_type(x),
        "repr": repr(x)[:500],
    }


def plain_attr_value(x):
    """
    Convert common layer attributes into JSON-safe values while preserving useful detail.
    """
    if x is None:
        return None

    if torch.is_tensor(x):
        try:
            if x.numel() <= 20:
                return x.detach().cpu().tolist()
            return summarize_value(x)
        except Exception:
            return summarize_value(x)

    if isinstance(x, torch.nn.Parameter):
        return summarize_value(x)

    if isinstance(x, (str, int, float, bool)):
        return x

    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            if isinstance(v, (str, int, float, bool)) or v is None:
                out.append(v)
            elif torch.is_tensor(v) or isinstance(v, torch.nn.Parameter) or hasattr(v, "shape"):
                out.append(summarize_value(v))
            else:
                out.append(str(v))
        return out

    if isinstance(x, dict):
        return {str(k): plain_attr_value(v) for k, v in x.items()}

    if hasattr(x, "shape"):
        return summarize_value(x)

    return str(x)


def summarize_last_layer_state(last_layer):
    """
    Capture the head fields that matter for checking set_classes side effects.
    """
    if last_layer is None:
        return None

    state = {
        "type": obj_type(last_layer),
        "class_name": last_layer.__class__.__name__,
    }

    for attr in ["nc", "no", "nl", "reg_max", "embed"]:
        state[attr] = plain_attr_value(safe_getattr(last_layer, attr, None))

    state["prompt_like_attrs"] = find_attr_tensors(last_layer)
    return state


def find_attr_tensors(obj, keywords=("pe", "text", "txt", "prompt", "embed", "clip", "savpe", "reprta")):
    """
    Search direct attributes only, not recursive.
    用于快速找到 YOLOE prompt / text embedding / pe 相关字段或方法。
    """
    found = []

    if obj is None:
        return found

    try:
        attr_names = dir(obj)
    except Exception:
        return found

    for name in attr_names:
        if name.startswith("__"):
            continue

        lname = name.lower()
        if not any(k in lname for k in keywords):
            continue

        try:
            value = getattr(obj, name)
        except Exception:
            continue

        item = {
            "attr": name,
            "python_type": type(value).__name__,
        }

        if callable(value):
            item["callable"] = True
            item["signature"] = safe_signature(value)
        else:
            item.update(summarize_value(value))

        found.append(item)

    return found


# ======================================================================================
# 模型结构相关
# ======================================================================================

def get_layers(inner_model):
    """
    For Ultralytics models, inner_model.model is usually a ModuleList/Sequential.
    """
    layers = safe_getattr(inner_model, "model", None)

    if layers is None:
        return []

    try:
        return list(layers)
    except Exception:
        return []


def summarize_layers(inner_model):
    layers = get_layers(inner_model)
    rows = []

    for i, m in enumerate(layers):
        total, trainable = count_params(m)

        row = {
            "idx": i,
            "type": m.__class__.__name__,
            "module": obj_type(m),
            "params": total,
            "params_M": round(total / 1e6, 6),
            "trainable": trainable,
            "trainable_M": round(trainable / 1e6, 6),
        }

        for attr in ["i", "f", "type", "np", "stride", "nc", "nl", "reg_max", "no", "nm", "npr"]:
            value = safe_getattr(m, attr, None)
            if value is None:
                continue

            try:
                if torch.is_tensor(value):
                    value = value.detach().cpu().tolist()
                else:
                    value = str(value)
            except Exception:
                value = str(value)

            row[attr] = value

        rows.append(row)

    return rows


def print_layer_table(layer_rows):
    log("")
    log("=" * 130)
    log("[LAYER SUMMARY]")
    log(f"{'idx':>3} | {'type':<30} | {'params(M)':>9} | {'from':<18} | extra")
    log("-" * 130)

    for r in layer_rows:
        extras = []
        for k in ["stride", "nc", "nl", "reg_max", "no", "nm", "npr"]:
            if k in r:
                extras.append(f"{k}={r[k]}")

        log(
            f"{r['idx']:>3} | "
            f"{r['type']:<30} | "
            f"{r['params'] / 1e6:>9.3f} | "
            f"{str(r.get('f', '')):<18} | "
            f"{', '.join(extras)}"
        )

    if layer_rows:
        last_idx = layer_rows[-1]["idx"]
        log("")
        log("[ROUGH STRUCTURE NOTE]")
        log("  Old branch assumption was roughly:")
        log("    backbone: layers 0-9")
        log("    neck:     layers 10-21")
        log(f"    head:     last layer, here layer {last_idx}")
        log("  In this clean branch, confirm by this table before interpreting freeze.")


def find_state_dict_prompt_keys(inner_model):
    """
    Search state_dict keys related to text/prompt/PE.
    """
    out = []

    if inner_model is None or not hasattr(inner_model, "state_dict"):
        return out

    try:
        sd = inner_model.state_dict()
    except Exception as e:
        return [{"error": repr(e)}]

    keywords = ["pe", "text", "txt", "prompt", "embed", "clip", "savpe", "reprta"]

    for k, v in sd.items():
        lk = k.lower()
        if any(x in lk for x in keywords):
            item = {"key": k}
            if torch.is_tensor(v):
                item["shape"] = list(v.shape)
                item["dtype"] = str(v.dtype)
            else:
                item["shape"] = None
                item["dtype"] = None
            out.append(item)

    return out


# ======================================================================================
# YOLOE prompt / class names 检查
# ======================================================================================

def try_make_text_embeddings(yolo_obj, inner_model, class_names):
    """
    Build text embeddings for class_names using the exposed YOLOE APIs.
    Returns:
      embeddings, success_record, all_attempts
    """
    attempts = []

    call_specs = [
        ("YOLO wrapper", yolo_obj, "get_text_pe(names)", (class_names,), {}),
        ("inner model", inner_model, "get_text_pe(names)", (class_names,), {}),
        ("YOLO wrapper", yolo_obj, "get_text_pe(names, cache_clip_model=False)", (class_names,), {"cache_clip_model": False}),
        ("inner model", inner_model, "get_text_pe(names, cache_clip_model=False)", (class_names,), {"cache_clip_model": False}),
    ]

    for label, obj, mode, args, kwargs in call_specs:
        if obj is None:
            attempts.append({
                "target": label,
                "has_get_text_pe": False,
                "mode": mode,
                "ok": False,
                "error": "target object is None",
            })
            continue

        fn = safe_getattr(obj, "get_text_pe", None)

        if not callable(fn):
            attempts.append({
                "target": label,
                "has_get_text_pe": False,
                "mode": mode,
                "ok": False,
                "error": "get_text_pe not found",
            })
            continue

        sig = safe_signature(fn)

        try:
            embeddings = fn(*args, **kwargs)
            record = {
                "target": label,
                "has_get_text_pe": True,
                "signature": sig,
                "mode": mode,
                "ok": True,
                "return_type": obj_type(embeddings),
                "summary": summarize_value(embeddings),
                "embedding_source": f"{label}.{mode}",
                "embedding_shape": shape_of(embeddings),
            }
            attempts.append(record)
            return embeddings, record, attempts
        except Exception as e:
            attempts.append({
                "target": label,
                "has_get_text_pe": True,
                "signature": sig,
                "mode": mode,
                "ok": False,
                "error": repr(e),
                "traceback": traceback.format_exc(),
            })

    return None, None, attempts


def try_call_set_classes(yolo_obj, inner_model, class_names):
    """
    Build text embeddings, then try set_classes on the wrapper first, then inner model.
    This only changes the in-memory model object and does not modify best.pt.
    """
    attempts = []
    embeddings, embedding_record, embedding_attempts = try_make_text_embeddings(yolo_obj, inner_model, class_names)

    embedding_source = None
    embedding_shape = None

    if embedding_record is not None:
        embedding_source = embedding_record.get("embedding_source")
        embedding_shape = embedding_record.get("embedding_shape")

    candidates = [
        ("YOLO wrapper", yolo_obj),
        ("inner model", inner_model),
    ]

    for label, obj in candidates:
        if obj is None:
            continue

        fn = safe_getattr(obj, "set_classes", None)

        if not callable(fn):
            attempts.append({
                "target": label,
                "has_set_classes": False,
                "signature": None,
                "mode": None,
                "embedding_source": embedding_source,
                "embedding_shape": embedding_shape,
                "ok": False,
                "error": "set_classes not found",
            })
            continue

        sig = safe_signature(fn)

        call_modes = []
        if embeddings is not None:
            call_modes.append(("set_classes(names, embeddings)", (class_names, embeddings)))
        call_modes.append(("set_classes(names)", (class_names,)))

        for mode, args in call_modes:
            try:
                ret = fn(*args)
                attempts.append({
                    "target": label,
                    "has_set_classes": True,
                    "signature": sig,
                    "mode": mode,
                    "embedding_source": embedding_source,
                    "embedding_shape": embedding_shape,
                    "ok": True,
                    "return_type": obj_type(ret),
                })
                return True, attempts, {
                    "ok": embedding_record is not None,
                    "embedding_source": embedding_source,
                    "embedding_shape": embedding_shape,
                    "attempts": embedding_attempts,
                }
            except Exception as e:
                attempts.append({
                    "target": label,
                    "has_set_classes": True,
                    "signature": sig,
                    "mode": mode,
                    "embedding_source": embedding_source,
                    "embedding_shape": embedding_shape,
                    "ok": False,
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                })

    return False, attempts, {
        "ok": embedding_record is not None,
        "embedding_source": embedding_source,
        "embedding_shape": embedding_shape,
        "attempts": embedding_attempts,
    }


def try_get_text_pe(yolo_obj, inner_model, class_names):
    """
    Probe get_text_pe safely.
    This is only a probe/print helper. set_classes builds its own embeddings.
    """
    attempts = []

    candidates = [
        ("YOLO wrapper", yolo_obj),
        ("inner model", inner_model),
    ]

    for label, obj in candidates:
        if obj is None:
            continue

        fn = safe_getattr(obj, "get_text_pe", None)

        if not callable(fn):
            attempts.append({
                "target": label,
                "has_get_text_pe": False,
                "ok": False,
                "error": "get_text_pe not found",
            })
            continue

        sig = safe_signature(fn)

        call_modes = [
            ("get_text_pe(names)", (class_names,), {}),
            ("get_text_pe(names, cache_clip_model=False)", (class_names,), {"cache_clip_model": False}),
            ("get_text_pe()", tuple(), {}),
        ]

        for mode, args, kwargs in call_modes:
            try:
                out = fn(*args, **kwargs)
                item = {
                    "target": label,
                    "has_get_text_pe": True,
                    "signature": sig,
                    "mode": mode,
                    "ok": True,
                    "return_type": obj_type(out),
                    "summary": summarize_value(out),
                    "embedding_shape": shape_of(out),
                }
                attempts.append(item)
                return attempts
            except Exception as e:
                attempts.append({
                    "target": label,
                    "has_get_text_pe": True,
                    "signature": sig,
                    "mode": mode,
                    "ok": False,
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                })

    return attempts


def get_names_from_wrapper_or_inner(yolo_obj, inner_model):
    return {
        "inner.names": to_plain_names(safe_getattr(inner_model, "names", None)),
        "wrapper.names": to_plain_names(safe_getattr(yolo_obj, "names", None)),
    }


def preferred_names(names_info):
    if not isinstance(names_info, dict):
        return names_info

    for key in ["inner.names", "wrapper.names"]:
        value = names_info.get(key)
        if value is not None:
            return value

    return None


def names_match_class_names(names_info, class_names):
    wanted = {i: str(v) for i, v in enumerate(class_names)}

    if isinstance(names_info, dict):
        for value in names_info.values():
            if value == wanted:
                return True

    return names_info == wanted


def get_task_candidates(yolo_obj, inner_model):
    inner_args = safe_getattr(inner_model, "args", None)

    if isinstance(inner_args, dict):
        inner_args_task = inner_args.get("task", None)
    else:
        inner_args_task = safe_getattr(inner_args, "task", None)

    return {
        "yolo.task": str(safe_getattr(yolo_obj, "task", None)),
        "inner.task": str(safe_getattr(inner_model, "task", None)),
        "inner.args.task": str(inner_args_task),
    }


# ======================================================================================
# raw checkpoint 检查
# ======================================================================================

def safe_torch_load(weights: Path):
    """
    兼容不同 PyTorch 版本。
    PyTorch 2.6+ 默认 weights_only=True 可能影响反序列化，这里显式尝试 weights_only=False。
    """
    try:
        return torch.load(str(weights), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(weights), map_location="cpu")


def inspect_raw_checkpoint(weights: Path):
    """
    Fallback / additional check:
    raw torch.load can show what is inside the checkpoint even if YOLO wrapper has issues.
    """
    log("")
    log("[RAW CHECKPOINT CHECK]")
    info = {}

    try:
        ckpt = safe_torch_load(weights)
        info["ckpt_type"] = obj_type(ckpt)

        if isinstance(ckpt, dict):
            info["top_level_keys"] = list(ckpt.keys())

            for key in ["model", "ema"]:
                m = ckpt.get(key, None)
                if m is not None:
                    info[f"{key}_type"] = obj_type(m)
                    info[f"{key}_names"] = to_plain_names(safe_getattr(m, "names", None))
                    info[f"{key}_args"] = str(safe_getattr(m, "args", None))[:1200]

                    layers = get_layers(m)
                    info[f"{key}_num_layers"] = len(layers)

                    if layers:
                        info[f"{key}_last_layer_type"] = obj_type(layers[-1])
                        info[f"{key}_last_layer_class"] = layers[-1].__class__.__name__
        else:
            info["repr"] = repr(ckpt)[:1200]

        log(json.dumps(info, ensure_ascii=False, indent=2))

    except Exception as e:
        info["error"] = repr(e)
        info["traceback"] = traceback.format_exc(limit=10)
        log("  raw torch.load failed:")
        log(json.dumps(info, ensure_ascii=False, indent=2))

    return info


# ======================================================================================
# 输出保存
# ======================================================================================

def write_compact_txt(report, layer_rows, save_txt: Path):
    lines = []

    lines.append("YOLOE best.pt inspection summary")
    lines.append("=" * 100)
    lines.append(f"time: {report.get('time')}")
    lines.append(f"checkpoint: {report.get('checkpoint_path')}")
    lines.append(f"repo root: {report.get('repo_root')}")
    lines.append("")
    lines.append(f"ultralytics imported from: {report.get('ultralytics_imported_from')}")
    lines.append(f"YOLO class: {report.get('yolo_class')}")
    lines.append("")
    lines.append(f"model class: {report.get('inner_model_type')}")
    lines.append(f"wrapper type: {report.get('wrapper_type')}")
    lines.append(f"inner model type: {report.get('inner_model_type')}")
    lines.append(f"task before: {report.get('task_before')}")
    lines.append(f"task after: {report.get('task_after')}")
    lines.append("")
    lines.append(f"names before: {report.get('names_before_preferred')}")
    lines.append(f"get_text_pe global4 ok: {report.get('get_text_pe_global4_ok')}")
    lines.append(f"global4 embedding shape: {report.get('global4_embedding_shape')}")
    lines.append(f"set_classes global4 ok: {report.get('set_classes_ok')}")
    lines.append(f"names after: {report.get('names_after_preferred')}")
    lines.append("")
    lines.append(f"last layer: {report.get('last_layer_type')}")
    lines.append(f"last layer type: {report.get('last_layer_type')}")
    lines.append(f"last layer class: {report.get('last_layer_class_name')}")
    lines.append(f"last_layer.nc before: {report.get('last_layer_nc_before')}")
    lines.append(f"last_layer.nc after: {report.get('last_layer_nc_after')}")
    lines.append(f"is YOLOESegModel guess: {report.get('is_yoloe_seg_model_guess')}")
    lines.append(f"is YOLOESegment head guess: {report.get('is_yoloe_segment_head_guess')}")
    lines.append("")
    lines.append(f"params: {report.get('params_total'):,} ({report.get('params_total_M'):.3f} M)")
    lines.append(f"total params: {report.get('params_total'):,} ({report.get('params_total_M'):.3f} M)")
    lines.append(f"trainable params: {report.get('params_trainable'):,} ({report.get('params_trainable_M'):.3f} M)")
    lines.append("")
    lines.append(f"has set_classes: {report.get('has_set_classes')}")
    lines.append(f"has get_text_pe: {report.get('has_get_text_pe')}")
    lines.append("")
    lines.append("set_classes attempts:")
    lines.append(json.dumps(report.get("set_classes_attempts"), ensure_ascii=False, indent=2))
    lines.append("")
    lines.append("get_text_pe attempts:")
    lines.append(json.dumps(report.get("get_text_pe_attempts"), ensure_ascii=False, indent=2))
    lines.append("")
    lines.append("Layer summary:")
    lines.append(f"{'idx':>3} | {'type':<30} | {'params(M)':>9} | {'from':<18} | extra")
    lines.append("-" * 120)

    for r in layer_rows:
        extras = []
        for k in ["stride", "nc", "nl", "reg_max", "no", "nm", "npr"]:
            if k in r:
                extras.append(f"{k}={r[k]}")

        lines.append(
            f"{r['idx']:>3} | "
            f"{r['type']:<30} | "
            f"{r['params'] / 1e6:>9.3f} | "
            f"{str(r.get('f', '')):<18} | "
            f"{', '.join(extras)}"
        )

    save_txt.parent.mkdir(parents=True, exist_ok=True)
    save_txt.write_text("\n".join(lines), encoding="utf-8")


# ======================================================================================
# main
# ======================================================================================

def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    setup_local_ultralytics_import(REPO_ROOT)

    weights = resolve_weights_path(WEIGHTS_PATH)
    class_names = list(GLOBAL4_CLASS_NAMES)

    log("")
    log("=" * 120)
    log("[YOLOE BEST.PT INSPECTION]")
    log(f"time:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"repo root:   {REPO_ROOT.resolve()}")
    log(f"checkpoint:  {weights}")
    log(f"class names: {class_names}")
    log("=" * 120)

    report = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "repo_root": REPO_ROOT.resolve().as_posix(),
        "checkpoint_path": weights.as_posix(),
        "class_names_for_set_classes": class_names,
    }

    # ----------------------------------------------------------------------------------
    # Import local ultralytics
    # ----------------------------------------------------------------------------------
    try:
        log("")
        log("[IMPORT CHECK]")
        from ultralytics import YOLO
        import ultralytics as ultralytics_pkg

        report["ultralytics_imported_from"] = str(getattr(ultralytics_pkg, "__file__", "UNKNOWN"))
        report["yolo_class"] = str(YOLO)

        log(f"  imported ultralytics from: {report['ultralytics_imported_from']}")
        log(f"  YOLO class: {report['yolo_class']}")

    except Exception:
        log("")
        log("[ERROR] Failed to import local ultralytics.YOLO")
        log("Expected local package path:")
        log(f"  {REPO_ROOT / 'ultralytics' / 'ultralytics' / '__init__.py'}")
        log("")
        log("Current sys.path first entries:")
        for x in sys.path[:10]:
            log(f"  {x}")
        traceback.print_exc()
        raise

    # ----------------------------------------------------------------------------------
    # Raw checkpoint check
    # ----------------------------------------------------------------------------------
    report["raw_checkpoint"] = inspect_raw_checkpoint(weights)

    # ----------------------------------------------------------------------------------
    # Load YOLO wrapper
    # ----------------------------------------------------------------------------------
    try:
        log("")
        log("[1] Loading YOLO wrapper...")
        yolo = YOLO(str(weights))
        log("[1] YOLO wrapper loaded successfully.")
    except Exception:
        log("")
        log("[ERROR] YOLO(str(weights)) failed.")
        traceback.print_exc()
        raise

    inner = safe_getattr(yolo, "model", None)

    report["wrapper_type"] = obj_type(yolo)
    report["inner_model_type"] = obj_type(inner)

    log("")
    log("[2] Wrapper / inner model type")
    log(f"YOLO wrapper type: {report['wrapper_type']}")
    log(f"inner model type:  {report['inner_model_type']}")

    # ----------------------------------------------------------------------------------
    # Basic info before set_classes
    # ----------------------------------------------------------------------------------
    log("")
    log("[3] Basic info before set_classes")

    task_before = get_task_candidates(yolo, inner)
    names_before = get_names_from_wrapper_or_inner(yolo, inner)

    report["task_before"] = task_before
    report["names_before"] = names_before
    report["names_before_preferred"] = preferred_names(names_before)

    log(f"task before:  {json.dumps(task_before, ensure_ascii=False)}")
    log(f"names before: {json.dumps(names_before, ensure_ascii=False)}")

    total_params, trainable_params = count_params(inner)

    report["params_total"] = total_params
    report["params_trainable"] = trainable_params
    report["params_total_M"] = total_params / 1e6
    report["params_trainable_M"] = trainable_params / 1e6

    log(f"params total:     {total_params:,} ({total_params / 1e6:.3f} M)")
    log(f"params trainable: {trainable_params:,} ({trainable_params / 1e6:.3f} M)")

    layers = get_layers(inner)
    last_layer = layers[-1] if layers else None

    report["num_layers"] = len(layers)
    report["last_layer_type"] = obj_type(last_layer)
    report["last_layer_class_name"] = last_layer.__class__.__name__ if last_layer is not None else None
    report["last_layer_state_before"] = summarize_last_layer_state(last_layer)
    report["last_layer_nc_before"] = plain_attr_value(safe_getattr(last_layer, "nc", None))

    log(f"num layers:       {len(layers)}")
    log(f"last layer type:  {report['last_layer_type']}")
    log(f"last_layer.nc before set_classes: {report['last_layer_nc_before']}")

    # ----------------------------------------------------------------------------------
    # YOLOE prompt capability
    # ----------------------------------------------------------------------------------
    log("")
    log("[4] YOLOE prompt capability")

    report["has_set_classes"] = {
        "wrapper": has_callable(yolo, "set_classes"),
        "inner_model": has_callable(inner, "set_classes"),
    }

    report["has_get_text_pe"] = {
        "wrapper": has_callable(yolo, "get_text_pe"),
        "inner_model": has_callable(inner, "get_text_pe"),
    }

    log(f"has set_classes: {report['has_set_classes']}")
    log(f"has get_text_pe: {report['has_get_text_pe']}")

    # ----------------------------------------------------------------------------------
    # Prompt attrs before
    # ----------------------------------------------------------------------------------
    log("")
    log("[5] Text/prompt-like attributes before set_classes")

    prompt_attrs_before = {
        "wrapper": find_attr_tensors(yolo),
        "inner_model": find_attr_tensors(inner),
        "last_layer": find_attr_tensors(last_layer),
    }

    report["prompt_attrs_before"] = prompt_attrs_before
    log(json.dumps(prompt_attrs_before, ensure_ascii=False, indent=2))

    # ----------------------------------------------------------------------------------
    # State dict prompt keys
    # ----------------------------------------------------------------------------------
    log("")
    log("[6] State dict keys related to text/prompt/PE")

    prompt_keys = find_state_dict_prompt_keys(inner)
    report["state_dict_prompt_keys"] = prompt_keys

    if prompt_keys:
        for x in prompt_keys[:150]:
            log(f"  {x.get('key')}: shape={x.get('shape')} dtype={x.get('dtype')}")
        if len(prompt_keys) > 150:
            log(f"  ... {len(prompt_keys) - 150} more keys omitted")
    else:
        log("  No obvious prompt/text/PE keys found in state_dict.")

    # ----------------------------------------------------------------------------------
    # set_classes
    # ----------------------------------------------------------------------------------
    log("")
    log("[7] Try set_classes(global4)")

    ok_set, set_attempts, set_embedding = try_call_set_classes(yolo, inner, class_names)

    report["set_classes_ok"] = ok_set
    report["set_classes_attempts"] = set_attempts
    report["set_classes_embedding"] = set_embedding
    report["get_text_pe_global4_ok"] = bool(set_embedding.get("ok")) if isinstance(set_embedding, dict) else False
    report["global4_embedding_shape"] = set_embedding.get("embedding_shape") if isinstance(set_embedding, dict) else None

    log("text embedding used by set_classes:")
    log(json.dumps(set_embedding, ensure_ascii=False, indent=2))
    log("set_classes attempts:")
    log(json.dumps(set_attempts, ensure_ascii=False, indent=2))

    # ----------------------------------------------------------------------------------
    # Basic after set_classes
    # ----------------------------------------------------------------------------------
    log("")
    log("[8] Basic info after set_classes")

    task_after = get_task_candidates(yolo, inner)
    names_after = get_names_from_wrapper_or_inner(yolo, inner)

    report["task_after"] = task_after
    report["names_after"] = names_after
    report["names_after_preferred"] = preferred_names(names_after)
    report["names_after_global4_ok"] = names_match_class_names(names_after, class_names)

    log(f"task after:  {json.dumps(task_after, ensure_ascii=False)}")
    log(f"names after: {json.dumps(names_after, ensure_ascii=False)}")

    # ----------------------------------------------------------------------------------
    # Global4 state verification after set_classes
    # ----------------------------------------------------------------------------------
    log("")
    log("[8.1] Global4 state verification after set_classes")

    layers_after_set_classes = get_layers(inner)
    last_layer_after_set_classes = layers_after_set_classes[-1] if layers_after_set_classes else None
    last_layer_state_after = summarize_last_layer_state(last_layer_after_set_classes)

    report["last_layer_state_after_set_classes"] = last_layer_state_after
    report["last_layer_nc_after"] = (
        last_layer_state_after.get("nc") if isinstance(last_layer_state_after, dict) else None
    )

    global4_state = {
        "names_after": names_after,
        "names_after_preferred": preferred_names(names_after),
        "names_after_global4_ok": report["names_after_global4_ok"],
        "set_classes_global4_ok": ok_set,
        "get_text_pe_global4_ok": report["get_text_pe_global4_ok"],
        "global4_embedding_shape": report["global4_embedding_shape"],
        "last_layer_nc_before": report["last_layer_nc_before"],
        "last_layer_state_after": last_layer_state_after,
    }
    report["global4_state_after_set_classes"] = global4_state

    log(json.dumps(global4_state, ensure_ascii=False, indent=2))

    nc_after = report["last_layer_nc_after"]
    if str(nc_after) == "4":
        log("OK: last_layer.nc is 4 after set_classes.")
    elif str(nc_after) == "3" and report["names_after_global4_ok"]:
        log("WARNING: names updated to 4 classes, but last_layer.nc is still 3.")
        log("This may mean set_classes only updates prompt/name state, not head nc attribute, or layer attrs are stale.")
        log("Do not force detect adaptation here.")
    else:
        log(f"NOTE: last_layer.nc after set_classes is {nc_after}. See JSON for full state.")

    # ----------------------------------------------------------------------------------
    # get_text_pe
    # ----------------------------------------------------------------------------------
    log("")
    log("[9] Try get_text_pe")

    get_text_pe_attempts = try_get_text_pe(yolo, inner, class_names)

    report["get_text_pe_attempts"] = get_text_pe_attempts

    log(json.dumps(get_text_pe_attempts, ensure_ascii=False, indent=2))

    # ----------------------------------------------------------------------------------
    # Prompt attrs after
    # ----------------------------------------------------------------------------------
    log("")
    log("[10] Text/prompt-like attributes after set_classes")

    prompt_attrs_after = {
        "wrapper": find_attr_tensors(yolo),
        "inner_model": find_attr_tensors(inner),
        "last_layer": find_attr_tensors(last_layer_after_set_classes),
    }

    report["prompt_attrs_after"] = prompt_attrs_after

    log(json.dumps(prompt_attrs_after, ensure_ascii=False, indent=2))

    # ----------------------------------------------------------------------------------
    # Layer summary
    # ----------------------------------------------------------------------------------
    log("")
    log("[11] Layer summary")

    layer_rows = summarize_layers(inner)
    report["layers"] = layer_rows

    print_layer_table(layer_rows)

    report["is_yoloe_seg_model_guess"] = "YOLOESegModel" in report["inner_model_type"]
    report["is_yoloe_segment_head_guess"] = (
        report["last_layer_class_name"] == "YOLOESegment"
        or "YOLOESegment" in report["last_layer_type"]
    )

    # ----------------------------------------------------------------------------------
    # Key checks
    # ----------------------------------------------------------------------------------
    log("")
    log("=" * 120)
    log("[KEY CHECKS]")
    log(f"model class is YOLOESegModel?  {report['is_yoloe_seg_model_guess']}")
    log(f"last layer is YOLOESegment?    {report['is_yoloe_segment_head_guess']}")
    log(f"task before:                   {report['task_before']}")
    log(f"task after:                    {report['task_after']}")
    log(f"names before:                  {report['names_before_preferred']}")
    log(f"get_text_pe global4 ok?         {report['get_text_pe_global4_ok']}")
    log(f"global4 embedding shape:        {report['global4_embedding_shape']}")
    log(f"set_classes global4 ok?         {report['set_classes_ok']}")
    log(f"names after:                   {report['names_after_preferred']}")
    log(f"last_layer.nc before:          {report['last_layer_nc_before']}")
    log(f"last_layer.nc after:           {report['last_layer_nc_after']}")
    log(f"total params:                  {report['params_total_M']:.3f} M")
    log("=" * 120)

    # ----------------------------------------------------------------------------------
    # Save reports
    # ----------------------------------------------------------------------------------
    SAVE_JSON.parent.mkdir(parents=True, exist_ok=True)
    SAVE_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    write_compact_txt(report, layer_rows, SAVE_TXT)

    log("")
    log("[OK] JSON report saved to:")
    log(f"  {SAVE_JSON}")
    log("[OK] TXT summary saved to:")
    log(f"  {SAVE_TXT}")

    log("")
    log("Next:")
    log("  1. Paste [KEY CHECKS].")
    log("  2. Paste [LAYER SUMMARY].")
    log("  3. If set_classes failed, paste [7] Try set_classes(global4).")


if __name__ == "__main__":
    try:
        main()
    finally:
        if PAUSE_AT_END:
            input("\nPress Enter to exit...")
