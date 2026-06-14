# tools/eval/probe_yoloeseg_detect_like_val.py
# -*- coding: utf-8 -*-

"""
Minimal YOLOESegModel validation probe.

Purpose:
1. Load upstream best.pt as-is.
2. Keep YOLOESegModel / YOLOESegment structure unchanged.
3. Initialize YOLOE global4 fixed-text prompts.
4. Try native model.val() on 5-column YOLO detect labels and print whether box metrics are available.

This script is intentionally diagnostic only:
- It does not train.
- It does not modify or save best.pt.
- It does not convert YOLOESegModel to detect.
- It does not replace YOLOESegment.
"""

from __future__ import annotations

import inspect
import json
import sys
import tempfile
import traceback
from pathlib import Path

import torch
import yaml


REPO_ROOT = Path(r"C:\Users\DOCTOR\Documents\GitHub\YOLOv8")
PACKAGE_ROOT = REPO_ROOT / "ultralytics"
WEIGHTS_PATH = REPO_ROOT / "ultralytics" / "best.pt"

GLOBAL4_CLASS_NAMES = ["ship", "harbor", "tank", "bridge"]

# User-requested path. Current clean repo may store these YAMLs directly under configs/.
REQUESTED_DATA_YAML = REPO_ROOT / "configs" / "datasets" / "global4_eval_ship.yaml"
DATA_YAML_FALLBACKS = [
    REQUESTED_DATA_YAML,
    REPO_ROOT / "configs" / "global4_eval_ship.yaml",
]

SHIP_SMALL_SPLIT_VAL_IMAGES = REPO_ROOT / "ultralytics" / "datasets" / "ship_small_split" / "images" / "val"

YAML_PRESETS = {
    "ship": [
        REPO_ROOT / "configs" / "datasets" / "global4_eval_ship.yaml",
        REPO_ROOT / "configs" / "global4_eval_ship.yaml",
    ],
    "bridge_rgb": [
        REPO_ROOT / "configs" / "datasets" / "global4_eval_bridge_rgb.yaml",
        REPO_ROOT / "configs" / "global4_eval_bridge_rgb.yaml",
    ],
    "bridge_sar": [
        REPO_ROOT / "configs" / "datasets" / "global4_eval_bridge_sar.yaml",
        REPO_ROOT / "configs" / "global4_eval_bridge_sar.yaml",
    ],
    "bridge_infr": [
        REPO_ROOT / "configs" / "datasets" / "global4_eval_bridge_infr.yaml",
        REPO_ROOT / "configs" / "global4_eval_bridge_infr.yaml",
    ],
    "bridge_all": [
        REPO_ROOT / "configs" / "datasets" / "global4_eval_bridge_all.yaml",
        REPO_ROOT / "configs" / "global4_eval_bridge_all.yaml",
    ],
}


def log(msg: object = "") -> None:
    print(msg, flush=True)


def setup_local_ultralytics_import() -> None:
    repo_root = REPO_ROOT.resolve()
    package_root = PACKAGE_ROOT.resolve()

    for p in [str(package_root), str(repo_root)]:
        while p in sys.path:
            sys.path.remove(p)

    sys.path.insert(0, str(package_root))
    sys.path.insert(1, str(repo_root))


def safe_getattr(obj: object, name: str, default=None):
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def obj_type(obj: object) -> str:
    if obj is None:
        return "None"
    return f"{obj.__class__.__module__}.{obj.__class__.__name__}"


def safe_signature(fn: object) -> str:
    try:
        return str(inspect.signature(fn))
    except Exception:
        return "signature unavailable"


def shape_of(x: object):
    if x is None:
        return None
    if torch.is_tensor(x):
        return list(x.shape)
    if isinstance(x, torch.nn.Parameter):
        return list(x.shape)
    if hasattr(x, "shape"):
        try:
            return list(x.shape)
        except Exception:
            return str(x.shape)
    return None


def summarize_tensor(x: object):
    if torch.is_tensor(x):
        return {
            "type": "torch.Tensor",
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "device": str(x.device),
        }
    if isinstance(x, torch.nn.Parameter):
        return {
            "type": "torch.nn.Parameter",
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "device": str(x.device),
        }
    return {
        "type": obj_type(x),
        "shape": shape_of(x),
        "repr": repr(x)[:500],
    }


def to_plain_names(names):
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


def get_layers(inner_model):
    layers = safe_getattr(inner_model, "model", None)
    if layers is None:
        return []
    try:
        return list(layers)
    except Exception:
        return []


def get_last_layer(inner_model):
    layers = get_layers(inner_model)
    return layers[-1] if layers else None


def get_names_state(yolo_obj, inner_model):
    return {
        "wrapper.names": to_plain_names(safe_getattr(yolo_obj, "names", None)),
        "inner.names": to_plain_names(safe_getattr(inner_model, "names", None)),
    }


def get_task_state(yolo_obj, inner_model):
    inner_args = safe_getattr(inner_model, "args", None)
    if isinstance(inner_args, dict):
        inner_args_task = inner_args.get("task")
    else:
        inner_args_task = safe_getattr(inner_args, "task", None)
    return {
        "wrapper.task": safe_getattr(yolo_obj, "task", None),
        "inner.task": safe_getattr(inner_model, "task", None),
        "inner.args.task": inner_args_task,
    }


def head_state(inner_model):
    last_layer = get_last_layer(inner_model)
    return {
        "last_layer_type": obj_type(last_layer),
        "last_layer_class": last_layer.__class__.__name__ if last_layer is not None else None,
        "last_layer.nc": safe_getattr(last_layer, "nc", None),
        "last_layer.no": safe_getattr(last_layer, "no", None),
        "last_layer.nl": safe_getattr(last_layer, "nl", None),
        "last_layer.reg_max": safe_getattr(last_layer, "reg_max", None),
        "last_layer.embed": safe_getattr(last_layer, "embed", None),
    }


def pe_state(inner_model):
    pe = safe_getattr(inner_model, "pe", None)
    if pe is None:
        return None
    return summarize_tensor(pe)


def resolve_data_yaml() -> Path:
    log("DATA_YAML candidates:")
    for p in DATA_YAML_FALLBACKS:
        log(f"  {p} exists={p.exists()}")
    for p in DATA_YAML_FALLBACKS:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(
        "No DATA_YAML candidate exists. Checked:\n"
        + "\n".join(f"  {p}" for p in DATA_YAML_FALLBACKS)
    )


def materialize_val_yaml_for_ultralytics(data_yaml: Path):
    """
    Ultralytics check_det_dataset() requires both train and val keys even for val-only use.
    If the source YAML is val-only, create a temporary YAML with train=val.
    The original YAML and dataset are not modified.
    """
    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if "val" not in data:
        raise SyntaxError(f"{data_yaml} has no 'val:' key.")

    patched = dict(data)
    notes = []
    val_paths = data["val"] if isinstance(data["val"], list) else [data["val"]]
    val_paths_exist = all(Path(str(p)).exists() for p in val_paths)

    if not val_paths_exist and SHIP_SMALL_SPLIT_VAL_IMAGES.exists():
        patched["val"] = [SHIP_SMALL_SPLIT_VAL_IMAGES.as_posix()]
        notes.append(
            f"source YAML val path is missing; using existing ship_small_split val path: {SHIP_SMALL_SPLIT_VAL_IMAGES}"
        )

    if "train" not in patched:
        patched["train"] = patched["val"]
        notes.append("source YAML is val-only; temporary YAML sets train=val for Ultralytics check_det_dataset().")

    train_paths = patched["train"] if isinstance(patched["train"], list) else [patched["train"]]
    patched_paths_exist = all(Path(str(p)).exists() for p in train_paths)
    patched_paths_exist = patched_paths_exist and all(
        Path(str(p)).exists() for p in (patched["val"] if isinstance(patched["val"], list) else [patched["val"]])
    )

    if not notes and patched_paths_exist and "train" in data and "val" in data:
        return data_yaml, None, data, notes

    tmp_dir = tempfile.TemporaryDirectory(prefix="yoloeseg_val_probe_")
    tmp_yaml = Path(tmp_dir.name) / data_yaml.name
    with tmp_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(patched, f, sort_keys=False, allow_unicode=True)

    return tmp_yaml, tmp_dir, patched, notes


def print_yaml_presets() -> None:
    log("YAML presets:")
    for name, paths in YAML_PRESETS.items():
        first_existing = next((p for p in paths if p.exists()), None)
        log(f"  {name}: {first_existing or paths[0]}")


def print_data_path_preflight(data: dict) -> None:
    log("DATA_YAML path preflight:")
    for key in ["train", "val"]:
        value = data.get(key)
        paths = value if isinstance(value, list) else [value]
        for p in paths:
            if p is None:
                log(f"  {key}: None")
                continue
            path = Path(str(p))
            log(f"  {key}: {path} exists={path.exists()}")


def set_global4_classes(yolo_obj, inner_model):
    names = list(GLOBAL4_CLASS_NAMES)

    get_text_pe = safe_getattr(yolo_obj, "get_text_pe", None)
    if not callable(get_text_pe):
        get_text_pe = safe_getattr(inner_model, "get_text_pe", None)

    if not callable(get_text_pe):
        raise AttributeError("Neither YOLO wrapper nor inner model has callable get_text_pe")

    log(f"get_text_pe signature: {safe_signature(get_text_pe)}")
    embeddings = get_text_pe(names)
    log(f"embedding summary: {json.dumps(summarize_tensor(embeddings), ensure_ascii=False)}")

    set_classes = safe_getattr(yolo_obj, "set_classes", None)
    set_target = "YOLO wrapper"
    if not callable(set_classes):
        set_classes = safe_getattr(inner_model, "set_classes", None)
        set_target = "inner model"

    if not callable(set_classes):
        raise AttributeError("Neither YOLO wrapper nor inner model has callable set_classes")

    log(f"set_classes target: {set_target}")
    log(f"set_classes signature: {safe_signature(set_classes)}")
    set_classes(names, embeddings)
    return embeddings, set_target


def print_metric_attr(metrics, attr_path: str) -> None:
    obj = metrics
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)
    log(f"{attr_path}: {obj}")


def print_metrics_summary(metrics) -> None:
    log("")
    log("[METRICS SUMMARY]")
    log(f"metrics object: {metrics}")
    log(f"metrics type: {obj_type(metrics)}")
    try:
        log(f"dir(metrics): {dir(metrics)}")
    except Exception as e:
        log(f"dir(metrics) failed: {repr(e)}")

    for attr in ["results_dict", "box"]:
        try:
            log(f"metrics.{attr}: {getattr(metrics, attr)}")
        except Exception as e:
            log(f"metrics.{attr}: ERROR {repr(e)}")

    for attr_path in ["box.map50", "box.map", "box.mp", "box.mr", "box.maps"]:
        try:
            print_metric_attr(metrics, attr_path)
        except Exception as e:
            log(f"metrics.{attr_path}: ERROR {repr(e)}")


def classify_val_error(exc: BaseException, tb_text: str) -> str:
    text = f"{repr(exc)}\n{tb_text}".lower()

    if any(s in text for s in ["no such file", "filenotfounderror", "does not exist", "not found", "images not found"]):
        return "1. dataloader / image / label path issue"

    if any(
        s in text
        for s in [
            "masks",
            "sem_masks",
            "segments",
            "segment",
            "polygon",
            "mask_iou",
            "batch[\"masks\"]",
            "batch['masks']",
            "augment.py",
        ]
    ):
        return "3. segment validator requires mask / segment labels and cannot directly consume 5-column detect labels"

    if any(s in text for s in ["dataset", "yaml", "names", "nc", "class", "label class", "exceeds dataset class count"]):
        return "2. YAML / names / nc / class id issue"

    if any(s in text for s in ["shape", "size mismatch", "dimension", "mat1", "mat2", "no", "nc", "indexerror"]):
        return "4. output shape / head nc / no mismatch"

    if any(s in text for s in ["get_text_pe", "set_classes", "embedding", "prompt", "clip", "pe"]):
        return "5. YOLOE text embedding / prompt initialization issue"

    return "6. other"


def main() -> int:
    setup_local_ultralytics_import()

    log("[IMPORT CHECK]")
    log(f"repo root: {REPO_ROOT.resolve()}")
    log(f"package root inserted first: {PACKAGE_ROOT.resolve()}")
    log(f"expected __init__.py: {PACKAGE_ROOT / 'ultralytics' / '__init__.py'}")
    log("first sys.path entries:")
    for p in sys.path[:5]:
        log(f"  {p}")

    from ultralytics import YOLO
    import ultralytics as ultralytics_pkg

    log(f"ultralytics imported from: {getattr(ultralytics_pkg, '__file__', 'UNKNOWN')}")
    log(f"YOLO class: {YOLO}")
    print_yaml_presets()
    source_data_yaml = resolve_data_yaml()
    data_yaml, tmp_yaml_dir, data_yaml_dict, yaml_notes = materialize_val_yaml_for_ultralytics(source_data_yaml)
    log(f"selected source DATA_YAML: {source_data_yaml}")
    log(f"DATA_YAML used for model.val: {data_yaml}")
    log(f"DATA_YAML keys: {list(data_yaml_dict.keys())}")
    for note in yaml_notes:
        log(note)
    print_data_path_preflight(data_yaml_dict)

    log("")
    log("[LOAD MODEL]")
    log(f"weights: {WEIGHTS_PATH}")
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"best.pt not found: {WEIGHTS_PATH}")
    model = YOLO(str(WEIGHTS_PATH))
    inner = safe_getattr(model, "model", None)
    log(f"YOLO wrapper type: {obj_type(model)}")
    log(f"inner model type: {obj_type(inner)}")

    log("")
    log("[BEFORE SET_CLASSES]")
    before_head = head_state(inner)
    log(f"task: {json.dumps(get_task_state(model, inner), ensure_ascii=False)}")
    log(f"names before: {json.dumps(get_names_state(model, inner), ensure_ascii=False)}")
    log(f"last_layer type: {before_head['last_layer_type']}")
    log(f"last_layer.nc before: {before_head['last_layer.nc']}")
    log(f"head before: {json.dumps(before_head, ensure_ascii=False, default=str)}")

    log("")
    log("[GLOBAL4 SET_CLASSES]")
    embeddings, set_target = set_global4_classes(model, inner)
    log(f"GLOBAL4 names: {GLOBAL4_CLASS_NAMES}")
    log(f"embedding shape: {shape_of(embeddings)}")
    log(f"set_classes used target: {set_target}")

    log("")
    log("[AFTER SET_CLASSES]")
    after_head = head_state(inner)
    log(f"names after: {json.dumps(get_names_state(model, inner), ensure_ascii=False)}")
    log(f"inner model pe shape if exists: {shape_of(safe_getattr(inner, 'pe', None))}")
    log(f"inner model pe summary if exists: {json.dumps(pe_state(inner), ensure_ascii=False, default=str)}")
    log(f"last_layer.nc after: {after_head['last_layer.nc']}")
    log(f"task after: {json.dumps(get_task_state(model, inner), ensure_ascii=False)}")
    log(f"last_layer type after: {after_head['last_layer_type']}")
    log(f"head after: {json.dumps(after_head, ensure_ascii=False, default=str)}")

    log("")
    log("[VAL PROBE]")
    log("calling native model.val(data=str(DATA_YAML), imgsz=640, batch=1, device='cpu', verbose=True, plots=False)")

    val_ok = False
    failure_type = None
    metrics = None
    try:
        metrics = model.val(
            data=str(data_yaml),
            imgsz=640,
            batch=1,
            device="cpu",
            verbose=True,
            plots=False,
        )
        val_ok = True
        print_metrics_summary(metrics)
    except Exception as e:
        tb_text = traceback.format_exc()
        failure_type = classify_val_error(e, tb_text)
        log("")
        log("[VAL ERROR]")
        log(f"error type: {type(e).__name__}")
        log(f"error repr: {repr(e)}")
        log(f"classified as: {failure_type}")
        log("full traceback:")
        log(tb_text)

    log("")
    log("[CONCLUSION]")
    log(f"best.pt loaded: True")
    log(f"global4 set_classes ok: True")
    log(f"native model.val ok: {val_ok}")
    log(f"model remains YOLOESegModel related: {'YOLOESegModel' in obj_type(inner)}")
    log(f"head remains YOLOESegment: {'YOLOESegment' in after_head['last_layer_type']}")
    log(f"task remains segment: {get_task_state(model, inner)}")
    log(f"last_layer.nc before -> after: {before_head['last_layer.nc']} -> {after_head['last_layer.nc']}")
    log(f"names after: {get_names_state(model, inner)}")
    log(f"embedding shape: {shape_of(embeddings)}")
    if val_ok:
        log("native model.val produced metrics; inspect [METRICS SUMMARY] for box P/R/AP fields.")
    else:
        log(f"native model.val failed type: {failure_type}")
        if failure_type and failure_type.startswith("3."):
            log(
                "Next step should be a minimal box-only eval path: take boxes/scores/classes from "
                "YOLOESegModel predictions, ignore masks, and evaluate with detect metrics."
            )
        else:
            log("Next step should address the classified failure only; do not change model structure in this probe.")

    if tmp_yaml_dir is not None:
        tmp_yaml_dir.cleanup()

    # A native val failure is a valid probe outcome when it has been classified above.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
