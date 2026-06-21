# tools/train/init_and_smoke_train_yoloe_2cls_student.py
# -*- coding: utf-8 -*-

"""
Initialize a true 2-class YOLOE student from a fresh YOLOE detect yaml, migrate
only shape-compatible weights from the original best.pt, then run a 1-epoch smoke train.

This script does not train the original best.pt directly, does not modify best.pt,
does not hard-edit segmentation loss, and does not fabricate masks.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import yaml


REPO_ROOT = Path(r"C:\Users\DOCTOR\Documents\GitHub\YOLOv8")
PACKAGE_ROOT = REPO_ROOT / "ultralytics"
DEFAULT_SOURCE_WEIGHTS = REPO_ROOT / "ultralytics" / "best.pt"
DEFAULT_MODEL_YAML = REPO_ROOT / "ultralytics" / "ultralytics" / "cfg" / "models" / "v8" / "yoloe-v8.yaml"
DEFAULT_DATA = REPO_ROOT / "ultralytics" / "datasets" / "bridge_ship_distill_2cls" / "data.yaml"
DEFAULT_INIT_DIR = REPO_ROOT / "runs" / "init" / "yoloe_2cls_detect_from_best_init" / "weights"
DEFAULT_INIT_WEIGHTS = DEFAULT_INIT_DIR / "yoloe_2cls_detect_student_init.pt"
STUDENT_NAMES = ["ship", "bridge"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def log(msg: object = "") -> None:
    print(msg, flush=True)


def setup_local_ultralytics_import() -> None:
    for p in [str(PACKAGE_ROOT.resolve()), str(REPO_ROOT.resolve())]:
        while p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, str(PACKAGE_ROOT.resolve()))
    sys.path.insert(1, str(REPO_ROOT.resolve()))


def parse_names(value: str) -> list[str]:
    names = [x.strip() for x in value.split(",") if x.strip()]
    if names != STUDENT_NAMES:
        raise ValueError(f"First smoke train only supports student names {STUDENT_NAMES}, got {names}")
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Init and smoke train a 2-class YOLOE student.")
    parser.add_argument("--source-weights", default=str(DEFAULT_SOURCE_WEIGHTS))
    parser.add_argument("--model-yaml", default=str(DEFAULT_MODEL_YAML))
    parser.add_argument("--task", choices=["detect", "segment"], default="detect")
    parser.add_argument("--student-names", default="ship,bridge")
    parser.add_argument("--init-weights", default=str(DEFAULT_INIT_WEIGHTS))
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", default=str(REPO_ROOT / "runs" / "train"))
    parser.add_argument("--name", default="smoke_yoloe_2cls_detect_distill")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--train-val", action="store_true", help="Allow Ultralytics native validation after training.")
    return parser.parse_args()


def labels_dir_from_images_dir(images_dir: Path) -> Path:
    parts = list(images_dir.parts)
    for i, part in enumerate(parts):
        if part.lower() == "images":
            parts[i] = "labels"
            return Path(*parts)
    return images_dir.parent.parent / "labels" / images_dir.name


def resolve_split_dirs(data_yaml: Path, data: dict, key: str) -> list[Path]:
    root = Path(str(data.get("path", data_yaml.parent))).expanduser()
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    values = data.get(key)
    if values is None:
        raise SyntaxError(f"{data_yaml} has no '{key}:' key.")
    values = values if isinstance(values, list) else [values]
    dirs = []
    for value in values:
        p = Path(str(value)).expanduser()
        if not p.is_absolute():
            p = root / p
        if not p.exists():
            raise FileNotFoundError(f"{key} images directory does not exist: {p}")
        if not p.is_dir():
            raise NotADirectoryError(f"{key} entry is not a directory: {p}")
        dirs.append(p.resolve())
    return dirs


def normalize_names(names_obj) -> dict[int, str]:
    if isinstance(names_obj, dict):
        return {int(k): str(v) for k, v in names_obj.items()}
    if isinstance(names_obj, list):
        return {i: str(v) for i, v in enumerate(names_obj)}
    raise TypeError(f"Unsupported names type: {type(names_obj)}")


def validate_data_yaml(data_yaml: Path) -> dict:
    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    names = normalize_names(data.get("names"))
    if int(data.get("nc", -1)) != 2 or names.get(0) != "ship" or names.get(1) != "bridge":
        raise ValueError(f"Expected nc=2 and names 0=ship, 1=bridge in {data_yaml}, got nc={data.get('nc')} names={names}")

    split_stats = {}
    class_hist = {0: 0, 1: 0}
    bad_rows = 0
    image_count = 0
    label_files = 0

    for split in ["train", "val"]:
        images_dirs = resolve_split_dirs(data_yaml, data, split)
        split_images = 0
        split_label_files = 0
        split_rows = 0
        for images_dir in images_dirs:
            labels_dir = labels_dir_from_images_dir(images_dir)
            if not labels_dir.exists():
                raise FileNotFoundError(f"labels directory does not exist for {split}: {labels_dir}")
            images = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
            split_images += len(images)
            for image_path in images:
                label_path = labels_dir / f"{image_path.stem}.txt"
                if not label_path.exists():
                    continue
                split_label_files += 1
                for row_idx, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        bad_rows += 1
                        raise ValueError(f"Non-5-column label row: {label_path}:{row_idx}: {line}")
                    try:
                        cls = int(float(parts[0]))
                        coords = [float(x) for x in parts[1:]]
                    except ValueError as exc:
                        bad_rows += 1
                        raise ValueError(f"Bad numeric label row: {label_path}:{row_idx}: {line}") from exc
                    if cls not in {0, 1}:
                        raise ValueError(f"Class id out of range for 2-class student: {label_path}:{row_idx}: {line}")
                    if any(x < 0.0 or x > 1.0 for x in coords):
                        raise ValueError(f"YOLO normalized coords out of [0,1]: {label_path}:{row_idx}: {line}")
                    class_hist[cls] += 1
                    split_rows += 1
        split_stats[split] = {
            "images": split_images,
            "label_files": split_label_files,
            "rows": split_rows,
            "image_dirs": [str(p) for p in images_dirs],
            "label_dirs": [str(labels_dir_from_images_dir(p)) for p in images_dirs],
        }
        image_count += split_images
        label_files += split_label_files

    return {
        "data_yaml": str(data_yaml),
        "names": names,
        "image_count": image_count,
        "label_files": label_files,
        "bad_rows": bad_rows,
        "class_hist": class_hist,
        "splits": split_stats,
    }


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def set_student_classes(yolo_obj, inner_model, names: list[str]):
    from tools.eval.eval_teacher_ship_external import safe_getattr, safe_signature, shape_of

    get_text_pe = safe_getattr(yolo_obj, "get_text_pe", None) or safe_getattr(inner_model, "get_text_pe", None)
    set_classes = safe_getattr(yolo_obj, "set_classes", None) or safe_getattr(inner_model, "set_classes", None)
    if not callable(get_text_pe) or not callable(set_classes):
        raise AttributeError("Student must expose get_text_pe and set_classes.")
    embeddings = get_text_pe(names)
    set_classes(names, embeddings)
    return {
        "pe_shape": shape_of(embeddings),
        "get_text_pe_signature": safe_signature(get_text_pe),
        "set_classes_signature": safe_signature(set_classes),
    }


def raw_yoloe_score_channel_forward(inner_model, x: torch.Tensor) -> int:
    from tools.eval.eval_teacher_ship_external import get_last_layer, obj_type, safe_getattr
    from ultralytics.nn.modules.head import YOLOEDetect, YOLOESegment

    y = []
    out = x
    head = get_last_layer(inner_model)
    for m in inner_model.model:
        if m.f != -1:
            out = y[m.f] if isinstance(m.f, int) else [out if j == -1 else y[j] for j in m.f]
        if m is head:
            if not isinstance(m, (YOLOEDetect, YOLOESegment)):
                raise AssertionError(f"Expected YOLOEDetect/YOLOESegment head, got {obj_type(m)}")
            feats = out
            bs = feats[0].shape[0]
            scores = []
            for i in range(m.nl):
                cls_feat = m.cv3[i](feats[i])
                if getattr(m, "is_fused", False):
                    score_i = m.cv4[i](cls_feat, None)
                else:
                    pe = safe_getattr(inner_model, "pe", None)
                    if pe is None:
                        raise RuntimeError("Non-fused YOLOE head requires inner_model.pe for text scores.")
                    pe = pe.to(device=cls_feat.device, dtype=cls_feat.dtype)
                    if pe.shape[0] != bs:
                        pe = pe.expand(bs, -1, -1)
                    score_i = m.cv4[i](cls_feat, pe)
                scores.append(score_i.reshape(bs, score_i.shape[1], -1))
            return int(torch.cat(scores, dim=-1).shape[1])
        out = m(out)
        y.append(out if m.i in inner_model.save else None)
    raise RuntimeError("YOLOE head was not reached.")


def probe_student(yolo_obj, names: list[str], imgsz: int, device: str, title: str) -> dict:
    from tools.eval.eval_teacher_ship_external import (
        get_names_state,
        get_task_state,
        head_state,
        obj_type,
        safe_getattr,
    )

    inner = safe_getattr(yolo_obj, "model", None)
    set_info = set_student_classes(yolo_obj, inner, names)
    head = head_state(inner)
    inner.eval().to(device)
    x = torch.zeros(1, 3, imgsz, imgsz, device=device)
    with torch.no_grad():
        score_nc = raw_yoloe_score_channel_forward(inner, x)
    result = {
        "title": title,
        "wrapper_type": obj_type(yolo_obj),
        "model_class": obj_type(inner),
        "task": get_task_state(yolo_obj, inner),
        "names": get_names_state(yolo_obj, inner),
        "last_layer_type": head.get("last_layer_type"),
        "last_layer.nc": head.get("last_layer.nc"),
        "last_layer.is_fused": head.get("last_layer.is_fused"),
        "pe_shape": set_info["pe_shape"],
        "score_channels_seen": [int(score_nc)],
    }
    print_probe(result)
    if result["score_channels_seen"] != [2] or result["last_layer.nc"] != 2:
        raise RuntimeError(f"{title}: expected score_channels_seen [2] and last_layer.nc 2, got {result}")
    return result


def print_probe(result: dict) -> None:
    log("")
    log(f"[{result['title']}]")
    log(f"YOLO wrapper type: {result['wrapper_type']}")
    log(f"model class: {result['model_class']}")
    log(f"task: {json.dumps(result['task'], ensure_ascii=False)}")
    log(f"names: {json.dumps(result['names'], ensure_ascii=False)}")
    log(f"last layer type: {result['last_layer_type']}")
    log(f"last_layer.nc: {result['last_layer.nc']}")
    log(f"last_layer.is_fused: {result['last_layer.is_fused']}")
    log(f"pe shape: {result['pe_shape']}")
    log(f"score channels seen: {result['score_channels_seen']}")


def migrate_compatible_weights(source_inner: torch.nn.Module, student_inner: torch.nn.Module) -> dict:
    src_sd = source_inner.float().state_dict()
    dst_sd = student_inner.state_dict()
    compatible = {}
    skipped = []
    missing = []

    for key, dst_tensor in dst_sd.items():
        src_tensor = src_sd.get(key)
        if src_tensor is None:
            missing.append(key)
            continue
        if tuple(src_tensor.shape) != tuple(dst_tensor.shape):
            skipped.append(
                {
                    "key": key,
                    "source_shape": list(src_tensor.shape),
                    "student_shape": list(dst_tensor.shape),
                }
            )
            continue
        compatible[key] = src_tensor.detach().to(dtype=dst_tensor.dtype)

    load_result = student_inner.load_state_dict(compatible, strict=False)
    source_keys = set(src_sd.keys())
    student_keys = set(dst_sd.keys())
    source_only = sorted(source_keys - student_keys)
    return {
        "source_param_count": count_params(source_inner),
        "student_param_count": count_params(student_inner),
        "source_key_count": len(src_sd),
        "student_key_count": len(dst_sd),
        "loaded_key_count": len(compatible),
        "skipped_key_count": len(skipped),
        "missing_student_key_count": len(missing),
        "source_only_key_count": len(source_only),
        "skipped_examples": skipped[:20],
        "missing_examples": missing[:20],
        "source_only_examples": source_only[:20],
        "load_missing_keys_examples": list(load_result.missing_keys)[:20],
        "load_unexpected_keys_examples": list(load_result.unexpected_keys)[:20],
    }


def save_init_checkpoint(yolo_obj, init_weights: Path, args: argparse.Namespace, names: list[str], migration: dict) -> None:
    from ultralytics import __version__

    init_weights.parent.mkdir(parents=True, exist_ok=True)
    model_copy = deepcopy(yolo_obj.model).half()
    ckpt = {
        "model": model_copy,
        "ema": None,
        "updates": None,
        "optimizer": None,
        "train_args": {
            "model": str(Path(args.model_yaml).resolve()),
            "data": str(Path(args.data).resolve()),
            "task": args.task,
            "nc": 2,
            "names": names,
        },
        "epoch": -1,
        "best_fitness": None,
        "date": datetime.now().isoformat(),
        "version": __version__,
        "license": "AGPL-3.0 License (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
        "source_weights": str(Path(args.source_weights).resolve()),
        "migration": migration,
    }
    torch.save(ckpt, init_weights)


def classify_train_error(text: str) -> str:
    s = text.lower()
    if "class" in s and ("out of" in s or "exceed" in s or "index" in s):
        return "class index / class channel mismatch"
    if "segment" in s or "mask" in s or "masks" in s:
        return "segmentation mask / segment label pipeline issue"
    if "shape" in s or "size mismatch" in s or "channels" in s:
        return "output shape / score channel mismatch"
    if "label" in s or "dataset" in s or "dataloader" in s:
        return "dataset / dataloader / label issue"
    return "other"


def find_train_weights(project: Path, name: str) -> dict:
    run_dir = project / name
    return {
        "run_dir": run_dir,
        "last": run_dir / "weights" / "last.pt",
        "best": run_dir / "weights" / "best.pt",
    }


def main() -> int:
    args = parse_args()
    if args.epochs != 1:
        raise ValueError("This smoke script is intentionally limited to --epochs 1 for the first run.")
    names = parse_names(args.student_names)
    setup_local_ultralytics_import()

    from ultralytics import YOLO
    from ultralytics.models.yolo.model import YOLOE
    from tools.eval.eval_teacher_ship_external import obj_type, safe_getattr

    source_weights = Path(args.source_weights).resolve()
    model_yaml = Path(args.model_yaml).resolve()
    init_weights = Path(args.init_weights).resolve()
    data_yaml = Path(args.data).resolve()
    project = Path(args.project).resolve()

    log("[IMPORT CHECK]")
    import ultralytics as ultralytics_pkg

    log(f"ultralytics imported from: {getattr(ultralytics_pkg, '__file__', 'UNKNOWN')}")
    log(f"first sys.path entries: {sys.path[:3]}")
    log(f"git commit: {get_git_commit()}")

    log("")
    log("[DATA CHECK]")
    data_stats = validate_data_yaml(data_yaml)
    log(json.dumps(data_stats, indent=2, ensure_ascii=False))

    log("")
    log("[LOAD SOURCE WEIGHTS]")
    log(f"source weights: {source_weights}")
    source_yolo = YOLO(str(source_weights))
    source_inner = safe_getattr(source_yolo, "model", None)
    log(f"source wrapper type: {obj_type(source_yolo)}")
    log(f"source model class: {obj_type(source_inner)}")
    log(f"source param count: {count_params(source_inner):,}")

    log("")
    log("[BUILD FRESH 2CLS STUDENT]")
    log(f"model yaml: {model_yaml}")
    log(f"student task: {args.task}")
    log(f"student names: {names}")
    student = YOLOE(str(model_yaml), task=args.task, verbose=False)
    init_probe = probe_student(student, names, args.imgsz, args.device, "INIT PROBE BEFORE MIGRATION")

    log("")
    log("[MIGRATE SHAPE-COMPATIBLE WEIGHTS]")
    migration = migrate_compatible_weights(source_inner, student.model)
    log(json.dumps(migration, indent=2, ensure_ascii=False))
    post_migrate_probe = probe_student(student, names, args.imgsz, args.device, "INIT PROBE AFTER MIGRATION")

    log("")
    log("[SAVE INIT WEIGHTS]")
    save_init_checkpoint(student, init_weights, args, names, migration)
    log(f"init weights saved: {init_weights}")

    log("")
    log("[SMOKE TRAIN]")
    log("native model.val called: False")
    log("teacher used for training: False")
    log("original best.pt modified: False")
    log(f"data: {data_yaml}")
    log(f"task: {args.task}")
    log(f"epochs: {args.epochs}")
    log(f"imgsz: {args.imgsz}")
    log(f"batch: {args.batch}")
    log(f"device: {args.device}")
    log(f"workers: {args.workers}")
    log(f"train val: {args.train_val}")
    log(f"project: {project}")
    log(f"name: {args.name}")

    train_model = YOLO(str(init_weights))
    set_student_classes(train_model, train_model.model, names)
    try:
        train_model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=str(project),
            name=args.name,
            exist_ok=args.exist_ok,
            val=args.train_val,
            plots=False,
            verbose=True,
        )
    except Exception:
        tb = traceback.format_exc()
        log("")
        log("[SMOKE TRAIN ERROR]")
        log(tb)
        log(f"failure type: {classify_train_error(tb)}")
        log("")
        log("[CONCLUSION]")
        log("Smoke train failed. The initialized 2-class student checkpoint was still saved for inspection.")
        log(f"init weights: {init_weights}")
        return 2

    weights = find_train_weights(project, args.name)
    trained_weights = weights["best"] if weights["best"].exists() else weights["last"]
    if not trained_weights.exists():
        raise FileNotFoundError(f"Smoke train completed but no trained weights found under {weights['run_dir'] / 'weights'}")

    log("")
    log("[TRAINED WEIGHTS]")
    log(f"run dir: {weights['run_dir']}")
    log(f"best: {weights['best']} exists={weights['best'].exists()}")
    log(f"last: {weights['last']} exists={weights['last'].exists()}")
    log(f"selected trained weights: {trained_weights}")

    log("")
    log("[POST-TRAIN PROBE]")
    trained = YOLOE(str(trained_weights), task=args.task, verbose=False)
    trained_probe = probe_student(trained, names, args.imgsz, args.device, "TRAINED STUDENT PROBE")

    log("")
    log("[CONCLUSION]")
    log("OK: built a true 2-class YOLOE student, migrated compatible best.pt weights, and completed 1 epoch smoke train.")
    log(f"init weights: {init_weights}")
    log(f"trained weights: {trained_weights}")
    log("class overflow: not seen")
    log("mask/segment label error: not seen")
    log("score channel error: not seen")
    log(f"initial score channels: {init_probe['score_channels_seen']}")
    log(f"post-migrate score channels: {post_migrate_probe['score_channels_seen']}")
    log(f"trained score channels: {trained_probe['score_channels_seen']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
