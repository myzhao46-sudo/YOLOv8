# tools/train/train_eval_yoloe_2cls_detect_100ep.py
# -*- coding: utf-8 -*-

"""
Run the first 100-epoch YOLOE detect 2-class student experiment and evaluate
external ship/bridge datasets.

This script uses YOLOE(..., task="detect"), keeps the student as 2 classes
(0 ship, 1 bridge), and does not use segmentation training, masks, DINO, replay,
slicing, or any old custom trainer.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch
import yaml


REPO_ROOT = Path(r"C:\Users\DOCTOR\Documents\GitHub\YOLOv8")
PACKAGE_ROOT = REPO_ROOT / "ultralytics"
PYTHON = Path(r"C:\Users\DOCTOR\anaconda3\envs\y8distill\python.exe")
INIT_WEIGHTS = REPO_ROOT / "runs" / "init" / "yoloe_2cls_detect_from_best_init" / "weights" / "yoloe_2cls_detect_student_init.pt"
TRAIN_DATA = REPO_ROOT / "ultralytics" / "datasets" / "bridge_ship_distill_2cls" / "data.yaml"
PROJECT = REPO_ROOT / "runs" / "train"
EXP_NAME = "yoloe_2cls_detect_distill_100ep_v1"
SHIP_EVAL_YAML = REPO_ROOT / "configs" / "datasets" / "external_ship_2cls_eval.yaml"
BRIDGE_EVAL_YAML = REPO_ROOT / "configs" / "datasets" / "external_bridge_2cls_eval.yaml"
COMBINED_EVAL_YAML = REPO_ROOT / "configs" / "datasets" / "external_ship_bridge_2cls_eval.yaml"
DATASETS_ROOT = REPO_ROOT / "ultralytics" / "datasets"
STUDENT_NAMES = ["ship", "bridge"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def setup_local_ultralytics_import() -> None:
    for p in [str(PACKAGE_ROOT.resolve()), str(REPO_ROOT.resolve())]:
        while p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, str(PACKAGE_ROOT.resolve()))
    sys.path.insert(1, str(REPO_ROOT.resolve()))


def log(msg: object = "") -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate YOLOE 2-class detect student for 100 epochs.")
    parser.add_argument("--init-weights", default=str(INIT_WEIGHTS))
    parser.add_argument("--data", default=str(TRAIN_DATA))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", default=str(PROJECT))
    parser.add_argument("--name", default=EXP_NAME)
    parser.add_argument("--exist-ok", action="store_true", default=True)
    parser.add_argument("--force-train", action="store_true", help="Retrain even if run weights already exist.")
    return parser.parse_args()


def choose_device(value: str) -> str:
    if value != "auto":
        return value
    return "0" if torch.cuda.is_available() else "cpu"


def torch_device_from_train_device(value: str) -> str:
    if value == "cpu":
        return "cpu"
    if value.isdigit():
        return f"cuda:{value}"
    return value


def list_images(images_dir: Path) -> list[Path]:
    return sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def labels_dir_from_images_dir(images_dir: Path) -> Path:
    parts = list(images_dir.parts)
    for i, part in enumerate(parts):
        if part.lower() == "images":
            parts[i] = "labels"
            return Path(*parts)
    return images_dir.parent.parent / "labels" / images_dir.name


def check_label_dir_for_images(images_dir: Path, expected_classes: set[int] | None = None) -> dict:
    labels_dir = labels_dir_from_images_dir(images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"labels directory not found: {labels_dir}")

    images = list_images(images_dir)
    class_hist: dict[int, int] = {}
    bad_rows = 0
    missing_labels = 0
    out_of_expected = 0
    label_files = 0
    rows = 0
    for image_path in images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing_labels += 1
            continue
        label_files += 1
        for row_idx, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                bad_rows += 1
                continue
            try:
                cls = int(float(parts[0]))
                coords = [float(x) for x in parts[1:]]
            except ValueError:
                bad_rows += 1
                continue
            if any(x < 0.0 or x > 1.0 for x in coords):
                bad_rows += 1
                continue
            if expected_classes is not None and cls not in expected_classes:
                out_of_expected += 1
            class_hist[cls] = class_hist.get(cls, 0) + 1
            rows += 1
    return {
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "image_count": len(images),
        "label_files": label_files,
        "missing_labels": missing_labels,
        "rows": rows,
        "class_hist": dict(sorted(class_hist.items())),
        "bad_rows": bad_rows,
        "out_of_expected_rows": out_of_expected,
    }


def check_external_group(root: Path, expected_class: int) -> dict:
    modalities = {
        "rgb": "extratest_rgb",
        "sar": "extratest_sar",
        "ir": "extratest_ir",
    }
    stats = {}
    total_hist: dict[int, int] = {}
    total_images = 0
    total_bad = 0
    total_out = 0
    for modality, dirname in modalities.items():
        item = check_label_dir_for_images(root / "images" / dirname, {expected_class})
        stats[modality] = item
        total_images += item["image_count"]
        total_bad += item["bad_rows"]
        total_out += item["out_of_expected_rows"]
        for cls, count in item["class_hist"].items():
            total_hist[int(cls)] = total_hist.get(int(cls), 0) + int(count)
    return {
        "root": str(root),
        "expected_class": expected_class,
        "modalities": stats,
        "total_images": total_images,
        "class_hist": dict(sorted(total_hist.items())),
        "bad_rows": total_bad,
        "out_of_expected_rows": total_out,
        "only_expected_class": total_bad == 0 and total_out == 0 and set(total_hist.keys()).issubset({expected_class}),
    }


def verify_eval_yaml(path: Path, expected_test_count: int) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    names = data.get("names", {})
    names = {int(k): str(v) for k, v in names.items()} if isinstance(names, dict) else {i: str(v) for i, v in enumerate(names)}
    if int(data.get("nc", -1)) != 2 or names != {0: "ship", 1: "bridge"}:
        raise ValueError(f"Bad eval YAML names/nc in {path}: nc={data.get('nc')} names={names}")
    test = data.get("test", [])
    if not isinstance(test, list) or len(test) != expected_test_count:
        raise ValueError(f"Bad test list in {path}: {test}")
    base = Path(str(data.get("path", path.parent)))
    for item in test:
        images_dir = base / str(item)
        if not images_dir.exists():
            raise FileNotFoundError(f"Eval images dir missing from {path}: {images_dir}")
    return {"path": str(path), "test": test, "nc": 2, "names": names}


def set_classes_2cls(model) -> list[int]:
    embeddings = model.get_text_pe(STUDENT_NAMES)
    model.set_classes(STUDENT_NAMES, embeddings)
    return list(embeddings.shape)


def metric_summary(metrics, class_ids: list[int]) -> dict:
    box = metrics.box
    ap_class_index = [int(x) for x in list(getattr(box, "ap_class_index", []))]
    out = {
        "results_dict": getattr(metrics, "results_dict", {}),
        "all": {
            "precision": float(box.mp),
            "recall": float(box.mr),
            "ap50": float(box.map50),
            "ap50_95": float(box.map),
        },
        "classes": {},
        "ap_class_index": ap_class_index,
    }
    nt_per_class = getattr(metrics, "nt_per_class", None)
    for cls_id in class_ids:
        if cls_id in ap_class_index:
            i = ap_class_index.index(cls_id)
            p, r, ap50, ap = metrics.class_result(i)
            instances = int(nt_per_class[cls_id]) if nt_per_class is not None and len(nt_per_class) > cls_id else None
            out["classes"][str(cls_id)] = {
                "name": STUDENT_NAMES[cls_id],
                "instances": instances,
                "precision": float(p),
                "recall": float(r),
                "ap50": float(ap50),
                "ap50_95": float(ap),
            }
        else:
            instances = int(nt_per_class[cls_id]) if nt_per_class is not None and len(nt_per_class) > cls_id else None
            out["classes"][str(cls_id)] = {
                "name": STUDENT_NAMES[cls_id],
                "instances": instances,
                "precision": 0.0,
                "recall": 0.0,
                "ap50": 0.0,
                "ap50_95": 0.0,
            }
    return out


def run_val(weights: Path, data_yaml: Path, label: str, device: str, imgsz: int, batch: int, workers: int):
    from ultralytics.models.yolo.model import YOLOE

    log("")
    log(f"[EXTERNAL VAL] {label}")
    log(f"weights: {weights}")
    log(f"data: {data_yaml}")
    model = YOLOE(str(weights), task="detect", verbose=False)
    pe_shape = set_classes_2cls(model)
    log(f"pe shape: {pe_shape}")
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        conf=0.001,
        iou=0.7,
        plots=True,
    )
    summary = metric_summary(metrics, [0, 1])
    log(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> int:
    args = parse_args()
    setup_local_ultralytics_import()

    from ultralytics.models.yolo.model import YOLOE
    from tools.train.init_and_smoke_train_yoloe_2cls_student import (
        classify_train_error,
        probe_student,
        validate_data_yaml,
    )

    device = choose_device(args.device)
    probe_device = torch_device_from_train_device(device)
    init_weights = Path(args.init_weights).resolve()
    data_yaml = Path(args.data).resolve()
    project = Path(args.project).resolve()
    run_dir = project / args.name
    summary_path = run_dir / "experiment_summary.json"

    log("[IMPORT CHECK]")
    import ultralytics as ultralytics_pkg

    log(f"python expected: {PYTHON}")
    log(f"ultralytics imported from: {getattr(ultralytics_pkg, '__file__', 'UNKNOWN')}")
    log(f"torch cuda available: {torch.cuda.is_available()}")
    log(f"device selected: {device}")
    log(f"probe device: {probe_device}")
    if torch.cuda.is_available():
        log(f"cuda name: {torch.cuda.get_device_name(0)}")

    log("")
    log("[TRAIN DATA CHECK]")
    train_stats = validate_data_yaml(data_yaml)
    log(json.dumps(train_stats, indent=2, ensure_ascii=False))

    log("")
    log("[EXTERNAL YAML CHECK]")
    yaml_stats = {
        "ship": verify_eval_yaml(SHIP_EVAL_YAML, 3),
        "bridge": verify_eval_yaml(BRIDGE_EVAL_YAML, 3),
        "combined": verify_eval_yaml(COMBINED_EVAL_YAML, 6),
    }
    log(json.dumps(yaml_stats, indent=2, ensure_ascii=False))

    log("")
    log("[EXTERNAL DATA CHECK]")
    external_stats = {
        "ship_external": check_external_group(DATASETS_ROOT / "ship_extratest_no_overlap", 0),
        "bridge_external": check_external_group(DATASETS_ROOT / "extracttest_bridge", 1),
    }
    log(json.dumps(external_stats, indent=2, ensure_ascii=False))
    if not external_stats["ship_external"]["only_expected_class"]:
        raise ValueError("Ship external labels are not clean class 0 only.")
    if not external_stats["bridge_external"]["only_expected_class"]:
        raise ValueError("Bridge external labels are not clean class 1 only.")

    log("")
    log("[INIT WEIGHTS PROBE]")
    init_model = YOLOE(str(init_weights), task="detect", verbose=False)
    init_probe = probe_student(init_model, STUDENT_NAMES, args.imgsz, probe_device, "INIT DETECT STUDENT PROBE")

    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    if (best.exists() or last.exists()) and not args.force_train:
        log("")
        log("[TRAIN 100EP]")
        log("Existing trained weights found; skipping retrain. Use --force-train to rerun.")
        train_error = None
    else:
        log("")
        log("[TRAIN 100EP]")
        log(f"init weights: {init_weights}")
        log(f"data: {data_yaml}")
        log(f"epochs: {args.epochs}")
        log(f"imgsz: {args.imgsz}")
        log(f"batch: {args.batch}")
        log(f"device: {device}")
        log(f"workers: {args.workers}")
        log(f"project: {project}")
        log(f"name: {args.name}")

        model = YOLOE(str(init_weights), task="detect", verbose=False)
        pe_shape = set_classes_2cls(model)
        log(f"train pe shape: {pe_shape}")
        try:
            model.train(
                data=str(data_yaml),
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device=device,
                workers=args.workers,
                project=str(project),
                name=args.name,
                exist_ok=args.exist_ok,
                val=True,
                plots=True,
            )
            train_error = None
        except Exception:
            tb = traceback.format_exc()
            log("")
            log("[TRAIN ERROR]")
            log(tb)
            train_error = {"traceback": tb, "failure_type": classify_train_error(tb)}
            run_dir.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps({"train_error": train_error}, indent=2, ensure_ascii=False), encoding="utf-8")
            return 2

    if not best.exists() and not last.exists():
        raise FileNotFoundError(f"No trained weights found under {run_dir / 'weights'}")
    selected = best if best.exists() else last

    log("")
    log("[POST-TRAIN PROBE]")
    post_probes = {}
    for name, path in [("best", best), ("last", last)]:
        if path.exists():
            probe_model = YOLOE(str(path), task="detect", verbose=False)
            post_probes[name] = probe_student(
                probe_model, STUDENT_NAMES, args.imgsz, probe_device, f"POST-TRAIN {name.upper()} PROBE"
            )

    eval_results = {
        "ship_external": run_val(selected, SHIP_EVAL_YAML, "ship_external", device, args.imgsz, args.batch, args.workers),
        "bridge_external": run_val(selected, BRIDGE_EVAL_YAML, "bridge_external", device, args.imgsz, args.batch, args.workers),
        "combined_external": run_val(selected, COMBINED_EVAL_YAML, "combined_external", device, args.imgsz, args.batch, args.workers),
    }

    summary = {
        "train_data": train_stats,
        "external_data": external_stats,
        "eval_yamls": yaml_stats,
        "training": {
            "epochs": args.epochs,
            "init_weights": str(init_weights),
            "run_dir": str(run_dir),
            "best": str(best),
            "best_exists": best.exists(),
            "last": str(last),
            "last_exists": last.exists(),
            "selected_weights": str(selected),
            "device": device,
            "train_error": train_error,
        },
        "init_probe": init_probe,
        "post_train_probe": post_probes,
        "eval_results": eval_results,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log("")
    log("[SUMMARY SAVED]")
    log(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
