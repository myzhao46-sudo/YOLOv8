from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "ultralytics"
STUDENT_NAMES = {0: "ship", 1: "bridge"}
IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_BRIDGE_WEIGHTS = (
    "runs/bridge_expert/yolov8m_total_bridge_split_img1024_ep150/weights/best.pt"
)
DEFAULT_BASE_MODEL = "yolov8m.pt"
DEFAULT_DATA = "ultralytics/datasets/bridge_ship_distill_2cls/data.yaml"
DEFAULT_PROJECT = "runs/student_2cls"
DEFAULT_NAME = "yolov8m_2cls_bridgeinit_ship_pseudo_bridge_gt_img1024_ep100"
DEFAULT_INIT = "runs/student_2cls/init/yolov8m_2cls_from_bridge_expert.pt"


def setup_import_paths() -> None:
    for value in (str(PACKAGE_ROOT.resolve()), str(PROJECT_ROOT.resolve())):
        while value in sys.path:
            sys.path.remove(value)
    sys.path.insert(0, str(PACKAGE_ROOT.resolve()))
    sys.path.insert(1, str(PROJECT_ROOT.resolve()))


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def git_info() -> dict[str, str | None]:
    result: dict[str, str | None] = {"branch": None, "commit": None}
    for key, command in (
        ("branch", ["git", "branch", "--show-current"]),
        ("commit", ["git", "rev-parse", "HEAD"]),
    ):
        try:
            result[key] = subprocess.check_output(
                command, cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return result


def normalize_names(names: Any) -> dict[int, str]:
    if isinstance(names, (list, tuple)):
        return {i: str(name) for i, name in enumerate(names)}
    if isinstance(names, dict):
        try:
            return {int(key): str(value) for key, value in names.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Dataset names must use integer keys, got {names!r}") from exc
    raise TypeError(f"Dataset names must be a list or mapping, got {names!r}")


def is_foreign_absolute(value: str) -> bool:
    return (os.name != "nt" and PureWindowsPath(value).is_absolute()) or (
        os.name == "nt" and PurePosixPath(value).is_absolute()
    )


def split_values(value: Any, key: str) -> list[str]:
    values = value if isinstance(value, list) else [value]
    if not values or any(not isinstance(item, str) or not item.strip() for item in values):
        raise ValueError(f"Dataset YAML {key!r} must be a path string or non-empty path list")
    return values


def split_exists_under(root: Path, value: Any) -> bool:
    try:
        return all((root / item).exists() for item in split_values(value, "split"))
    except (TypeError, ValueError):
        return False


def resolve_dataset_root(data_yaml: Path, data: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    raw = data.get("path")
    resolution: dict[str, Any] = {"input_path": raw, "rebased_foreign_path": False}
    if raw is None or str(raw).strip() == "":
        root = data_yaml.parent.resolve()
    else:
        value = str(raw)
        if is_foreign_absolute(value):
            candidate = data_yaml.parent.resolve()
            if not split_exists_under(candidate, data.get("train")) or not split_exists_under(
                candidate, data.get("val")
            ):
                raise FileNotFoundError(
                    f"Dataset YAML path belongs to another OS ({value!r}) and train/val cannot be "
                    f"safely rebased to the YAML directory {candidate}. Provide a Linux-valid 2-class data YAML."
                )
            root = candidate
            resolution["rebased_foreign_path"] = True
        else:
            path = Path(value).expanduser()
            if path.is_absolute():
                root = path.resolve()
            else:
                candidates = [
                    (data_yaml.parent / path).resolve(),
                    (PROJECT_ROOT / path).resolve(),
                    (Path.cwd() / path).resolve(),
                ]
                root = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    if not root.is_dir():
        raise FileNotFoundError(f"Resolved dataset root does not exist: {root}")
    resolution["resolved_path"] = str(root)
    return root, resolution


def resolve_split_paths(root: Path, value: Any, key: str) -> list[Path]:
    resolved = []
    for item in split_values(value, key):
        if is_foreign_absolute(item):
            raise FileNotFoundError(
                f"Dataset YAML {key} path belongs to another OS and cannot be used here: {item}"
            )
        path = Path(item).expanduser()
        path = path.resolve() if path.is_absolute() else (root / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Dataset {key} image path does not exist: {path}")
        resolved.append(path)
    return resolved


def images_from_source(source: Path) -> list[Path]:
    if source.is_dir():
        return sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if source.is_file() and source.suffix.lower() == ".txt":
        images = []
        for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
            value = raw.strip()
            if not value:
                continue
            if is_foreign_absolute(value):
                raise FileNotFoundError(
                    f"Foreign-OS image path in {source}:{line_number}: {value}"
                )
            path = Path(value).expanduser()
            path = path.resolve() if path.is_absolute() else (source.parent / path).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Image listed by {source}:{line_number} does not exist: {path}")
            images.append(path)
        return images
    raise ValueError(f"Image source must be a directory or .txt file list: {source}")


def label_path_for_image(image: Path) -> Path:
    parts = list(image.parts)
    image_indexes = [i for i, value in enumerate(parts) if value.lower() == "images"]
    if not image_indexes:
        raise ValueError(
            f"Cannot derive label path because image path has no 'images' component: {image}"
        )
    parts[image_indexes[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def scan_labels(images_by_split: dict[str, list[Path]], seed: int, sample_count: int = 20) -> dict[str, Any]:
    class_histogram = {0: 0, 1: 0}
    split_histograms = {split: {0: 0, 1: 0} for split in images_by_split}
    split_label_files = {split: 0 for split in images_by_split}
    split_empty_labels = {split: 0 for split in images_by_split}
    row_count = 0
    missing_labels = []
    label_files = []
    for split, images in images_by_split.items():
        if not images:
            raise ValueError(f"Dataset {split} split contains no supported images")
        for image in images:
            label = label_path_for_image(image)
            if not label.is_file():
                missing_labels.append(str(label))
                continue
            label_files.append(label)
            split_label_files[split] += 1
            label_rows = 0
            for line_number, raw in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
                line = raw.strip()
                if not line:
                    continue
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(
                        f"Label must have exactly 5 columns, got {len(fields)} at {label}:{line_number}"
                    )
                try:
                    class_value = float(fields[0])
                    coordinates = [float(value) for value in fields[1:]]
                except ValueError as exc:
                    raise ValueError(f"Non-numeric label at {label}:{line_number}: {line}") from exc
                class_id = int(class_value)
                if class_value != class_id or class_id not in STUDENT_NAMES:
                    raise ValueError(
                        f"Label class must be integer 0 or 1, got {fields[0]!r} at {label}:{line_number}"
                    )
                if any(value < 0.0 or value > 1.0 for value in coordinates):
                    raise ValueError(f"Normalized coordinates must be in [0,1] at {label}:{line_number}")
                if coordinates[2] <= 0.0 or coordinates[3] <= 0.0:
                    raise ValueError(f"Label width/height must be positive at {label}:{line_number}")
                class_histogram[class_id] += 1
                split_histograms[split][class_id] += 1
                row_count += 1
                label_rows += 1
            if label_rows == 0:
                split_empty_labels[split] += 1

    if not label_files or row_count == 0:
        raise ValueError("No non-empty YOLO detect labels were found for train/val images")
    unique_labels = sorted(set(label_files))
    rng = random.Random(seed)
    sampled = rng.sample(unique_labels, min(sample_count, len(unique_labels)))
    return {
        "rows_checked": row_count,
        "label_files_checked": len(unique_labels),
        "class_histogram": class_histogram,
        "split_histograms": split_histograms,
        "split_label_files": split_label_files,
        "split_empty_labels": split_empty_labels,
        "missing_label_files": len(missing_labels),
        "missing_label_examples": missing_labels[:20],
        "random_label_sample": [str(path) for path in sampled],
        "classes_valid": True,
    }


def validate_and_write_runtime_data(
    data_yaml: Path,
    runtime_yaml: Path,
    seed: int,
    require_both_classes_in_train: bool = False,
    require_both_classes_in_val: bool = False,
) -> dict[str, Any]:
    if not data_yaml.is_file():
        raise FileNotFoundError(f"2-class dataset YAML does not exist: {data_yaml}")
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {data_yaml}")
    if "train" not in data or "val" not in data:
        raise ValueError(f"Dataset YAML must contain train and val: {data_yaml}")
    names = normalize_names(data.get("names"))
    nc = int(data.get("nc", len(names)))
    if nc != 2 or names != STUDENT_NAMES:
        raise ValueError(
            f"Dataset must define nc=2 and names={STUDENT_NAMES}, got nc={nc}, names={names}"
        )

    root, root_resolution = resolve_dataset_root(data_yaml, data)
    train_paths = resolve_split_paths(root, data["train"], "train")
    val_paths = resolve_split_paths(root, data["val"], "val")
    images_by_split = {
        "train": [image for source in train_paths for image in images_from_source(source)],
        "val": [image for source in val_paths for image in images_from_source(source)],
    }
    label_check = scan_labels(images_by_split, seed=seed)
    requirements = {
        "train": require_both_classes_in_train,
        "val": require_both_classes_in_val,
    }
    for split, required in requirements.items():
        if not required:
            continue
        histogram = label_check["split_histograms"][split]
        missing_classes = [class_id for class_id in STUDENT_NAMES if histogram[class_id] == 0]
        if missing_classes:
            details = ", ".join(
                f"class {class_id} {STUDENT_NAMES[class_id]} = 0" for class_id in missing_classes
            )
            raise ValueError(f"Dataset {split} must contain both classes; {details}")

    runtime_data = dict(data)
    runtime_data["path"] = str(root)
    runtime_data["train"] = str(train_paths[0]) if len(train_paths) == 1 else [str(x) for x in train_paths]
    runtime_data["val"] = str(val_paths[0]) if len(val_paths) == 1 else [str(x) for x in val_paths]
    runtime_data["nc"] = 2
    runtime_data["names"] = {0: "ship", 1: "bridge"}
    runtime_yaml.parent.mkdir(parents=True, exist_ok=True)
    runtime_yaml.write_text(
        yaml.safe_dump(runtime_data, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return {
        "user_input_data": str(data_yaml),
        "resolved_data": str(root),
        "runtime_data_yaml": str(runtime_yaml),
        "root_resolution": root_resolution,
        "names": names,
        "nc": nc,
        "train": [str(path) for path in train_paths],
        "val": [str(path) for path in val_paths],
        "train_images": len(images_by_split["train"]),
        "val_images": len(images_by_split["val"]),
        "label_check": label_check,
        "require_both_classes_in_train": require_both_classes_in_train,
        "require_both_classes_in_val": require_both_classes_in_val,
    }


def inspect_init_checkpoint(path: Path) -> dict[str, Any]:
    from ultralytics import YOLO
    from ultralytics.nn.modules.head import Detect
    from ultralytics.nn.tasks import DetectionModel

    model = YOLO(str(path), verbose=False)
    inner = model.model
    if type(inner) is not DetectionModel or type(inner.model[-1]) is not Detect:
        raise TypeError(
            f"Init checkpoint is not an ordinary DetectionModel + Detect: "
            f"{type(inner).__name__} + {type(inner.model[-1]).__name__}"
        )
    head = inner.model[-1]
    names = normalize_names(inner.names)
    if int(head.nc) != 2 or names != STUDENT_NAMES:
        raise ValueError(f"Init checkpoint must be nc=2 names={STUDENT_NAMES}, got nc={head.nc}, names={names}")
    metadata = model.ckpt.get("bridge_init", {})
    if not metadata.get("bridge_class_row_copied"):
        raise ValueError("Init checkpoint does not record a successful bridge class 0 -> class 1 copy")
    return {
        "path": str(path),
        "model_class": f"{type(inner).__module__}.{type(inner).__name__}",
        "head_class": f"{type(head).__module__}.{type(head).__name__}",
        "nc": int(head.nc),
        "names": names,
        "bridge_class_row_copied": True,
        "copied_layers": metadata.get("copied_layers", []),
    }


def read_training_results(run_dir: Path) -> dict[str, Any]:
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    results_csv = run_dir / "results.csv"
    result: dict[str, Any] = {
        "output_best": str(best),
        "output_best_exists": best.is_file(),
        "output_last": str(last),
        "output_last_exists": last.is_file(),
        "results_csv": str(results_csv),
        "results_csv_exists": results_csv.is_file(),
        "last_results_rows": [],
        "map50_column": None,
        "map50_nonzero": None,
    }
    if not results_csv.is_file():
        return result
    with results_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result["last_results_rows"] = rows[-5:]
    if rows:
        columns = rows[0].keys()
        map50_column = next(
            (
                key
                for key in columns
                if "map50" in key.lower() and "95" not in key.lower()
            ),
            None,
        )
        result["map50_column"] = map50_column
        if map50_column:
            values = []
            for row in rows:
                try:
                    values.append(float(row[map50_column]))
                except (TypeError, ValueError):
                    continue
            result["map50_nonzero"] = any(value > 0.0 for value in values)
            result["last_map50"] = values[-1] if values else None
            result["max_map50"] = max(values) if values else None
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize and train an ordinary YOLOv8m 2-class ship/bridge detector."
    )
    parser.add_argument("--bridge-weights", default=DEFAULT_BRIDGE_WEIGHTS)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--init-checkpoint", default=DEFAULT_INIT)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--freeze", type=int, default=10)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument(
        "--require-both-classes-in-train",
        action="store_true",
        help="Fail before initialization/training unless train contains class 0 and class 1 boxes.",
    )
    parser.add_argument(
        "--require-both-classes-in-val",
        action="store_true",
        help="Fail before initialization/training unless val contains class 0 and class 1 boxes.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Generate/verify init checkpoint and runtime YAML, but do not start training.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_import_paths()
    from tools.train.init_yolov8m_2cls_from_bridge_expert import initialize_student

    bridge_weights = resolve_path(args.bridge_weights)
    base_model = resolve_path(args.base_model)
    data_yaml = resolve_path(args.data)
    init_checkpoint = resolve_path(args.init_checkpoint)
    project = resolve_path(args.project)
    runtime_yaml = project / "runtime_bridge_ship_distill_2cls.yaml"
    expected_run_dir = project / args.name
    summary: dict[str, Any] = {
        "task": "bridge-initialized yolov8m 2-class student first training",
        "student_classes": {"0": "ship", "1": "bridge"},
        "project_root": str(PROJECT_ROOT),
        "git": git_info(),
        "bridge_init_source": str(bridge_weights),
        "base_model": str(base_model),
        "data": str(data_yaml),
        "runtime_data_yaml": str(runtime_yaml),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "freeze": args.freeze,
        "device": args.device,
        "workers": args.workers,
        "project": str(project),
        "name": args.name,
        "init_checkpoint": str(init_checkpoint),
        "expected_run_dir": str(expected_run_dir),
        "started_at": datetime.now().isoformat(),
        "preflight_only": args.preflight_only,
        "require_both_classes_in_train": args.require_both_classes_in_train,
        "require_both_classes_in_val": args.require_both_classes_in_val,
        "status": "running",
    }
    actual_run_dir: Path | None = None
    train_model: Any | None = None
    try:
        if not bridge_weights.is_file():
            raise FileNotFoundError(
                f"Bridge-only expert weights are missing: {bridge_weights}. Upload the specified best.pt; "
                "no substitute model will be used."
            )
        if not base_model.is_file():
            raise FileNotFoundError(
                f"Base YOLOv8m model is missing: {base_model}. No substitute model will be used."
            )
        if not data_yaml.is_file():
            raise FileNotFoundError(
                f"Required 2-class data YAML is missing: {data_yaml}. Provide the correct YAML; "
                "no alternate dataset will be selected."
            )

        project.mkdir(parents=True, exist_ok=True)
        data_check = validate_and_write_runtime_data(
            data_yaml,
            runtime_yaml,
            seed=args.seed,
            require_both_classes_in_train=args.require_both_classes_in_train,
            require_both_classes_in_val=args.require_both_classes_in_val,
        )
        summary["data_check"] = data_check
        summary["resolved_data"] = data_check["resolved_data"]
        summary["names"] = data_check["names"]
        summary["nc"] = data_check["nc"]
        summary["train"] = data_check["train"]
        summary["val"] = data_check["val"]

        init_summary = initialize_student(
            bridge_weights,
            base_model,
            init_checkpoint,
            init_checkpoint.with_name(f"{init_checkpoint.stem}_summary.json"),
        )
        summary["initialization"] = {
            "status": init_summary["status"],
            "old_nc": init_summary["old_nc"],
            "new_nc": init_summary["new_nc"],
            "names": init_summary["names"],
            "num_shape_compatible_loaded": init_summary["num_shape_compatible_loaded"],
            "num_skipped": init_summary["num_skipped"],
            "bridge_class_row_copied": init_summary["bridge_class_row_copied"],
            "copied_layers": init_summary["copied_layers"],
        }
        summary["init_checkpoint_check"] = inspect_init_checkpoint(init_checkpoint)
        summary["preflight_checks_passed"] = True

        print("\n[TRAINING PREFLIGHT]", flush=True)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        if args.preflight_only:
            summary["status"] = "preflight_passed"
            summary["completed_at"] = datetime.now().isoformat()
            preflight_summary = project / f"{args.name}_preflight_summary.json"
            preflight_summary.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"Preflight-only mode: training was not started. Summary: {preflight_summary}", flush=True)
            return 0

        from ultralytics import YOLO

        train_model = YOLO(str(init_checkpoint), verbose=False)
        train_model.train(
            data=str(runtime_yaml),
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=str(project),
            name=args.name,
            seed=args.seed,
            patience=args.patience,
            close_mosaic=args.close_mosaic,
            freeze=args.freeze,
            pretrained=False,
            exist_ok=args.exist_ok,
        )
        trainer_save_dir = getattr(getattr(train_model, "trainer", None), "save_dir", None)
        actual_run_dir = Path(trainer_save_dir).resolve() if trainer_save_dir else expected_run_dir
        output_check = read_training_results(actual_run_dir)
        summary["run_dir"] = str(actual_run_dir)
        summary["training_outputs"] = output_check
        summary.update(
            {
                "output_best": output_check["output_best"],
                "output_last": output_check["output_last"],
                "results_csv": output_check["results_csv"],
            }
        )
        if not (
            output_check["output_best_exists"]
            and output_check["output_last_exists"]
            and output_check["results_csv_exists"]
        ):
            raise RuntimeError(f"Training returned but required output files are missing: {output_check}")
        summary["status"] = "completed"
        summary["completed_at"] = datetime.now().isoformat()
        print("\n[TRAINING OUTPUT CHECK]", flush=True)
        print(json.dumps(output_check, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as exc:
        summary["status"] = "failed"
        summary["failed_at"] = datetime.now().isoformat()
        summary["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if not args.preflight_only or summary["status"] == "failed":
            if actual_run_dir is None and train_model is not None:
                trainer_save_dir = getattr(getattr(train_model, "trainer", None), "save_dir", None)
                actual_run_dir = Path(trainer_save_dir).resolve() if trainer_save_dir else None
            if actual_run_dir is not None:
                actual_run_dir.mkdir(parents=True, exist_ok=True)
                summary_path = actual_run_dir / "train_bridgeinit_summary.json"
            else:
                project.mkdir(parents=True, exist_ok=True)
                summary_path = project / f"{args.name}_failed_summary.json"
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"Training summary: {summary_path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
