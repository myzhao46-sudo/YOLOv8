from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}
CLASS_NAMES = {0: "ship", 1: "bridge"}
OPTIONAL_TEST_SPLITS = ("test_ship", "test_bridge")


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def is_foreign_absolute(value: str) -> bool:
    return (os.name != "nt" and PureWindowsPath(value).is_absolute()) or (
        os.name == "nt" and PurePosixPath(value).is_absolute()
    )


def normalize_names(value: Any) -> dict[int, str]:
    if isinstance(value, (list, tuple)):
        return {i: str(name) for i, name in enumerate(value)}
    if isinstance(value, dict):
        try:
            return {int(key): str(name) for key, name in value.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Class names must use integer keys, got {value!r}") from exc
    raise TypeError(f"Dataset names must be a list or mapping, got {value!r}")


def as_path_values(value: Any, key: str) -> list[str]:
    values = value if isinstance(value, list) else [value]
    if not values or any(not isinstance(item, str) or not item.strip() for item in values):
        raise ValueError(f"Dataset YAML {key!r} must be a path string or non-empty list")
    return values


def paths_exist_under(root: Path, value: Any) -> bool:
    try:
        return all((root / item).exists() for item in as_path_values(value, "split"))
    except (TypeError, ValueError):
        return False


def resolve_yaml_root(data_yaml: Path, data: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    raw_path = data.get("path")
    details: dict[str, Any] = {"input_path": raw_path, "rebased_foreign_path": False}
    if raw_path is None or not str(raw_path).strip():
        root = data_yaml.parent.resolve()
    else:
        raw = str(raw_path)
        if is_foreign_absolute(raw):
            candidate = data_yaml.parent.resolve()
            if not paths_exist_under(candidate, data.get("train")) or not paths_exist_under(
                candidate, data.get("val")
            ):
                raise FileNotFoundError(
                    f"Dataset YAML contains a path for another OS ({raw!r}) and cannot be safely "
                    f"rebased to {candidate}"
                )
            root = candidate
            details["rebased_foreign_path"] = True
        else:
            path = Path(raw).expanduser()
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
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    details["resolved_path"] = str(root)
    return root, details


def resolve_split_sources(root: Path, value: Any, split: str) -> list[Path]:
    sources = []
    for raw in as_path_values(value, split):
        if is_foreign_absolute(raw):
            raise FileNotFoundError(f"Dataset {split} path belongs to another OS: {raw}")
        path = Path(raw).expanduser()
        path = path.resolve() if path.is_absolute() else (root / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Dataset {split} image source does not exist: {path}")
        sources.append(path)
    return sources


def load_dataset_definition(data_value: str | None, root_value: str | None) -> dict[str, Any]:
    if bool(data_value) == bool(root_value):
        raise ValueError("Specify exactly one of --data or --root")
    if data_value:
        data_yaml = resolve_project_path(data_value)
        if not data_yaml.is_file():
            raise FileNotFoundError(f"Dataset YAML does not exist: {data_yaml}")
        data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Dataset YAML must contain a mapping: {data_yaml}")
        for key in ("train", "val"):
            if key not in data:
                raise ValueError(f"Dataset YAML is missing required key {key!r}: {data_yaml}")
        names = normalize_names(data.get("names"))
        nc = int(data.get("nc", len(names)))
        if nc != 2 or names != CLASS_NAMES:
            raise ValueError(f"Expected nc=2 names={CLASS_NAMES}, got nc={nc}, names={names}")
        root, root_details = resolve_yaml_root(data_yaml, data)
        split_sources = {
            "train": resolve_split_sources(root, data["train"], "train"),
            "val": resolve_split_sources(root, data["val"], "val"),
        }
    else:
        data_yaml = None
        root = resolve_project_path(root_value or "")
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {root}")
        root_details = {"input_path": root_value, "resolved_path": str(root), "rebased_foreign_path": False}
        split_sources = {
            "train": [root / "images" / "train"],
            "val": [root / "images" / "val"],
        }
        for split, sources in split_sources.items():
            if not sources[0].is_dir():
                raise FileNotFoundError(f"Dataset {split} image directory does not exist: {sources[0]}")

    for split in OPTIONAL_TEST_SPLITS:
        source = root / "images" / split
        if source.is_dir():
            split_sources[split] = [source]
    return {
        "data_yaml": str(data_yaml) if data_yaml else None,
        "root": root,
        "root_resolution": root_details,
        "split_sources": split_sources,
    }


def images_from_source(source: Path) -> list[Path]:
    if source.is_dir():
        return sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if source.is_file() and source.suffix.lower() == ".txt":
        result = []
        for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
            value = raw.strip()
            if not value:
                continue
            if is_foreign_absolute(value):
                raise FileNotFoundError(f"Foreign-OS image path in {source}:{line_number}: {value}")
            path = Path(value).expanduser()
            path = path.resolve() if path.is_absolute() else (source.parent / path).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Image in {source}:{line_number} does not exist: {path}")
            result.append(path)
        return result
    raise ValueError(f"Image source must be a directory or .txt image list: {source}")


def label_for_image(image: Path) -> Path:
    parts = list(image.parts)
    indexes = [index for index, part in enumerate(parts) if part.lower() == "images"]
    if not indexes:
        raise ValueError(f"Cannot derive label path; image has no 'images' path component: {image}")
    parts[indexes[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def new_split_stats(sources: list[Path]) -> dict[str, Any]:
    return {
        "image_sources": [str(source) for source in sources],
        "images": 0,
        "label_files": 0,
        "boxes": 0,
        "class_histogram": {"0": 0, "1": 0},
        "class_0_ship_boxes": 0,
        "class_1_bridge_boxes": 0,
        "missing_labels": 0,
        "empty_labels": 0,
        "bad_rows": 0,
        "missing_label_examples": [],
        "bad_row_examples": [],
    }


def scan_split(sources: list[Path]) -> dict[str, Any]:
    stats = new_split_stats(sources)
    images = []
    for source in sources:
        images.extend(images_from_source(source))
    unique_images = sorted(set(path.resolve() for path in images))
    stats["images"] = len(unique_images)
    for image in unique_images:
        label = label_for_image(image)
        if not label.is_file():
            stats["missing_labels"] += 1
            if len(stats["missing_label_examples"]) < 20:
                stats["missing_label_examples"].append(str(label))
            continue
        stats["label_files"] += 1
        nonempty_rows = 0
        for line_number, raw in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            nonempty_rows += 1
            try:
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f"expected 5 columns, got {len(fields)}")
                class_value = float(fields[0])
                class_id = int(class_value)
                if class_value != class_id or class_id not in CLASS_NAMES:
                    raise ValueError(f"class id must be integer 0 or 1, got {fields[0]!r}")
                coordinates = [float(value) for value in fields[1:]]
                if any(value < 0.0 or value > 1.0 for value in coordinates):
                    raise ValueError("coordinates outside [0,1]")
                if coordinates[2] <= 0.0 or coordinates[3] <= 0.0:
                    raise ValueError("width/height must be positive")
                stats["boxes"] += 1
                stats["class_histogram"][str(class_id)] += 1
            except Exception as exc:
                stats["bad_rows"] += 1
                if len(stats["bad_row_examples"]) < 20:
                    stats["bad_row_examples"].append(f"{label}:{line_number}: {line} | {exc}")
        if nonempty_rows == 0:
            stats["empty_labels"] += 1
    stats["class_0_ship_boxes"] = stats["class_histogram"]["0"]
    stats["class_1_bridge_boxes"] = stats["class_histogram"]["1"]
    return stats


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = ["# Two-Class Dataset Split Check", ""]
    lines.append(f"- Status: `{summary['status']}`")
    lines.append(f"- Dataset root: `{summary['dataset_root']}`")
    lines.append(f"- Require train/val both classes: `{summary['require_train_val_both_classes']}`")
    lines.append("")
    lines.append("| split | images | labels | boxes | ship (0) | bridge (1) | missing | empty | bad rows |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for split, stats in summary["splits"].items():
        lines.append(
            f"| {split} | {stats['images']} | {stats['label_files']} | {stats['boxes']} | "
            f"{stats['class_0_ship_boxes']} | {stats['class_1_bridge_boxes']} | "
            f"{stats['missing_labels']} | {stats['empty_labels']} | {stats['bad_rows']} |"
        )
    lines.extend(["", "## Failures", ""])
    if summary["failures"]:
        lines.extend(f"- {failure}" for failure in summary["failures"])
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate class and label integrity for a 2-class YOLO dataset.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", help="Dataset data.yaml path")
    source.add_argument("--root", help="Dataset root containing images/train and images/val")
    parser.add_argument("--require-train-val-both-classes", action="store_true")
    parser.add_argument("--out-dir", default="runs/check_2cls_dataset/default")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = resolve_project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "data_input": args.data,
        "root_input": args.root,
        "require_train_val_both_classes": args.require_train_val_both_classes,
        "status": "running",
        "failures": [],
    }
    try:
        definition = load_dataset_definition(args.data, args.root)
        summary.update(
            {
                "data_yaml": definition["data_yaml"],
                "dataset_root": str(definition["root"]),
                "root_resolution": definition["root_resolution"],
                "splits": {
                    split: scan_split(sources) for split, sources in definition["split_sources"].items()
                },
            }
        )
        for split, stats in summary["splits"].items():
            if stats["missing_labels"]:
                summary["failures"].append(f"{split}: missing labels = {stats['missing_labels']}")
            if stats["bad_rows"]:
                summary["failures"].append(f"{split}: bad rows = {stats['bad_rows']}")
        if args.require_train_val_both_classes:
            for split in ("train", "val"):
                stats = summary["splits"][split]
                if stats["class_0_ship_boxes"] == 0:
                    summary["failures"].append(f"{split} class 0 ship = 0")
                if stats["class_1_bridge_boxes"] == 0:
                    summary["failures"].append(f"{split} class 1 bridge = 0")
        summary["status"] = "failed" if summary["failures"] else "passed"
    except Exception as exc:
        summary["status"] = "failed"
        summary["failures"].append(f"{type(exc).__name__}: {exc}")

    json_path = out_dir / "check_2cls_dataset_summary.json"
    md_path = out_dir / "check_2cls_dataset_summary.md"
    summary["summary_json"] = str(json_path)
    summary["summary_md"] = str(md_path)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
