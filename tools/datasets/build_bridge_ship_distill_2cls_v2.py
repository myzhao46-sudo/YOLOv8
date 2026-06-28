from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "ultralytics"
DATASETS_ROOT = PACKAGE_ROOT / "datasets"
IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}
CLASS_NAMES = {0: "ship", 1: "bridge"}
MODALITIES = ("rgb", "sar", "ir")

DEFAULT_SHIP_TRAIN_IMAGES = "ultralytics/datasets/ship_small_split/images/train"
DEFAULT_SHIP_TRAIN_LABELS = "ultralytics/datasets/ship_teacher_pseudo_conf025/labels/train"
DEFAULT_SHIP_VAL_IMAGES = "ultralytics/datasets/ship_small_split/images/val"
DEFAULT_SHIP_VAL_LABELS = "ultralytics/datasets/ship_small_split/labels/val"
DEFAULT_BRIDGE_TRAINVAL_ROOT = "ultralytics/datasets/total_bridge_trainval"
DEFAULT_BRIDGE_TEST_ROOT = "ultralytics/datasets/total_bridge_test"
DEFAULT_SHIP_TEST_ROOT = "ultralytics/datasets/ship_extratest_no_overlap"
DEFAULT_OUT_ROOT = "ultralytics/datasets/bridge_ship_distill_2cls_v2"
DEFAULT_SHIP_TEACHER = "ultralytics/best.pt"


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def require_dir(path: Path, description: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{description} directory does not exist: {path}")


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} file does not exist: {path}")


def list_images(path: Path) -> list[Path]:
    require_dir(path, "Image source")
    images = sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"No supported images found in: {path}")
    return images


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_one_to_one(images: list[Path], labels_dir: Path, description: str) -> None:
    require_dir(labels_dir, f"{description} labels")
    image_stems = {image.stem for image in images}
    label_stems = {label.stem for label in labels_dir.iterdir() if label.is_file() and label.suffix.lower() == ".txt"}
    missing = sorted(image_stems - label_stems)
    extra = sorted(label_stems - image_stems)
    if missing or extra:
        raise ValueError(
            f"{description} image/label stems are not one-to-one: "
            f"missing_labels={missing[:20]}, extra_labels={extra[:20]}"
        )


def read_and_remap_label(
    label: Path,
    expected_source_class: int,
    target_class: int,
    allow_empty: bool,
    description: str,
) -> tuple[list[str], int]:
    require_file(label, f"{description} label")
    output = []
    for line_number, raw in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{description} label must have 5 columns at {label}:{line_number}: {line}")
        try:
            class_value = float(fields[0])
            class_id = int(class_value)
            coordinates = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise ValueError(f"Non-numeric {description} label at {label}:{line_number}: {line}") from exc
        if class_value != class_id or class_id != expected_source_class:
            raise ValueError(
                f"{description} source class must be {expected_source_class}, got {fields[0]!r} "
                f"at {label}:{line_number}"
            )
        if any(value < 0.0 or value > 1.0 for value in coordinates):
            raise ValueError(f"Coordinates outside [0,1] at {label}:{line_number}")
        if coordinates[2] <= 0.0 or coordinates[3] <= 0.0:
            raise ValueError(f"Non-positive width/height at {label}:{line_number}")
        output.append(f"{target_class} {' '.join(fields[1:])}")
    if not output and not allow_empty:
        raise ValueError(f"{description} label is empty: {label}")
    return output, len(output)


def materialize_image(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, target)
    elif mode == "hardlink":
        os.link(source, target)
    elif mode == "symlink":
        target.symlink_to(source.resolve())
    else:
        raise ValueError(f"Unsupported copy mode: {mode}")


def modality_from_name(name: str) -> str:
    lower = name.lower()
    if "rgb" in lower or "dior" in lower:
        return "rgb"
    if "sar" in lower or "msar" in lower or "masr" in lower:
        return "sar"
    if "ir" in lower or "infr" in lower or "massmind" in lower:
        return "ir"
    raise ValueError(f"Cannot infer modality from name: {name}")


def setup_local_ultralytics_import() -> None:
    for value in (str(PACKAGE_ROOT.resolve()), str(PROJECT_ROOT.resolve())):
        while value in sys.path:
            sys.path.remove(value)
    sys.path.insert(0, str(PACKAGE_ROOT.resolve()))
    sys.path.insert(1, str(PROJECT_ROOT.resolve()))


def generate_ship_pseudo_labels(
    image_paths: list[Path], output_dir: Path, args: argparse.Namespace
) -> dict[str, Any]:
    """Run the frozen YOLOE box branch and emit class-0 five-column labels."""
    setup_local_ultralytics_import()
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.modules.head import YOLOESegment
    from ultralytics.nn.tasks import YOLOESegModel

    from tools.eval.eval_teacher_ship_external import collect_predictions, safe_getattr

    teacher_weights = resolve_path(args.ship_teacher_weights)
    require_file(teacher_weights, "Ship teacher weights")
    teacher = YOLO(str(teacher_weights))
    inner = teacher.model
    if type(inner) is not YOLOESegModel or type(inner.model[-1]) is not YOLOESegment:
        raise TypeError(
            f"Ship teacher must be YOLOESegModel + YOLOESegment, got "
            f"{type(inner).__name__} + {type(inner.model[-1]).__name__}"
        )

    names = ["ship", "harbor", "tank"]
    get_text_pe = safe_getattr(teacher, "get_text_pe", None)
    if not callable(get_text_pe):
        get_text_pe = safe_getattr(inner, "get_text_pe", None)
    if not callable(get_text_pe):
        raise AttributeError("YOLOE teacher has no callable get_text_pe")
    embeddings = get_text_pe(names)
    set_classes = safe_getattr(teacher, "set_classes", None)
    if not callable(set_classes):
        set_classes = safe_getattr(inner, "set_classes", None)
    if not callable(set_classes):
        raise AttributeError("YOLOE teacher has no callable set_classes")
    set_classes(names, embeddings)
    for parameter in inner.parameters():
        parameter.requires_grad_(False)

    device_value = str(args.device)
    if device_value.isdigit():
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {device_value} requested but CUDA is unavailable")
        device = torch.device(f"cuda:{device_value}")
    else:
        device = torch.device(device_value)
    inference_args = SimpleNamespace(
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        conf=args.ship_conf,
        pred_iou=args.iou,
        max_det=args.max_det,
        target_class=0,
    )
    predictions, score_channels, total_boxes = collect_predictions(inner, image_paths, inference_args)
    if score_channels != {3}:
        raise RuntimeError(f"Expected three fused YOLOE score channels, got {sorted(score_channels)}")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for prediction in predictions:
        grouped[prediction["image_key"]].append(prediction)

    output_dir.mkdir(parents=True, exist_ok=True)
    empty_images = []
    boxes = 0
    for image in image_paths:
        with Image.open(image) as opened:
            width, height = opened.size
        rows = []
        for prediction in sorted(grouped[str(image.resolve())], key=lambda item: item["conf"], reverse=True):
            x1, y1, x2, y2 = prediction["xyxy"]
            x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
            y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
            box_width, box_height = max(0.0, x2 - x1), max(0.0, y2 - y1)
            if box_width <= 0.0 or box_height <= 0.0:
                continue
            center_x = ((x1 + x2) / 2.0) / width
            center_y = ((y1 + y2) / 2.0) / height
            rows.append(
                f"0 {center_x:.8f} {center_y:.8f} {box_width / width:.8f} {box_height / height:.8f}"
            )
        if not rows:
            empty_images.append(str(image.resolve()))
        boxes += len(rows)
        (output_dir / f"{image.stem}.txt").write_text(
            "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8"
        )
    if empty_images and not args.allow_empty_ship_pseudo:
        raise ValueError(
            f"Frozen teacher produced {len(empty_images)} empty ship pseudo labels at conf={args.ship_conf}; "
            f"examples={empty_images[:20]}; "
            "lower --ship-conf or explicitly pass --allow-empty-ship-pseudo"
        )
    return {
        "mode": "generated",
        "teacher_weights": str(teacher_weights),
        "teacher_names": names,
        "set_classes_used_embeddings": True,
        "images": len(image_paths),
        "boxes": boxes,
        "empty_labels": len(empty_images),
        "empty_label_images": empty_images,
        "score_channels": sorted(score_channels),
        "all_class_post_nms_boxes": total_boxes,
        "conf": args.ship_conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "device": str(device),
        "teacher_frozen": True,
    }


class DatasetBuilder:
    def __init__(self, staging_root: Path, final_root: Path, copy_mode: str):
        self.staging_root = staging_root
        self.final_root = final_root
        self.copy_mode = copy_mode
        self.records: list[dict[str, Any]] = []
        self.target_names: set[tuple[str, str]] = set()

    def add(
        self,
        source_image: Path,
        source_label: Path,
        split: str,
        target_stem: str,
        source_type: str,
        modality: str,
        source_dataset: str,
        expected_source_class: int,
        target_class: int,
        allow_empty: bool = False,
        generated_source_label: bool = False,
    ) -> None:
        key = (split, target_stem.lower())
        if key in self.target_names:
            raise ValueError(f"Duplicate target stem in {split}: {target_stem}")
        self.target_names.add(key)
        rows, box_count = read_and_remap_label(
            source_label,
            expected_source_class=expected_source_class,
            target_class=target_class,
            allow_empty=allow_empty,
            description=source_type,
        )
        target_image_relative = Path("images") / split / f"{target_stem}{source_image.suffix.lower()}"
        target_label_relative = Path("labels") / split / f"{target_stem}.txt"
        materialize_image(source_image, self.staging_root / target_image_relative, self.copy_mode)
        target_label = self.staging_root / target_label_relative
        target_label.parent.mkdir(parents=True, exist_ok=True)
        target_label.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        self.records.append(
            {
                "split": split,
                "source_type": source_type,
                "source_image": str(source_image.resolve()),
                "source_label": (
                    str((self.final_root / target_label_relative).resolve())
                    if generated_source_label
                    else str(source_label.resolve())
                ),
                "target_image": str((self.final_root / target_image_relative).resolve()),
                "target_label": str((self.final_root / target_label_relative).resolve()),
                "class_ids": str(target_class) if box_count else "",
                "num_boxes": box_count,
                "modality": modality,
                "source_dataset": source_dataset,
                "hash_sha256": sha256(source_image),
            }
        )


def bridge_candidates(root: Path, split: str, modality: str) -> list[tuple[Path, Path]]:
    image_dir = root / "images" / f"{split}_{modality}"
    label_dir = root / "labels" / f"{split}_{modality}"
    images = list_images(image_dir)
    validate_one_to_one(images, label_dir, f"bridge {split} {modality}")
    candidates = []
    for image in images:
        label = label_dir / f"{image.stem}.txt"
        read_and_remap_label(label, 0, 1, False, f"bridge {split} {modality}")
        candidates.append((image, label))
    return candidates


def add_ship_test(builder: DatasetBuilder, root: Path) -> None:
    images_root, labels_root = root / "images", root / "labels"
    require_dir(images_root, "Ship test images root")
    require_dir(labels_root, "Ship test labels root")
    for image_dir in sorted(path for path in images_root.iterdir() if path.is_dir()):
        modality = modality_from_name(image_dir.name)
        label_dir = labels_root / image_dir.name
        images = list_images(image_dir)
        validate_one_to_one(images, label_dir, f"ship test {modality}")
        for image in images:
            builder.add(
                image,
                label_dir / f"{image.stem}.txt",
                split="test_ship",
                target_stem=f"ship_test_{image.stem}",
                source_type="ship_gt",
                modality=modality,
                source_dataset="ship_extratest_no_overlap",
                expected_source_class=0,
                target_class=0,
            )


def add_bridge_test(builder: DatasetBuilder, root: Path) -> None:
    for modality in MODALITIES:
        image_dir = root / "images" / f"test_{modality}"
        label_dir = root / "labels" / f"test_{modality}"
        images = list_images(image_dir)
        validate_one_to_one(images, label_dir, f"bridge test {modality}")
        for image in images:
            builder.add(
                image,
                label_dir / f"{image.stem}.txt",
                split="test_bridge",
                target_stem=f"bridge_test_{modality}_{image.stem}",
                source_type="bridge_gt",
                modality=modality,
                source_dataset="total_bridge_test",
                expected_source_class=0,
                target_class=1,
            )


def overlap_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_split[record["split"]][record["hash_sha256"]].append(record["source_image"])
    pairs = (
        ("train", "val"),
        ("train", "test_ship"),
        ("train", "test_bridge"),
        ("val", "test_ship"),
        ("val", "test_bridge"),
    )
    report = {}
    for left, right in pairs:
        common = sorted(set(by_split[left]) & set(by_split[right]))
        report[f"{left}_vs_{right}"] = {
            "count": len(common),
            "examples": [
                {
                    "sha256": digest,
                    "left": by_split[left][digest],
                    "right": by_split[right][digest],
                }
                for digest in common[:20]
            ],
        }
    return report


def write_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "split",
        "source_type",
        "source_image",
        "source_label",
        "target_image",
        "target_label",
        "class_ids",
        "num_boxes",
        "modality",
        "source_dataset",
        "hash_sha256",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def build_summary(
    records: list[dict[str, Any]], args: argparse.Namespace, final_root: Path, pseudo: dict[str, Any]
) -> dict[str, Any]:
    counts = Counter(record["split"] for record in records)
    boxes = Counter()
    histograms: dict[str, Counter] = defaultdict(Counter)
    modalities: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        count = int(record["num_boxes"])
        boxes[record["split"]] += count
        if record["class_ids"]:
            histograms[record["split"]][record["class_ids"]] += count
        if record["source_type"] == "bridge_gt" and record["split"] in {"train", "val"}:
            modalities[record["split"]][record["modality"]] += 1
    overlaps = overlap_report(records)
    if any(item["count"] for item in overlaps.values()):
        raise RuntimeError(f"SHA256 split leakage detected: {overlaps}")
    ship_train = [record for record in records if record["split"] == "train" and record["source_type"] == "ship_teacher_pseudo"]
    ship_val = [record for record in records if record["split"] == "val" and record["source_type"] == "ship_gt"]
    bridge_train = [record for record in records if record["split"] == "train" and record["source_type"] == "bridge_gt"]
    bridge_val = [record for record in records if record["split"] == "val" and record["source_type"] == "bridge_gt"]
    sampled_bridge_images = {
        split: {
            modality: [
                record["source_image"]
                for record in records
                if record["split"] == split
                and record["source_type"] == "bridge_gt"
                and record["modality"] == modality
            ]
            for modality in MODALITIES
        }
        for split in ("train", "val")
    }
    return {
        "dataset": "bridge_ship_distill_2cls_v2",
        "created_at": datetime.now().isoformat(),
        "classes": {"0": "ship", "1": "bridge"},
        "copy_mode": args.copy_mode,
        "ship_train_images": len(ship_train),
        "ship_train_boxes": sum(int(record["num_boxes"]) for record in ship_train),
        "ship_val_images": len(ship_val),
        "ship_val_boxes": sum(int(record["num_boxes"]) for record in ship_val),
        "bridge_train_images": len(bridge_train),
        "bridge_train_boxes": sum(int(record["num_boxes"]) for record in bridge_train),
        "bridge_val_images": len(bridge_val),
        "bridge_val_boxes": sum(int(record["num_boxes"]) for record in bridge_val),
        "test_ship_images": counts["test_ship"],
        "test_ship_boxes": boxes["test_ship"],
        "test_bridge_images": counts["test_bridge"],
        "test_bridge_boxes": boxes["test_bridge"],
        "train_class_histogram": {"0": histograms["train"]["0"], "1": histograms["train"]["1"]},
        "val_class_histogram": {"0": histograms["val"]["0"], "1": histograms["val"]["1"]},
        "test_ship_class_histogram": {
            "0": histograms["test_ship"]["0"],
            "1": histograms["test_ship"]["1"],
        },
        "test_bridge_class_histogram": {
            "0": histograms["test_bridge"]["0"],
            "1": histograms["test_bridge"]["1"],
        },
        "bridge_train_per_modality": dict(modalities["train"]),
        "bridge_val_per_modality": dict(modalities["val"]),
        "sampled_bridge_images": sampled_bridge_images,
        "ship_pseudo_mode": pseudo["mode"],
        "ship_pseudo_details": pseudo,
        "ship_teacher_weights": pseudo.get("teacher_weights"),
        "ship_pseudo_conf": args.ship_conf,
        "seed": args.seed,
        "sha256_overlap": overlaps,
        "no_total_bridge_test_in_train_val": True,
        "no_ship_test_in_train_val": True,
        "bridge_labels_remapped_0_to_1": True,
        "ship_labels_kept_as_0": True,
        "data_yaml": str((final_root / "data.yaml").resolve()),
        "manifest_csv": str((final_root / "manifest.csv").resolve()),
        "build_summary_json": str((final_root / "build_summary.json").resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a clean controlled ship/bridge 2-class YOLO dataset.")
    parser.add_argument("--ship-train-images", default=DEFAULT_SHIP_TRAIN_IMAGES)
    parser.add_argument("--ship-train-labels", default=DEFAULT_SHIP_TRAIN_LABELS)
    parser.add_argument("--ship-val-images", default=DEFAULT_SHIP_VAL_IMAGES)
    parser.add_argument("--ship-val-labels", default=DEFAULT_SHIP_VAL_LABELS)
    parser.add_argument("--bridge-trainval-root", default=DEFAULT_BRIDGE_TRAINVAL_ROOT)
    parser.add_argument("--bridge-test-root", default=DEFAULT_BRIDGE_TEST_ROOT)
    parser.add_argument("--ship-test-root", default=DEFAULT_SHIP_TEST_ROOT)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--bridge-train-per-modality", type=int, default=24)
    parser.add_argument("--bridge-val-per-modality", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-mode", choices=("copy", "symlink", "hardlink"), default="copy")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-empty-ship-pseudo", action="store_true")
    parser.add_argument("--generate-ship-pseudo", action="store_true")
    parser.add_argument("--ship-teacher-weights", default=DEFAULT_SHIP_TEACHER)
    parser.add_argument("--ship-conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--max-det", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bridge_train_per_modality <= 0 or args.bridge_val_per_modality <= 0:
        raise ValueError("Bridge samples per modality must be positive")
    ship_train_images_dir = resolve_path(args.ship_train_images)
    ship_train_labels_input = resolve_path(args.ship_train_labels)
    ship_val_images_dir = resolve_path(args.ship_val_images)
    ship_val_labels_dir = resolve_path(args.ship_val_labels)
    bridge_trainval_root = resolve_path(args.bridge_trainval_root)
    bridge_test_root = resolve_path(args.bridge_test_root)
    ship_test_root = resolve_path(args.ship_test_root)
    final_root = resolve_path(args.out_root)
    staging_root = final_root.parent / f".{final_root.name}.building"

    if final_root.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {final_root}")
    if staging_root.exists():
        raise FileExistsError(f"Staging directory already exists; inspect and remove it manually: {staging_root}")
    for path, description in (
        (ship_train_images_dir, "Ship train images"),
        (ship_val_images_dir, "Ship val images"),
        (ship_val_labels_dir, "Ship val labels"),
        (bridge_trainval_root, "Bridge trainval root"),
        (bridge_test_root, "Bridge test root"),
        (ship_test_root, "Ship test root"),
    ):
        require_dir(path, description)

    ship_train_images = list_images(ship_train_images_dir)
    ship_val_images = list_images(ship_val_images_dir)
    validate_one_to_one(ship_val_images, ship_val_labels_dir, "ship val GT")
    pseudo_temp: tempfile.TemporaryDirectory[str] | None = None
    try:
        if args.generate_ship_pseudo:
            pseudo_temp = tempfile.TemporaryDirectory(prefix="ship_pseudo_v2_")
            ship_train_labels_dir = Path(pseudo_temp.name).resolve()
            pseudo_details = generate_ship_pseudo_labels(ship_train_images, ship_train_labels_dir, args)
        else:
            ship_train_labels_dir = ship_train_labels_input
            validate_one_to_one(ship_train_images, ship_train_labels_dir, "ship train pseudo")
            empty_labels = 0
            pseudo_boxes = 0
            for image in ship_train_images:
                _, count = read_and_remap_label(
                    ship_train_labels_dir / f"{image.stem}.txt",
                    expected_source_class=0,
                    target_class=0,
                    allow_empty=args.allow_empty_ship_pseudo,
                    description="ship_teacher_pseudo",
                )
                empty_labels += int(count == 0)
                pseudo_boxes += count
            pseudo_details = {
                "mode": "reused",
                "labels_dir": str(ship_train_labels_dir),
                "teacher_weights": None,
                "images": len(ship_train_images),
                "boxes": pseudo_boxes,
                "empty_labels": empty_labels,
                "conf": args.ship_conf,
            }

        rng = random.Random(args.seed)
        selected_bridge_train = {}
        selected_bridge_val = {}
        for modality in MODALITIES:
            train_candidates = bridge_candidates(bridge_trainval_root, "train", modality)
            val_candidates = bridge_candidates(bridge_trainval_root, "val", modality)
            if len(train_candidates) < args.bridge_train_per_modality:
                raise ValueError(
                    f"Not enough bridge train {modality} candidates: "
                    f"need {args.bridge_train_per_modality}, have {len(train_candidates)}"
                )
            if len(val_candidates) < args.bridge_val_per_modality:
                raise ValueError(
                    f"Not enough bridge val {modality} candidates: "
                    f"need {args.bridge_val_per_modality}, have {len(val_candidates)}"
                )
            selected_bridge_train[modality] = sorted(
                rng.sample(train_candidates, args.bridge_train_per_modality), key=lambda item: item[0].name
            )
            selected_bridge_val[modality] = sorted(
                rng.sample(val_candidates, args.bridge_val_per_modality), key=lambda item: item[0].name
            )

        staging_root.mkdir(parents=True)
        builder = DatasetBuilder(staging_root, final_root, args.copy_mode)
        for image in ship_train_images:
            builder.add(
                image,
                ship_train_labels_dir / f"{image.stem}.txt",
                split="train",
                target_stem=f"ship_train_{image.stem}",
                source_type="ship_teacher_pseudo",
                modality=modality_from_name(image.stem),
                source_dataset="ship_small_split",
                expected_source_class=0,
                target_class=0,
                allow_empty=args.allow_empty_ship_pseudo,
                generated_source_label=pseudo_details["mode"] == "generated",
            )
        for modality in MODALITIES:
            for image, label in selected_bridge_train[modality]:
                builder.add(
                    image,
                    label,
                    split="train",
                    target_stem=f"bridge_train_{modality}_{image.stem}",
                    source_type="bridge_gt",
                    modality=modality,
                    source_dataset="total_bridge_trainval",
                    expected_source_class=0,
                    target_class=1,
                )
        for image in ship_val_images:
            builder.add(
                image,
                ship_val_labels_dir / f"{image.stem}.txt",
                split="val",
                target_stem=f"ship_val_{image.stem}",
                source_type="ship_gt",
                modality=modality_from_name(image.stem),
                source_dataset="ship_small_split",
                expected_source_class=0,
                target_class=0,
            )
        for modality in MODALITIES:
            for image, label in selected_bridge_val[modality]:
                builder.add(
                    image,
                    label,
                    split="val",
                    target_stem=f"bridge_val_{modality}_{image.stem}",
                    source_type="bridge_gt",
                    modality=modality,
                    source_dataset="total_bridge_trainval",
                    expected_source_class=0,
                    target_class=1,
                )
        add_ship_test(builder, ship_test_root)
        add_bridge_test(builder, bridge_test_root)

        manifest_path = staging_root / "manifest.csv"
        write_manifest(manifest_path, builder.records)
        summary = build_summary(builder.records, args, final_root, pseudo_details)
        (staging_root / "build_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        data = {
            "path": str(final_root),
            "train": "images/train",
            "val": "images/val",
            "nc": 2,
            "names": {0: "ship", 1: "bridge"},
        }
        (staging_root / "data.yaml").write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )

        if final_root.exists():
            shutil.rmtree(final_root)
        staging_root.replace(final_root)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise
    finally:
        if pseudo_temp is not None:
            pseudo_temp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
