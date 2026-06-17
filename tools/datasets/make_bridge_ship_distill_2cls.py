# tools/datasets/make_bridge_ship_distill_2cls.py
# -*- coding: utf-8 -*-

"""
Create a flat 2-class distillation dataset:
  class 0: ship teacher pseudo labels
  class 1: bridge ground-truth labels

The script copies data into a new dataset, writes data.yaml and manifest.csv, and
strictly accepts only 5-column YOLO detect labels. It does not modify source data.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

import yaml


REPO_ROOT = Path(r"C:\Users\DOCTOR\Documents\GitHub\YOLOv8")
DEFAULT_BRIDGE_ROOT = REPO_ROOT / "ultralytics" / "datasets" / "bridge_small_labels1"
DEFAULT_SHIP_IMAGES = REPO_ROOT / "ultralytics" / "datasets" / "ship_small_split" / "images" / "train"
DEFAULT_SHIP_LABELS = REPO_ROOT / "ultralytics" / "datasets" / "ship_teacher_pseudo_conf025" / "labels" / "train"
DEFAULT_OUT = REPO_ROOT / "ultralytics" / "datasets" / "bridge_ship_distill_2cls"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Merge bridge GT and ship teacher pseudo into 2-class dataset.")
    parser.add_argument("--bridge-root", default=str(DEFAULT_BRIDGE_ROOT))
    parser.add_argument("--ship-images", default=str(DEFAULT_SHIP_IMAGES))
    parser.add_argument("--ship-labels", default=str(DEFAULT_SHIP_LABELS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=0, help="Reserved for future split logic; not used in v1.")
    return parser.parse_args()


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_").lower()


def list_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"images directory not found: {images_dir}")
    return sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def validate_label_file(label_path: Path, allowed_classes: set[int], source_type: str) -> dict[int, int]:
    hist = {0: 0, 1: 0}
    if not label_path.exists():
        raise FileNotFoundError(f"missing label for {source_type}: {label_path}")
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"non-5-column label row in {label_path}:{line_no}: {line}")
        try:
            cls = int(float(parts[0]))
            _ = [float(x) for x in parts[1:]]
        except Exception as e:
            raise ValueError(f"bad numeric label row in {label_path}:{line_no}: {line}") from e
        if cls not in allowed_classes:
            raise ValueError(
                f"class id {cls} not allowed for {source_type} in {label_path}:{line_no}; allowed={sorted(allowed_classes)}"
            )
        if cls not in {0, 1}:
            raise ValueError(f"class id {cls} would exceed 2-class student range in {label_path}:{line_no}")
        hist[cls] += 1
    return hist


def copy_pair(
    image_path: Path,
    label_path: Path,
    out_root: Path,
    split: str,
    prefix: str,
    source_type: str,
    modality: str,
    writer: csv.DictWriter,
    allowed_classes: set[int],
) -> dict:
    hist = validate_label_file(label_path, allowed_classes, source_type)
    new_name = f"{prefix}_{image_path.stem}{image_path.suffix.lower()}"
    new_label_name = f"{prefix}_{image_path.stem}.txt"
    out_image = out_root / "images" / split / new_name
    out_label = out_root / "labels" / split / new_label_name
    out_image.parent.mkdir(parents=True, exist_ok=True)
    out_label.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, out_image)
    shutil.copy2(label_path, out_label)
    writer.writerow(
        {
            "new_image_path": str(out_image),
            "new_label_path": str(out_label),
            "source_image_path": str(image_path),
            "source_label_path": str(label_path),
            "source_type": source_type,
            "modality": modality,
            "split": split,
        }
    )
    return {"images": 1, "class0": hist[0], "class1": hist[1], "bad_rows": 0}


def modality_from_name(name: str) -> str:
    lower = name.lower()
    if "rgb" in lower or "dior" in lower:
        return "RGB"
    if "sar" in lower or "msar" in lower or "masr" in lower:
        return "SAR"
    if "ir" in lower or "infr" in lower or "massmind" in lower:
        return "IR"
    return "unknown"


def iter_bridge_items(bridge_root: Path):
    for dataset_dir in sorted(p for p in bridge_root.iterdir() if p.is_dir()):
        for split in ["train", "val"]:
            images_dir = dataset_dir / "images" / split
            labels_dir = dataset_dir / "labels" / split
            if not images_dir.exists():
                continue
            for image_path in list_images(images_dir):
                yield {
                    "image": image_path,
                    "label": labels_dir / f"{image_path.stem}.txt",
                    "split": split,
                    "prefix": slug(dataset_dir.name),
                    "modality": modality_from_name(dataset_dir.name),
                }


def iter_ship_items(ship_images: Path, ship_labels: Path):
    for image_path in list_images(ship_images):
        yield {
            "image": image_path,
            "label": ship_labels / f"{image_path.stem}.txt",
            "split": "train",
            "prefix": "ship_teacher",
            "modality": "ship",
        }


def write_data_yaml(out_root: Path) -> Path:
    data = {
        "path": out_root.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "nc": 2,
        "names": {0: "ship", 1: "bridge"},
    }
    path = out_root / "data.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def add_stats(total: dict, stat: dict, split: str, source_type: str) -> None:
    total["images"] += stat["images"]
    total["class0"] += stat["class0"]
    total["class1"] += stat["class1"]
    total["bad_rows"] += stat["bad_rows"]
    total[f"{split}_images"] += stat["images"]
    total[f"{source_type}_images"] += stat["images"]


def main() -> int:
    args = parse_args()
    bridge_root = Path(args.bridge_root).resolve()
    ship_images = Path(args.ship_images).resolve()
    ship_labels = Path(args.ship_labels).resolve()
    out_root = Path(args.out).resolve()

    if out_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists, use --overwrite to replace: {out_root}")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_path = out_root / "manifest.csv"
    total = {
        "images": 0,
        "train_images": 0,
        "val_images": 0,
        "bridge_gt_images": 0,
        "ship_teacher_pseudo_images": 0,
        "class0": 0,
        "class1": 0,
        "bad_rows": 0,
    }

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "new_image_path",
                "new_label_path",
                "source_image_path",
                "source_label_path",
                "source_type",
                "modality",
                "split",
            ],
        )
        writer.writeheader()

        bridge_count = 0
        for item in iter_bridge_items(bridge_root):
            stat = copy_pair(
                item["image"],
                item["label"],
                out_root,
                item["split"],
                item["prefix"],
                "bridge_gt",
                item["modality"],
                writer,
                allowed_classes={1},
            )
            add_stats(total, stat, item["split"], "bridge_gt")
            bridge_count += 1

        ship_count = 0
        for item in iter_ship_items(ship_images, ship_labels):
            stat = copy_pair(
                item["image"],
                item["label"],
                out_root,
                item["split"],
                item["prefix"],
                "ship_teacher_pseudo",
                item["modality"],
                writer,
                allowed_classes={0},
            )
            add_stats(total, stat, item["split"], "ship_teacher_pseudo")
            ship_count += 1

    data_yaml = write_data_yaml(out_root)

    print("[MERGE SUMMARY]", flush=True)
    print(f"bridge root: {bridge_root}", flush=True)
    print(f"ship images: {ship_images}", flush=True)
    print(f"ship labels: {ship_labels}", flush=True)
    print(f"out: {out_root}", flush=True)
    print(f"bridge images: {bridge_count}", flush=True)
    print(f"ship pseudo images: {ship_count}", flush=True)
    print(f"train images: {total['train_images']}", flush=True)
    print(f"val images: {total['val_images']}", flush=True)
    print(f"class histogram: {{0: {total['class0']}, 1: {total['class1']}}}", flush=True)
    print(f"bad rows: {total['bad_rows']}", flush=True)
    print(f"manifest: {manifest_path}", flush=True)
    print(f"data.yaml: {data_yaml}", flush=True)
    print("bridge labels kept as class 1: True", flush=True)
    print("ship pseudo labels kept as class 0: True", flush=True)
    print("no class id outside {0,1}: True", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
