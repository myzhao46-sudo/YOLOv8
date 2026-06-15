# tools/data/make_ship_extratest_no_overlap.py
# -*- coding: utf-8 -*-

"""
Create an external ship extra-test dataset from the original ship dataset,
excluding all images already used by the current ship_small_split train/val.

Goal:
  - Source raw ship dataset:
      E:/YOLODATA/Ship
      E:/YOLODATA/Ship/labels

  - Current used small split:
      C:/Users/DOCTOR/Documents/GitHub/YOLOv8/ultralytics/datasets/ship_small_split

  - Output:
      C:/Users/DOCTOR/Documents/GitHub/YOLOv8/ultralytics/datasets/ship_extratest_no_overlap

  - Select up to 100 images per modality:
      rgb / sar / ir

  - Labels:
      YOLO detect 5-column format:
        cls cx cy w h
      force cls = 0 for ship in global4 namespace.

This script:
  - Does NOT modify original data.
  - Does NOT modify ship_small_split.
  - Does NOT train.
  - Does NOT touch model weights.
  - Only creates a new external test dataset folder and YAML files.

Output structure:
  ship_extratest_no_overlap/
    images/extratest_rgb
    labels/extratest_rgb
    images/extratest_sar
    labels/extratest_sar
    images/extratest_ir
    labels/extratest_ir
    selection_summary.json

YAML files:
  configs/datasets/global4_eval_ship_extratest_rgb.yaml
  configs/datasets/global4_eval_ship_extratest_sar.yaml
  configs/datasets/global4_eval_ship_extratest_ir.yaml
  configs/datasets/global4_eval_ship_extratest_all.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


# =============================================================================
# Fixed default paths
# =============================================================================

REPO_ROOT = Path(r"C:\Users\DOCTOR\Documents\GitHub\YOLOv8")

SRC_ROOT = Path(r"E:\YOLODATA\Ship")
SRC_LABELS_DIR = Path(r"E:\YOLODATA\Ship\labels")

CURRENT_SMALL_ROOT = REPO_ROOT / r"ultralytics\datasets\ship_small_split"

DST_ROOT = REPO_ROOT / r"ultralytics\datasets\ship_extratest_no_overlap"
CONFIGS_ROOT = REPO_ROOT / r"configs\datasets"

GLOBAL4_NAMES = {
    0: "ship",
    1: "harbor",
    2: "tank",
    3: "bridge",
}

TARGET_CLASS_ID = 0
PER_MODALITY = 100
SEED = 42

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# =============================================================================
# Basic helpers
# =============================================================================

def log(msg: object = "") -> None:
    print(msg, flush=True)


def norm_path(p: Path) -> str:
    """Use forward slashes for Ultralytics YAML on Windows."""
    return p.resolve().as_posix()


def infer_modality(path: Path) -> str:
    """
    Infer modality from filename.

    Expected examples:
      rgb_564.jpg       -> rgb
      sar_0031427.jpg   -> sar
      ir_1_4.jpg        -> ir

    Modify this function if your filename convention changes.
    """
    stem = path.stem.lower()

    if stem.startswith("rgb"):
        return "rgb"

    if stem.startswith("sar"):
        return "sar"

    if (
        stem.startswith("ir")
        or stem.startswith("infr")
        or stem.startswith("infra")
        or stem.startswith("thermal")
        or stem.startswith("lwir")
    ):
        return "ir"

    return "unknown"


def is_inside_labels_dir(path: Path, labels_dir: Path) -> bool:
    try:
        path.resolve().relative_to(labels_dir.resolve())
        return True
    except Exception:
        return False


def find_images_recursive(root: Path, labels_dir: Path | None = None) -> List[Path]:
    """
    Recursively find images under root, skipping labels directory.
    This is intentional because the original ship dataset may have:
      images/
      images/train/
      images/val/
      images/test/
      or other nested folders.
    """
    images = []

    for p in root.rglob("*"):
        if not p.is_file():
            continue

        if p.suffix.lower() not in IMG_EXTS:
            continue

        if labels_dir is not None and labels_dir.exists():
            if is_inside_labels_dir(p, labels_dir):
                continue

        images.append(p)

    return sorted(images)


def image_sha1(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def collect_used_small_images(small_root: Path, use_hash: bool = False) -> Dict:
    """
    Collect used image names/stems from ship_small_split train/val.
    Name/stem exclusion is the main rule because the user says names should remain the same.
    Optional hash exclusion is available for renamed duplicate safety.
    """
    used_images = find_images_recursive(small_root, labels_dir=None)

    used_names = set()
    used_stems = set()
    used_hashes = set()

    for img in used_images:
        used_names.add(img.name.lower())
        used_stems.add(img.stem.lower())

        if use_hash:
            try:
                used_hashes.add(image_sha1(img))
            except Exception as e:
                log(f"[WARN] Failed hashing used image {img}: {e}")

    return {
        "count": len(used_images),
        "names": used_names,
        "stems": used_stems,
        "hashes": used_hashes,
    }


def is_overlapping_with_small(img: Path, used: Dict, use_hash: bool = False) -> Tuple[bool, str]:
    if img.name.lower() in used["names"]:
        return True, "same filename"

    if img.stem.lower() in used["stems"]:
        return True, "same stem"

    if use_hash:
        try:
            h = image_sha1(img)
            if h in used["hashes"]:
                return True, "same sha1"
        except Exception as e:
            return True, f"hash failed: {e}"

    return False, ""


def build_label_index(labels_dir: Path) -> Dict[str, Path]:
    """
    Build stem -> label path index.

    Supports flat labels:
      E:/YOLODATA/Ship/labels/xxx.txt

    Also supports nested labels recursively if they exist.
    """
    label_index = {}

    for p in sorted(labels_dir.rglob("*.txt")):
        key = p.stem.lower()
        if key in label_index:
            log(f"[WARN] Duplicate label stem found, keeping first: {key}")
            log(f"       first: {label_index[key]}")
            log(f"       skip : {p}")
            continue
        label_index[key] = p

    return label_index


def clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def read_image_size(img_path: Path) -> Tuple[int, int]:
    with Image.open(img_path) as im:
        return im.size  # width, height


# =============================================================================
# Label conversion
# =============================================================================

def parse_label_to_yolo5_lines(
    src_label: Path,
    img_path: Path,
    force_class: int = 0,
) -> Tuple[List[str], int, int]:
    """
    Convert source label into YOLO detect 5-col lines:
      cls cx cy w h

    Supports:
      5-col YOLO:
        cls cx cy w h
      9-col polygon / DOTA-like:
        cls x1 y1 x2 y2 x3 y3 x4 y4

    Handles normalized coords and simple pixel coords.
    Output class id is forced to force_class.

    Returns:
      out_lines, kept_count, skipped_count
    """
    img_w, img_h = read_image_size(img_path)

    out_lines = []
    kept = 0
    skipped = 0

    if not src_label.exists():
        return out_lines, kept, 1

    for line_no, line in enumerate(src_label.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue

        parts = line.split()

        try:
            if len(parts) == 5:
                # cls cx cy w h
                vals = list(map(float, parts[1:5]))
                cx, cy, w, h = vals

                # If values look like pixel coords, normalize them.
                if max(abs(cx), abs(cy), abs(w), abs(h)) > 1.5:
                    cx = cx / img_w
                    w = w / img_w
                    cy = cy / img_h
                    h = h / img_h

                x1 = cx - w / 2.0
                y1 = cy - h / 2.0
                x2 = cx + w / 2.0
                y2 = cy + h / 2.0

            elif len(parts) >= 9:
                # cls x1 y1 x2 y2 x3 y3 x4 y4
                coords = list(map(float, parts[1:9]))
                xs = coords[0::2]
                ys = coords[1::2]

                # If polygon coords look like pixels, normalize.
                if max(max(map(abs, xs)), max(map(abs, ys))) > 1.5:
                    xs = [x / img_w for x in xs]
                    ys = [y / img_h for y in ys]

                x1 = min(xs)
                x2 = max(xs)
                y1 = min(ys)
                y2 = max(ys)

            else:
                skipped += 1
                continue

            x1 = clip01(x1)
            y1 = clip01(y1)
            x2 = clip01(x2)
            y2 = clip01(y2)

            bw = x2 - x1
            bh = y2 - y1

            if bw <= 0 or bh <= 0:
                skipped += 1
                continue

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            out_lines.append(f"{force_class} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}\n")
            kept += 1

        except Exception as e:
            log(f"[WARN] Failed parsing {src_label} line {line_no}: {line} | {e}")
            skipped += 1

    return out_lines, kept, skipped


def write_converted_label(
    src_label: Path,
    dst_label: Path,
    img_path: Path,
    force_class: int = 0,
) -> Tuple[int, int]:
    lines, kept, skipped = parse_label_to_yolo5_lines(src_label, img_path, force_class)
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    dst_label.write_text("".join(lines), encoding="utf-8")
    return kept, skipped


# =============================================================================
# Dataset creation
# =============================================================================

def select_candidates(
    src_root: Path,
    src_labels_dir: Path,
    small_root: Path,
    per_modality: int,
    seed: int,
    use_hash: bool = False,
) -> Tuple[Dict[str, List[Tuple[Path, Path]]], Dict]:
    """
    Return selected images grouped by modality:
      modality -> [(img_path, label_path), ...]
    """
    if not src_root.exists():
        raise FileNotFoundError(f"SRC_ROOT not found: {src_root}")

    if not src_labels_dir.exists():
        raise FileNotFoundError(f"SRC_LABELS_DIR not found: {src_labels_dir}")

    if not small_root.exists():
        raise FileNotFoundError(f"CURRENT_SMALL_ROOT not found: {small_root}")

    used = collect_used_small_images(small_root, use_hash=use_hash)
    label_index = build_label_index(src_labels_dir)
    raw_images = find_images_recursive(src_root, labels_dir=src_labels_dir)

    stats = {
        "src_root": str(src_root),
        "src_labels_dir": str(src_labels_dir),
        "small_root": str(small_root),
        "raw_image_count": len(raw_images),
        "raw_label_count": len(label_index),
        "used_small_image_count": used["count"],
        "excluded_overlap": 0,
        "excluded_overlap_reasons": defaultdict(int),
        "excluded_unknown_modality": 0,
        "excluded_missing_label": 0,
        "excluded_empty_or_bad_label": 0,
        "available_by_modality": {},
        "selected_by_modality": {},
        "shortage_by_modality": {},
    }

    groups = defaultdict(list)
    unknown_examples = []
    missing_label_examples = []
    bad_label_examples = []
    overlap_examples = []

    for img in raw_images:
        overlap, reason = is_overlapping_with_small(img, used, use_hash=use_hash)
        if overlap:
            stats["excluded_overlap"] += 1
            stats["excluded_overlap_reasons"][reason] += 1
            if len(overlap_examples) < 20:
                overlap_examples.append(f"{img} | {reason}")
            continue

        modality = infer_modality(img)
        if modality == "unknown":
            stats["excluded_unknown_modality"] += 1
            if len(unknown_examples) < 20:
                unknown_examples.append(str(img))
            continue

        label = label_index.get(img.stem.lower())
        if label is None:
            stats["excluded_missing_label"] += 1
            if len(missing_label_examples) < 20:
                missing_label_examples.append(str(img))
            continue

        # Pre-check: require at least one valid box.
        try:
            _, kept, _ = parse_label_to_yolo5_lines(label, img, TARGET_CLASS_ID)
        except Exception as e:
            kept = 0
            if len(bad_label_examples) < 20:
                bad_label_examples.append(f"{img} | {label} | {e}")

        if kept <= 0:
            stats["excluded_empty_or_bad_label"] += 1
            if len(bad_label_examples) < 20:
                bad_label_examples.append(f"{img} | {label}")
            continue

        groups[modality].append((img, label))

    rng = random.Random(seed)
    selected = {}

    for modality in ["rgb", "sar", "ir"]:
        items = groups.get(modality, [])
        rng.shuffle(items)

        n_available = len(items)
        n_select = min(per_modality, n_available)

        selected[modality] = sorted(items[:n_select], key=lambda x: x[0].name.lower())

        stats["available_by_modality"][modality] = n_available
        stats["selected_by_modality"][modality] = n_select
        stats["shortage_by_modality"][modality] = max(0, per_modality - n_available)

    stats["examples"] = {
        "overlap_examples": overlap_examples,
        "unknown_modality_examples": unknown_examples,
        "missing_label_examples": missing_label_examples,
        "bad_label_examples": bad_label_examples,
    }

    # Convert defaultdicts to plain dicts for JSON.
    stats["excluded_overlap_reasons"] = dict(stats["excluded_overlap_reasons"])

    return selected, stats


def copy_selected_dataset(
    selected: Dict[str, List[Tuple[Path, Path]]],
    dst_root: Path,
    overwrite: bool = False,
) -> Dict:
    if dst_root.exists():
        if overwrite:
            log(f"[INFO] Removing existing output dataset: {dst_root}")
            shutil.rmtree(dst_root)
        else:
            raise FileExistsError(
                f"Output already exists: {dst_root}\n"
                f"Use --overwrite only if you want to recreate this output dataset."
            )

    copy_stats = {}

    for modality, items in selected.items():
        split_name = f"extratest_{modality}"

        dst_img_dir = dst_root / "images" / split_name
        dst_lbl_dir = dst_root / "labels" / split_name

        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        boxes = 0
        skipped = 0

        for img, label in items:
            dst_img = dst_img_dir / img.name
            dst_lbl = dst_lbl_dir / f"{img.stem}.txt"

            shutil.copy2(img, dst_img)
            kept, bad = write_converted_label(
                src_label=label,
                dst_label=dst_lbl,
                img_path=img,
                force_class=TARGET_CLASS_ID,
            )

            boxes += kept
            skipped += bad

        copy_stats[modality] = {
            "images": len(items),
            "boxes": boxes,
            "skipped_label_rows": skipped,
            "images_dir": str(dst_img_dir),
            "labels_dir": str(dst_lbl_dir),
        }

    return copy_stats


def write_global4_yaml(yaml_path: Path, val_image_dirs: List[Path]) -> None:
    """
    Write eval YAML. Include train=val because some Ultralytics dataset checks
    require both train and val fields even for val-only workflows.
    """
    lines = []

    lines.append("train:")
    for p in val_image_dirs:
        lines.append(f"  - {norm_path(p)}")
    lines.append("")

    lines.append("val:")
    for p in val_image_dirs:
        lines.append(f"  - {norm_path(p)}")
    lines.append("")

    lines.append("nc: 4")
    lines.append("")
    lines.append("names:")
    for k, v in GLOBAL4_NAMES.items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"[OK] wrote YAML: {yaml_path}")


def write_yamls(dst_root: Path, configs_root: Path) -> Dict[str, str]:
    rgb_dir = dst_root / "images" / "extratest_rgb"
    sar_dir = dst_root / "images" / "extratest_sar"
    ir_dir = dst_root / "images" / "extratest_ir"

    yamls = {
        "rgb": configs_root / "global4_eval_ship_extratest_rgb.yaml",
        "sar": configs_root / "global4_eval_ship_extratest_sar.yaml",
        "ir": configs_root / "global4_eval_ship_extratest_ir.yaml",
        "all": configs_root / "global4_eval_ship_extratest_all.yaml",
    }

    write_global4_yaml(yamls["rgb"], [rgb_dir])
    write_global4_yaml(yamls["sar"], [sar_dir])
    write_global4_yaml(yamls["ir"], [ir_dir])
    write_global4_yaml(yamls["all"], [rgb_dir, sar_dir, ir_dir])

    return {k: str(v) for k, v in yamls.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", type=str, default=str(SRC_ROOT))
    parser.add_argument("--src-labels-dir", type=str, default=str(SRC_LABELS_DIR))
    parser.add_argument("--small-root", type=str, default=str(CURRENT_SMALL_ROOT))
    parser.add_argument("--dst-root", type=str, default=str(DST_ROOT))
    parser.add_argument("--configs-root", type=str, default=str(CONFIGS_ROOT))
    parser.add_argument("--per-modality", type=int, default=PER_MODALITY)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--hash-check", action="store_true", help="Also exclude duplicates by SHA1 image hash.")
    parser.add_argument("--overwrite", action="store_true", help="Recreate output dataset folder if it already exists.")
    args = parser.parse_args()

    src_root = Path(args.src_root)
    src_labels_dir = Path(args.src_labels_dir)
    small_root = Path(args.small_root)
    dst_root = Path(args.dst_root)
    configs_root = Path(args.configs_root)

    log("=" * 100)
    log("Make ship extratest dataset with no overlap against ship_small_split")
    log("=" * 100)
    log(f"src_root       = {src_root}")
    log(f"src_labels_dir = {src_labels_dir}")
    log(f"small_root     = {small_root}")
    log(f"dst_root       = {dst_root}")
    log(f"configs_root   = {configs_root}")
    log(f"per_modality   = {args.per_modality}")
    log(f"seed           = {args.seed}")
    log(f"hash_check     = {args.hash_check}")
    log(f"force_class    = {TARGET_CLASS_ID}  # ship in global4")
    log("=" * 100)

    selected, select_stats = select_candidates(
        src_root=src_root,
        src_labels_dir=src_labels_dir,
        small_root=small_root,
        per_modality=args.per_modality,
        seed=args.seed,
        use_hash=args.hash_check,
    )

    log("")
    log("[SELECTION SUMMARY]")
    log(json.dumps(select_stats, ensure_ascii=False, indent=2))

    for modality in ["rgb", "sar", "ir"]:
        available = select_stats["available_by_modality"].get(modality, 0)
        selected_n = select_stats["selected_by_modality"].get(modality, 0)
        shortage = select_stats["shortage_by_modality"].get(modality, 0)

        if shortage > 0:
            log(
                f"[WARN] {modality}: only {available} available after removing overlap; "
                f"selected {selected_n}, shortage {shortage}."
            )
        else:
            log(f"[OK] {modality}: selected {selected_n}/{args.per_modality}.")

    log("")
    log("[COPY DATASET]")
    copy_stats = copy_selected_dataset(
        selected=selected,
        dst_root=dst_root,
        overwrite=args.overwrite,
    )
    log(json.dumps(copy_stats, ensure_ascii=False, indent=2))

    log("")
    log("[WRITE YAML]")
    yaml_stats = write_yamls(dst_root=dst_root, configs_root=configs_root)

    summary = {
        "select_stats": select_stats,
        "copy_stats": copy_stats,
        "yaml_stats": yaml_stats,
        "output_dataset": str(dst_root),
        "class_id_for_ship": TARGET_CLASS_ID,
        "global4_names": GLOBAL4_NAMES,
    }

    summary_path = dst_root / "selection_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    log("")
    log("[DONE]")
    log(f"Output dataset: {dst_root}")
    log(f"Summary JSON:   {summary_path}")
    log("YAML files:")
    for k, v in yaml_stats.items():
        log(f"  {k}: {v}")

    log("")
    log("Next possible eval YAML:")
    log(f"  {yaml_stats['all']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())