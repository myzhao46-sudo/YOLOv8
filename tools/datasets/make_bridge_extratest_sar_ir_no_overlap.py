# tools/datasets/make_bridge_extratest_sar_ir_no_overlap.py
# -*- coding: utf-8 -*-

"""
Create bridge external extra-test dataset for the current 2-class YOLOE student.

Current student class space:
  0 ship
  1 bridge

This script only handles SAR + IR/LWIR first.
It intentionally does NOT process DIOR/RGB XML yet.

Default source datasets:
  SAR/MSAR:
    E:/YOLODATA/bridge_sar_msar_yolo_cls3_300/images/train
    labels inferred from:
    E:/YOLODATA/bridge_sar_msar_yolo_cls3_300/labels/train

  IR/MassMIND:
    E:/YOLODATA/MassMIND_bridge_yolo/images/train
    labels inferred from:
    E:/YOLODATA/MassMIND_bridge_yolo/labels/train

No-leak reference:
  E:/YOLODATA/bridgeAll

Output:
  C:/Users/DOCTOR/Documents/GitHub/YOLOv8/ultralytics/datasets/extracttest_bridge

Output structure:
  extracttest_bridge/
    images/extratest_sar
    labels/extratest_sar
    images/extratest_ir
    labels/extratest_ir
    data.yaml
    data_sar.yaml
    data_ir.yaml
    manifest.csv
    selection_summary.json

Also writes optional config YAMLs:
  configs/datasets/student2_eval_bridge_extratest_sar.yaml
  configs/datasets/student2_eval_bridge_extratest_ir.yaml
  configs/datasets/student2_eval_bridge_extratest_all.yaml

Important:
  - Does NOT modify source datasets.
  - Does NOT modify E:/YOLODATA/bridgeAll.
  - Does NOT train.
  - Forces every output label class id to 1, because bridge is class 1
    in the current 2-class student.
  - Excludes overlap against bridgeAll by exact filename, exact stem,
    and normalized numeric/source IDs extracted from filenames.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from PIL import Image


# =============================================================================
# Defaults
# =============================================================================

REPO_ROOT = Path(r"C:\Users\DOCTOR\Documents\GitHub\YOLOv8")

SAR_IMAGE_ROOTS = [
    Path(r"E:\YOLODATA\bridge_sar_msar_yolo_cls3_300\images\train"),
    # If you later decide val can also be used as candidate source, pass:
    # --sar-image-roots E:\YOLODATA\bridge_sar_msar_yolo_cls3_300\images\train E:\YOLODATA\bridge_sar_msar_yolo_cls3_300\images\val
]

IR_IMAGE_ROOTS = [
    Path(r"E:\YOLODATA\MassMIND_bridge_yolo\images\train"),
]

USED_BRIDGEALL_ROOT = Path(r"E:\YOLODATA\bridgeAll")

DST_ROOT = REPO_ROOT / r"ultralytics\datasets\extracttest_bridge"
CONFIGS_ROOT = REPO_ROOT / r"configs\datasets"

PER_MODALITY = 100
SEED = 42
TARGET_CLASS_ID = 1

STUDENT2_NAMES = {
    0: "ship",
    1: "bridge",
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# =============================================================================
# Utilities
# =============================================================================

def log(msg: object = "") -> None:
    print(msg, flush=True)


def norm_path(p: Path) -> str:
    return p.resolve().as_posix()


def as_paths(xs: Sequence[str | Path]) -> List[Path]:
    return [Path(x) for x in xs]


def find_images_recursive(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def image_sha1(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_image_size(img_path: Path) -> Tuple[int, int]:
    with Image.open(img_path) as im:
        return im.size  # width, height


def clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def infer_family_from_path(path: Path) -> str:
    """
    Infer dataset/source family from path text.
    Used to avoid false exclusion when two unrelated datasets share numeric ids.
    """
    s = path.as_posix().lower()

    if "massmind" in s or "lwir" in s or "infr" in s or "infra" in s or "thermal" in s:
        return "ir"

    # Put MSAR before generic SAR.
    if "msar" in s or "/sar" in s or "\\sar" in str(path).lower() or "sar_" in s:
        return "sar"

    if "dior" in s or "rgb" in s:
        return "rgb"

    return "unknown"


def canonical_id_keys(path: Path) -> Set[str]:
    """
    Build robust id keys from filename/stem.

    Examples:
      msar_000123.jpg     -> {"msar_000123", "000123", "123", ...}
      a00158952.png       -> {"a00158952", "00158952", "158952", ...}
      MassMIND_a00123.jpg -> {"massmind_a00123", "a00123", "00123", "123", ...}

    We keep several variants because the same original image may have gained
    prefixes when moved into bridgeAll.
    """
    stem = path.stem.lower()
    base = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")

    keys: Set[str] = set()
    if base:
        keys.add(base)

    prefixes = [
        "msar_bridge_", "msar_", "sar_bridge_", "sar_",
        "massmind_bridge_", "massmind_", "infr_bridge_", "infr_", "infra_",
        "lwir_bridge_", "lwir_", "ir_bridge_", "ir_",
        "bridge_", "dior_bridge_", "dior_", "rgb_bridge_", "rgb_",
    ]

    stripped_variants = {base}
    changed = True
    while changed:
        changed = False
        new_items = set(stripped_variants)
        for x in stripped_variants:
            for pref in prefixes:
                if x.startswith(pref) and len(x) > len(pref):
                    y = x[len(pref):]
                    if y and y not in new_items:
                        new_items.add(y)
                        changed = True
        stripped_variants = new_items

    for x in stripped_variants:
        if x:
            keys.add(x)

    # Numeric groups are the most important "编号" key.
    digit_groups = re.findall(r"\d+", base)
    for g in digit_groups:
        if len(g) >= 2:
            keys.add(g)
            # Add integer-normalized version to handle leading zero changes.
            try:
                keys.add(str(int(g)))
            except Exception:
                pass

    if len(digit_groups) >= 2:
        joined = "_".join(digit_groups)
        keys.add(joined)
        try:
            keys.add("_".join(str(int(g)) for g in digit_groups))
        except Exception:
            pass

    return {k for k in keys if k}


def replace_images_with_labels_path(img_path: Path) -> Optional[Path]:
    """
    Try direct image->label path inference:
      .../images/train/xxx.jpg -> .../labels/train/xxx.txt
      .../Images/train/xxx.jpg -> .../labels/train/xxx.txt
    """
    parts = list(img_path.parts)
    lowered = [p.lower() for p in parts]

    # Use the last "images" component if multiple exist.
    idx = None
    for i, p in enumerate(lowered):
        if p == "images" or p == "image":
            idx = i

    if idx is None:
        return None

    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def labels_root_from_image_root(image_root: Path) -> Optional[Path]:
    """
    Infer labels root from an image root:
      E:/x/images/train -> E:/x/labels
      E:/x/images       -> E:/x/labels
    """
    parts = list(image_root.parts)
    lowered = [p.lower() for p in parts]

    idx = None
    for i, p in enumerate(lowered):
        if p == "images" or p == "image":
            idx = i

    if idx is None:
        return None

    # Dataset root is before "images"; label root is dataset_root/labels.
    return Path(*parts[:idx]) / "labels"


def build_label_index(label_roots: Iterable[Path]) -> Dict[str, Path]:
    """
    Build a stem -> txt label index from one or multiple label roots.
    If duplicate stems exist, keep the first and warn.
    """
    index: Dict[str, Path] = {}

    for root in label_roots:
        if root is None or not root.exists():
            continue

        for p in sorted(root.rglob("*.txt")):
            key = p.stem.lower()
            if key in index:
                log(f"[WARN] Duplicate label stem, keeping first: {key}")
                log(f"       first: {index[key]}")
                log(f"       skip : {p}")
                continue
            index[key] = p

    return index


def find_label_for_image(img: Path, label_index: Dict[str, Path]) -> Optional[Path]:
    direct = replace_images_with_labels_path(img)
    if direct is not None and direct.exists():
        return direct

    return label_index.get(img.stem.lower())


# =============================================================================
# Label conversion
# =============================================================================

def parse_label_to_yolo5_lines(
    src_label: Path,
    img_path: Path,
    force_class: int,
) -> Tuple[List[str], int, int]:
    """
    Convert label to YOLO detect 5-column:
      cls cx cy w h

    Supports:
      5-col YOLO:
        cls cx cy w h
      9-col polygon / DOTA-like:
        cls x1 y1 x2 y2 x3 y3 x4 y4

    Handles normalized coords and simple pixel coords.
    Output cls is always force_class.
    """
    img_w, img_h = read_image_size(img_path)

    out_lines: List[str] = []
    kept = 0
    skipped = 0

    if not src_label.exists():
        return out_lines, kept, 1

    text = src_label.read_text(encoding="utf-8", errors="ignore")

    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue

        parts = line.split()

        try:
            if len(parts) == 5:
                cx, cy, bw, bh = map(float, parts[1:5])

                # If values look like pixels, normalize them.
                if max(abs(cx), abs(cy), abs(bw), abs(bh)) > 1.5:
                    cx = cx / img_w
                    bw = bw / img_w
                    cy = cy / img_h
                    bh = bh / img_h

                x1 = cx - bw / 2.0
                y1 = cy - bh / 2.0
                x2 = cx + bw / 2.0
                y2 = cy + bh / 2.0

            elif len(parts) >= 9:
                coords = list(map(float, parts[1:9]))
                xs = coords[0::2]
                ys = coords[1::2]

                # If polygon coords look like pixels, normalize.
                if max(max(abs(x) for x in xs), max(abs(y) for y in ys)) > 1.5:
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
    force_class: int,
) -> Tuple[int, int]:
    lines, kept, skipped = parse_label_to_yolo5_lines(src_label, img_path, force_class)
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    dst_label.write_text("".join(lines), encoding="utf-8")
    return kept, skipped


# =============================================================================
# Overlap collection
# =============================================================================

@dataclass
class UsedIndex:
    image_count: int
    names: Set[str]
    stems: Set[str]
    ids_by_family: Dict[str, Set[str]]
    hashes: Set[str]


def collect_used_index(used_root: Path, use_hash: bool = False) -> UsedIndex:
    used_images = find_images_recursive(used_root)

    names: Set[str] = set()
    stems: Set[str] = set()
    ids_by_family: Dict[str, Set[str]] = defaultdict(set)
    hashes: Set[str] = set()

    for img in used_images:
        names.add(img.name.lower())
        stems.add(img.stem.lower())

        family = infer_family_from_path(img)
        ids_by_family[family].update(canonical_id_keys(img))

        if use_hash:
            try:
                hashes.add(image_sha1(img))
            except Exception as e:
                log(f"[WARN] Failed hashing used image {img}: {e}")

    return UsedIndex(
        image_count=len(used_images),
        names=names,
        stems=stems,
        ids_by_family={k: set(v) for k, v in ids_by_family.items()},
        hashes=hashes,
    )


def is_overlapping_with_used(
    img: Path,
    used: UsedIndex,
    family: str,
    use_hash: bool = False,
) -> Tuple[bool, str]:
    if img.name.lower() in used.names:
        return True, "same filename"

    if img.stem.lower() in used.stems:
        return True, "same stem"

    keys = canonical_id_keys(img)
    same_family_used_ids = used.ids_by_family.get(family, set())
    id_hit = keys.intersection(same_family_used_ids)
    if id_hit:
        preview = ",".join(sorted(id_hit)[:5])
        return True, f"same source-id key: {preview}"

    if use_hash:
        try:
            h = image_sha1(img)
            if h in used.hashes:
                return True, "same sha1"
        except Exception as e:
            return True, f"hash failed: {e}"

    return False, ""


# =============================================================================
# Selection and copy
# =============================================================================

@dataclass
class Candidate:
    modality: str
    family: str
    img: Path
    label: Path
    boxes: int
    id_keys: Set[str]


def collect_candidates_for_modality(
    modality: str,
    image_roots: List[Path],
    label_roots: List[Path],
    used: UsedIndex,
    target_class: int,
    use_hash: bool,
) -> Tuple[List[Candidate], Dict]:
    image_roots_existing = [p for p in image_roots if p.exists()]
    missing_roots = [str(p) for p in image_roots if not p.exists()]

    if not image_roots_existing:
        raise FileNotFoundError(
            f"No existing image root for modality={modality}. Tried:\n"
            + "\n".join(str(p) for p in image_roots)
        )

    # If user did not specify label roots explicitly, infer from image roots.
    inferred_label_roots = []
    for r in image_roots_existing:
        lr = labels_root_from_image_root(r)
        if lr is not None:
            inferred_label_roots.append(lr)

    all_label_roots = list(label_roots) + inferred_label_roots
    label_index = build_label_index(all_label_roots)

    raw_images: List[Path] = []
    for root in image_roots_existing:
        raw_images.extend(find_images_recursive(root))

    stats = {
        "modality": modality,
        "image_roots": [str(p) for p in image_roots],
        "image_roots_missing": missing_roots,
        "label_roots_used_for_index": [str(p) for p in all_label_roots if p is not None],
        "raw_images": len(raw_images),
        "label_index_size": len(label_index),
        "excluded_overlap": 0,
        "excluded_overlap_reasons": defaultdict(int),
        "excluded_missing_label": 0,
        "excluded_empty_or_bad_label": 0,
        "available": 0,
        "examples": {
            "overlap": [],
            "missing_label": [],
            "bad_label": [],
        },
    }

    candidates: List[Candidate] = []

    for img in sorted(raw_images):
        family = infer_family_from_path(img)
        if family == "unknown":
            # For safety, bind unknown source to the requested modality.
            family = modality

        overlap, reason = is_overlapping_with_used(
            img=img,
            used=used,
            family=family,
            use_hash=use_hash,
        )
        if overlap:
            stats["excluded_overlap"] += 1
            stats["excluded_overlap_reasons"][reason] += 1
            if len(stats["examples"]["overlap"]) < 20:
                stats["examples"]["overlap"].append(f"{img} | {reason}")
            continue

        label = find_label_for_image(img, label_index)
        if label is None:
            stats["excluded_missing_label"] += 1
            if len(stats["examples"]["missing_label"]) < 20:
                stats["examples"]["missing_label"].append(str(img))
            continue

        try:
            _, kept, _ = parse_label_to_yolo5_lines(label, img, force_class=target_class)
        except Exception as e:
            kept = 0
            if len(stats["examples"]["bad_label"]) < 20:
                stats["examples"]["bad_label"].append(f"{img} | {label} | {e}")

        if kept <= 0:
            stats["excluded_empty_or_bad_label"] += 1
            if len(stats["examples"]["bad_label"]) < 20:
                stats["examples"]["bad_label"].append(f"{img} | {label}")
            continue

        candidates.append(
            Candidate(
                modality=modality,
                family=family,
                img=img,
                label=label,
                boxes=kept,
                id_keys=canonical_id_keys(img),
            )
        )

    stats["available"] = len(candidates)
    stats["excluded_overlap_reasons"] = dict(stats["excluded_overlap_reasons"])

    return candidates, stats


def select_per_modality(
    candidates_by_modality: Dict[str, List[Candidate]],
    per_modality: int,
    seed: int,
) -> Tuple[Dict[str, List[Candidate]], Dict]:
    rng = random.Random(seed)
    selected: Dict[str, List[Candidate]] = {}
    stats = {}

    for modality, candidates in candidates_by_modality.items():
        items = list(candidates)
        rng.shuffle(items)

        n = min(per_modality, len(items))
        chosen = sorted(items[:n], key=lambda c: c.img.name.lower())
        selected[modality] = chosen

        stats[modality] = {
            "available": len(items),
            "selected": n,
            "shortage": max(0, per_modality - len(items)),
        }

    return selected, stats


def unique_destination_name(dst_dir: Path, original_name: str, used_names: Set[str]) -> str:
    stem = Path(original_name).stem
    suffix = Path(original_name).suffix

    name = original_name
    i = 1
    while name.lower() in used_names or (dst_dir / name).exists():
        name = f"{stem}_{i:03d}{suffix}"
        i += 1

    used_names.add(name.lower())
    return name


def copy_selected_dataset(
    selected: Dict[str, List[Candidate]],
    dst_root: Path,
    target_class: int,
    overwrite: bool,
) -> Tuple[Dict, List[Dict[str, str]]]:
    if dst_root.exists():
        if overwrite:
            log(f"[INFO] Removing existing output dataset: {dst_root}")
            shutil.rmtree(dst_root)
        else:
            raise FileExistsError(
                f"Output already exists: {dst_root}\n"
                f"Use --overwrite only if you want to recreate this dataset."
            )

    manifest_rows: List[Dict[str, str]] = []
    copy_stats = {}

    for modality, items in selected.items():
        split_name = f"extratest_{modality}"

        dst_img_dir = dst_root / "images" / split_name
        dst_lbl_dir = dst_root / "labels" / split_name
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        used_dst_names: Set[str] = set()
        total_boxes = 0
        total_skipped = 0

        for idx, cand in enumerate(items):
            dst_img_name = unique_destination_name(dst_img_dir, cand.img.name, used_dst_names)
            dst_img = dst_img_dir / dst_img_name
            dst_lbl = dst_lbl_dir / f"{Path(dst_img_name).stem}.txt"

            shutil.copy2(cand.img, dst_img)
            kept, skipped = write_converted_label(
                src_label=cand.label,
                dst_label=dst_lbl,
                img_path=cand.img,
                force_class=target_class,
            )

            total_boxes += kept
            total_skipped += skipped

            manifest_rows.append({
                "modality": modality,
                "family": cand.family,
                "src_image": str(cand.img),
                "src_label": str(cand.label),
                "dst_image": str(dst_img),
                "dst_label": str(dst_lbl),
                "dst_image_name": dst_img.name,
                "boxes": str(kept),
                "skipped_label_rows": str(skipped),
                "id_keys_preview": "|".join(sorted(cand.id_keys)[:10]),
            })

        copy_stats[modality] = {
            "images": len(items),
            "boxes": total_boxes,
            "skipped_label_rows": total_skipped,
            "images_dir": str(dst_img_dir),
            "labels_dir": str(dst_lbl_dir),
        }

    return copy_stats, manifest_rows


def write_manifest(dst_root: Path, rows: List[Dict[str, str]]) -> Path:
    path = dst_root / "manifest.csv"

    fieldnames = [
        "modality",
        "family",
        "src_image",
        "src_label",
        "dst_image",
        "dst_label",
        "dst_image_name",
        "boxes",
        "skipped_label_rows",
        "id_keys_preview",
    ]

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    return path


def write_student2_yaml(yaml_path: Path, val_image_dirs: List[Path]) -> None:
    """
    Write 2-class eval YAML for current student:
      0 ship
      1 bridge

    train is set equal to val because some Ultralytics checks expect train.
    """
    lines: List[str] = []

    lines.append("train:")
    for p in val_image_dirs:
        lines.append(f"  - {norm_path(p)}")
    lines.append("")

    lines.append("val:")
    for p in val_image_dirs:
        lines.append(f"  - {norm_path(p)}")
    lines.append("")

    lines.append("nc: 2")
    lines.append("")
    lines.append("names:")
    for k, v in STUDENT2_NAMES.items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"[OK] wrote YAML: {yaml_path}")


def write_yamls(dst_root: Path, configs_root: Path) -> Dict[str, str]:
    sar_dir = dst_root / "images" / "extratest_sar"
    ir_dir = dst_root / "images" / "extratest_ir"

    dataset_yamls = {
        "sar": dst_root / "data_sar.yaml",
        "ir": dst_root / "data_ir.yaml",
        "all": dst_root / "data.yaml",
    }

    config_yamls = {
        "sar_config": configs_root / "student2_eval_bridge_extratest_sar.yaml",
        "ir_config": configs_root / "student2_eval_bridge_extratest_ir.yaml",
        "all_config": configs_root / "student2_eval_bridge_extratest_all.yaml",
    }

    write_student2_yaml(dataset_yamls["sar"], [sar_dir])
    write_student2_yaml(dataset_yamls["ir"], [ir_dir])
    write_student2_yaml(dataset_yamls["all"], [sar_dir, ir_dir])

    write_student2_yaml(config_yamls["sar_config"], [sar_dir])
    write_student2_yaml(config_yamls["ir_config"], [ir_dir])
    write_student2_yaml(config_yamls["all_config"], [sar_dir, ir_dir])

    return {k: str(v) for k, v in {**dataset_yamls, **config_yamls}.items()}


def verify_output_labels(dst_root: Path, target_class: int) -> Dict:
    label_files = sorted((dst_root / "labels").rglob("*.txt"))

    stats = {
        "label_files": len(label_files),
        "rows": 0,
        "class_hist": defaultdict(int),
        "bad_rows": 0,
        "bad_examples": [],
        "class_id_gt_1": 0,
    }

    for p in label_files:
        for line_no, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            stats["rows"] += 1

            try:
                cls = int(float(parts[0]))
                if len(parts) != 5:
                    raise ValueError(f"expected 5 cols, got {len(parts)}")
                vals = list(map(float, parts[1:]))
                if cls != target_class:
                    raise ValueError(f"expected cls={target_class}, got cls={cls}")
                if any(v < 0 or v > 1 for v in vals):
                    raise ValueError(f"coords outside [0,1]: {vals}")
                if vals[2] <= 0 or vals[3] <= 0:
                    raise ValueError(f"non-positive wh: {vals[2:]}")
                if cls > 1:
                    stats["class_id_gt_1"] += 1
                stats["class_hist"][cls] += 1
            except Exception as e:
                stats["bad_rows"] += 1
                if len(stats["bad_examples"]) < 20:
                    stats["bad_examples"].append(f"{p}:{line_no}: {line} | {e}")

    stats["class_hist"] = dict(stats["class_hist"])
    return stats


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sar-image-roots",
        nargs="+",
        default=[str(p) for p in SAR_IMAGE_ROOTS],
        help="One or more SAR/MSAR image roots to scan.",
    )
    parser.add_argument(
        "--ir-image-roots",
        nargs="+",
        default=[str(p) for p in IR_IMAGE_ROOTS],
        help="One or more IR/LWIR/MassMIND image roots to scan.",
    )
    parser.add_argument(
        "--sar-label-roots",
        nargs="*",
        default=[],
        help="Optional explicit SAR label roots. If omitted, inferred from images->labels.",
    )
    parser.add_argument(
        "--ir-label-roots",
        nargs="*",
        default=[],
        help="Optional explicit IR label roots. If omitted, inferred from images->labels.",
    )
    parser.add_argument("--used-root", type=str, default=str(USED_BRIDGEALL_ROOT))
    parser.add_argument("--dst-root", type=str, default=str(DST_ROOT))
    parser.add_argument("--configs-root", type=str, default=str(CONFIGS_ROOT))
    parser.add_argument("--per-modality", type=int, default=PER_MODALITY)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--target-class", type=int, default=TARGET_CLASS_ID)
    parser.add_argument("--hash-check", action="store_true", help="Also exclude exact duplicate image bytes by SHA1.")
    parser.add_argument("--overwrite", action="store_true", help="Recreate output folder if it already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Only print selection stats; do not copy/write output.")
    parser.add_argument("--strict-count", action="store_true", help="Fail if any modality has fewer than --per-modality candidates.")

    args = parser.parse_args()

    sar_image_roots = as_paths(args.sar_image_roots)
    ir_image_roots = as_paths(args.ir_image_roots)
    sar_label_roots = as_paths(args.sar_label_roots)
    ir_label_roots = as_paths(args.ir_label_roots)

    used_root = Path(args.used_root)
    dst_root = Path(args.dst_root)
    configs_root = Path(args.configs_root)

    log("=" * 100)
    log("Make bridge SAR/IR extratest dataset with no overlap against E:/YOLODATA/bridgeAll")
    log("=" * 100)
    log(f"sar_image_roots = {[str(p) for p in sar_image_roots]}")
    log(f"ir_image_roots  = {[str(p) for p in ir_image_roots]}")
    log(f"sar_label_roots = {[str(p) for p in sar_label_roots]}")
    log(f"ir_label_roots  = {[str(p) for p in ir_label_roots]}")
    log(f"used_root       = {used_root}")
    log(f"dst_root        = {dst_root}")
    log(f"configs_root    = {configs_root}")
    log(f"per_modality    = {args.per_modality}")
    log(f"seed            = {args.seed}")
    log(f"target_class    = {args.target_class}  # bridge in current 2-class student")
    log(f"hash_check      = {args.hash_check}")
    log(f"dry_run         = {args.dry_run}")
    log("=" * 100)

    if not used_root.exists():
        raise FileNotFoundError(f"used_root not found: {used_root}")

    used = collect_used_index(used_root=used_root, use_hash=args.hash_check)

    used_summary = {
        "used_root": str(used_root),
        "used_image_count": used.image_count,
        "used_family_id_counts": {k: len(v) for k, v in used.ids_by_family.items()},
        "hash_count": len(used.hashes),
    }

    log("")
    log("[USED bridgeAll SUMMARY]")
    log(json.dumps(used_summary, ensure_ascii=False, indent=2))

    sar_candidates, sar_stats = collect_candidates_for_modality(
        modality="sar",
        image_roots=sar_image_roots,
        label_roots=sar_label_roots,
        used=used,
        target_class=args.target_class,
        use_hash=args.hash_check,
    )

    ir_candidates, ir_stats = collect_candidates_for_modality(
        modality="ir",
        image_roots=ir_image_roots,
        label_roots=ir_label_roots,
        used=used,
        target_class=args.target_class,
        use_hash=args.hash_check,
    )

    candidates_by_modality = {
        "sar": sar_candidates,
        "ir": ir_candidates,
    }

    selected, select_stats = select_per_modality(
        candidates_by_modality=candidates_by_modality,
        per_modality=args.per_modality,
        seed=args.seed,
    )

    full_select_summary = {
        "used_summary": used_summary,
        "sar_collect_stats": sar_stats,
        "ir_collect_stats": ir_stats,
        "select_stats": select_stats,
    }

    log("")
    log("[SELECTION SUMMARY]")
    log(json.dumps(full_select_summary, ensure_ascii=False, indent=2))

    for modality in ["sar", "ir"]:
        s = select_stats[modality]
        if s["shortage"] > 0:
            msg = (
                f"[WARN] {modality}: only {s['available']} available after no-overlap filtering; "
                f"selected {s['selected']}, shortage {s['shortage']}."
            )
            if args.strict_count:
                raise RuntimeError(msg)
            log(msg)
        else:
            log(f"[OK] {modality}: selected {s['selected']}/{args.per_modality}.")

    if args.dry_run:
        log("")
        log("[DRY RUN DONE] No files were copied.")
        return 0

    log("")
    log("[COPY DATASET]")
    copy_stats, manifest_rows = copy_selected_dataset(
        selected=selected,
        dst_root=dst_root,
        target_class=args.target_class,
        overwrite=args.overwrite,
    )
    log(json.dumps(copy_stats, ensure_ascii=False, indent=2))

    manifest_path = write_manifest(dst_root=dst_root, rows=manifest_rows)

    log("")
    log("[WRITE YAML]")
    yaml_stats = write_yamls(dst_root=dst_root, configs_root=configs_root)

    log("")
    log("[VERIFY OUTPUT LABELS]")
    verify_stats = verify_output_labels(dst_root=dst_root, target_class=args.target_class)
    log(json.dumps(verify_stats, ensure_ascii=False, indent=2))

    summary = {
        "used_summary": used_summary,
        "sar_collect_stats": sar_stats,
        "ir_collect_stats": ir_stats,
        "select_stats": select_stats,
        "copy_stats": copy_stats,
        "verify_stats": verify_stats,
        "yaml_stats": yaml_stats,
        "manifest_csv": str(manifest_path),
        "output_dataset": str(dst_root),
        "target_class_for_bridge": args.target_class,
        "student2_names": STUDENT2_NAMES,
        "notes": [
            "This dataset is for current 2-class student: 0 ship, 1 bridge.",
            "All output labels are forced to class 1.",
            "RGB/DIOR XML source is intentionally not processed in this script.",
            "Overlap is excluded against E:/YOLODATA/bridgeAll by filename, stem, source-id keys, and optionally sha1.",
        ],
    }

    summary_path = dst_root / "selection_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    log("")
    log("[DONE]")
    log(f"Output dataset: {dst_root}")
    log(f"Manifest CSV:   {manifest_path}")
    log(f"Summary JSON:   {summary_path}")
    log("YAML files:")
    for k, v in yaml_stats.items():
        log(f"  {k}: {v}")

    log("")
    log("Recommended eval YAML for bridge extratest all:")
    log(f"  {yaml_stats['all']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
