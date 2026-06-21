# tools/datasets/make_bridge_rgb_extratest_no_overlap.py
# -*- coding: utf-8 -*-
"""
Create RGB/DIOR bridge external extra-test dataset for the current 2-class YOLOE student.

Student class space:
  0 ship
  1 bridge

RGB source:
  images:      E:/YOLODATA/bridge_rgb_all/JPEGImages-test
  annotations: E:/YOLODATA/bridge_rgb_all/Annotations/Horizontal Bounding Boxes

No-leak reference:
  E:/YOLODATA/bridgeAll

Default output:
  C:/Users/DOCTOR/Documents/GitHub/YOLOv8/ultralytics/datasets/extracttest_bridge

This script:
  - parses DIOR/VOC-style HBB XML
  - keeps only object name "bridge"
  - converts bndbox to YOLO detect 5-column labels
  - forces class id = 1 for the current student
  - selects up to 100 non-overlapping RGB images
  - does NOT delete existing SAR/IR extratest folders
  - writes data_rgb.yaml and updates combined data.yaml
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from PIL import Image


REPO_ROOT = Path(r"C:\Users\DOCTOR\Documents\GitHub\YOLOv8")
RGB_IMAGE_DIR = Path(r"E:\YOLODATA\bridge_rgb_all\JPEGImages-test")
RGB_XML_DIR = Path(r"E:\YOLODATA\bridge_rgb_all\Annotations\Horizontal Bounding Boxes")
USED_BRIDGEALL_ROOT = Path(r"E:\YOLODATA\bridgeAll")

DST_ROOT = REPO_ROOT / r"ultralytics\datasets\extracttest_bridge"
CONFIGS_ROOT = REPO_ROOT / r"configs\datasets"

PER_MODALITY = 100
SEED = 42
TARGET_CLASS_ID = 1

STUDENT2_NAMES = {0: "ship", 1: "bridge"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def log(msg: object = "") -> None:
    print(msg, flush=True)


def norm_path(p: Path) -> str:
    return p.resolve().as_posix()


def find_images_recursive(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)


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
        return im.size


def clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def xml_text(node: Optional[ET.Element]) -> Optional[str]:
    if node is None or node.text is None:
        return None
    s = node.text.strip()
    return s if s else None


def xml_float(parent: Optional[ET.Element], tag: str) -> Optional[float]:
    if parent is None:
        return None
    s = xml_text(parent.find(tag))
    if s is None:
        return None
    try:
        return float(s)
    except Exception:
        return None


def infer_family(path: Path) -> str:
    s = path.as_posix().lower()
    ws = str(path).lower()
    if "dior" in s or "rgb" in s or "jpegimages" in s:
        return "rgb"
    if "massmind" in s or "lwir" in s or "infr" in s or "infra" in s or "thermal" in s:
        return "ir"
    if "msar" in s or "/sar" in s or "\\sar" in ws or "sar_" in s:
        return "sar"
    return "unknown"


def canonical_id_keys(path: Path) -> Set[str]:
    """
    Robust ID keys for no-leak filtering.
    17581.jpg, rgb_17581.jpg, dior_bridge_rgb_17581.jpg should share key 17581.
    """
    stem = path.stem.lower()
    base = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    keys: Set[str] = set()
    if base:
        keys.add(base)

    prefixes = [
        "dior_bridge_rgb_", "dior_rgb_", "dior_bridge_", "dior_",
        "rgb_bridge_", "bridge_rgb_", "rgb_", "bridge_",
        "msar_bridge_", "msar_", "sar_bridge_", "sar_",
        "massmind_bridge_", "massmind_", "infr_bridge_", "infr_",
        "infra_", "lwir_bridge_", "lwir_", "ir_bridge_", "ir_",
    ]

    variants = {base}
    changed = True
    while changed:
        changed = False
        new_variants = set(variants)
        for x in variants:
            for pref in prefixes:
                if x.startswith(pref) and len(x) > len(pref):
                    y = x[len(pref):]
                    if y and y not in new_variants:
                        new_variants.add(y)
                        changed = True
        variants = new_variants

    keys.update(v for v in variants if v)

    digit_groups = re.findall(r"\d+", base)
    for g in digit_groups:
        if len(g) >= 2:
            keys.add(g)
            try:
                keys.add(str(int(g)))
            except Exception:
                pass

    if len(digit_groups) >= 2:
        keys.add("_".join(digit_groups))
        try:
            keys.add("_".join(str(int(g)) for g in digit_groups))
        except Exception:
            pass

    return {k for k in keys if k}


@dataclass
class UsedIndex:
    image_count: int
    names: Set[str]
    stems: Set[str]
    ids_by_family: Dict[str, Set[str]]
    hashes: Set[str]


def collect_used_index(used_root: Path, use_hash: bool = False) -> UsedIndex:
    images = find_images_recursive(used_root)
    names: Set[str] = set()
    stems: Set[str] = set()
    ids_by_family: Dict[str, Set[str]] = defaultdict(set)
    hashes: Set[str] = set()

    for img in images:
        names.add(img.name.lower())
        stems.add(img.stem.lower())
        ids_by_family[infer_family(img)].update(canonical_id_keys(img))

        if use_hash:
            try:
                hashes.add(image_sha1(img))
            except Exception as e:
                log(f"[WARN] hash failed for used image: {img} | {e}")

    return UsedIndex(
        image_count=len(images),
        names=names,
        stems=stems,
        ids_by_family={k: set(v) for k, v in ids_by_family.items()},
        hashes=hashes,
    )


def is_overlap(img: Path, used: UsedIndex, family: str = "rgb", use_hash: bool = False) -> Tuple[bool, str]:
    if img.name.lower() in used.names:
        return True, "same filename"
    if img.stem.lower() in used.stems:
        return True, "same stem"

    keys = canonical_id_keys(img)

    # Compare with RGB family and unknown family. unknown is needed for flattened bridgeAll layouts.
    used_ids = set()
    used_ids.update(used.ids_by_family.get(family, set()))
    used_ids.update(used.ids_by_family.get("unknown", set()))

    hit = keys.intersection(used_ids)
    if hit:
        return True, "same source-id key: " + ",".join(sorted(hit)[:5])

    if use_hash:
        try:
            h = image_sha1(img)
            if h in used.hashes:
                return True, "same sha1"
        except Exception as e:
            return True, f"hash failed: {e}"

    return False, ""


def build_image_index(image_dir: Path) -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    for img in find_images_recursive(image_dir):
        idx.setdefault(img.name.lower(), img)
        idx.setdefault(img.stem.lower(), img)
    return idx


def find_image_for_xml(xml_path: Path, image_index: Dict[str, Path]) -> Optional[Path]:
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return None

    filename = xml_text(root.find("filename"))
    if filename:
        hit = image_index.get(filename.lower())
        if hit:
            return hit
        hit = image_index.get(Path(filename).stem.lower())
        if hit:
            return hit

    return image_index.get(xml_path.stem.lower())


@dataclass
class Box:
    cx: float
    cy: float
    w: float
    h: float


@dataclass
class Candidate:
    img: Path
    xml: Path
    boxes: List[Box]
    image_w: int
    image_h: int
    id_keys: Set[str]


def parse_bridge_boxes(xml_path: Path, img_path: Path, object_name: str = "bridge") -> Tuple[List[Box], int, int, int]:
    root = ET.parse(xml_path).getroot()

    size = root.find("size")
    xml_w = xml_float(size, "width")
    xml_h = xml_float(size, "height")

    if xml_w is None or xml_h is None or xml_w <= 0 or xml_h <= 0:
        img_w, img_h = read_image_size(img_path)
    else:
        img_w, img_h = int(round(xml_w)), int(round(xml_h))

    boxes: List[Box] = []
    skipped = 0

    for obj in root.findall("object"):
        name = xml_text(obj.find("name"))
        if name is None:
            skipped += 1
            continue

        if name.strip().lower() != object_name.lower():
            continue

        bnd = obj.find("bndbox")
        if bnd is None:
            skipped += 1
            continue

        xmin = xml_float(bnd, "xmin")
        ymin = xml_float(bnd, "ymin")
        xmax = xml_float(bnd, "xmax")
        ymax = xml_float(bnd, "ymax")

        if None in (xmin, ymin, xmax, ymax):
            skipped += 1
            continue

        x1, x2 = sorted([float(xmin), float(xmax)])
        y1, y2 = sorted([float(ymin), float(ymax)])

        x1 = clip01(x1 / img_w)
        x2 = clip01(x2 / img_w)
        y1 = clip01(y1 / img_h)
        y2 = clip01(y2 / img_h)

        bw = x2 - x1
        bh = y2 - y1

        if bw <= 0 or bh <= 0:
            skipped += 1
            continue

        boxes.append(Box(cx=(x1 + x2) / 2, cy=(y1 + y2) / 2, w=bw, h=bh))

    return boxes, img_w, img_h, skipped


def collect_candidates(
    image_dir: Path,
    xml_dir: Path,
    used: UsedIndex,
    object_name: str,
    use_hash: bool,
) -> Tuple[List[Candidate], Dict]:
    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir not found: {image_dir}")
    if not xml_dir.exists():
        raise FileNotFoundError(f"xml_dir not found: {xml_dir}")

    image_index = build_image_index(image_dir)
    xml_files = sorted(xml_dir.rglob("*.xml"))

    stats = {
        "image_dir": str(image_dir),
        "xml_dir": str(xml_dir),
        "image_index_size": len(image_index),
        "xml_files": len(xml_files),
        "excluded_missing_image": 0,
        "excluded_parse_error": 0,
        "excluded_no_target_object": 0,
        "excluded_overlap": 0,
        "excluded_overlap_reasons": defaultdict(int),
        "available": 0,
        "total_target_boxes_available": 0,
        "examples": {
            "missing_image": [],
            "parse_error": [],
            "no_target_object": [],
            "overlap": [],
        },
    }

    candidates: List[Candidate] = []

    for xml in xml_files:
        img = find_image_for_xml(xml, image_index)
        if img is None:
            stats["excluded_missing_image"] += 1
            if len(stats["examples"]["missing_image"]) < 20:
                stats["examples"]["missing_image"].append(str(xml))
            continue

        overlap, reason = is_overlap(img, used, family="rgb", use_hash=use_hash)
        if overlap:
            stats["excluded_overlap"] += 1
            stats["excluded_overlap_reasons"][reason] += 1
            if len(stats["examples"]["overlap"]) < 20:
                stats["examples"]["overlap"].append(f"{img} | {xml} | {reason}")
            continue

        try:
            boxes, w, h, _ = parse_bridge_boxes(xml, img, object_name=object_name)
        except Exception as e:
            stats["excluded_parse_error"] += 1
            if len(stats["examples"]["parse_error"]) < 20:
                stats["examples"]["parse_error"].append(f"{xml} | {e}")
            continue

        if not boxes:
            stats["excluded_no_target_object"] += 1
            if len(stats["examples"]["no_target_object"]) < 20:
                stats["examples"]["no_target_object"].append(str(xml))
            continue

        candidates.append(Candidate(img=img, xml=xml, boxes=boxes, image_w=w, image_h=h, id_keys=canonical_id_keys(img)))
        stats["total_target_boxes_available"] += len(boxes)

    stats["available"] = len(candidates)
    stats["excluded_overlap_reasons"] = dict(stats["excluded_overlap_reasons"])
    return candidates, stats


def select_candidates(candidates: List[Candidate], n: int, seed: int) -> Tuple[List[Candidate], Dict]:
    rng = random.Random(seed)
    items = list(candidates)
    rng.shuffle(items)
    selected = sorted(items[: min(n, len(items))], key=lambda c: c.img.name.lower())
    stats = {
        "available": len(candidates),
        "selected": len(selected),
        "shortage": max(0, n - len(candidates)),
        "selected_boxes": sum(len(c.boxes) for c in selected),
    }
    return selected, stats


def unique_name(dst_dir: Path, original_name: str, used_names: Set[str]) -> str:
    stem = Path(original_name).stem
    suffix = Path(original_name).suffix
    name = original_name
    i = 1
    while name.lower() in used_names or (dst_dir / name).exists():
        name = f"{stem}_{i:03d}{suffix}"
        i += 1
    used_names.add(name.lower())
    return name


def write_label(path: Path, boxes: List[Box], cls: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{cls} {b.cx:.8f} {b.cy:.8f} {b.w:.8f} {b.h:.8f}\n" for b in boxes]
    path.write_text("".join(lines), encoding="utf-8")


def copy_rgb(selected: List[Candidate], dst_root: Path, target_class: int, overwrite_rgb: bool) -> Tuple[Dict, List[Dict[str, str]]]:
    dst_img_dir = dst_root / "images" / "extratest_rgb"
    dst_lbl_dir = dst_root / "labels" / "extratest_rgb"

    if overwrite_rgb:
        if dst_img_dir.exists():
            log(f"[INFO] Removing existing RGB image dir: {dst_img_dir}")
            shutil.rmtree(dst_img_dir)
        if dst_lbl_dir.exists():
            log(f"[INFO] Removing existing RGB label dir: {dst_lbl_dir}")
            shutil.rmtree(dst_lbl_dir)
    else:
        if dst_img_dir.exists() or dst_lbl_dir.exists():
            raise FileExistsError(
                f"RGB output exists:\n  {dst_img_dir}\n  {dst_lbl_dir}\n"
                f"Use --overwrite-rgb to recreate only RGB output."
            )

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    used_dst_names: Set[str] = set()
    total_boxes = 0

    for cand in selected:
        dst_name = unique_name(dst_img_dir, cand.img.name, used_dst_names)
        dst_img = dst_img_dir / dst_name
        dst_lbl = dst_lbl_dir / f"{Path(dst_name).stem}.txt"

        shutil.copy2(cand.img, dst_img)
        write_label(dst_lbl, cand.boxes, cls=target_class)
        total_boxes += len(cand.boxes)

        rows.append({
            "modality": "rgb",
            "src_image": str(cand.img),
            "src_xml": str(cand.xml),
            "dst_image": str(dst_img),
            "dst_label": str(dst_lbl),
            "dst_image_name": dst_img.name,
            "image_w": str(cand.image_w),
            "image_h": str(cand.image_h),
            "boxes": str(len(cand.boxes)),
            "id_keys_preview": "|".join(sorted(cand.id_keys)[:10]),
        })

    stats = {"rgb": {"images": len(selected), "boxes": total_boxes, "images_dir": str(dst_img_dir), "labels_dir": str(dst_lbl_dir)}}
    return stats, rows


def write_manifest(dst_root: Path, rows: List[Dict[str, str]]) -> Path:
    path = dst_root / "manifest_rgb.csv"
    fields = ["modality", "src_image", "src_xml", "dst_image", "dst_label", "dst_image_name", "image_w", "image_h", "boxes", "id_keys_preview"]

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    return path


def write_student2_yaml(yaml_path: Path, val_image_dirs: List[Path]) -> None:
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


def existing_extratest_image_dirs(dst_root: Path) -> Dict[str, Path]:
    out = {}
    for name in ["rgb", "sar", "ir"]:
        p = dst_root / "images" / f"extratest_{name}"
        if p.exists() and any(x.is_file() and x.suffix.lower() in IMG_EXTS for x in p.rglob("*")):
            out[name] = p
    return out


def write_yamls(dst_root: Path, configs_root: Path) -> Dict[str, str]:
    dirs = existing_extratest_image_dirs(dst_root)
    rgb_dir = dst_root / "images" / "extratest_rgb"

    if "rgb" not in dirs:
        raise FileNotFoundError(f"RGB output dir has no images: {rgb_dir}")

    combined = [dirs[k] for k in ["rgb", "sar", "ir"] if k in dirs]

    yamls = {
        "rgb": dst_root / "data_rgb.yaml",
        "all": dst_root / "data.yaml",
        "rgb_config": configs_root / "student2_eval_bridge_extratest_rgb.yaml",
        "all_config": configs_root / "student2_eval_bridge_extratest_all.yaml",
    }

    write_student2_yaml(yamls["rgb"], [rgb_dir])
    write_student2_yaml(yamls["all"], combined)
    write_student2_yaml(yamls["rgb_config"], [rgb_dir])
    write_student2_yaml(yamls["all_config"], combined)

    return {k: str(v) for k, v in yamls.items()}


def verify_rgb_labels(dst_root: Path, target_class: int) -> Dict:
    label_dir = dst_root / "labels" / "extratest_rgb"
    files = sorted(label_dir.rglob("*.txt")) if label_dir.exists() else []

    stats = {
        "label_files": len(files),
        "rows": 0,
        "class_hist": defaultdict(int),
        "bad_rows": 0,
        "bad_examples": [],
        "class_id_gt_1": 0,
    }

    for p in files:
        for line_no, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            stats["rows"] += 1
            parts = line.split()

            try:
                cls = int(float(parts[0]))
                vals = list(map(float, parts[1:]))
                if len(parts) != 5:
                    raise ValueError(f"expected 5 cols, got {len(parts)}")
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str, default=str(RGB_IMAGE_DIR))
    parser.add_argument("--xml-dir", type=str, default=str(RGB_XML_DIR))
    parser.add_argument("--used-root", type=str, default=str(USED_BRIDGEALL_ROOT))
    parser.add_argument("--dst-root", type=str, default=str(DST_ROOT))
    parser.add_argument("--configs-root", type=str, default=str(CONFIGS_ROOT))
    parser.add_argument("--per-modality", type=int, default=PER_MODALITY)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--target-class", type=int, default=TARGET_CLASS_ID)
    parser.add_argument("--object-name", type=str, default="bridge")
    parser.add_argument("--hash-check", action="store_true", help="Also exclude exact duplicate image bytes by SHA1.")
    parser.add_argument("--overwrite-rgb", action="store_true", help="Recreate only images/labels/extratest_rgb.")
    parser.add_argument("--dry-run", action="store_true", help="Only print selection stats; do not copy/write output.")
    parser.add_argument("--strict-count", action="store_true", help="Fail if fewer than --per-modality RGB candidates remain.")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    xml_dir = Path(args.xml_dir)
    used_root = Path(args.used_root)
    dst_root = Path(args.dst_root)
    configs_root = Path(args.configs_root)

    log("=" * 100)
    log("Make bridge RGB/DIOR extratest dataset with no overlap against bridgeAll")
    log("=" * 100)
    log(f"image_dir      = {image_dir}")
    log(f"xml_dir        = {xml_dir}")
    log(f"used_root      = {used_root}")
    log(f"dst_root       = {dst_root}")
    log(f"configs_root   = {configs_root}")
    log(f"per_modality   = {args.per_modality}")
    log(f"seed           = {args.seed}")
    log(f"object_name    = {args.object_name}")
    log(f"target_class   = {args.target_class}  # bridge in current 2-class student")
    log(f"hash_check     = {args.hash_check}")
    log(f"dry_run        = {args.dry_run}")
    log("=" * 100)

    if not used_root.exists():
        raise FileNotFoundError(f"used_root not found: {used_root}")

    used = collect_used_index(used_root, use_hash=args.hash_check)
    used_summary = {
        "used_root": str(used_root),
        "used_image_count": used.image_count,
        "used_family_id_counts": {k: len(v) for k, v in used.ids_by_family.items()},
        "hash_count": len(used.hashes),
    }

    log("")
    log("[USED bridgeAll SUMMARY]")
    log(json.dumps(used_summary, ensure_ascii=False, indent=2))

    candidates, collect_stats = collect_candidates(
        image_dir=image_dir,
        xml_dir=xml_dir,
        used=used,
        object_name=args.object_name,
        use_hash=args.hash_check,
    )

    selected, select_stats = select_candidates(candidates, n=args.per_modality, seed=args.seed)

    summary_for_print = {
        "used_summary": used_summary,
        "collect_stats": collect_stats,
        "select_stats": select_stats,
    }

    log("")
    log("[SELECTION SUMMARY]")
    log(json.dumps(summary_for_print, ensure_ascii=False, indent=2))

    if select_stats["shortage"] > 0:
        msg = (
            f"[WARN] rgb: only {select_stats['available']} available after no-overlap filtering; "
            f"selected {select_stats['selected']}, shortage {select_stats['shortage']}."
        )
        if args.strict_count:
            raise RuntimeError(msg)
        log(msg)
    else:
        log(f"[OK] rgb: selected {select_stats['selected']}/{args.per_modality}.")

    if args.dry_run:
        log("")
        log("[DRY RUN DONE] No files were copied.")
        return 0

    log("")
    log("[COPY RGB DATASET]")
    copy_stats, rows = copy_rgb(
        selected=selected,
        dst_root=dst_root,
        target_class=args.target_class,
        overwrite_rgb=args.overwrite_rgb,
    )
    log(json.dumps(copy_stats, ensure_ascii=False, indent=2))

    manifest_path = write_manifest(dst_root, rows)

    log("")
    log("[WRITE YAML]")
    yaml_stats = write_yamls(dst_root, configs_root)

    log("")
    log("[VERIFY OUTPUT LABELS]")
    verify_stats = verify_rgb_labels(dst_root, target_class=args.target_class)
    log(json.dumps(verify_stats, ensure_ascii=False, indent=2))

    summary = {
        "used_summary": used_summary,
        "collect_stats": collect_stats,
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
            "All output RGB labels are forced to class 1.",
            "This script only handles DIOR/RGB VOC-style HBB XML.",
            "Existing SAR/IR extratest folders are not deleted.",
            "Combined data.yaml includes existing extratest_rgb/sar/ir image dirs if present.",
            "Overlap is excluded against E:/YOLODATA/bridgeAll by filename, stem, source-id keys, and optionally sha1.",
        ],
    }

    summary_path = dst_root / "selection_summary_rgb.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    log("")
    log("[DONE]")
    log(f"Output dataset root: {dst_root}")
    log(f"RGB images:          {dst_root / 'images' / 'extratest_rgb'}")
    log(f"RGB labels:          {dst_root / 'labels' / 'extratest_rgb'}")
    log(f"Manifest CSV:        {manifest_path}")
    log(f"Summary JSON:        {summary_path}")
    log("YAML files:")
    for k, v in yaml_stats.items():
        log(f"  {k}: {v}")

    log("")
    log("Recommended RGB-only eval YAML:")
    log(f"  {yaml_stats['rgb']}")
    log("Recommended combined eval YAML:")
    log(f"  {yaml_stats['all']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
