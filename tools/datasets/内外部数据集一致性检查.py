# tools/datasets/check_bridge_extratest_no_overlap.py
# -*- coding: utf-8 -*-

"""
Check bridge external test dataset against bridge training dataset.

Purpose:
  1. Check whether extracttest_bridge structure looks valid.
  2. Check whether bridge_small_labels1 structure looks valid.
  3. Check whether labels are YOLO detect 5-column format.
  4. Check whether class ids match current 2-class student:
       0 ship
       1 bridge
     For this bridge-only check, expected class id is 1.
  5. Check whether extracttest_bridge overlaps with bridge_small_labels1
     by:
       - exact filename
       - exact stem
       - canonical numeric/source IDs extracted from filenames
       - optional SHA1 hash check

Default paths:
  test/extratest:
    C:/Users/DOCTOR/Documents/GitHub/YOLOv8/ultralytics/datasets/extracttest_bridge

  train/used bridge small:
    C:/Users/DOCTOR/Documents/GitHub/YOLOv8/ultralytics/datasets/bridge_small_labels1

This script:
  - Does NOT modify images.
  - Does NOT modify labels.
  - Does NOT train.
  - Only writes a check report JSON/CSV into the test root unless --no-write-report is used.

Expected test structure:
  extracttest_bridge/
    images/extratest_rgb
    images/extratest_sar
    images/extratest_ir
    labels/extratest_rgb
    labels/extratest_sar
    labels/extratest_ir

Expected train structure is allowed to be nested, e.g.:
  bridge_small_labels1/
    DIOR_bridge_rgb30/images/train
    DIOR_bridge_rgb30/images/val
    DIOR_bridge_rgb30/labels/train
    DIOR_bridge_rgb30/labels/val
    MSAR_bridge_sar30/...
    MassMIND_bridge_infr30/...
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


# =============================================================================
# Defaults
# =============================================================================

REPO_ROOT = Path(r"C:\Users\DOCTOR\Documents\GitHub\YOLOv8")

TEST_ROOT = REPO_ROOT / r"ultralytics\datasets\extracttest_bridge"
TRAIN_ROOT = REPO_ROOT / r"ultralytics\datasets\bridge_small_labels1"

EXPECTED_CLASS_ID = 1
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# =============================================================================
# Helpers
# =============================================================================

def log(msg: object = "") -> None:
    print(msg, flush=True)


def find_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def find_labels(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.txt") if p.is_file())


def image_sha1(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def infer_label_path_from_image(img: Path) -> Optional[Path]:
    """
    Infer label path by replacing the last images/image component with labels.

    Works for:
      extracttest_bridge/images/extratest_rgb/xxx.jpg
        -> extracttest_bridge/labels/extratest_rgb/xxx.txt

      bridge_small_labels1/DIOR_bridge_rgb30/images/train/xxx.jpg
        -> bridge_small_labels1/DIOR_bridge_rgb30/labels/train/xxx.txt
    """
    parts = list(img.parts)
    lowered = [p.lower() for p in parts]

    idx = None
    for i, p in enumerate(lowered):
        if p in {"images", "image"}:
            idx = i

    if idx is None:
        return None

    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def infer_family_from_path(path: Path) -> str:
    s = path.as_posix().lower()
    ws = str(path).lower()

    if "dior" in s or "rgb" in s or "jpegimages" in s or "extratest_rgb" in s:
        return "rgb"

    if "massmind" in s or "infr" in s or "infra" in s or "lwir" in s or "thermal" in s or "extratest_ir" in s:
        return "ir"

    if "msar" in s or "/sar" in s or "\\sar" in ws or "sar_" in s or "extratest_sar" in s:
        return "sar"

    return "unknown"


def infer_split_or_subset(path: Path) -> str:
    parts = [p.lower() for p in path.parts]

    for key in ["extratest_rgb", "extratest_sar", "extratest_ir", "train", "val", "test"]:
        if key in parts:
            return key

    return "unknown"


def canonical_id_keys(path: Path) -> Set[str]:
    """
    Build robust ID keys from filename/stem.

    This is deliberately similar to the extratest making scripts:
      17581.jpg                 -> 17581
      rgb_17581.jpg             -> rgb_17581, 17581
      dior_bridge_rgb_17581.jpg -> dior_bridge_rgb_17581, 17581
      msar_000123.jpg           -> msar_000123, 000123, 123
      massmind_a00158952.jpg    -> massmind_a00158952, a00158952, 00158952, 158952
    """
    stem = path.stem.lower()
    base = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")

    keys: Set[str] = set()
    if base:
        keys.add(base)

    prefixes = [
        "dior_bridge_rgb_", "dior_rgb_", "dior_bridge_", "dior_",
        "rgb_bridge_", "rgb_", "bridge_rgb_", "bridge_",
        "msar_bridge_", "msar_", "sar_bridge_", "sar_",
        "massmind_bridge_", "massmind_", "infr_bridge_", "infr_",
        "infra_bridge_", "infra_", "lwir_bridge_", "lwir_",
        "ir_bridge_", "ir_",
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

    digit_groups = re.findall(r"\d+", base)
    for g in digit_groups:
        if len(g) >= 2:
            keys.add(g)
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


def rel_to(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


# =============================================================================
# Label checking
# =============================================================================

@dataclass
class LabelCheck:
    exists: bool
    rows: int
    class_hist: Dict[int, int]
    bad_rows: int
    bad_examples: List[str]


def check_label_file(label_path: Path, expected_class: int) -> LabelCheck:
    if not label_path.exists():
        return LabelCheck(
            exists=False,
            rows=0,
            class_hist={},
            bad_rows=1,
            bad_examples=[f"missing label: {label_path}"],
        )

    rows = 0
    class_hist: Dict[int, int] = defaultdict(int)
    bad_rows = 0
    bad_examples: List[str] = []

    text = label_path.read_text(encoding="utf-8", errors="ignore")

    for line_no, line in enumerate(text.splitlines(), start=1):
        raw = line
        line = line.strip()

        if not line:
            continue

        rows += 1
        parts = line.split()

        try:
            if len(parts) != 5:
                raise ValueError(f"expected 5 columns, got {len(parts)}")

            cls = int(float(parts[0]))
            vals = list(map(float, parts[1:5]))

            class_hist[cls] += 1

            if cls != expected_class:
                raise ValueError(f"expected class {expected_class}, got {cls}")

            cx, cy, w, h = vals

            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                raise ValueError(f"coords outside [0,1]: {vals}")

            if w <= 0 or h <= 0:
                raise ValueError(f"non-positive width/height: w={w}, h={h}")

        except Exception as e:
            bad_rows += 1
            if len(bad_examples) < 20:
                bad_examples.append(f"{label_path}:{line_no}: {raw} | {e}")

    return LabelCheck(
        exists=True,
        rows=rows,
        class_hist=dict(class_hist),
        bad_rows=bad_rows,
        bad_examples=bad_examples,
    )


@dataclass
class ImageRecord:
    root_kind: str
    image: Path
    label: Optional[Path]
    family: str
    subset: str
    name: str
    stem: str
    id_keys: Set[str]
    sha1: Optional[str]
    label_exists: bool
    boxes: int
    class_hist: Dict[int, int]
    bad_label_rows: int


def collect_records(
    root: Path,
    root_kind: str,
    expected_class: int,
    use_hash: bool = False,
) -> Tuple[List[ImageRecord], Dict]:
    images = find_images(root)
    all_labels = find_labels(root)

    records: List[ImageRecord] = []

    stats = {
        "root": str(root),
        "root_kind": root_kind,
        "exists": root.exists(),
        "images": len(images),
        "labels_txt": len(all_labels),
        "images_missing_labels": 0,
        "labels_without_images": 0,
        "empty_label_files": 0,
        "total_boxes": 0,
        "class_hist": defaultdict(int),
        "bad_label_rows": 0,
        "bad_label_examples": [],
        "by_family": defaultdict(lambda: {"images": 0, "labels": 0, "boxes": 0}),
        "by_subset": defaultdict(lambda: {"images": 0, "labels": 0, "boxes": 0}),
        "examples_missing_labels": [],
        "examples_orphan_labels": [],
    }

    expected_label_paths: Set[Path] = set()

    for img in images:
        label = infer_label_path_from_image(img)
        if label is not None:
            expected_label_paths.add(label.resolve())

        family = infer_family_from_path(img)
        subset = infer_split_or_subset(img)

        if label is None:
            lc = LabelCheck(
                exists=False,
                rows=0,
                class_hist={},
                bad_rows=1,
                bad_examples=[f"cannot infer label path from image: {img}"],
            )
        else:
            lc = check_label_file(label, expected_class=expected_class)

        if not lc.exists:
            stats["images_missing_labels"] += 1
            if len(stats["examples_missing_labels"]) < 20:
                stats["examples_missing_labels"].append(str(img))

        if lc.exists and lc.rows == 0:
            stats["empty_label_files"] += 1

        stats["total_boxes"] += lc.rows
        stats["bad_label_rows"] += lc.bad_rows

        for k, v in lc.class_hist.items():
            stats["class_hist"][k] += v

        if lc.bad_examples:
            room = 20 - len(stats["bad_label_examples"])
            if room > 0:
                stats["bad_label_examples"].extend(lc.bad_examples[:room])

        stats["by_family"][family]["images"] += 1
        if lc.exists:
            stats["by_family"][family]["labels"] += 1
        stats["by_family"][family]["boxes"] += lc.rows

        stats["by_subset"][subset]["images"] += 1
        if lc.exists:
            stats["by_subset"][subset]["labels"] += 1
        stats["by_subset"][subset]["boxes"] += lc.rows

        sha1 = None
        if use_hash:
            try:
                sha1 = image_sha1(img)
            except Exception as e:
                sha1 = f"ERROR:{e}"

        records.append(
            ImageRecord(
                root_kind=root_kind,
                image=img,
                label=label,
                family=family,
                subset=subset,
                name=img.name.lower(),
                stem=img.stem.lower(),
                id_keys=canonical_id_keys(img),
                sha1=sha1,
                label_exists=lc.exists,
                boxes=lc.rows,
                class_hist=lc.class_hist,
                bad_label_rows=lc.bad_rows,
            )
        )

    # Orphan labels: txt files that do not correspond to any found image by inferred path.
    for lbl in all_labels:
        if lbl.resolve() not in expected_label_paths:
            # This can also catch metadata txt files outside labels dirs; still useful.
            stats["labels_without_images"] += 1
            if len(stats["examples_orphan_labels"]) < 20:
                stats["examples_orphan_labels"].append(str(lbl))

    stats["class_hist"] = dict(stats["class_hist"])
    stats["by_family"] = {k: dict(v) for k, v in stats["by_family"].items()}
    stats["by_subset"] = {k: dict(v) for k, v in stats["by_subset"].items()}

    return records, stats


# =============================================================================
# Overlap checking
# =============================================================================

def build_train_indices(train_records: List[ImageRecord]) -> Dict:
    by_name: Dict[str, List[ImageRecord]] = defaultdict(list)
    by_stem: Dict[str, List[ImageRecord]] = defaultdict(list)
    by_family_id: Dict[str, Dict[str, List[ImageRecord]]] = defaultdict(lambda: defaultdict(list))
    by_unknown_id: Dict[str, List[ImageRecord]] = defaultdict(list)
    by_sha1: Dict[str, List[ImageRecord]] = defaultdict(list)

    for r in train_records:
        by_name[r.name].append(r)
        by_stem[r.stem].append(r)

        for k in r.id_keys:
            by_family_id[r.family][k].append(r)
            if r.family == "unknown":
                by_unknown_id[k].append(r)

        if r.sha1 and not r.sha1.startswith("ERROR:"):
            by_sha1[r.sha1].append(r)

    return {
        "by_name": by_name,
        "by_stem": by_stem,
        "by_family_id": by_family_id,
        "by_unknown_id": by_unknown_id,
        "by_sha1": by_sha1,
    }


def find_overlaps(
    test_records: List[ImageRecord],
    train_records: List[ImageRecord],
    use_hash: bool = False,
) -> Tuple[List[Dict[str, str]], Dict]:
    idx = build_train_indices(train_records)

    rows: List[Dict[str, str]] = []

    stats = {
        "overlap_pairs": 0,
        "test_images_with_overlap": 0,
        "by_reason": defaultdict(int),
        "by_family": defaultdict(int),
    }

    for tr in test_records:
        hits: List[Tuple[str, ImageRecord, str]] = []

        for rr in idx["by_name"].get(tr.name, []):
            hits.append(("same filename", rr, tr.name))

        for rr in idx["by_stem"].get(tr.stem, []):
            hits.append(("same stem", rr, tr.stem))

        # Family-aware ID hit. Include unknown family too, because some copied data
        # may have lost folder-level modality information.
        for key in tr.id_keys:
            for rr in idx["by_family_id"].get(tr.family, {}).get(key, []):
                hits.append(("same source-id key", rr, key))
            for rr in idx["by_unknown_id"].get(key, []):
                hits.append(("same source-id key in unknown family", rr, key))

        if use_hash and tr.sha1 and not tr.sha1.startswith("ERROR:"):
            for rr in idx["by_sha1"].get(tr.sha1, []):
                hits.append(("same sha1", rr, tr.sha1))

        # Deduplicate pairs.
        seen = set()
        unique_hits: List[Tuple[str, ImageRecord, str]] = []
        for reason, rr, key in hits:
            dedup_key = (reason, str(rr.image), key)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            unique_hits.append((reason, rr, key))

        if unique_hits:
            stats["test_images_with_overlap"] += 1

        for reason, rr, key in unique_hits:
            stats["overlap_pairs"] += 1
            stats["by_reason"][reason] += 1
            stats["by_family"][tr.family] += 1

            rows.append({
                "reason": reason,
                "key": key,
                "test_family": tr.family,
                "test_subset": tr.subset,
                "test_image": str(tr.image),
                "test_label": str(tr.label) if tr.label else "",
                "train_family": rr.family,
                "train_subset": rr.subset,
                "train_image": str(rr.image),
                "train_label": str(rr.label) if rr.label else "",
            })

    stats["by_reason"] = dict(stats["by_reason"])
    stats["by_family"] = dict(stats["by_family"])

    return rows, stats


# =============================================================================
# Structure expectations
# =============================================================================

def check_expected_test_structure(test_root: Path) -> Dict:
    expected = {
        "images/extratest_rgb": test_root / "images" / "extratest_rgb",
        "labels/extratest_rgb": test_root / "labels" / "extratest_rgb",
        "images/extratest_sar": test_root / "images" / "extratest_sar",
        "labels/extratest_sar": test_root / "labels" / "extratest_sar",
        "images/extratest_ir": test_root / "images" / "extratest_ir",
        "labels/extratest_ir": test_root / "labels" / "extratest_ir",
    }

    out = {}
    for k, p in expected.items():
        if p.exists():
            if k.startswith("images/"):
                n = len(find_images(p))
            else:
                n = len(find_labels(p))
            out[k] = {"exists": True, "count": n, "path": str(p)}
        else:
            out[k] = {"exists": False, "count": 0, "path": str(p)}

    return out


def check_train_structure(train_root: Path) -> Dict:
    """
    Training bridge_small_labels1 can be nested. We summarize each dataset folder.
    """
    out = {
        "root": str(train_root),
        "exists": train_root.exists(),
        "top_level_children": [],
    }

    if not train_root.exists():
        return out

    for child in sorted(p for p in train_root.iterdir() if p.is_dir()):
        item = {
            "name": child.name,
            "path": str(child),
            "images_train": len(find_images(child / "images" / "train")),
            "labels_train": len(find_labels(child / "labels" / "train")),
            "images_val": len(find_images(child / "images" / "val")),
            "labels_val": len(find_labels(child / "labels" / "val")),
            "images_total": len(find_images(child / "images")),
            "labels_total": len(find_labels(child / "labels")),
        }
        out["top_level_children"].append(item)

    return out


def write_overlap_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    fieldnames = [
        "reason",
        "key",
        "test_family",
        "test_subset",
        "test_image",
        "test_label",
        "train_family",
        "train_subset",
        "train_image",
        "train_label",
    ]

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument("--test-root", type=str, default=str(TEST_ROOT))
    parser.add_argument("--train-root", type=str, default=str(TRAIN_ROOT))
    parser.add_argument("--expected-class", type=int, default=EXPECTED_CLASS_ID)
    parser.add_argument("--hash-check", action="store_true", help="Also compare image SHA1 hashes. Slower but stricter.")
    parser.add_argument("--no-write-report", action="store_true", help="Do not write JSON/CSV reports.")
    parser.add_argument("--strict", action="store_true", help="Exit with code 2 if serious issues are found.")

    args = parser.parse_args()

    test_root = Path(args.test_root)
    train_root = Path(args.train_root)

    log("=" * 100)
    log("Check bridge extratest vs bridge_small_labels1")
    log("=" * 100)
    log(f"test_root      = {test_root}")
    log(f"train_root     = {train_root}")
    log(f"expected_class = {args.expected_class}  # bridge in current 2-class student")
    log(f"hash_check     = {args.hash_check}")
    log("=" * 100)

    test_structure = check_expected_test_structure(test_root)
    train_structure = check_train_structure(train_root)

    test_records, test_stats = collect_records(
        root=test_root,
        root_kind="test",
        expected_class=args.expected_class,
        use_hash=args.hash_check,
    )

    train_records, train_stats = collect_records(
        root=train_root,
        root_kind="train",
        expected_class=args.expected_class,
        use_hash=args.hash_check,
    )

    overlap_rows, overlap_stats = find_overlaps(
        test_records=test_records,
        train_records=train_records,
        use_hash=args.hash_check,
    )

    serious_issues = []

    if not test_root.exists():
        serious_issues.append(f"test root does not exist: {test_root}")
    if not train_root.exists():
        serious_issues.append(f"train root does not exist: {train_root}")

    if test_stats["images"] <= 0:
        serious_issues.append("test images count is 0")
    if train_stats["images"] <= 0:
        serious_issues.append("train images count is 0")

    if test_stats["images_missing_labels"] > 0:
        serious_issues.append(f"test images missing labels: {test_stats['images_missing_labels']}")
    if train_stats["images_missing_labels"] > 0:
        serious_issues.append(f"train images missing labels: {train_stats['images_missing_labels']}")

    if test_stats["bad_label_rows"] > 0:
        serious_issues.append(f"test bad label rows: {test_stats['bad_label_rows']}")
    if train_stats["bad_label_rows"] > 0:
        serious_issues.append(f"train bad label rows: {train_stats['bad_label_rows']}")

    if overlap_stats["overlap_pairs"] > 0:
        serious_issues.append(f"overlap pairs found: {overlap_stats['overlap_pairs']}")

    report = {
        "test_root": str(test_root),
        "train_root": str(train_root),
        "expected_class": args.expected_class,
        "hash_check": args.hash_check,
        "test_structure": test_structure,
        "train_structure": train_structure,
        "test_stats": test_stats,
        "train_stats": train_stats,
        "overlap_stats": overlap_stats,
        "overlap_examples": overlap_rows[:50],
        "serious_issues": serious_issues,
        "verdict": "PASS" if not serious_issues else "FAIL",
        "notes": [
            "This check is for current 2-class student: 0 ship, 1 bridge.",
            "Expected bridge class id is 1 in both bridge_small_labels1 and extracttest_bridge.",
            "Overlap check uses filename, stem, canonical source-id keys, and optionally SHA1.",
            "No source images or labels are modified by this script.",
        ],
    }

    log("")
    log("[TEST STRUCTURE]")
    log(json.dumps(test_structure, ensure_ascii=False, indent=2))

    log("")
    log("[TRAIN STRUCTURE]")
    log(json.dumps(train_structure, ensure_ascii=False, indent=2))

    log("")
    log("[TEST STATS]")
    log(json.dumps(test_stats, ensure_ascii=False, indent=2))

    log("")
    log("[TRAIN STATS]")
    log(json.dumps(train_stats, ensure_ascii=False, indent=2))

    log("")
    log("[OVERLAP STATS]")
    log(json.dumps(overlap_stats, ensure_ascii=False, indent=2))

    log("")
    log("[VERDICT]")
    if serious_issues:
        log("FAIL")
        for x in serious_issues:
            log(f"  - {x}")
    else:
        log("PASS")
        log("  - no overlap found")
        log("  - labels look like YOLO 5-column")
        log(f"  - class ids match expected class {args.expected_class}")

    if not args.no_write_report:
        test_root.mkdir(parents=True, exist_ok=True)

        report_json = test_root / "check_extratest_vs_bridge_small_labels1.json"
        report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        overlap_csv = test_root / "check_overlap_pairs.csv"
        write_overlap_csv(overlap_csv, overlap_rows)

        log("")
        log("[REPORT WRITTEN]")
        log(f"JSON: {report_json}")
        log(f"CSV:  {overlap_csv}")

    if args.strict and serious_issues:
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
