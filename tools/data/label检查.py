# tools/data/check_yolo_labels.py
from pathlib import Path
import argparse
import math
import json


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


DATASETS_TEMPLATE = [
    {
        "name": "ship",
        "relative_root": "{ship_name}",
        "expected_cls": 0,
        "splits": ["train", "val"],
    },
    {
        "name": "bridge_rgb",
        "relative_root": "bridge_small/DIOR_bridge_rgb30",
        "expected_cls": 3,
        "splits": ["train", "val"],
    },
    {
        "name": "bridge_sar",
        "relative_root": "bridge_small/{sar_name}",
        "expected_cls": 3,
        "splits": ["train", "val"],
    },
    {
        "name": "bridge_infr",
        "relative_root": "bridge_small/MassMIND_bridge_infr30",
        "expected_cls": 3,
        "splits": ["train", "val"],
    },
]


def find_images(image_dir: Path):
    if not image_dir.exists():
        return []
    return sorted([p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS])


def parse_label_file(label_path: Path):
    rows = []
    if not label_path.exists():
        return rows

    text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return rows

    for line_idx, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        rows.append((line_idx, parts))
    return rows


def is_number(x: str):
    try:
        float(x)
        return True
    except Exception:
        return False


def check_one_dataset(datasets_root: Path, item: dict, max_errors_show: int = 30):
    name = item["name"]
    root = datasets_root / item["relative_root"]
    expected_cls = item["expected_cls"]
    splits = item["splits"]

    result = {
        "name": name,
        "root": root.as_posix(),
        "expected_cls": expected_cls,
        "splits": {},
        "errors": [],
        "warnings": [],
    }

    print("=" * 100)
    print(f"[DATASET] {name}")
    print(f"  root: {root}")
    print(f"  expected class id: {expected_cls}")

    for split in splits:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split

        images = find_images(image_dir)
        labels = sorted(label_dir.glob("*.txt")) if label_dir.exists() else []

        image_stems = {p.stem for p in images}
        label_stems = {p.stem for p in labels}

        missing_labels = sorted(image_stems - label_stems)
        extra_labels = sorted(label_stems - image_stems)

        split_stat = {
            "image_dir": image_dir.as_posix(),
            "label_dir": label_dir.as_posix(),
            "num_images": len(images),
            "num_labels": len(labels),
            "missing_labels": len(missing_labels),
            "extra_labels": len(extra_labels),
            "num_boxes": 0,
            "empty_label_files": 0,
            "bad_format_rows": 0,
            "bad_class_rows": 0,
            "bad_coord_rows": 0,
            "polygon_9col_rows": 0,
        }

        if not image_dir.exists():
            result["errors"].append(f"[{name}/{split}] image dir not found: {image_dir}")
        if not label_dir.exists():
            result["errors"].append(f"[{name}/{split}] label dir not found: {label_dir}")

        for stem in missing_labels[:max_errors_show]:
            result["errors"].append(f"[{name}/{split}] missing label for image stem: {stem}")

        for stem in extra_labels[:max_errors_show]:
            result["errors"].append(f"[{name}/{split}] extra label without image stem: {stem}")

        for label_path in labels:
            rows = parse_label_file(label_path)
            if len(rows) == 0:
                split_stat["empty_label_files"] += 1
                result["warnings"].append(f"[{name}/{split}] empty label file: {label_path}")
                continue

            for line_idx, parts in rows:
                if len(parts) == 9:
                    split_stat["polygon_9col_rows"] += 1
                    split_stat["bad_format_rows"] += 1
                    result["errors"].append(
                        f"[{name}/{split}] 9-col polygon row, should be YOLO 5-col: "
                        f"{label_path}:{line_idx}"
                    )
                    continue

                if len(parts) != 5:
                    split_stat["bad_format_rows"] += 1
                    result["errors"].append(
                        f"[{name}/{split}] bad column count={len(parts)}, expected 5: "
                        f"{label_path}:{line_idx}"
                    )
                    continue

                if not all(is_number(x) for x in parts):
                    split_stat["bad_format_rows"] += 1
                    result["errors"].append(
                        f"[{name}/{split}] non-numeric row: {label_path}:{line_idx}"
                    )
                    continue

                cls_f, cx, cy, w, h = map(float, parts)

                if not float(cls_f).is_integer():
                    split_stat["bad_class_rows"] += 1
                    result["errors"].append(
                        f"[{name}/{split}] class id is not integer: {label_path}:{line_idx} -> {cls_f}"
                    )
                    continue

                cls_id = int(cls_f)
                if cls_id != expected_cls:
                    split_stat["bad_class_rows"] += 1
                    result["errors"].append(
                        f"[{name}/{split}] wrong class id, expected {expected_cls}, got {cls_id}: "
                        f"{label_path}:{line_idx}"
                    )

                coords = [cx, cy, w, h]
                if not all(math.isfinite(v) for v in coords):
                    split_stat["bad_coord_rows"] += 1
                    result["errors"].append(
                        f"[{name}/{split}] non-finite coord: {label_path}:{line_idx}"
                    )
                    continue

                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                    split_stat["bad_coord_rows"] += 1
                    result["errors"].append(
                        f"[{name}/{split}] coord out of [0,1]: {label_path}:{line_idx} -> "
                        f"{cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                    )

                if w <= 0.0 or h <= 0.0:
                    split_stat["bad_coord_rows"] += 1
                    result["errors"].append(
                        f"[{name}/{split}] invalid box size w/h <= 0: {label_path}:{line_idx} -> "
                        f"w={w:.6f}, h={h:.6f}"
                    )

                split_stat["num_boxes"] += 1

        result["splits"][split] = split_stat

        print(f"\n  [{split}]")
        print(f"    images:            {split_stat['num_images']}")
        print(f"    labels:            {split_stat['num_labels']}")
        print(f"    boxes:             {split_stat['num_boxes']}")
        print(f"    missing labels:    {split_stat['missing_labels']}")
        print(f"    extra labels:      {split_stat['extra_labels']}")
        print(f"    empty label files: {split_stat['empty_label_files']}")
        print(f"    bad format rows:   {split_stat['bad_format_rows']}")
        print(f"    bad class rows:    {split_stat['bad_class_rows']}")
        print(f"    bad coord rows:    {split_stat['bad_coord_rows']}")
        print(f"    9-col rows:        {split_stat['polygon_9col_rows']}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help="YOLOv8 repo root. Default: auto infer from this script path.",
    )
    parser.add_argument("--ship-name", type=str, default="ship_small_split")
    parser.add_argument("--sar-name", type=str, default="MASR_bridge_sar30")
    parser.add_argument(
        "--save-json",
        type=str,
        default="runs/data_check/global4_label_check.json",
    )
    args = parser.parse_args()

    if args.repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    else:
        repo_root = Path(args.repo_root).resolve()

    datasets_root = repo_root / "ultralytics" / "datasets"

    datasets = []
    for x in DATASETS_TEMPLATE:
        y = dict(x)
        y["relative_root"] = y["relative_root"].format(
            ship_name=args.ship_name,
            sar_name=args.sar_name,
        )
        datasets.append(y)

    all_results = []
    all_errors = []
    all_warnings = []

    print(f"Repo root:     {repo_root}")
    print(f"Datasets root: {datasets_root}")

    for item in datasets:
        res = check_one_dataset(datasets_root, item)
        all_results.append(res)
        all_errors.extend(res["errors"])
        all_warnings.extend(res["warnings"])

    save_path = repo_root / args.save_json
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print("[SUMMARY]")
    print(f"  total errors:   {len(all_errors)}")
    print(f"  total warnings: {len(all_warnings)}")
    print(f"  json report:    {save_path}")

    if all_errors:
        print("\nFirst errors:")
        for e in all_errors[:50]:
            print("  ERROR:", e)

    if all_warnings:
        print("\nFirst warnings:")
        for w in all_warnings[:30]:
            print("  WARN:", w)

    if all_errors:
        raise SystemExit(1)

    print("\n[OK] All label checks passed.")


if __name__ == "__main__":
    main()