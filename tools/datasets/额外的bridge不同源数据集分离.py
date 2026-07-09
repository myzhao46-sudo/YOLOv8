import argparse
import csv
import json
import shutil
from pathlib import Path

import yaml


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_yaml(path: Path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def infer_src_root_from_train_images(train_images: Path) -> Path:
    """
    Input:
      E:/YOLODATA/NWPU VHR-10.v1i.yolov8/train/images

    Return:
      E:/YOLODATA/NWPU VHR-10.v1i.yolov8
    """
    train_images = train_images.resolve()

    if train_images.name.lower() != "images":
        raise ValueError(f"--train-images should point to an images folder, got: {train_images}")

    split_dir = train_images.parent
    if split_dir.name.lower() not in {"train", "valid", "val", "test"}:
        raise ValueError(f"Parent of images should be train/valid/val/test, got: {split_dir}")

    return split_dir.parent.resolve()


def find_split_image_dir(src_root: Path, split: str) -> Path | None:
    """
    Roboflow commonly uses:
      train/images
      valid/images
      test/images

    Some datasets use val/images.
    """
    candidates = []

    if split == "val":
        candidates.append(src_root / "valid" / "images")
        candidates.append(src_root / "val" / "images")
    elif split == "valid":
        candidates.append(src_root / "valid" / "images")
        candidates.append(src_root / "val" / "images")
    else:
        candidates.append(src_root / split / "images")

    for c in candidates:
        if c.exists() and c.is_dir():
            return c.resolve()

    return None


def image_to_label_path(image_path: Path) -> Path:
    """
    Convert:
      .../train/images/xxx.jpg

    To:
      .../train/labels/xxx.txt
    """
    parts = list(image_path.parts)
    for i, p in enumerate(parts):
        if p.lower() == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")

    raise ValueError(f"Cannot infer label path from image path: {image_path}")


def parse_yolo_line(line: str, label_path: Path, line_idx: int):
    parts = line.strip().split()

    if len(parts) != 5:
        raise ValueError(
            f"{label_path}:{line_idx} expected YOLO 5 columns, got {len(parts)}: {line}"
        )

    try:
        cls = int(float(parts[0]))
        x = float(parts[1])
        y = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
    except Exception:
        raise ValueError(f"{label_path}:{line_idx} cannot parse numbers: {line}")

    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
        raise ValueError(f"{label_path}:{line_idx} coords out of [0,1]: {line}")

    if w <= 0 or h <= 0:
        raise ValueError(f"{label_path}:{line_idx} non-positive width/height: {line}")

    return cls, x, y, w, h


def write_output_data_yaml(out_root: Path, target_class_id: int):
    """
    target_class_id=0:
      bridge-only expert:
        nc: 1
        names: [bridge]

    target_class_id=1:
      late fusion final 2-class space:
        nc: 2
        names: [ship, bridge]
    """
    if target_class_id == 0:
        nc = 1
        names = ["bridge"]
    elif target_class_id == 1:
        nc = 2
        names = ["ship", "bridge"]
    else:
        nc = target_class_id + 1
        names = [f"class_{i}" for i in range(nc)]
        names[target_class_id] = "bridge"

    data = {
        "path": str(out_root.resolve()).replace("\\", "/"),
        "train": "images/test",
        "val": "images/test",
        "test": "images/test",
        "nc": nc,
        "names": names,
    }

    with (out_root / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(
        description="Extract one source class from Roboflow YOLOv8 NWPU dataset into a bridge-only external test set."
    )

    parser.add_argument(
        "--src-root",
        type=Path,
        default=None,
        help="Dataset root, e.g. E:/YOLODATA/NWPU VHR-10.v1i.yolov8",
    )

    parser.add_argument(
        "--train-images",
        type=Path,
        default=None,
        help="Train images folder, e.g. E:/YOLODATA/NWPU VHR-10.v1i.yolov8/train/images. "
             "If provided, src-root will be inferred.",
    )

    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Output dataset root, e.g. E:/YOLODATA/external_bridge_nwpu_cls0",
    )

    parser.add_argument(
        "--source-class-id",
        type=int,
        default=9,
        help="Class id in source YOLO labels to extract. You asked for label=9, so default is 9.",
    )

    parser.add_argument(
        "--target-class-id",
        type=int,
        default=0,
        help="Class id in output labels. Use 0 for bridge-only expert, 1 for late fusion 2-class space.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output folder if it already exists.",
    )

    args = parser.parse_args()

    if args.src_root is None and args.train_images is None:
        raise ValueError("Please provide either --src-root or --train-images")

    if args.train_images is not None:
        src_root = infer_src_root_from_train_images(args.train_images)
    else:
        src_root = args.src_root.resolve()

    out_root = args.out_root.resolve()

    data_yaml_path = src_root / "data.yaml"
    src_yaml = load_yaml(data_yaml_path)

    if out_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output folder already exists: {out_root}. Use --overwrite.")
        shutil.rmtree(out_root)

    out_img_dir = out_root / "images" / "test"
    out_lab_dir = out_root / "labels" / "test"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lab_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "src_root": str(src_root),
        "data_yaml": str(data_yaml_path) if data_yaml_path.exists() else None,
        "source_class_id": args.source_class_id,
        "target_class_id": args.target_class_id,
        "note": "Only images containing source_class_id are copied. Only that class is kept and remapped.",
        "source_yaml_nc": src_yaml.get("nc"),
        "source_yaml_names": src_yaml.get("names"),
        "total_scanned_images": 0,
        "total_written_images": 0,
        "total_written_boxes": 0,
        "class_box_counts_all": {},
        "splits": {},
        "missing_label_files": [],
        "empty_label_files": [],
        "bad_label_lines": [],
    }

    manifest_rows = []

    for split in ["train", "valid", "val", "test"]:
        img_dir = find_split_image_dir(src_root, split)
        if img_dir is None:
            continue

        split_key = img_dir.parent.name
        if split_key in summary["splits"]:
            continue

        split_summary = {
            "image_dir": str(img_dir),
            "scanned_images": 0,
            "written_images": 0,
            "written_boxes": 0,
        }

        image_paths = sorted(
            p for p in img_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        )

        for img_path in image_paths:
            summary["total_scanned_images"] += 1
            split_summary["scanned_images"] += 1

            label_path = image_to_label_path(img_path)

            if not label_path.exists():
                summary["missing_label_files"].append(str(label_path))
                continue

            raw_lines = [
                line.strip()
                for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                if line.strip()
            ]

            if not raw_lines:
                summary["empty_label_files"].append(str(label_path))
                continue

            kept_boxes = []

            for line_idx, line in enumerate(raw_lines, start=1):
                try:
                    cls, x, y, w, h = parse_yolo_line(line, label_path, line_idx)
                except Exception as e:
                    summary["bad_label_lines"].append(str(e))
                    continue

                cls_key = str(cls)
                summary["class_box_counts_all"][cls_key] = (
                    summary["class_box_counts_all"].get(cls_key, 0) + 1
                )

                if cls == args.source_class_id:
                    kept_boxes.append((args.target_class_id, x, y, w, h))

            if not kept_boxes:
                continue

            # Add split prefix to avoid duplicate names across train/valid/test.
            out_stem = f"{split_key}_{img_path.stem}"
            out_img_path = out_img_dir / f"{out_stem}{img_path.suffix.lower()}"
            out_lab_path = out_lab_dir / f"{out_stem}.txt"

            # Collision protection.
            if out_img_path.exists() or out_lab_path.exists():
                idx = 1
                while True:
                    new_stem = f"{out_stem}_{idx}"
                    out_img_path = out_img_dir / f"{new_stem}{img_path.suffix.lower()}"
                    out_lab_path = out_lab_dir / f"{new_stem}.txt"
                    if not out_img_path.exists() and not out_lab_path.exists():
                        break
                    idx += 1

            shutil.copy2(img_path, out_img_path)

            with out_lab_path.open("w", encoding="utf-8") as f:
                for cls, x, y, w, h in kept_boxes:
                    f.write(f"{cls} {x:.8f} {y:.8f} {w:.8f} {h:.8f}\n")

            summary["total_written_images"] += 1
            summary["total_written_boxes"] += len(kept_boxes)

            split_summary["written_images"] += 1
            split_summary["written_boxes"] += len(kept_boxes)

            manifest_rows.append({
                "source_split": split_key,
                "source_image": str(img_path),
                "source_label": str(label_path),
                "output_image": str(out_img_path),
                "output_label": str(out_lab_path),
                "kept_boxes": len(kept_boxes),
                "source_class_id": args.source_class_id,
                "target_class_id": args.target_class_id,
            })

        summary["splits"][split_key] = split_summary

    write_output_data_yaml(out_root, args.target_class_id)

    with (out_root / "build_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with (out_root / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "source_split",
            "source_image",
            "source_label",
            "output_image",
            "output_label",
            "kept_boxes",
            "source_class_id",
            "target_class_id",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("=" * 80)
    print("NWPU bridge extraction finished")
    print("=" * 80)
    print(f"Source root:       {src_root}")
    print(f"Output root:       {out_root}")
    print(f"Source class id:   {args.source_class_id}")
    print(f"Target class id:   {args.target_class_id}")
    print(f"Scanned images:    {summary['total_scanned_images']}")
    print(f"Written images:    {summary['total_written_images']}")
    print(f"Written boxes:     {summary['total_written_boxes']}")
    print()
    print("All source class box counts:")
    for k, v in sorted(summary["class_box_counts_all"].items(), key=lambda x: int(x[0])):
        print(f"  class {k}: {v}")
    print()
    print(f"Summary JSON:      {out_root / 'build_summary.json'}")
    print(f"Manifest CSV:      {out_root / 'manifest.csv'}")
    print(f"Output data.yaml:  {out_root / 'data.yaml'}")

    if summary["total_written_images"] == 0:
        print()
        print("[WARNING] No images were written.")
        print("This usually means source-class-id is wrong.")
        print("For Roboflow YOLOv8, NWPU bridge may be class 8 instead of 9.")
        print("Try rerun with: --source-class-id 8")


if __name__ == "__main__":
    main()