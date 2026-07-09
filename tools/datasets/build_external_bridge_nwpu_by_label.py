"""Build a bridge-only external YOLO test set from an NWPU Roboflow package."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "valid", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract one source class from an NWPU Roboflow YOLO dataset, remap it, "
            "and write a bridge-only external test set."
        )
    )
    parser.add_argument("--src-root", type=Path, help="Source dataset root.")
    parser.add_argument(
        "--train-images",
        type=Path,
        help="Source train/images directory; its dataset root is inferred automatically.",
    )
    parser.add_argument("--out-root", type=Path, required=True, help="Output dataset root.")
    parser.add_argument("--source-class-id", type=int, default=9)
    parser.add_argument("--target-class-id", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.src_root is None and args.train_images is None:
        parser.error("one of --src-root or --train-images is required")
    if args.source_class_id < 0 or args.target_class_id < 0:
        parser.error("class IDs must be non-negative integers")
    return args


def infer_src_root(train_images: Path) -> Path:
    train_images = train_images.expanduser().resolve()
    if train_images.name.lower() != "images":
        raise ValueError(f"--train-images must end in an images directory: {train_images}")
    if train_images.parent.name.lower() != "train":
        raise ValueError(f"--train-images must point to train/images: {train_images}")
    return train_images.parent.parent


def parse_label_line(line: str, label_path: Path, line_number: int) -> tuple[int, float, float, float, float]:
    fields = line.split()
    if len(fields) != 5:
        raise ValueError(f"expected 5 columns, got {len(fields)}")

    try:
        class_value = float(fields[0])
        coordinates = tuple(float(value) for value in fields[1:])
    except ValueError as exc:
        raise ValueError("contains a non-numeric value") from exc

    if not class_value.is_integer() or class_value < 0:
        raise ValueError(f"invalid class id {fields[0]!r}")
    if not all(0.0 <= value <= 1.0 for value in coordinates):
        raise ValueError("coordinates must all be in [0, 1]")

    x_center, y_center, width, height = coordinates
    if width <= 0.0 or height <= 0.0:
        raise ValueError("width and height must be greater than 0")
    return int(class_value), x_center, y_center, width, height


def label_for_image(image_path: Path, image_dir: Path) -> Path:
    relative_path = image_path.relative_to(image_dir)
    return image_dir.parent.joinpath("labels", relative_path).with_suffix(".txt")


def unique_output_paths(
    output_image_dir: Path, output_label_dir: Path, split: str, source_image: Path
) -> tuple[Path, Path]:
    base_stem = f"{split}_{source_image.stem}"
    suffix = source_image.suffix.lower()
    counter = 0
    while True:
        stem = base_stem if counter == 0 else f"{base_stem}_{counter}"
        output_image = output_image_dir / f"{stem}{suffix}"
        output_label = output_label_dir / f"{stem}.txt"
        if not output_image.exists() and not output_label.exists():
            return output_image, output_label
        counter += 1


def write_data_yaml(out_root: Path, target_class_id: int) -> None:
    if target_class_id == 0:
        nc = 1
        names = ["bridge"]
    else:
        nc = target_class_id + 1
        names = [f"class_{index}" for index in range(nc)]
        names[target_class_id] = "bridge"

    yaml_text = (
        f'path: "{out_root.as_posix()}"\n'
        "train: images/test\n"
        "val: images/test\n"
        "test: images/test\n"
        f"nc: {nc}\n"
        f"names: [{', '.join(names)}]\n"
    )
    (out_root / "data.yaml").write_text(yaml_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    src_root = (
        infer_src_root(args.train_images)
        if args.train_images is not None
        else args.src_root.expanduser().resolve()
    )
    out_root = args.out_root.expanduser().resolve()

    if not src_root.is_dir():
        raise FileNotFoundError(f"Source dataset root does not exist: {src_root}")
    if out_root == src_root or src_root in out_root.parents:
        raise ValueError("--out-root must not be the source root or inside the source dataset")

    if args.src_root is not None and args.train_images is not None:
        explicit_src_root = args.src_root.expanduser().resolve()
        if explicit_src_root != src_root:
            print(
                f"[WARNING] Ignoring --src-root {explicit_src_root}; "
                f"--train-images inferred {src_root}",
                flush=True,
            )

    if out_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite to replace it: {out_root}")
        print(f"Removing existing output directory: {out_root}", flush=True)
        shutil.rmtree(out_root)

    output_image_dir = out_root / "images" / "test"
    output_label_dir = out_root / "labels" / "test"
    output_image_dir.mkdir(parents=True)
    output_label_dir.mkdir(parents=True)

    class_counts: Counter[int] = Counter()
    summary: dict = {
        "src_root": str(src_root),
        "out_root": str(out_root),
        "source_class_id": args.source_class_id,
        "target_class_id": args.target_class_id,
        "total_scanned_images": 0,
        "total_written_images": 0,
        "total_written_boxes": 0,
        "class_box_counts_all": {},
        "splits": {},
        "missing_label_files": [],
        "empty_label_files": [],
        "bad_label_lines": [],
    }
    manifest_rows: list[dict] = []

    for split in SPLITS:
        image_dir = src_root / split / "images"
        if not image_dir.is_dir():
            continue

        print(f"Starting split scan: {split} ({image_dir})", flush=True)
        image_paths = sorted(
            path
            for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        print(f"Found {len(image_paths)} images in split: {split}", flush=True)
        split_summary = {"scanned_images": 0, "written_images": 0, "written_boxes": 0}

        for image_path in image_paths:
            summary["total_scanned_images"] += 1
            split_summary["scanned_images"] += 1
            label_path = label_for_image(image_path, image_dir)

            if not label_path.is_file():
                summary["missing_label_files"].append(str(label_path))
                continue

            nonempty_lines = [
                (number, line.strip())
                for number, line in enumerate(
                    label_path.read_text(encoding="utf-8-sig", errors="replace").splitlines(), 1
                )
                if line.strip()
            ]
            if not nonempty_lines:
                summary["empty_label_files"].append(str(label_path))
                continue

            kept_boxes: list[tuple[int, float, float, float, float]] = []
            for line_number, line in nonempty_lines:
                try:
                    class_id, x_center, y_center, width, height = parse_label_line(
                        line, label_path, line_number
                    )
                except ValueError as exc:
                    summary["bad_label_lines"].append(
                        {
                            "label": str(label_path),
                            "line_number": line_number,
                            "line": line,
                            "error": str(exc),
                        }
                    )
                    continue

                class_counts[class_id] += 1
                if class_id == args.source_class_id:
                    kept_boxes.append(
                        (args.target_class_id, x_center, y_center, width, height)
                    )

            if not kept_boxes:
                continue

            output_image, output_label = unique_output_paths(
                output_image_dir, output_label_dir, split, image_path
            )
            shutil.copy2(image_path, output_image)
            output_label.write_text(
                "".join(
                    f"{class_id} {x:.8f} {y:.8f} {w:.8f} {h:.8f}\n"
                    for class_id, x, y, w, h in kept_boxes
                ),
                encoding="utf-8",
            )

            kept_count = len(kept_boxes)
            summary["total_written_images"] += 1
            summary["total_written_boxes"] += kept_count
            split_summary["written_images"] += 1
            split_summary["written_boxes"] += kept_count
            manifest_rows.append(
                {
                    "source_split": split,
                    "source_image": str(image_path),
                    "source_label": str(label_path),
                    "output_image": str(output_image),
                    "output_label": str(output_label),
                    "kept_boxes": kept_count,
                }
            )

        summary["splits"][split] = split_summary

    summary["class_box_counts_all"] = {
        str(class_id): count for class_id, count in sorted(class_counts.items())
    }
    write_data_yaml(out_root, args.target_class_id)
    (out_root / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (out_root / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as file:
        fieldnames = [
            "source_split",
            "source_image",
            "source_label",
            "output_image",
            "output_label",
            "kept_boxes",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Final output directory: {out_root}", flush=True)
    print(f"Written images: {summary['total_written_images']}", flush=True)
    print(f"Written boxes: {summary['total_written_boxes']}", flush=True)
    print("class_box_counts_all:", flush=True)
    for class_id, count in summary["class_box_counts_all"].items():
        print(f"  class {class_id}: {count}", flush=True)
    if summary["total_written_images"] == 0:
        print(
            "[WARNING] No images were written. The source class ID may be wrong; "
            "try --source-class-id 8.",
            flush=True,
        )


if __name__ == "__main__":
    main()
