"""Render YOLO labels onto image copies for manual dataset inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
COLORS = (
    (0, 220, 255),
    (255, 120, 0),
    (60, 220, 60),
    (220, 60, 220),
    (255, 80, 80),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw YOLO boxes on image copies.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-images", type=int, default=50)
    args = parser.parse_args()
    if args.max_images <= 0:
        parser.error("--max-images must be greater than 0")
    return args


def read_class_names(data_root: Path) -> dict[int, str]:
    yaml_path = data_root / "data.yaml"
    if not yaml_path.is_file():
        return {}
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    names = data.get("names", [])
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(index): str(name) for index, name in names.items()}
    return {}


def read_image(path: Path) -> np.ndarray | None:
    """Read paths robustly on Windows, including paths containing non-ASCII text."""
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def write_image(path: Path, image: np.ndarray) -> bool:
    extension = path.suffix.lower() or ".jpg"
    success, encoded = cv2.imencode(extension, image)
    if not success:
        return False
    encoded.tofile(path)
    return True


def parse_box(line: str) -> tuple[int, float, float, float, float]:
    fields = line.split()
    if len(fields) != 5:
        raise ValueError(f"expected 5 columns, got {len(fields)}")
    values = [float(field) for field in fields]
    class_value = values[0]
    if not class_value.is_integer() or class_value < 0:
        raise ValueError(f"invalid class id {fields[0]!r}")
    x_center, y_center, width, height = values[1:]
    if not all(0.0 <= value <= 1.0 for value in values[1:]):
        raise ValueError("coordinates must all be in [0, 1]")
    if width <= 0.0 or height <= 0.0:
        raise ValueError("width and height must be greater than 0")
    return int(class_value), x_center, y_center, width, height


def draw_box(
    image: np.ndarray,
    box: tuple[int, float, float, float, float],
    class_names: dict[int, str],
) -> None:
    class_id, x_center, y_center, box_width, box_height = box
    image_height, image_width = image.shape[:2]
    x1 = max(0, min(image_width - 1, round((x_center - box_width / 2) * image_width)))
    y1 = max(0, min(image_height - 1, round((y_center - box_height / 2) * image_height)))
    x2 = max(0, min(image_width - 1, round((x_center + box_width / 2) * image_width)))
    y2 = max(0, min(image_height - 1, round((y_center + box_height / 2) * image_height)))
    color = COLORS[class_id % len(COLORS)]
    thickness = max(2, round(min(image_width, image_height) / 300))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    class_name = class_names.get(class_id, "unknown")
    text = f"{class_id} {class_name}"
    font_scale = max(0.5, min(image_width, image_height) / 900)
    text_thickness = max(1, thickness - 1)
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
    )
    text_top = max(0, y1 - text_height - baseline - 6)
    text_bottom = min(image_height - 1, text_top + text_height + baseline + 6)
    text_right = min(image_width - 1, x1 + text_width + 8)
    cv2.rectangle(image, (x1, text_top), (text_right, text_bottom), color, -1)
    cv2.putText(
        image,
        text,
        (x1 + 4, text_bottom - baseline - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        text_thickness,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    image_dir = data_root / "images" / args.split
    label_dir = data_root / "labels" / args.split

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label directory does not exist: {label_dir}")
    if out_root == data_root or data_root in out_root.parents:
        raise ValueError("--out-root must not be the dataset root or inside the dataset")

    out_root.mkdir(parents=True, exist_ok=True)
    class_names = read_class_names(data_root)
    all_images = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    selected_images = all_images[: args.max_images]
    summary: dict = {
        "data_root": str(data_root),
        "out_root": str(out_root),
        "split": args.split,
        "max_images": args.max_images,
        "available_images": len(all_images),
        "selected_images": len(selected_images),
        "written_previews": 0,
        "drawn_boxes": 0,
        "class_names": {str(index): name for index, name in sorted(class_names.items())},
        "missing_label_files": [],
        "empty_label_files": [],
        "bad_label_lines": [],
        "unreadable_images": [],
        "write_failures": [],
        "preview_files": [],
    }

    print(f"Available images: {len(all_images)}", flush=True)
    print(f"Selected images: {len(selected_images)}", flush=True)
    for image_path in selected_images:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            summary["missing_label_files"].append(str(label_path))
            continue

        lines = [
            (line_number, line.strip())
            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8-sig", errors="replace").splitlines(), 1
            )
            if line.strip()
        ]
        if not lines:
            summary["empty_label_files"].append(str(label_path))
            continue

        image = read_image(image_path)
        if image is None:
            summary["unreadable_images"].append(str(image_path))
            continue

        drawn_for_image = 0
        for line_number, line in lines:
            try:
                box = parse_box(line)
            except (ValueError, OverflowError) as exc:
                summary["bad_label_lines"].append(
                    {
                        "label": str(label_path),
                        "line_number": line_number,
                        "line": line,
                        "error": str(exc),
                    }
                )
                continue
            draw_box(image, box, class_names)
            drawn_for_image += 1

        output_path = out_root / image_path.name
        if not write_image(output_path, image):
            summary["write_failures"].append(str(output_path))
            continue
        summary["written_previews"] += 1
        summary["drawn_boxes"] += drawn_for_image
        summary["preview_files"].append(str(output_path))

    summary_path = out_root / "preview_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Written previews: {summary['written_previews']}", flush=True)
    print(f"Drawn boxes: {summary['drawn_boxes']}", flush=True)
    print(f"Preview output directory: {out_root}", flush=True)
    print(f"Preview summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
