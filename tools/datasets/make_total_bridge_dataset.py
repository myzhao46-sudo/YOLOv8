from __future__ import annotations

import sys
sys.path.insert(0, r"C:/Users/DOCTOR/Documents/GitHub/YOLOv8/ultralytics")
sys.path.insert(1, r"C:/Users/DOCTOR/Documents/GitHub/YOLOv8")

import argparse
import csv
import json
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(r"C:/Users/DOCTOR/Documents/GitHub/YOLOv8")
DEFAULT_MSAR_IMAGE_DIR = Path(r"E:/YOLODATA/MSAR-1.0 dataset/JPEGImages")
DEFAULT_MSAR_XML_DIR = Path(r"E:/YOLODATA/MSAR-1.0 dataset/Annotations")
DEFAULT_IR_IMAGE_ROOT = Path(r"E:/YOLODATA/MassMIND_bridge_yolo/images/train")
DEFAULT_IR_LABEL_ROOT = Path(r"E:/YOLODATA/MassMIND_bridge_yolo/labels/train")
DEFAULT_RGB_IMAGE_DIR = Path(r"E:/YOLODATA/bridge_rgb_all/JPEGImages-test")
DEFAULT_RGB_XML_DIR = Path(r"E:/YOLODATA/bridge_rgb_all/Annotations/Horizontal Bounding Boxes")
DEFAULT_DST_ROOT = REPO_ROOT / "ultralytics" / "datasets" / "total_bridge"
DEFAULT_CONFIGS_ROOT = REPO_ROOT / "configs" / "datasets"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
XML_ENCODINGS = ("utf-8", "utf-8-sig", "gbk", "gb2312")
BRIDGE_NAMES = {"桥梁", "bridge", "Bridge", "BRIDGE"}
MANIFEST_FIELDS = [
    "modality",
    "src_image",
    "src_label",
    "src_xml",
    "dst_image",
    "dst_label",
    "boxes",
    "skipped_label_rows",
]


@dataclass
class Box:
    cx: float
    cy: float
    w: float
    h: float


@dataclass
class Sample:
    modality: str
    src_image: Path
    src_label: Path | None
    src_xml: Path | None
    boxes: list[Box]
    skipped_label_rows: int


def log(msg: object = "") -> None:
    print(msg, flush=True)


def norm_path(path: Path) -> str:
    return path.resolve().as_posix()


def clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def read_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def find_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMG_EXTS)


def build_image_index(image_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for image in find_images(image_dir):
        index.setdefault(image.name.lower(), image)
        index.setdefault(image.stem.lower(), image)
    return index


def read_xml_root(xml_path: Path) -> ET.Element:
    data = xml_path.read_bytes()
    last_error = None
    for encoding in XML_ENCODINGS:
        try:
            return ET.fromstring(data.decode(encoding))
        except Exception as exc:
            last_error = exc
    try:
        return ET.fromstring(data.decode("utf-8", errors="ignore"))
    except Exception as exc:
        raise ValueError(f"Failed to parse XML {xml_path}: {last_error!r}; fallback={exc!r}") from exc


def xml_text(node: ET.Element | None) -> str | None:
    if node is None or node.text is None:
        return None
    text = node.text.strip()
    return text or None


def xml_float(parent: ET.Element | None, tag: str) -> float | None:
    if parent is None:
        return None
    text = xml_text(parent.find(tag))
    if text is None:
        return None
    try:
        return float(text)
    except Exception:
        return None


def image_for_xml(xml_path: Path, image_index: dict[str, Path], root: ET.Element) -> Path | None:
    filename = xml_text(root.find("filename"))
    if filename:
        hit = image_index.get(filename.lower())
        if hit:
            return hit
        hit = image_index.get(Path(filename).stem.lower())
        if hit:
            return hit
    return image_index.get(xml_path.stem.lower())


def wh_from_xml_or_image(root: ET.Element, image_path: Path) -> tuple[int, int]:
    size = root.find("size")
    width = xml_float(size, "width")
    height = xml_float(size, "height")
    if width is None or height is None or width <= 0 or height <= 0:
        return read_image_size(image_path)
    return int(round(width)), int(round(height))


def xyxy_to_box(x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int) -> Box | None:
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = clip01(x1 / img_w)
    x2 = clip01(x2 / img_w)
    y1 = clip01(y1 / img_h)
    y2 = clip01(y2 / img_h)
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return None
    return Box((x1 + x2) / 2.0, (y1 + y2) / 2.0, w, h)


def parse_voc_bridge_boxes(xml_path: Path, image_path: Path, allowed_names: set[str]) -> tuple[list[Box], int, int]:
    root = read_xml_root(xml_path)
    img_w, img_h = wh_from_xml_or_image(root, image_path)
    boxes: list[Box] = []
    bad_boxes = 0
    for obj in root.findall("object"):
        name = xml_text(obj.find("name"))
        if name not in allowed_names:
            continue
        bnd = obj.find("bndbox")
        xmin = xml_float(bnd, "xmin")
        ymin = xml_float(bnd, "ymin")
        xmax = xml_float(bnd, "xmax")
        ymax = xml_float(bnd, "ymax")
        if None in (xmin, ymin, xmax, ymax):
            bad_boxes += 1
            continue
        box = xyxy_to_box(float(xmin), float(ymin), float(xmax), float(ymax), img_w, img_h)
        if box is None:
            bad_boxes += 1
            continue
        boxes.append(box)
    return boxes, bad_boxes, img_w * img_h


def parse_ir_label(label_path: Path, image_path: Path) -> tuple[list[Box], int]:
    img_w, img_h = read_image_size(image_path)
    boxes: list[Box] = []
    skipped = 0
    if not label_path.exists():
        return boxes, 1
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            if len(parts) == 5:
                cx, cy, bw, bh = map(float, parts[1:5])
                if max(abs(cx), abs(cy), abs(bw), abs(bh)) > 1.5:
                    cx, bw = cx / img_w, bw / img_w
                    cy, bh = cy / img_h, bh / img_h
                x1, y1 = cx - bw / 2.0, cy - bh / 2.0
                x2, y2 = cx + bw / 2.0, cy + bh / 2.0
            elif len(parts) == 9:
                coords = list(map(float, parts[1:9]))
                xs = coords[0::2]
                ys = coords[1::2]
                if max(max(abs(x) for x in xs), max(abs(y) for y in ys)) > 1.5:
                    xs = [x / img_w for x in xs]
                    ys = [y / img_h for y in ys]
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)
            else:
                skipped += 1
                continue
            x1, y1, x2, y2 = clip01(x1), clip01(y1), clip01(x2), clip01(y2)
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                skipped += 1
                continue
            boxes.append(Box((x1 + x2) / 2.0, (y1 + y2) / 2.0, w, h))
        except Exception:
            skipped += 1
    return boxes, skipped


def init_modality_stats() -> dict:
    return {
        "xml_files": 0,
        "raw_images": 0,
        "images_found": 0,
        "missing_images": 0,
        "missing_labels": 0,
        "images_copied": 0,
        "boxes": 0,
        "no_target_object": 0,
        "parse_errors": 0,
        "bad_boxes": 0,
        "bad_rows": 0,
        "empty_labels": 0,
        "class_histogram": {},
        "examples": {
            "missing_images": [],
            "missing_labels": [],
            "parse_errors": [],
            "no_target_object": [],
        },
    }


def add_example(stats: dict, key: str, value: object, limit: int = 20) -> None:
    if len(stats["examples"][key]) < limit:
        stats["examples"][key].append(str(value))


def collect_xml_source(modality: str, image_dir: Path, xml_dir: Path, allowed_names: set[str]) -> tuple[list[Sample], dict]:
    if not image_dir.exists():
        raise FileNotFoundError(f"{modality} image dir not found: {image_dir}")
    if not xml_dir.exists():
        raise FileNotFoundError(f"{modality} XML dir not found: {xml_dir}")
    stats = init_modality_stats()
    image_index = build_image_index(image_dir)
    xml_files = sorted(xml_dir.rglob("*.xml"))
    stats["xml_files"] = len(xml_files)
    samples: list[Sample] = []
    for xml_path in xml_files:
        try:
            root = read_xml_root(xml_path)
            image_path = image_for_xml(xml_path, image_index, root)
            if image_path is None:
                stats["missing_images"] += 1
                add_example(stats, "missing_images", xml_path)
                continue
            stats["images_found"] += 1
            boxes, bad_boxes, _ = parse_voc_bridge_boxes(xml_path, image_path, allowed_names)
            stats["bad_boxes"] += bad_boxes
            if not boxes:
                stats["no_target_object"] += 1
                add_example(stats, "no_target_object", xml_path)
                continue
            samples.append(Sample(modality, image_path, None, xml_path, boxes, bad_boxes))
            stats["boxes"] += len(boxes)
            stats["images_copied"] += 1
            stats["class_histogram"][0] = stats["class_histogram"].get(0, 0) + len(boxes)
        except Exception as exc:
            stats["parse_errors"] += 1
            add_example(stats, "parse_errors", f"{xml_path} | {exc}")
    return samples, stats


def collect_ir_source(image_root: Path, label_root: Path) -> tuple[list[Sample], dict]:
    if not image_root.exists():
        raise FileNotFoundError(f"IR image root not found: {image_root}")
    if not label_root.exists():
        raise FileNotFoundError(f"IR label root not found: {label_root}")
    stats = init_modality_stats()
    images = find_images(image_root)
    stats["raw_images"] = len(images)
    samples: list[Sample] = []
    for image_path in images:
        label_path = label_root / f"{image_path.stem}.txt"
        if not label_path.exists():
            stats["missing_labels"] += 1
            add_example(stats, "missing_labels", image_path)
            continue
        boxes, skipped = parse_ir_label(label_path, image_path)
        stats["bad_rows"] += skipped
        if not boxes:
            stats["empty_labels"] += 1
            add_example(stats, "no_target_object", label_path)
            continue
        samples.append(Sample("ir", image_path, label_path, None, boxes, skipped))
        stats["boxes"] += len(boxes)
        stats["images_copied"] += 1
        stats["class_histogram"][0] = stats["class_histogram"].get(0, 0) + len(boxes)
    return samples, stats


def unique_destination(dst_dir: Path, prefixed_name: str, used_names: set[str]) -> Path:
    stem = Path(prefixed_name).stem
    suffix = Path(prefixed_name).suffix
    name = prefixed_name
    i = 1
    while name.lower() in used_names or (dst_dir / name).exists():
        name = f"{stem}_{i:03d}{suffix}"
        i += 1
    used_names.add(name.lower())
    return dst_dir / name


def write_label(path: Path, boxes: list[Box]) -> None:
    lines = [f"0 {b.cx:.8f} {b.cy:.8f} {b.w:.8f} {b.h:.8f}\n" for b in boxes]
    path.write_text("".join(lines), encoding="utf-8")


def prepare_output(dst_root: Path, overwrite: bool, dry_run: bool) -> None:
    if dry_run:
        return
    if dst_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {dst_root}. Use --overwrite to rebuild.")
        resolved = dst_root.resolve()
        expected = DEFAULT_DST_ROOT.resolve()
        if resolved != expected:
            raise RuntimeError(f"Refusing to delete unexpected output path: {resolved}")
        shutil.rmtree(resolved)
    for split in ("train_rgb", "train_sar", "train_ir"):
        (dst_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_samples(dst_root: Path, grouped: dict[str, list[Sample]], dry_run: bool) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    copy_stats = {}
    split_by_modality = {"rgb": "train_rgb", "sar": "train_sar", "ir": "train_ir"}
    prefix_by_modality = {"rgb": "rgb_", "sar": "sar_", "ir": "ir_"}
    for modality, samples in grouped.items():
        split = split_by_modality[modality]
        dst_img_dir = dst_root / "images" / split
        dst_lbl_dir = dst_root / "labels" / split
        used_names: set[str] = set()
        boxes_total = 0
        skipped_total = 0
        for sample in samples:
            dst_img = unique_destination(dst_img_dir, prefix_by_modality[modality] + sample.src_image.name, used_names)
            dst_label = dst_lbl_dir / f"{dst_img.stem}.txt"
            boxes_total += len(sample.boxes)
            skipped_total += sample.skipped_label_rows
            rows.append(
                {
                    "modality": modality,
                    "src_image": str(sample.src_image),
                    "src_label": str(sample.src_label or ""),
                    "src_xml": str(sample.src_xml or ""),
                    "dst_image": str(dst_img),
                    "dst_label": str(dst_label),
                    "boxes": len(sample.boxes),
                    "skipped_label_rows": sample.skipped_label_rows,
                }
            )
            if not dry_run:
                shutil.copy2(sample.src_image, dst_img)
                write_label(dst_label, sample.boxes)
        copy_stats[modality] = {
            "images": len(samples),
            "boxes": boxes_total,
            "skipped_label_rows": skipped_total,
            "images_dir": str(dst_img_dir),
            "labels_dir": str(dst_lbl_dir),
        }
    return rows, copy_stats


def write_manifest(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})


def yaml_text(dst_root: Path) -> str:
    train_dirs = [
        dst_root / "images" / "train_rgb",
        dst_root / "images" / "train_sar",
        dst_root / "images" / "train_ir",
    ]
    lines = ["train:"]
    lines.extend(f"  - {norm_path(path)}" for path in train_dirs)
    lines.append("")
    lines.append("val:")
    lines.extend(f"  - {norm_path(path)}" for path in train_dirs)
    lines.extend(["", "nc: 1", "names:", "  0: bridge", ""])
    return "\n".join(lines)


def write_yamls(dst_root: Path, configs_root: Path) -> dict:
    dataset_yaml = dst_root / "data.yaml"
    config_yaml = configs_root / "total_bridge.yaml"
    text = yaml_text(dst_root)
    dataset_yaml.write_text(text, encoding="utf-8")
    config_yaml.parent.mkdir(parents=True, exist_ok=True)
    config_yaml.write_text(text, encoding="utf-8")
    return {"dataset_yaml": str(dataset_yaml), "config_yaml": str(config_yaml)}


def verify_output(dst_root: Path) -> dict:
    stats = {
        "image_count": 0,
        "label_files": 0,
        "rows": 0,
        "class_histogram": {},
        "missing_label_files": 0,
        "bad_rows": 0,
        "bad_examples": [],
        "all_images_have_labels": True,
        "all_labels_5_columns": True,
        "all_class_id_zero": True,
        "class_id_gt_zero": False,
        "coords_in_0_1": True,
        "positive_wh": True,
    }
    for image_dir in sorted((dst_root / "images").glob("train_*")):
        label_dir = dst_root / "labels" / image_dir.name
        for image in find_images(image_dir):
            stats["image_count"] += 1
            label = label_dir / f"{image.stem}.txt"
            if not label.exists():
                stats["missing_label_files"] += 1
                stats["all_images_have_labels"] = False
                if len(stats["bad_examples"]) < 20:
                    stats["bad_examples"].append(f"missing label: {image}")
                continue
            stats["label_files"] += 1
            for line_no, line in enumerate(label.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                parts = line.split()
                if not parts:
                    continue
                stats["rows"] += 1
                try:
                    if len(parts) != 5:
                        stats["all_labels_5_columns"] = False
                        raise ValueError(f"expected 5 columns, got {len(parts)}")
                    cls = int(float(parts[0]))
                    values = [float(v) for v in parts[1:]]
                    stats["class_histogram"][cls] = stats["class_histogram"].get(cls, 0) + 1
                    if cls != 0:
                        stats["all_class_id_zero"] = False
                    if cls > 0:
                        stats["class_id_gt_zero"] = True
                    if any(v < 0 or v > 1 for v in values):
                        stats["coords_in_0_1"] = False
                        raise ValueError("coordinate outside [0,1]")
                    if values[2] <= 0 or values[3] <= 0:
                        stats["positive_wh"] = False
                        raise ValueError("non-positive width/height")
                except Exception as exc:
                    stats["bad_rows"] += 1
                    if len(stats["bad_examples"]) < 20:
                        stats["bad_examples"].append(f"{label}:{line_no}: {line} | {exc}")
    return stats


def flatten_summary(summary: dict) -> None:
    for modality in ("sar", "rgb", "ir"):
        prefix = f"{modality}_"
        modality_stats = summary["modalities"].get(modality, {})
        for key, value in modality_stats.items():
            if key != "examples":
                summary[prefix + key] = value


def main() -> int:
    parser = argparse.ArgumentParser(description="Build bridge-only total_bridge YOLO detect dataset from raw RGB/SAR/IR sources.")
    parser.add_argument("--msar-image-dir", default=str(DEFAULT_MSAR_IMAGE_DIR))
    parser.add_argument("--msar-xml-dir", default=str(DEFAULT_MSAR_XML_DIR))
    parser.add_argument("--ir-image-root", default=str(DEFAULT_IR_IMAGE_ROOT))
    parser.add_argument("--ir-label-root", default=str(DEFAULT_IR_LABEL_ROOT))
    parser.add_argument("--rgb-image-dir", default=str(DEFAULT_RGB_IMAGE_DIR))
    parser.add_argument("--rgb-xml-dir", default=str(DEFAULT_RGB_XML_DIR))
    parser.add_argument("--dst-root", default=str(DEFAULT_DST_ROOT))
    parser.add_argument("--configs-root", default=str(DEFAULT_CONFIGS_ROOT))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    msar_image_dir = Path(args.msar_image_dir)
    msar_xml_dir = Path(args.msar_xml_dir)
    ir_image_root = Path(args.ir_image_root)
    ir_label_root = Path(args.ir_label_root)
    rgb_image_dir = Path(args.rgb_image_dir)
    rgb_xml_dir = Path(args.rgb_xml_dir)
    dst_root = Path(args.dst_root)
    configs_root = Path(args.configs_root)

    log("=" * 100)
    log("Build total_bridge bridge-only YOLO detect dataset")
    log("=" * 100)
    for name, value in {
        "msar_image_dir": msar_image_dir,
        "msar_xml_dir": msar_xml_dir,
        "ir_image_root": ir_image_root,
        "ir_label_root": ir_label_root,
        "rgb_image_dir": rgb_image_dir,
        "rgb_xml_dir": rgb_xml_dir,
        "dst_root": dst_root,
        "configs_root": configs_root,
        "dry_run": args.dry_run,
        "overwrite": args.overwrite,
    }.items():
        log(f"{name}: {value}")
    log("=" * 100)

    prepare_output(dst_root, overwrite=args.overwrite, dry_run=args.dry_run)

    sar_samples, sar_stats = collect_xml_source("sar", msar_image_dir, msar_xml_dir, BRIDGE_NAMES)
    rgb_samples, rgb_stats = collect_xml_source("rgb", rgb_image_dir, rgb_xml_dir, {"bridge", "Bridge", "BRIDGE"})
    ir_samples, ir_stats = collect_ir_source(ir_image_root, ir_label_root)
    grouped = {"rgb": rgb_samples, "sar": sar_samples, "ir": ir_samples}

    manifest_rows, copy_stats = copy_samples(dst_root, grouped, dry_run=args.dry_run)

    summary = {
        "dry_run": args.dry_run,
        "sources": {
            "sar_msar_images": str(msar_image_dir),
            "sar_msar_xml": str(msar_xml_dir),
            "ir_images": str(ir_image_root),
            "ir_labels": str(ir_label_root),
            "rgb_images": str(rgb_image_dir),
            "rgb_xml": str(rgb_xml_dir),
        },
        "output_dataset": str(dst_root),
        "modalities": {"rgb": rgb_stats, "sar": sar_stats, "ir": ir_stats},
        "copy_stats": copy_stats,
        "manifest_csv": str(dst_root / "manifest.csv"),
        "notes": [
            "Bridge-only expert dataset.",
            "All output labels are YOLO detect 5-column rows with class id 0.",
            "No no-overlap filtering is performed.",
            "The old E:/YOLODATA/bridge_sar_msar_yolo_cls3_300 subset is not read.",
            "Original datasets are not modified.",
        ],
    }
    flatten_summary(summary)

    if not args.dry_run:
        write_manifest(dst_root / "manifest.csv", manifest_rows)
        yaml_paths = write_yamls(dst_root, configs_root)
        summary["yaml_paths"] = yaml_paths
        verify_stats = verify_output(dst_root)
        summary["verify_output"] = verify_stats
        (dst_root / "selection_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        summary["yaml_paths"] = {
            "dataset_yaml": str(dst_root / "data.yaml"),
            "config_yaml": str(configs_root / "total_bridge.yaml"),
        }

    total_images = sum(len(samples) for samples in grouped.values())
    total_boxes = sum(len(sample.boxes) for samples in grouped.values() for sample in samples)
    class_hist = {0: total_boxes}

    log("")
    log("[SUMMARY]")
    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log("")
    log("[COUNTS]")
    log(f"RGB images / boxes: {len(rgb_samples)} / {rgb_stats['boxes']}")
    log(f"SAR images / boxes: {len(sar_samples)} / {sar_stats['boxes']}")
    log(f"IR images / boxes:  {len(ir_samples)} / {ir_stats['boxes']}")
    log(f"ALL images / boxes: {total_images} / {total_boxes}")
    log(f"Class histogram: {class_hist}")
    log(f"Output dataset: {dst_root}")
    log(f"YAML: {configs_root / 'total_bridge.yaml'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
