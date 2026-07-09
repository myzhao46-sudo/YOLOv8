from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "ultralytics"
for _path in (str(PACKAGE_ROOT.resolve()), str(PROJECT_ROOT.resolve())):
    while _path in sys.path:
        sys.path.remove(_path)
sys.path.insert(0, str(PACKAGE_ROOT.resolve()))
sys.path.insert(1, str(PROJECT_ROOT.resolve()))

from ultralytics import YOLO  # noqa: E402


IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}
TARGET_CLASS_ID = 0
TARGET_CLASS_NAME = "bridge"
MATCH_IOU = 0.50
ULTRALYTICS_VAL_CONF = 0.001


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML mapping: {path}")
    return data


def list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    images = sorted(p.resolve() for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"No supported images found in: {directory}")
    return images


def list_label_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Label directory does not exist: {directory}")
    return sorted(p.resolve() for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".txt")


def label_for_image(image: Path) -> Path:
    parts = list(image.parts)
    indexes = [index for index, part in enumerate(parts) if part.lower() == "images"]
    if not indexes:
        raise ValueError(f"Cannot derive label path from image without an images component: {image}")
    parts[indexes[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def normalize_split_items(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


def resolve_yaml_fields(data_yaml: Path) -> dict[str, Any]:
    data = load_yaml(data_yaml)
    dataset_path = data.get("path", data_yaml.parent)
    dataset_root = resolve_path(dataset_path) if not Path(str(dataset_path)).is_absolute() else Path(str(dataset_path)).resolve()
    fields: dict[str, Any] = {"dataset_root": dataset_root, "splits": {}}
    for split in ("train", "val", "test"):
        split_items = normalize_split_items(data.get(split))
        resolved = []
        for item in split_items:
            split_path = Path(item)
            resolved.append(split_path.resolve() if split_path.is_absolute() else (dataset_root / split_path).resolve())
        fields["splits"][split] = {
            "raw": data.get(split),
            "resolved": resolved,
            "exists": [path.is_dir() for path in resolved],
        }
    fields["data"] = data
    return fields


def resolve_split_from_data(data_yaml: Path, split: str = "test") -> tuple[list[Path], dict[str, Any], dict[str, Any]]:
    resolved = resolve_yaml_fields(data_yaml)
    image_dirs = resolved["splits"][split]["resolved"]
    if not image_dirs:
        raise KeyError(f"{data_yaml} does not define split '{split}'")
    for image_dir in image_dirs:
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Resolved {split} image directory does not exist: {image_dir}")
    return image_dirs, resolved["data"], resolved


def list_images_from_dirs(image_dirs: list[Path]) -> list[Path]:
    images: list[Path] = []
    seen: set[str] = set()
    for directory in image_dirs:
        for image in list_images(directory):
            key = str(image)
            if key in seen:
                raise RuntimeError(f"Duplicate image in resolved split directories: {image}")
            seen.add(key)
            images.append(image)
    return sorted(images)


def read_yolo_labels(images: list[Path]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    labels_by_image: dict[str, list[dict[str, Any]]] = {}
    image_stems_by_label_dir: dict[Path, set[str]] = {}
    label_dirs: set[Path] = set()
    stats: dict[str, Any] = {
        "images": len(images),
        "label_files": 0,
        "gt_boxes": 0,
        "class_histogram": {"0": 0},
        "class_box_counts": {"0": 0},
        "missing_labels": 0,
        "orphan_labels": [],
        "empty_labels": 0,
        "bad_rows": [],
    }
    for image in images:
        label = label_for_image(image)
        label_dirs.add(label.parent)
        image_stems_by_label_dir.setdefault(label.parent, set()).add(label.stem)
        with Image.open(image) as opened:
            width, height = opened.size
        rows: list[dict[str, Any]] = []
        if not label.is_file():
            stats["missing_labels"] += 1
            labels_by_image[str(image)] = rows
            continue
        stats["label_files"] += 1
        lines = label.read_text(encoding="utf-8").splitlines()
        nonempty = [line.strip() for line in lines if line.strip()]
        if not nonempty:
            stats["empty_labels"] += 1
        for line_number, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            fields = stripped.split()
            try:
                if len(fields) != 5:
                    raise ValueError(f"expected 5 columns, got {len(fields)}")
                class_id = int(float(fields[0]))
                cx, cy, bw, bh = [float(v) for v in fields[1:]]
                if class_id != TARGET_CLASS_ID:
                    raise ValueError(f"expected class 0 bridge, got {class_id}")
                if not all(0.0 <= v <= 1.0 for v in (cx, cy, bw, bh)):
                    raise ValueError("box coordinates are not normalized to [0, 1]")
                if bw <= 0 or bh <= 0:
                    raise ValueError("box width/height must be positive")
            except Exception as exc:
                stats["bad_rows"].append({"label": str(label), "line": line_number, "text": stripped, "error": str(exc)})
                continue
            x1 = (cx - bw / 2.0) * width
            y1 = (cy - bh / 2.0) * height
            x2 = (cx + bw / 2.0) * width
            y2 = (cy + bh / 2.0) * height
            rows.append(
                {
                    "class_id": class_id,
                    "class_name": TARGET_CLASS_NAME,
                    "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "label": str(label),
                }
            )
            stats["gt_boxes"] += 1
            stats["class_histogram"]["0"] += 1
            stats["class_box_counts"]["0"] += 1
        labels_by_image[str(image)] = rows
    for label_dir in sorted(label_dirs):
        image_stems = image_stems_by_label_dir.get(label_dir, set())
        for label_file in list_label_files(label_dir):
            if label_file.stem not in image_stems:
                stats["orphan_labels"].append(str(label_file))
    return labels_by_image, stats


def bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def collect_predictions(
    model: YOLO,
    images: list[Path],
    imgsz: int,
    device: str,
    conf: float,
    iou: float,
    batch: int,
) -> dict[str, list[dict[str, Any]]]:
    predictions: dict[str, list[dict[str, Any]]] = {}
    for start in range(0, len(images), batch):
        chunk = images[start : start + batch]
        results = model.predict(
            source=[str(p) for p in chunk],
            imgsz=imgsz,
            device=device,
            conf=conf,
            iou=iou,
            verbose=False,
            save=False,
            stream=False,
        )
        for image, result in zip(chunk, results):
            rows: list[dict[str, Any]] = []
            boxes = getattr(result, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.detach().cpu().numpy()
                confs = boxes.conf.detach().cpu().numpy()
                classes = boxes.cls.detach().cpu().numpy().astype(int)
                for box, score, cls in zip(xyxy, confs, classes):
                    if int(cls) != TARGET_CLASS_ID:
                        continue
                    rows.append(
                        {
                            "class_id": TARGET_CLASS_ID,
                            "class_name": TARGET_CLASS_NAME,
                            "score": float(score),
                            "xyxy": [float(v) for v in box.tolist()],
                        }
                    )
            rows.sort(key=lambda item: item["score"], reverse=True)
            predictions[str(image.resolve())] = rows
    return predictions


def match_fixed_conf(
    ground_truth: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    tp = fp = fn = 0
    per_image: dict[str, dict[str, Any]] = {}
    pred_counts = []
    for image, gt_rows in ground_truth.items():
        pred_rows = sorted(predictions.get(image, []), key=lambda item: item["score"], reverse=True)
        pred_counts.append(len(pred_rows))
        matched_gt: set[int] = set()
        pred_matches: list[dict[str, Any]] = []
        image_tp = image_fp = 0
        best_unmatched_iou = 0.0
        for pred_index, pred in enumerate(pred_rows):
            best_gt = -1
            best_iou = 0.0
            for gt_index, gt in enumerate(gt_rows):
                if gt_index in matched_gt:
                    continue
                iou_value = bbox_iou(pred["xyxy"], gt["xyxy"])
                if iou_value > best_iou:
                    best_iou = iou_value
                    best_gt = gt_index
            best_unmatched_iou = max(best_unmatched_iou, best_iou)
            if best_gt >= 0 and best_iou >= MATCH_IOU:
                matched_gt.add(best_gt)
                tp += 1
                image_tp += 1
                pred_matches.append({"prediction_index": pred_index, "matched": True, "iou": best_iou})
            else:
                fp += 1
                image_fp += 1
                pred_matches.append({"prediction_index": pred_index, "matched": False, "iou": best_iou})
        image_fn = len(gt_rows) - len(matched_gt)
        fn += image_fn
        per_image[image] = {
            "tp": image_tp,
            "fp": image_fp,
            "fn": image_fn,
            "gt": len(gt_rows),
            "pred": len(pred_rows),
            "best_unmatched_iou": best_unmatched_iou,
            "prediction_matches": pred_matches,
        }
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {
        "match_iou": MATCH_IOU,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predictions": int(sum(pred_counts)),
        "average_predictions_per_image": float(np.mean(pred_counts)) if pred_counts else 0.0,
    }
    return metrics, per_image


def draw_boxes(
    image_path: Path,
    gt_rows: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
    out_path: Path,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    def draw_label(box: list[float], text: str, color: tuple[int, int, int]) -> None:
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        y_text = max(0.0, y1 - th - 4)
        draw.rectangle([x1, y_text, x1 + tw + 4, y_text + th + 4], fill=color)
        draw.text((x1 + 2, y_text + 2), text, fill=(255, 255, 255), font=font)

    for gt in gt_rows:
        draw_label(gt["xyxy"], "GT bridge", (0, 170, 0))
    for pred in pred_rows:
        draw_label(pred["xyxy"], f"Pred bridge conf={pred['score']:.2f}", (220, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def save_gt_previews(
    images: list[Path],
    ground_truth: dict[str, list[dict[str, Any]]],
    out_dir: Path,
    max_count: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    chosen = sorted(images, key=lambda p: (-len(ground_truth[str(p.resolve())]), p.name))[:max_count]
    for image in chosen:
        key = str(image.resolve())
        draw_boxes(image, ground_truth[key], [], out_dir / image.name)
    return {"dir": str(out_dir), "saved": len(chosen)}


def save_predictions_jsonl(path: Path, images: list[Path], predictions: dict[str, list[dict[str, Any]]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for image in images:
            with Image.open(image) as opened:
                width, height = opened.size
            record = {
                "image": str(image),
                "width": width,
                "height": height,
                "predictions": predictions.get(str(image.resolve()), []),
            }
            handle.write(json.dumps(to_jsonable(record), ensure_ascii=False) + "\n")


def save_visuals(
    images: list[Path],
    ground_truth: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
    per_image: dict[str, dict[str, Any]],
    vis_dir: Path,
    max_vis: int,
) -> dict[str, Any]:
    if max_vis <= 0:
        return {"saved": 0}
    rng = random.Random(42)
    ordered = sorted(
        images,
        key=lambda p: (
            per_image[str(p.resolve())]["fp"] + per_image[str(p.resolve())]["fn"],
            max([pred["score"] for pred in predictions.get(str(p.resolve()), [])] or [0.0]),
        ),
        reverse=True,
    )
    if len(ordered) > max_vis:
        error_heavy = ordered[: max_vis // 2]
        rest = ordered[max_vis // 2 :]
        rng.shuffle(rest)
        chosen = error_heavy + rest[: max_vis - len(error_heavy)]
    else:
        chosen = ordered
    for image in chosen[:max_vis]:
        key = str(image.resolve())
        draw_boxes(image, ground_truth[key], predictions.get(key, []), vis_dir / image.name)
    return {"saved": min(len(chosen), max_vis), "dir": str(vis_dir)}


def save_error_examples(
    images: list[Path],
    ground_truth: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
    per_image: dict[str, dict[str, Any]],
    out_root: Path,
    max_each: int,
) -> dict[str, Any]:
    examples: dict[str, list[Path]] = {
        "false_positive_examples": [],
        "false_negative_examples": [],
        "low_iou_examples": [],
    }
    for image in images:
        key = str(image.resolve())
        stat = per_image[key]
        if stat["fp"] > 0:
            examples["false_positive_examples"].append(image)
        if stat["fn"] > 0:
            examples["false_negative_examples"].append(image)
        if stat["fp"] > 0 and stat["fn"] > 0 and 0.0 < stat["best_unmatched_iou"] < MATCH_IOU:
            examples["low_iou_examples"].append(image)

    saved: dict[str, Any] = {}
    for name, candidates in examples.items():
        target = mkdir(out_root / name)
        ranked = sorted(
            candidates,
            key=lambda p: (
                per_image[str(p.resolve())]["fp"] + per_image[str(p.resolve())]["fn"],
                per_image[str(p.resolve())]["best_unmatched_iou"],
            ),
            reverse=True,
        )[:max_each]
        for image in ranked:
            key = str(image.resolve())
            draw_boxes(image, ground_truth[key], predictions.get(key, []), target / image.name)
        saved[name] = {"dir": str(target), "saved": len(ranked)}
    return saved


def extract_ultralytics_metrics(metrics: Any) -> dict[str, Any]:
    box = getattr(metrics, "box", None)
    results_dict = getattr(metrics, "results_dict", {})
    output = {
        "results_dict": dict(results_dict) if isinstance(results_dict, dict) else results_dict,
        "precision": float(getattr(box, "mp", 0.0)) if box is not None else None,
        "recall": float(getattr(box, "mr", 0.0)) if box is not None else None,
        "ap50": float(getattr(box, "map50", 0.0)) if box is not None else None,
        "map50_95": float(getattr(box, "map", 0.0)) if box is not None else None,
        "maps": to_jsonable(getattr(box, "maps", None)) if box is not None else None,
    }
    return output


def write_report(path: Path, summary: dict[str, Any]) -> None:
    fixed = summary["fixed_conf_metrics"]
    official = summary["ultralytics_val"]
    lines = [
        "# Bridge expert external NWPU evaluation",
        "",
        "No training was performed. Model weights and source dataset were not modified.",
        "",
        "## Inputs",
        "",
        f"- Dataset: `{summary['data']}`",
        f"- Weights: `{summary['weights']}`",
        f"- Split: `test`",
        f"- Images: {summary['dataset_stats']['images']}",
        f"- GT boxes: {summary['dataset_stats']['gt_boxes']}",
        f"- Test image dirs: `{summary['test_image_dirs']}`",
        f"- Missing labels: {summary['dataset_stats']['missing_labels']}",
        f"- Orphan labels: {len(summary['dataset_stats']['orphan_labels'])}",
        f"- Empty labels: {summary['dataset_stats']['empty_labels']}",
        f"- Bad label rows: {len(summary['dataset_stats']['bad_rows'])}",
        "",
        "## Ultralytics official val metrics",
        "",
        "- Source: `model.val(data=..., split=\"test\", imgsz=1024, device=0)`.",
        f"- val conf used for AP: {summary['ultralytics_val_conf']}",
        f"- IoU/NMS: {summary['iou']}",
        f"- Precision: {official['precision']:.6f}",
        f"- Recall: {official['recall']:.6f}",
        f"- AP50: {official['ap50']:.6f}",
        f"- mAP50-95: {official['map50_95']:.6f}",
        "",
        f"Historical total_bridge_test RGB AP50 reference: {summary['historical_total_bridge_rgb_ap50']:.4f}",
        "",
        "## Fixed conf detection statistics",
        "",
        "- Source: script-run `model.predict()` followed by IoU=0.5 same-class greedy matching.",
        "- These fixed-threshold P/R values are intentionally separate from Ultralytics val P/R.",
        "- AP50 and mAP50-95 should be read from Ultralytics val; fixed TP/FP/FN explains false positives and misses.",
        f"- fixed conf: {summary['conf']}",
        f"- match IoU: {fixed['match_iou']}",
        f"- TP: {fixed['tp']}",
        f"- FP: {fixed['fp']}",
        f"- FN: {fixed['fn']}",
        f"- Precision: {fixed['precision']:.6f}",
        f"- Recall: {fixed['recall']:.6f}",
        f"- F1: {fixed['f1']:.6f}",
        f"- Predictions: {fixed['predictions']}",
        f"- Average predictions/image: {fixed['average_predictions_per_image']:.6f}",
        "",
        "## Outputs",
        "",
        f"- Predictions: `{summary['predictions_jsonl']}`",
        f"- GT preview: `{summary['gt_preview']['dir']}`",
        f"- Visualizations: `{summary['vis_dir']}`",
        f"- Error examples: `{summary['error_examples_dir']}`",
        "",
        "## Interpretation guide",
        "",
        "- If NWPU AP50 is close to or higher than 0.659, bridge expert has some RGB cross-source generalization.",
        "- If NWPU AP50 is far below 0.659, the expert is distribution-dependent and external RGB generalization is weak.",
        "- Low precision with acceptable recall means external false positives are severe and calibration/gating is needed.",
        "- Low recall means external bridge appearance differs enough to cause missed detections.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate bridge-only YOLOv8 expert on external NWPU bridge test set.")
    parser.add_argument("--weights", required=True, help="Path to bridge-only YOLOv8 weights.")
    parser.add_argument("--data", required=True, help="Path to bridge-only data.yaml with test split.")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25, help="Fixed confidence threshold for predict statistics.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU for val/predict.")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", default=str(PROJECT_ROOT / "runs" / "eval"))
    parser.add_argument("--name", default="bridge_external_nwpu_cls1_as_bridge")
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--save-md", action="store_true")
    parser.add_argument("--save-preds", action="store_true")
    parser.add_argument("--max-vis", type=int, default=50)
    parser.add_argument("--gt-preview-dir", default=None, help="Optional directory for GT-only preview images.")
    parser.add_argument("--max-gt-preview", type=int, default=30)
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = resolve_path(args.weights)
    data_yaml = resolve_path(args.data)
    project = resolve_path(args.project)
    run_dir = project / args.name
    if run_dir.exists() and not args.exist_ok:
        # Do not delete old evaluations; reuse directory and overwrite known summary files only.
        print(f"[WARN] Output directory exists, known output files may be overwritten: {run_dir}", flush=True)
    mkdir(run_dir)
    predictions_dir = mkdir(run_dir / "predictions")
    vis_dir = mkdir(run_dir / "vis")
    error_dir = mkdir(run_dir / "errors")

    if not weights.is_file():
        raise FileNotFoundError(f"Missing weights: {weights}")
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Missing data.yaml: {data_yaml}")
    image_dirs, data, resolved_yaml = resolve_split_from_data(data_yaml, "test")
    images = list_images_from_dirs(image_dirs)
    ground_truth, dataset_stats = read_yolo_labels(images)
    if dataset_stats["missing_labels"] or dataset_stats["bad_rows"] or dataset_stats["orphan_labels"]:
        raise RuntimeError(f"Dataset preflight failed: {dataset_stats}")
    if int(data.get("nc", -1)) != 1:
        raise RuntimeError(f"Expected nc=1 bridge-only data.yaml, got nc={data.get('nc')}")
    names = data.get("names")
    if isinstance(names, list):
        name0 = names[0] if names else None
    elif isinstance(names, dict):
        name0 = names.get(0, names.get("0"))
    else:
        name0 = None
    if str(name0).lower() != TARGET_CLASS_NAME:
        raise RuntimeError(f"Expected names[0]=bridge, got names={names!r}")

    print(f"[INFO] Dataset images={dataset_stats['images']} gt_boxes={dataset_stats['gt_boxes']}", flush=True)
    print(f"[INFO] Test image dirs={[str(p) for p in image_dirs]}", flush=True)
    print(f"[INFO] Loading YOLOv8 bridge expert: {weights}", flush=True)
    model = YOLO(str(weights))
    model_model = getattr(model, "model", None)
    model_task = getattr(model, "task", None)
    model_name = type(model_model).__name__ if model_model is not None else None
    head = getattr(model_model, "model", [None])[-1] if model_model is not None and hasattr(model_model, "model") else None
    head_name = type(head).__name__ if head is not None else None
    head_nc = getattr(head, "nc", None)
    model_names = getattr(model_model, "names", getattr(model, "names", None))
    if model_task != "detect" or model_name != "DetectionModel" or head_name != "Detect" or int(head_nc) != 1:
        raise RuntimeError(
            f"Expected ordinary YOLOv8 DetectionModel+Detect nc=1, got task={model_task}, "
            f"model={model_name}, head={head_name}, nc={head_nc}"
        )
    print(f"[INFO] Model OK: task={model_task}, model={model_name}, head={head_name}, nc={head_nc}, names={model_names}", flush=True)

    print("[INFO] Running Ultralytics official val on split=test (AP conf=0.001)...", flush=True)
    val_metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=ULTRALYTICS_VAL_CONF,
        iou=args.iou,
        project=str(run_dir),
        name="ultralytics_val",
        exist_ok=True,
        verbose=True,
        plots=True,
    )
    official = extract_ultralytics_metrics(val_metrics)

    print(f"[INFO] Running fixed-conf predict: conf={args.conf}, iou={args.iou}", flush=True)
    predictions = collect_predictions(model, images, args.imgsz, args.device, args.conf, args.iou, args.batch)
    fixed_metrics, per_image = match_fixed_conf(ground_truth, predictions)

    predictions_jsonl = predictions_dir / "predictions_conf025.jsonl"
    if args.save_preds:
        save_predictions_jsonl(predictions_jsonl, images, predictions)
    gt_preview_dir = resolve_path(args.gt_preview_dir) if args.gt_preview_dir else (run_dir / "gt_preview")
    gt_preview_summary = save_gt_previews(images, ground_truth, gt_preview_dir, args.max_gt_preview)
    visual_summary = save_visuals(images, ground_truth, predictions, per_image, vis_dir, args.max_vis)
    error_summary = save_error_examples(images, ground_truth, predictions, per_image, error_dir, min(args.max_vis, 50))

    score_bins = Counter()
    for rows in predictions.values():
        for pred in rows:
            score = pred["score"]
            if score >= 0.75:
                score_bins[">=0.75"] += 1
            elif score >= 0.50:
                score_bins["0.50-0.75"] += 1
            elif score >= 0.25:
                score_bins["0.25-0.50"] += 1
            else:
                score_bins["<0.25"] += 1

    summary = {
        "task": "bridge-only YOLOv8 expert external NWPU bridge evaluation",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "weights": str(weights),
        "data": str(data_yaml),
        "image_dir": str(image_dirs[0]) if len(image_dirs) == 1 else None,
        "test_image_dirs": [str(path) for path in image_dirs],
        "split": "test",
        "imgsz": args.imgsz,
        "device": args.device,
        "conf": args.conf,
        "iou": args.iou,
        "batch": args.batch,
        "ultralytics_val_conf": ULTRALYTICS_VAL_CONF,
        "historical_total_bridge_rgb_ap50": 0.6590,
        "model_info": {
            "task": model_task,
            "model_type": model_name,
            "head_type": head_name,
            "head_nc": head_nc,
            "names": model_names,
        },
        "dataset_yaml": data,
        "resolved_yaml_fields": {
            "dataset_root": str(resolved_yaml["dataset_root"]),
            "splits": {
                split: {
                    "raw": value["raw"],
                    "resolved": [str(path) for path in value["resolved"]],
                    "exists": value["exists"],
                }
                for split, value in resolved_yaml["splits"].items()
            },
        },
        "dataset_stats": dataset_stats,
        "ultralytics_val": official,
        "fixed_conf_metrics": fixed_metrics,
        "prediction_score_bins": dict(score_bins),
        "predictions_jsonl": str(predictions_jsonl) if args.save_preds else None,
        "gt_preview": gt_preview_summary,
        "vis_dir": str(vis_dir),
        "visualizations": visual_summary,
        "error_examples_dir": str(error_dir),
        "error_examples": error_summary,
        "notes": [
            "No training was performed.",
            "Model weights were not modified.",
            "NWPU external data was evaluated only and was not mixed into training.",
            "Ultralytics val AP uses conf=0.001; fixed-conf statistics use the user-provided --conf threshold.",
        ],
    }

    metrics_json = run_dir / "metrics.json"
    report_md = run_dir / "report.md"
    if args.save_json:
        metrics_json.write_text(json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    if args.save_md:
        write_report(report_md, summary)
    print(json.dumps(to_jsonable({
        "images": dataset_stats["images"],
        "gt_boxes": dataset_stats["gt_boxes"],
        "ultralytics_ap50": official["ap50"],
        "ultralytics_map50_95": official["map50_95"],
        "fixed_conf": fixed_metrics,
        "metrics_json": str(metrics_json),
        "report_md": str(report_md),
    }), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
