from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "ultralytics"
for _path in (str(PACKAGE_ROOT.resolve()), str(PROJECT_ROOT.resolve())):
    while _path in sys.path:
        sys.path.remove(_path)
sys.path.insert(0, str(PACKAGE_ROOT.resolve()))
sys.path.insert(1, str(PROJECT_ROOT.resolve()))

import torch


CLASS_NAMES = {0: "ship", 1: "bridge"}
IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}
IOU_THRESHOLDS = [round(value, 2) for value in np.arange(0.50, 0.96, 0.05)]
DEFAULT_SHIP_WEIGHTS = "ultralytics/best.pt"
DEFAULT_BRIDGE_WEIGHTS = "runs/bridge_expert/yolov8m_total_bridge_split_img1024_ep150/weights/best.pt"
DEFAULT_DATASET = "ultralytics/datasets/bridge_ship_distill_2cls_v2"
DEFAULT_PROJECT = "runs/late_fusion"
DEFAULT_NAME = "yoloe_ship_yolov8m_bridge_v1"


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def git_info() -> dict[str, str | None]:
    result: dict[str, str | None] = {"branch": None, "commit": None}
    for key, command in (
        ("branch", ["git", "branch", "--show-current"]),
        ("commit", ["git", "rev-parse", "HEAD"]),
    ):
        try:
            result[key] = subprocess.check_output(
                command, cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return result


def qualified_name(obj: object) -> str:
    cls = type(obj)
    return f"{cls.__module__}.{cls.__name__}"


def normalize_names(value: Any) -> dict[int, str]:
    if isinstance(value, (list, tuple)):
        return {index: str(name) for index, name in enumerate(value)}
    if isinstance(value, dict):
        return {int(key): str(name) for key, name in value.items()}
    raise TypeError(f"Unsupported model names value: {value!r}")


def parse_device(value: str) -> torch.device:
    text = str(value)
    if text.isdigit():
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {text} requested but CUDA is unavailable")
        return torch.device(f"cuda:{text}")
    return torch.device(text)


def list_images(path: Path) -> list[Path]:
    if not path.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {path}")
    images = sorted(item.resolve() for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"No supported images found in: {path}")
    return images


def label_for_image(image: Path) -> Path:
    parts = list(image.parts)
    indexes = [index for index, part in enumerate(parts) if part.lower() == "images"]
    if not indexes:
        raise ValueError(f"Cannot derive label path from image without an images component: {image}")
    parts[indexes[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def basename_any(value: str) -> str:
    return value.replace("\\", "/").rsplit("/", 1)[-1]


def load_manifest_modalities(dataset_root: Path) -> dict[str, str]:
    path = dataset_root / "manifest.csv"
    mapping = {}
    if not path.is_file():
        return mapping
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") == "test_bridge" and row.get("target_image"):
                mapping[basename_any(row["target_image"])] = row.get("modality", "unknown").lower()
    return mapping


def infer_bridge_modality(image: Path, manifest_modalities: dict[str, str]) -> str:
    manifest_value = manifest_modalities.get(image.name)
    if manifest_value in {"rgb", "sar", "ir"}:
        return manifest_value
    name = image.name.lower()
    for modality in ("rgb", "sar", "ir"):
        if name.startswith(f"bridge_test_{modality}_"):
            return modality
    return "unknown"


def read_ground_truth(
    images: list[Path], source_split: str, expected_class: int, manifest_modalities: dict[str, str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    stats = {
        "images": len(images),
        "label_files": 0,
        "boxes": 0,
        "class_histogram": {"0": 0, "1": 0},
        "missing_labels": 0,
        "bad_rows": 0,
        "empty_labels": 0,
        "examples": [],
    }
    for image in images:
        label = label_for_image(image)
        with Image.open(image) as opened:
            width, height = opened.size
        boxes = []
        if not label.is_file():
            stats["missing_labels"] += 1
            stats["examples"].append(f"missing label: {label}")
        else:
            stats["label_files"] += 1
            for line_number, raw in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    fields = line.split()
                    if len(fields) != 5:
                        raise ValueError(f"expected 5 columns, got {len(fields)}")
                    class_value = float(fields[0])
                    class_id = int(class_value)
                    coordinates = [float(value) for value in fields[1:]]
                    if class_value != class_id or class_id != expected_class:
                        raise ValueError(f"expected only class {expected_class}, got {fields[0]!r}")
                    if any(value < 0.0 or value > 1.0 for value in coordinates):
                        raise ValueError("coordinates outside [0,1]")
                    if coordinates[2] <= 0.0 or coordinates[3] <= 0.0:
                        raise ValueError("width/height must be positive")
                    cx, cy, box_width, box_height = coordinates
                    x1 = max(0.0, (cx - box_width / 2.0) * width)
                    y1 = max(0.0, (cy - box_height / 2.0) * height)
                    x2 = min(float(width), (cx + box_width / 2.0) * width)
                    y2 = min(float(height), (cy + box_height / 2.0) * height)
                    boxes.append({"class_id": class_id, "class_name": CLASS_NAMES[class_id], "xyxy": [x1, y1, x2, y2]})
                    stats["boxes"] += 1
                    stats["class_histogram"][str(class_id)] += 1
                except Exception as exc:
                    stats["bad_rows"] += 1
                    if len(stats["examples"]) < 20:
                        stats["examples"].append(f"{label}:{line_number}: {line} | {exc}")
        if not boxes:
            stats["empty_labels"] += 1
        modality = (
            infer_bridge_modality(image, manifest_modalities) if source_split == "test_bridge" else "ship"
        )
        records.append(
            {
                "image": str(image),
                "image_key": str(image.resolve()),
                "width": width,
                "height": height,
                "source_split": source_split,
                "modality": modality,
                "label": str(label),
                "ground_truth": boxes,
            }
        )
    return records, stats


def preflight_dataset(dataset_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    manifest_modalities = load_manifest_modalities(dataset_root)
    ship_images = list_images(dataset_root / "images" / "test_ship")
    bridge_images = list_images(dataset_root / "images" / "test_bridge")
    ship_records, ship_stats = read_ground_truth(ship_images, "test_ship", 0, manifest_modalities)
    bridge_records, bridge_stats = read_ground_truth(bridge_images, "test_bridge", 1, manifest_modalities)
    for split, stats in (("test_ship", ship_stats), ("test_bridge", bridge_stats)):
        if stats["missing_labels"] or stats["bad_rows"]:
            raise ValueError(f"{split} label preflight failed: {stats}")
    if ship_stats["class_histogram"] != {"0": 694, "1": 0}:
        raise ValueError(f"Unexpected test_ship class histogram: {ship_stats['class_histogram']}")
    if bridge_stats["class_histogram"] != {"0": 0, "1": 798}:
        raise ValueError(f"Unexpected test_bridge class histogram: {bridge_stats['class_histogram']}")
    return ship_records + bridge_records, {
        "dataset_root": str(dataset_root),
        "test_ship": ship_stats,
        "test_bridge": bridge_stats,
        "manifest_found": (dataset_root / "manifest.csv").is_file(),
        "bridge_modality_histogram": dict(Counter(record["modality"] for record in bridge_records)),
    }


def prepare_ship_expert(weights: Path):
    from ultralytics import YOLO
    from ultralytics.nn.modules.head import YOLOESegment
    from ultralytics.nn.tasks import YOLOESegModel

    if not weights.is_file():
        raise FileNotFoundError(f"Ship expert weights do not exist: {weights}")
    wrapper = YOLO(str(weights))
    inner = wrapper.model
    if type(inner) is not YOLOESegModel or type(inner.model[-1]) is not YOLOESegment:
        raise TypeError(
            f"Ship expert must be YOLOESegModel + YOLOESegment, got "
            f"{qualified_name(inner)} + {qualified_name(inner.model[-1])}"
        )
    names = ["ship", "harbor", "tank"]
    get_text_pe = getattr(wrapper, "get_text_pe", None) or getattr(inner, "get_text_pe", None)
    set_classes = getattr(wrapper, "set_classes", None) or getattr(inner, "set_classes", None)
    if not callable(get_text_pe) or not callable(set_classes):
        raise AttributeError("Ship YOLOE expert must provide get_text_pe and set_classes")
    embeddings = get_text_pe(names)
    set_classes(names, embeddings)
    for parameter in inner.parameters():
        parameter.requires_grad_(False)
    return wrapper, inner, {
        "weights": str(weights),
        "wrapper_class": qualified_name(wrapper),
        "model_class": qualified_name(inner),
        "head_class": qualified_name(inner.model[-1]),
        "task": wrapper.task,
        "names": names,
        "text_embedding_shape": list(embeddings.shape),
        "set_classes_used_embeddings": True,
        "frozen": all(not parameter.requires_grad for parameter in inner.parameters()),
        "native_model_val_called": False,
        "output_mapping": "YOLOE class 0 ship -> final class 0 ship",
    }


def prepare_bridge_expert(weights: Path):
    from ultralytics import YOLO
    from ultralytics.nn.modules.head import Detect
    from ultralytics.nn.tasks import DetectionModel

    if not weights.is_file():
        raise FileNotFoundError(f"Bridge expert weights do not exist: {weights}")
    wrapper = YOLO(str(weights), verbose=False)
    inner = wrapper.model
    if type(inner) is not DetectionModel or type(inner.model[-1]) is not Detect:
        raise TypeError(
            f"Bridge expert must be ordinary DetectionModel + Detect, got "
            f"{qualified_name(inner)} + {qualified_name(inner.model[-1])}"
        )
    head = inner.model[-1]
    names = normalize_names(inner.names)
    if wrapper.task != "detect" or int(head.nc) != 1 or names != {0: "bridge"}:
        raise ValueError(
            f"Bridge expert must be detect nc=1 names={{0: 'bridge'}}, got "
            f"task={wrapper.task}, nc={head.nc}, names={names}"
        )
    for parameter in inner.parameters():
        parameter.requires_grad_(False)
    return wrapper, inner, {
        "weights": str(weights),
        "wrapper_class": qualified_name(wrapper),
        "model_class": qualified_name(inner),
        "head_class": qualified_name(head),
        "task": wrapper.task,
        "nc": int(head.nc),
        "names": names,
        "frozen": all(not parameter.requires_grad for parameter in inner.parameters()),
        "output_mapping": "bridge expert class 0 bridge -> final class 1 bridge",
    }


def infer_ship(
    inner: object, records: list[dict[str, Any]], args: argparse.Namespace, device: torch.device
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    from tools.eval.eval_teacher_ship_external import collect_predictions

    image_paths = [Path(record["image"]) for record in records]
    helper_args = SimpleNamespace(
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        conf=args.ship_conf,
        pred_iou=args.nms_iou,
        max_det=args.max_det,
        target_class=0,
    )
    grouped: dict[str, list[dict[str, Any]]] = {record["image_key"]: [] for record in records}
    score_channels = set()
    total_all_class_boxes = 0
    progress_chunk = max(args.batch, 64)
    started = time.time()
    for start in range(0, len(image_paths), progress_chunk):
        chunk = image_paths[start : start + progress_chunk]
        predictions, channels, all_boxes = collect_predictions(inner, chunk, helper_args)
        score_channels.update(channels)
        total_all_class_boxes += all_boxes
        for prediction in predictions:
            grouped[prediction["image_key"]].append(
                {
                    "class_id": 0,
                    "class_name": "ship",
                    "score": float(prediction["conf"]),
                    "xyxy": [float(value) for value in prediction["xyxy"]],
                    "source_expert": "ship_yoloe",
                }
            )
        print(f"[ship expert] {min(start + len(chunk), len(image_paths))}/{len(image_paths)} images", flush=True)
    if score_channels != {3}:
        raise RuntimeError(f"Ship expert must emit three fused score channels, got {sorted(score_channels)}")
    return grouped, {
        "images": len(image_paths),
        "ship_predictions": sum(len(predictions) for predictions in grouped.values()),
        "all_class_post_nms_boxes": total_all_class_boxes,
        "score_channels_seen": sorted(score_channels),
        "seconds": time.time() - started,
    }


def infer_bridge(
    wrapper: object, records: list[dict[str, Any]], args: argparse.Namespace
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    image_paths = [Path(record["image"]) for record in records]
    grouped: dict[str, list[dict[str, Any]]] = {record["image_key"]: [] for record in records}
    started = time.time()
    for start in range(0, len(image_paths), args.batch):
        chunk = image_paths[start : start + args.batch]
        results = wrapper.predict(
            source=[str(path) for path in chunk],
            imgsz=args.imgsz,
            conf=args.bridge_conf,
            iou=args.nms_iou,
            max_det=args.max_det,
            classes=[0],
            device=args.device,
            batch=len(chunk),
            verbose=False,
            save=False,
        )
        if len(results) != len(chunk):
            raise RuntimeError(f"Bridge predict returned {len(results)} results for {len(chunk)} images")
        for image_path, result in zip(chunk, results):
            image_key = str(image_path.resolve())
            if result.boxes is None:
                continue
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            scores = result.boxes.conf.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            for box, score, class_id in zip(xyxy, scores, classes):
                if class_id != 0:
                    raise RuntimeError(f"Bridge expert emitted unexpected class {class_id}")
                grouped[image_key].append(
                    {
                        "class_id": 1,
                        "class_name": "bridge",
                        "score": float(score),
                        "xyxy": [float(value) for value in box.tolist()],
                        "source_expert": "bridge_yolov8m",
                    }
                )
        if start == 0 or start + len(chunk) == len(image_paths) or (start // args.batch + 1) % 8 == 0:
            print(f"[bridge expert] {min(start + len(chunk), len(image_paths))}/{len(image_paths)} images", flush=True)
    return grouped, {
        "images": len(image_paths),
        "bridge_predictions": sum(len(predictions) for predictions in grouped.values()),
        "seconds": time.time() - started,
    }


def box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float64)
    top_left = np.maximum(box[:2], boxes[:, :2])
    bottom_right = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.maximum(0.0, bottom_right - top_left)
    intersection_area = intersection[:, 0] * intersection[:, 1]
    box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return intersection_area / np.maximum(box_area + boxes_area - intersection_area, 1e-12)


def nms_predictions(predictions: list[dict[str, Any]], iou_threshold: float, mode: str, max_det: int) -> list[dict[str, Any]]:
    if not predictions:
        return []
    groups: list[list[dict[str, Any]]]
    if mode == "class-aware":
        groups = [
            [prediction for prediction in predictions if prediction["class_id"] == class_id]
            for class_id in CLASS_NAMES
        ]
    elif mode == "class-agnostic":
        groups = [predictions]
    else:
        raise ValueError(f"Unsupported fusion NMS mode: {mode}")
    kept = []
    for group in groups:
        pending = sorted(group, key=lambda item: item["score"], reverse=True)
        kept_in_group = []
        while pending:
            current = pending.pop(0)
            kept_in_group.append(current)
            if len(kept_in_group) >= max_det:
                break
            if not pending:
                continue
            other_boxes = np.asarray([item["xyxy"] for item in pending], dtype=np.float64)
            ious = box_iou_one_to_many(np.asarray(current["xyxy"], dtype=np.float64), other_boxes)
            pending = [item for item, iou in zip(pending, ious) if iou <= iou_threshold]
        kept.extend(kept_in_group)
    ordered = sorted(kept, key=lambda item: item["score"], reverse=True)
    # Class-aware fusion must not let one class consume the other class's detection budget.
    # Each expert/class is capped independently above. Class-agnostic fusion has one global group.
    return ordered if mode == "class-aware" else ordered[:max_det]


def fuse_predictions(
    records: list[dict[str, Any]],
    ship_predictions: dict[str, list[dict[str, Any]]],
    bridge_predictions: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
) -> dict[str, list[dict[str, Any]]]:
    fused = {}
    for record in records:
        key = record["image_key"]
        combined = ship_predictions[key] + bridge_predictions[key]
        fused[key] = nms_predictions(combined, args.nms_iou, args.fusion_nms, args.max_det)
    return fused


def match_class(
    records: list[dict[str, Any]],
    predictions: dict[str, list[dict[str, Any]]],
    class_id: int,
    iou_threshold: float,
    minimum_score: float | None = None,
) -> dict[str, Any]:
    ground_truth = {}
    total_gt = 0
    selected_keys = {record["image_key"] for record in records}
    for record in records:
        boxes = [item["xyxy"] for item in record["ground_truth"] if item["class_id"] == class_id]
        ground_truth[record["image_key"]] = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        total_gt += len(boxes)
    flat_predictions = []
    for image_key in selected_keys:
        for prediction in predictions[image_key]:
            if prediction["class_id"] == class_id and (
                minimum_score is None or prediction["score"] >= minimum_score
            ):
                flat_predictions.append((image_key, prediction))
    flat_predictions.sort(key=lambda item: item[1]["score"], reverse=True)
    matched = {image_key: set() for image_key in selected_keys}
    true_positive = np.zeros(len(flat_predictions), dtype=np.float64)
    false_positive = np.zeros(len(flat_predictions), dtype=np.float64)
    scores = np.zeros(len(flat_predictions), dtype=np.float64)
    for index, (image_key, prediction) in enumerate(flat_predictions):
        scores[index] = prediction["score"]
        gt_boxes = ground_truth[image_key]
        if gt_boxes.size == 0:
            false_positive[index] = 1.0
            continue
        ious = box_iou_one_to_many(np.asarray(prediction["xyxy"], dtype=np.float64), gt_boxes)
        for gt_index in np.argsort(-ious):
            if ious[gt_index] < iou_threshold:
                break
            if int(gt_index) not in matched[image_key]:
                matched[image_key].add(int(gt_index))
                true_positive[index] = 1.0
                break
        if true_positive[index] == 0.0:
            false_positive[index] = 1.0
    tp = int(true_positive.sum())
    fp = int(false_positive.sum())
    return {
        "total_gt": total_gt,
        "prediction_count": len(flat_predictions),
        "tp": tp,
        "fp": fp,
        "fn": total_gt - tp,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "scores": scores,
    }


def compute_ap(total_gt: int, true_positive: np.ndarray, false_positive: np.ndarray) -> float | None:
    if total_gt == 0:
        return None
    if true_positive.size == 0:
        return 0.0
    tp_cumulative = np.cumsum(true_positive)
    fp_cumulative = np.cumsum(false_positive)
    recall = tp_cumulative / max(total_gt, 1)
    precision = tp_cumulative / np.maximum(tp_cumulative + fp_cumulative, 1e-12)
    modified_recall = np.concatenate(([0.0], recall, [1.0]))
    modified_precision = np.concatenate(([1.0], precision, [0.0]))
    modified_precision = np.flip(np.maximum.accumulate(np.flip(modified_precision)))
    x = np.linspace(0.0, 1.0, 101)
    return float(np.trapz(np.interp(x, modified_recall, modified_precision), x))


def evaluate_class(
    records: list[dict[str, Any]], predictions: dict[str, list[dict[str, Any]]], class_id: int, report_conf: float
) -> dict[str, Any]:
    ap_by_iou = {}
    for threshold in IOU_THRESHOLDS:
        match = match_class(records, predictions, class_id, threshold)
        ap_by_iou[f"{threshold:.2f}"] = compute_ap(
            match["total_gt"], match["true_positive"], match["false_positive"]
        )
    report = match_class(records, predictions, class_id, 0.50, minimum_score=report_conf)
    precision = report["tp"] / max(report["tp"] + report["fp"], 1)
    recall = report["tp"] / max(report["total_gt"], 1) if report["total_gt"] else None
    valid_aps = [value for value in ap_by_iou.values() if value is not None]
    return {
        "class_id": class_id,
        "class_name": CLASS_NAMES[class_id],
        "gt": report["total_gt"],
        "predictions_all": match_class(records, predictions, class_id, 0.50)["prediction_count"],
        "ap50": ap_by_iou["0.50"],
        "map50_95": float(np.mean(valid_aps)) if valid_aps else None,
        "ap_by_iou": ap_by_iou,
        "report_conf": report_conf,
        "predictions_at_report_conf": report["prediction_count"],
        "tp": report["tp"],
        "fp": report["fp"],
        "fn": report["fn"],
        "precision": precision,
        "recall": recall,
    }


def false_positive_diagnostic(
    records: list[dict[str, Any]], predictions: dict[str, list[dict[str, Any]]], class_id: int, report_conf: float
) -> dict[str, Any]:
    image_keys = {record["image_key"] for record in records}
    counts_all = {}
    counts_report = {}
    for image_key in image_keys:
        all_count = sum(1 for item in predictions[image_key] if item["class_id"] == class_id)
        report_count = sum(
            1
            for item in predictions[image_key]
            if item["class_id"] == class_id and item["score"] >= report_conf
        )
        if all_count:
            counts_all[image_key] = all_count
        if report_count:
            counts_report[image_key] = report_count
    return {
        "class_id": class_id,
        "class_name": CLASS_NAMES[class_id],
        "note": "This class has no GT in this split, so every prediction is an off-target false positive.",
        "predictions_all": sum(counts_all.values()),
        "images_with_predictions_all": len(counts_all),
        "predictions_at_report_conf": sum(counts_report.values()),
        "images_with_predictions_at_report_conf": len(counts_report),
        "top_images_at_report_conf": [
            {"image": image, "count": count}
            for image, count in sorted(counts_report.items(), key=lambda item: item[1], reverse=True)[:20]
        ],
    }


def evaluate_subset(
    split_name: str, records: list[dict[str, Any]], predictions: dict[str, list[dict[str, Any]]], report_conf: float
) -> dict[str, Any]:
    gt_histogram = Counter(
        item["class_id"] for record in records for item in record["ground_truth"]
    )
    prediction_histogram_all = Counter(
        item["class_id"] for record in records for item in predictions[record["image_key"]]
    )
    prediction_histogram_report = Counter(
        item["class_id"]
        for record in records
        for item in predictions[record["image_key"]]
        if item["score"] >= report_conf
    )
    present_classes = [class_id for class_id in CLASS_NAMES if gt_histogram[class_id] > 0]
    per_class = {
        str(class_id): evaluate_class(records, predictions, class_id, report_conf)
        for class_id in present_classes
    }
    all_report_matches = [
        match_class(records, predictions, class_id, 0.50, minimum_score=report_conf)
        for class_id in CLASS_NAMES
    ]
    total_tp = sum(item["tp"] for item in all_report_matches)
    total_fp = sum(item["fp"] for item in all_report_matches)
    total_fn = sum(item["fn"] for item in all_report_matches)
    class_ap50 = [per_class[str(class_id)]["ap50"] for class_id in present_classes]
    class_map = [per_class[str(class_id)]["map50_95"] for class_id in present_classes]
    metrics = {
        "split": split_name,
        "image_count": len(records),
        "classes": {str(key): value for key, value in CLASS_NAMES.items()},
        "gt_histogram": {str(class_id): gt_histogram[class_id] for class_id in CLASS_NAMES},
        "prediction_histogram_all": {
            str(class_id): prediction_histogram_all[class_id] for class_id in CLASS_NAMES
        },
        "prediction_histogram_at_report_conf": {
            str(class_id): prediction_histogram_report[class_id] for class_id in CLASS_NAMES
        },
        "iou_thresholds": IOU_THRESHOLDS,
        "report_conf": report_conf,
        "overall": {
            "ap_classes": present_classes,
            "ap50": float(np.mean(class_ap50)) if class_ap50 else None,
            "map50_95": float(np.mean(class_map)) if class_map else None,
            "precision": total_tp / max(total_tp + total_fp, 1),
            "recall": total_tp / max(total_tp + total_fn, 1),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "note": (
                "AP is macro-averaged over classes with GT. Overall fixed-threshold P/R includes both final "
                "classes, so off-target expert predictions are counted as FP."
            ),
        },
        "per_class": per_class,
        "off_target_false_positives": {},
    }
    for class_id in CLASS_NAMES:
        if gt_histogram[class_id] == 0:
            metrics["off_target_false_positives"][str(class_id)] = false_positive_diagnostic(
                records, predictions, class_id, report_conf
            )
    return metrics


def prediction_record(
    record: dict[str, Any], predictions: list[dict[str, Any]], field_name: str = "predictions"
) -> dict[str, Any]:
    return {
        "image": record["image"],
        "width": record["width"],
        "height": record["height"],
        "source_split": record["source_split"],
        "modality": record["modality"],
        field_name: predictions,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_metrics_markdown(path: Path, metrics: dict[str, Any]) -> None:
    lines = [f"# {metrics['split']} Late Fusion Metrics", ""]
    lines.append(f"- Images: `{metrics['image_count']}`")
    lines.append(f"- Report confidence: `{metrics['report_conf']}`")
    lines.append("")
    lines.append("| scope | GT | AP50 | mAP50-95 | P | R | TP | FP | FN |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    overall = metrics["overall"]
    lines.append(
        f"| all | {sum(metrics['gt_histogram'].values())} | {overall['ap50']:.6f} | "
        f"{overall['map50_95']:.6f} | {overall['precision']:.6f} | {overall['recall']:.6f} | "
        f"{overall['tp']} | {overall['fp']} | {overall['fn']} |"
    )
    for item in metrics["per_class"].values():
        lines.append(
            f"| {item['class_name']} | {item['gt']} | {item['ap50']:.6f} | {item['map50_95']:.6f} | "
            f"{item['precision']:.6f} | {item['recall']:.6f} | {item['tp']} | {item['fp']} | {item['fn']} |"
        )
    lines.extend(["", "## Off-target false positives", ""])
    if metrics["off_target_false_positives"]:
        for item in metrics["off_target_false_positives"].values():
            lines.append(
                f"- {item['class_name']}: all={item['predictions_all']}, "
                f"at report conf={item['predictions_at_report_conf']}"
            )
    else:
        lines.append("- None (both classes have GT in this split).")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_modality_outputs(
    metrics_dir: Path,
    bridge_records: list[dict[str, Any]],
    fused: dict[str, list[dict[str, Any]]],
    report_conf: float,
) -> tuple[dict[str, Any], dict[str, str]]:
    by_modality = {}
    for modality in ("rgb", "sar", "ir", "unknown"):
        selected = [record for record in bridge_records if record["modality"] == modality]
        if selected:
            by_modality[modality.upper()] = evaluate_subset(
                f"test_bridge_{modality}", selected, fused, report_conf
            )
    by_modality["ALL"] = evaluate_subset("test_bridge", bridge_records, fused, report_conf)
    json_path = metrics_dir / "test_bridge_by_modality.json"
    csv_path = metrics_dir / "test_bridge_by_modality.csv"
    md_path = metrics_dir / "test_bridge_by_modality.md"
    json_path.write_text(json.dumps(by_modality, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["modality", "images", "gt", "ap50", "map50_95", "precision", "recall", "tp", "fp", "fn"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for modality, metrics in by_modality.items():
            bridge = metrics["per_class"]["1"]
            writer.writerow(
                {
                    "modality": modality,
                    "images": metrics["image_count"],
                    "gt": bridge["gt"],
                    "ap50": bridge["ap50"],
                    "map50_95": bridge["map50_95"],
                    "precision": bridge["precision"],
                    "recall": bridge["recall"],
                    "tp": bridge["tp"],
                    "fp": bridge["fp"],
                    "fn": bridge["fn"],
                }
            )
    lines = ["# Bridge Late Fusion Metrics by Modality", ""]
    lines.append("| modality | images | GT | AP50 | mAP50-95 | P | R |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for modality, metrics in by_modality.items():
        bridge = metrics["per_class"]["1"]
        lines.append(
            f"| {modality} | {metrics['image_count']} | {bridge['gt']} | {bridge['ap50']:.6f} | "
            f"{bridge['map50_95']:.6f} | {bridge['precision']:.6f} | {bridge['recall']:.6f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return by_modality, {"json": str(json_path), "csv": str(csv_path), "md": str(md_path)}


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "images": metrics["image_count"],
        "gt_histogram": metrics["gt_histogram"],
        "prediction_histogram_all": metrics["prediction_histogram_all"],
        "overall": metrics["overall"],
        "per_class": metrics["per_class"],
        "off_target_false_positives": metrics["off_target_false_positives"],
    }


def write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = ["# Ship + Bridge Late Fusion Upper Bound", ""]
    for key in (
        "status",
        "ship_expert",
        "bridge_expert",
        "dataset_root",
        "fusion_nms",
        "nms_iou",
        "ship_conf",
        "bridge_conf",
        "report_conf",
    ):
        lines.append(f"- {key}: `{summary.get(key)}`")
    lines.extend(["", "| split | AP50 | mAP50-95 | P | R | TP | FP | FN |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for split in ("test_ship", "test_bridge", "test_combined"):
        metrics = summary.get(split)
        if not metrics:
            continue
        overall = metrics["overall"]
        lines.append(
            f"| {split} | {overall['ap50']:.6f} | {overall['map50_95']:.6f} | "
            f"{overall['precision']:.6f} | {overall['recall']:.6f} | "
            f"{overall['tp']} | {overall['fp']} | {overall['fn']} |"
        )
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in summary.get("notes", []))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate frozen YOLOE ship + YOLOv8m bridge prediction-level late fusion.")
    parser.add_argument("--ship-weights", default=DEFAULT_SHIP_WEIGHTS)
    parser.add_argument("--bridge-weights", default=DEFAULT_BRIDGE_WEIGHTS)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--ship-conf", type=float, default=0.001)
    parser.add_argument("--bridge-conf", type=float, default=0.001)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--fusion-nms", choices=("class-aware", "class-agnostic"), default="class-aware")
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--report-conf", type=float, default=0.25)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--save-preds", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch <= 0 or args.imgsz <= 0 or args.max_det <= 0:
        raise ValueError("batch, imgsz, and max_det must be positive")
    if not (0.0 <= args.ship_conf <= 1.0 and 0.0 <= args.bridge_conf <= 1.0):
        raise ValueError("Expert confidence thresholds must be in [0,1]")
    if not (0.0 < args.nms_iou <= 1.0 and 0.0 <= args.report_conf <= 1.0):
        raise ValueError("NMS IoU/report confidence is outside its valid range")

    ship_weights = resolve_path(args.ship_weights)
    bridge_weights = resolve_path(args.bridge_weights)
    dataset_root = resolve_path(args.dataset_root)
    project = resolve_path(args.project)
    run_dir = project / args.name
    if run_dir.exists() and not args.exist_ok:
        raise FileExistsError(f"Output run already exists; use a new --name or pass --exist-ok: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = run_dir / "metrics"
    predictions_dir = run_dir / "predictions"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "late_fusion_summary.json"
    summary_md_path = run_dir / "late_fusion_summary.md"
    summary: dict[str, Any] = {
        "task": "late fusion upper bound: YOLOE ship expert + YOLOv8m bridge expert",
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "project_root": str(PROJECT_ROOT),
        "git": git_info(),
        "ship_expert": str(ship_weights),
        "bridge_expert": str(bridge_weights),
        "dataset_root": str(dataset_root),
        "classes": {"0": "ship", "1": "bridge"},
        "fusion_nms": args.fusion_nms,
        "nms_iou": args.nms_iou,
        "ship_conf": args.ship_conf,
        "bridge_conf": args.bridge_conf,
        "report_conf": args.report_conf,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "max_det": args.max_det,
        "save_predictions": args.save_preds,
        "training_performed": False,
    }
    try:
        device = parse_device(args.device)
        records, dataset_preflight = preflight_dataset(dataset_root)
        summary["dataset_preflight"] = dataset_preflight
        ship_wrapper, ship_inner, ship_preflight = prepare_ship_expert(ship_weights)
        bridge_wrapper, bridge_inner, bridge_preflight = prepare_bridge_expert(bridge_weights)
        summary["ship_expert_preflight"] = ship_preflight
        summary["bridge_expert_preflight"] = bridge_preflight
        print("[preflight] dataset and both frozen experts passed", flush=True)

        ship_predictions, ship_inference = infer_ship(ship_inner, records, args, device)
        summary["ship_inference"] = ship_inference
        ship_inner.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        bridge_predictions, bridge_inference = infer_bridge(bridge_wrapper, records, args)
        summary["bridge_inference"] = bridge_inference
        bridge_inner.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        fused = fuse_predictions(records, ship_predictions, bridge_predictions, args)
        ship_records = [record for record in records if record["source_split"] == "test_ship"]
        bridge_records = [record for record in records if record["source_split"] == "test_bridge"]
        split_records = {
            "test_ship": ship_records,
            "test_bridge": bridge_records,
            "test_combined": records,
        }
        metric_paths = {}
        full_metrics = {}
        for split, selected_records in split_records.items():
            metrics = evaluate_subset(split, selected_records, fused, args.report_conf)
            full_metrics[split] = metrics
            json_path = metrics_dir / f"{split}_metrics.json"
            md_path = metrics_dir / f"{split}_metrics.md"
            json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            write_metrics_markdown(md_path, metrics)
            metric_paths[split] = {"json": str(json_path), "md": str(md_path)}
            summary[split] = compact_metrics(metrics)

        bridge_by_modality, modality_paths = write_modality_outputs(
            metrics_dir, bridge_records, fused, args.report_conf
        )
        summary["bridge_by_modality"] = {
            modality: compact_metrics(metrics) for modality, metrics in bridge_by_modality.items()
        }
        summary["metric_files"] = metric_paths
        summary["bridge_modality_files"] = modality_paths

        runtime_manifest = [
            {
                "image": record["image"],
                "label": record["label"],
                "width": record["width"],
                "height": record["height"],
                "source_split": record["source_split"],
                "modality": record["modality"],
                "ground_truth_boxes": len(record["ground_truth"]),
            }
            for record in records
        ]
        runtime_manifest_path = run_dir / "runtime_manifest.json"
        runtime_manifest_path.write_text(
            json.dumps(runtime_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary["runtime_manifest"] = str(runtime_manifest_path)

        if args.save_preds:
            prediction_files = {
                "ship": predictions_dir / "ship_predictions.jsonl",
                "bridge": predictions_dir / "bridge_predictions.jsonl",
                "fused": predictions_dir / "fused_predictions.jsonl",
                "ground_truth": predictions_dir / "ground_truth.jsonl",
            }
            write_jsonl(
                prediction_files["ship"],
                [prediction_record(record, ship_predictions[record["image_key"]]) for record in records],
            )
            write_jsonl(
                prediction_files["bridge"],
                [prediction_record(record, bridge_predictions[record["image_key"]]) for record in records],
            )
            write_jsonl(
                prediction_files["fused"],
                [prediction_record(record, fused[record["image_key"]]) for record in records],
            )
            write_jsonl(
                prediction_files["ground_truth"],
                [prediction_record(record, record["ground_truth"], field_name="ground_truth") for record in records],
            )
            summary["prediction_files"] = {key: str(value) for key, value in prediction_files.items()}

        summary["notes"] = [
            "No training was performed.",
            "YOLOE best.pt was frozen and used only for ship prediction.",
            "YOLOE native model.val() was not called.",
            "Bridge expert was frozen and used only for bridge prediction.",
            "This is prediction-level late fusion, not a single student model.",
            "AP uses predictions retained above each expert's low confidence threshold and COCO-style 101-point interpolation.",
            "Fixed-threshold overall P/R includes off-target class predictions as false positives.",
            "total_bridge_test is internal held-out, not external bridge generalization.",
        ]
        summary["status"] = "completed"
        summary["completed_at"] = datetime.now().isoformat()
        summary["elapsed_seconds"] = time.time() - datetime.fromisoformat(summary["started_at"]).timestamp()
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        write_summary_markdown(summary_md_path, summary)
        print(json.dumps({
            "status": summary["status"],
            "test_ship": summary["test_ship"]["overall"],
            "test_bridge": summary["test_bridge"]["overall"],
            "test_combined": summary["test_combined"]["overall"],
            "summary": str(summary_path),
        }, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as exc:
        summary["status"] = "failed"
        summary["failed_at"] = datetime.now().isoformat()
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        write_summary_markdown(summary_md_path, summary)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
