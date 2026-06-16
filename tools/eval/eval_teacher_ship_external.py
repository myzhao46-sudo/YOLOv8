# tools/eval/eval_teacher_ship_external.py
# -*- coding: utf-8 -*-

"""
Parameterized box-only evaluation for the original YOLOE teacher/best.pt on external ship datasets.

This script keeps the YOLOESegModel / YOLOESegment structure unchanged. It does not call model.val(),
does not train, does not save checkpoints, and does not convert segmentation labels. It reads only
5-column YOLO detect labels, ignores masks, and reports box-only metrics for one target class
(default: class 0 ship).
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image


REPO_ROOT = Path(r"C:\Users\DOCTOR\Documents\GitHub\YOLOv8")
PACKAGE_ROOT = REPO_ROOT / "ultralytics"
DEFAULT_WEIGHTS = REPO_ROOT / "ultralytics" / "best.pt"
DEFAULT_DATA = REPO_ROOT / "configs" / "datasets" / "global4_eval_ship_extratest_all.yaml"

GLOBAL4_CLASS_NAMES = ["ship", "harbor", "tank", "bridge"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CSV_FIELDS = [
    "exp_id",
    "date",
    "git_commit",
    "weights",
    "data_yaml",
    "modality",
    "images_dir",
    "labels_dir",
    "image_count",
    "missing_label_files",
    "bad_label_rows",
    "non_target_rows",
    "target_class",
    "target_name",
    "gt_count",
    "pred_count",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "ap50",
    "imgsz",
    "conf",
    "pred_iou",
    "match_iou",
    "max_det",
    "batch",
    "device",
    "model_class",
    "task",
    "last_layer_type",
    "set_classes_ok",
    "pe_shape",
    "score_channels_seen",
    "native_val_called",
    "masks_ignored",
    "notes",
]


def log(msg: object = "") -> None:
    print(msg, flush=True)


def setup_local_ultralytics_import() -> None:
    repo_root = REPO_ROOT.resolve()
    package_root = PACKAGE_ROOT.resolve()
    for p in [str(package_root), str(repo_root)]:
        while p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, str(package_root))
    sys.path.insert(1, str(repo_root))


def safe_getattr(obj: object, name: str, default=None):
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def obj_type(obj: object) -> str:
    if obj is None:
        return "None"
    return f"{obj.__class__.__module__}.{obj.__class__.__name__}"


def safe_signature(fn: object) -> str:
    try:
        return str(inspect.signature(fn))
    except Exception:
        return "signature unavailable"


def shape_of(x: object):
    if x is None:
        return None
    if torch.is_tensor(x):
        return list(x.shape)
    if hasattr(x, "shape"):
        try:
            return list(x.shape)
        except Exception:
            return str(x.shape)
    return None


def to_plain_names(names):
    if names is None:
        return None
    if isinstance(names, dict):
        out = {}
        for k, v in names.items():
            try:
                kk = int(k)
            except Exception:
                kk = str(k)
            out[kk] = str(v)
        return out
    if isinstance(names, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names)}
    return str(names)


def get_layers(inner_model):
    layers = safe_getattr(inner_model, "model", None)
    if layers is None:
        return []
    try:
        return list(layers)
    except Exception:
        return []


def get_last_layer(inner_model):
    layers = get_layers(inner_model)
    return layers[-1] if layers else None


def get_task_state(yolo_obj, inner_model) -> dict:
    inner_args = safe_getattr(inner_model, "args", None)
    inner_args_task = inner_args.get("task") if isinstance(inner_args, dict) else safe_getattr(inner_args, "task", None)
    return {
        "wrapper.task": safe_getattr(yolo_obj, "task", None),
        "inner.task": safe_getattr(inner_model, "task", None),
        "inner.args.task": inner_args_task,
    }


def get_names_state(yolo_obj, inner_model) -> dict:
    return {
        "wrapper.names": to_plain_names(safe_getattr(yolo_obj, "names", None)),
        "inner.names": to_plain_names(safe_getattr(inner_model, "names", None)),
    }


def head_state(inner_model) -> dict:
    last_layer = get_last_layer(inner_model)
    return {
        "last_layer_type": obj_type(last_layer),
        "last_layer_class": last_layer.__class__.__name__ if last_layer is not None else None,
        "last_layer.nc": safe_getattr(last_layer, "nc", None),
        "last_layer.no": safe_getattr(last_layer, "no", None),
        "last_layer.nl": safe_getattr(last_layer, "nl", None),
        "last_layer.reg_max": safe_getattr(last_layer, "reg_max", None),
        "last_layer.embed": safe_getattr(last_layer, "embed", None),
        "last_layer.is_fused": safe_getattr(last_layer, "is_fused", None),
    }


def read_data_yaml(data_yaml: Path) -> dict:
    if not data_yaml.exists():
        raise FileNotFoundError(f"DATA YAML not found: {data_yaml}")
    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "val" not in data:
        raise SyntaxError(f"{data_yaml} has no 'val:' key.")
    return data


def resolve_val_dirs(data_yaml: Path) -> list[Path]:
    data = read_data_yaml(data_yaml)
    val = data["val"]
    values = val if isinstance(val, list) else [val]
    dirs = []
    for v in values:
        p = Path(str(v)).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"val images directory does not exist: {p}")
        if not p.is_dir():
            raise NotADirectoryError(f"val entry is not a directory: {p}")
        dirs.append(p.resolve())
    return dirs


def labels_dir_from_images_dir(images_dir: Path) -> Path:
    parts = list(images_dir.parts)
    for i, part in enumerate(parts):
        if part.lower() == "images":
            parts[i] = "labels"
            return Path(*parts)
    return images_dir.parent.parent / "labels" / images_dir.name


def list_images(images_dir: Path) -> list[Path]:
    images = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if not images:
        raise FileNotFoundError(f"No images found in {images_dir}")
    return images


def read_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as im:
        return im.size


def make_letterbox_rgb(image_path: Path, size: int) -> tuple[np.ndarray, dict]:
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        orig_w, orig_h = im.size
        scale = min(size / orig_w, size / orig_h)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        resized = im.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (size, size), (114, 114, 114))
        pad_x = (size - new_w) / 2.0
        pad_y = (size - new_h) / 2.0
        canvas.paste(resized, (int(round(pad_x)), int(round(pad_y))))
    return np.ascontiguousarray(np.asarray(canvas)), {
        "orig_w": orig_w,
        "orig_h": orig_h,
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
    }


def unletterbox_xyxy(box: np.ndarray, meta: dict) -> list[float]:
    x1 = (float(box[0]) - meta["pad_x"]) / meta["scale"]
    y1 = (float(box[1]) - meta["pad_y"]) / meta["scale"]
    x2 = (float(box[2]) - meta["pad_x"]) / meta["scale"]
    y2 = (float(box[3]) - meta["pad_y"]) / meta["scale"]
    orig_w = meta["orig_w"]
    orig_h = meta["orig_h"]
    return [
        float(max(0.0, min(orig_w, x1))),
        float(max(0.0, min(orig_h, y1))),
        float(max(0.0, min(orig_w, x2))),
        float(max(0.0, min(orig_h, y2))),
    ]


def xywhn_to_xyxy_pixels(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int) -> list[float]:
    x1 = (cx - w / 2.0) * img_w
    y1 = (cy - h / 2.0) * img_h
    x2 = (cx + w / 2.0) * img_w
    y2 = (cy + h / 2.0) * img_h
    return [
        float(max(0.0, min(img_w, x1))),
        float(max(0.0, min(img_h, y1))),
        float(max(0.0, min(img_w, x2))),
        float(max(0.0, min(img_h, y2))),
    ]


def read_gt_boxes(
    label_path: Path,
    img_w: int,
    img_h: int,
    target_cls: int,
    strict_label: bool,
) -> tuple[list[list[float]], int, int]:
    boxes = []
    skipped_non_target = 0
    skipped_bad = 0
    if not label_path.exists():
        return boxes, skipped_non_target, skipped_bad

    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            if strict_label:
                raise ValueError(f"Non-5-column label row: {label_path}:{line_no}: {line}")
            skipped_bad += 1
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:])
        except Exception as e:
            if strict_label:
                raise ValueError(f"Bad numeric label row: {label_path}:{line_no}: {line}") from e
            skipped_bad += 1
            continue
        if cls != target_cls:
            skipped_non_target += 1
            continue
        boxes.append(xywhn_to_xyxy_pixels(cx, cy, w, h, img_w, img_h))
    return boxes, skipped_non_target, skipped_bad


def box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter
    return inter / np.maximum(union, 1e-9)


def compute_ap_from_pr(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if recalls.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_predictions(predictions: list[dict], gt_by_image: dict[str, list[list[float]]], match_iou: float) -> dict:
    gt_arrays = {k: np.asarray(v, dtype=np.float32).reshape(-1, 4) for k, v in gt_by_image.items()}
    matched = {k: np.zeros((len(v),), dtype=bool) for k, v in gt_arrays.items()}
    total_gt = int(sum(len(v) for v in gt_arrays.values()))
    preds = sorted(predictions, key=lambda x: x["conf"], reverse=True)
    tp_flags = []
    fp_flags = []
    for pred in preds:
        image_key = pred["image_key"]
        gt = gt_arrays.get(image_key, np.zeros((0, 4), dtype=np.float32))
        box = np.asarray(pred["xyxy"], dtype=np.float32)
        if gt.shape[0] == 0:
            tp_flags.append(0.0)
            fp_flags.append(1.0)
            continue
        ious = box_iou_one_to_many(box, gt)
        best_idx = int(np.argmax(ious)) if ious.size else -1
        best_iou = float(ious[best_idx]) if best_idx >= 0 else 0.0
        if best_iou >= match_iou and not matched[image_key][best_idx]:
            matched[image_key][best_idx] = True
            tp_flags.append(1.0)
            fp_flags.append(0.0)
        else:
            tp_flags.append(0.0)
            fp_flags.append(1.0)

    tp_cum = np.cumsum(np.asarray(tp_flags, dtype=np.float32))
    fp_cum = np.cumsum(np.asarray(fp_flags, dtype=np.float32))
    tp = int(tp_cum[-1]) if tp_cum.size else 0
    fp = int(fp_cum[-1]) if fp_cum.size else 0
    fn = int(total_gt - tp)
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(total_gt, 1))
    recalls = tp_cum / max(total_gt, 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    return {
        "gt_count": total_gt,
        "prediction_count": len(preds),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "ap50": compute_ap_from_pr(recalls, precisions),
    }


def set_global4_classes(model, inner_model):
    names = list(GLOBAL4_CLASS_NAMES)
    get_text_pe = safe_getattr(model, "get_text_pe", None)
    if not callable(get_text_pe):
        get_text_pe = safe_getattr(inner_model, "get_text_pe", None)
    if not callable(get_text_pe):
        raise AttributeError("get_text_pe not found on wrapper or inner model")
    embeddings = get_text_pe(names)

    set_classes = safe_getattr(model, "set_classes", None)
    target = "YOLO wrapper"
    if not callable(set_classes):
        set_classes = safe_getattr(inner_model, "set_classes", None)
        target = "inner model"
    if not callable(set_classes):
        raise AttributeError("set_classes not found on wrapper or inner model")
    set_classes(names, embeddings)
    return embeddings, target, safe_signature(get_text_pe), safe_signature(set_classes)


def tensor_from_rgb_images(images: list[np.ndarray]) -> torch.Tensor:
    arr = np.stack(images).transpose(0, 3, 1, 2)
    return torch.from_numpy(np.ascontiguousarray(arr)).float() / 255.0


def raw_yoloeseg_fused_forward(inner_model, x: torch.Tensor) -> tuple[torch.Tensor, int]:
    from ultralytics.nn.modules.head import YOLOESegment

    y = []
    out = x
    head = get_last_layer(inner_model)
    for m in inner_model.model:
        if m.f != -1:
            out = y[m.f] if isinstance(m.f, int) else [out if j == -1 else y[j] for j in m.f]
        if m is head:
            assert isinstance(m, YOLOESegment), f"Expected YOLOESegment head, got {obj_type(m)}"
            feats = out
            bs = feats[0].shape[0]
            boxes = []
            scores = []
            for i in range(m.nl):
                box_i = m.cv2[i](feats[i]).view(bs, 4 * m.reg_max, -1)
                cls_feat = m.cv3[i](feats[i])
                if getattr(m, "is_fused", False):
                    score_i = m.cv4[i](cls_feat, None)
                else:
                    pe = safe_getattr(inner_model, "pe", None)
                    if pe is None:
                        raise RuntimeError("Non-fused YOLOE head requires inner_model.pe for text scores.")
                    pe = pe.to(device=cls_feat.device, dtype=cls_feat.dtype)
                    if pe.shape[0] != bs:
                        pe = pe.expand(bs, -1, -1)
                    score_i = m.cv4[i](cls_feat, pe)
                boxes.append(box_i)
                scores.append(score_i.reshape(bs, score_i.shape[1], -1))
            scores = torch.cat(scores, dim=-1)
            preds = {"boxes": torch.cat(boxes, dim=-1), "scores": scores, "feats": feats}
            dbox = m._get_decode_boxes(preds)
            return torch.cat((dbox, scores.sigmoid()), dim=1), int(scores.shape[1])
        out = m(out)
        y.append(out if m.i in inner_model.save else None)
    raise RuntimeError("YOLOESegment head was not reached.")


def collect_predictions(inner_model, image_paths: list[Path], args) -> tuple[list[dict], set[int], int]:
    from ultralytics.utils import nms

    inner_model.eval().to(args.device)
    predictions = []
    total_boxes = 0
    score_channels_seen = set()
    with torch.no_grad():
        for start in range(0, len(image_paths), args.batch):
            batch_paths = image_paths[start : start + args.batch]
            batch_imgs = []
            metas = {}
            for p in batch_paths:
                arr, meta = make_letterbox_rgb(p, args.imgsz)
                batch_imgs.append(arr)
                metas[str(p.resolve())] = meta
            batch_tensor = tensor_from_rgb_images(batch_imgs).to(args.device)
            raw_pred, score_nc = raw_yoloeseg_fused_forward(inner_model, batch_tensor)
            score_channels_seen.add(score_nc)
            dets = nms.non_max_suppression(
                raw_pred,
                conf_thres=args.conf,
                iou_thres=args.pred_iou,
                classes=None,
                agnostic=False,
                max_det=args.max_det,
                nc=score_nc,
            )
            for image_path, det in zip(batch_paths, dets):
                if det is None or det.shape[0] == 0:
                    continue
                det_np = det.detach().cpu().numpy()
                total_boxes += int(det_np.shape[0])
                image_key = str(image_path.resolve())
                for row in det_np:
                    k = int(row[5])
                    if k != args.target_class:
                        continue
                    predictions.append(
                        {
                            "image_key": image_key,
                            "xyxy": unletterbox_xyxy(row[:4], metas[image_key]),
                            "conf": float(row[4]),
                            "cls": k,
                        }
                    )
    return predictions, score_channels_seen, total_boxes


def collect_ground_truth(image_paths: list[Path], labels_dir: Path, args) -> tuple[dict[str, list[list[float]]], dict]:
    gt_by_image = {}
    bad_rows = 0
    non_target_rows = 0
    missing_label_files = 0
    for image_path in image_paths:
        img_w, img_h = read_image_size(image_path)
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing_label_files += 1
        boxes, skipped_non_target, skipped_bad = read_gt_boxes(
            label_path, img_w, img_h, args.target_class, args.strict_label
        )
        gt_by_image[str(image_path.resolve())] = boxes
        non_target_rows += skipped_non_target
        bad_rows += skipped_bad
    return gt_by_image, {
        "image_count": len(image_paths),
        "missing_label_files": missing_label_files,
        "bad_label_rows": bad_rows,
        "non_target_rows": non_target_rows,
        "target_gt_boxes": sum(len(v) for v in gt_by_image.values()),
    }


def infer_modality(images_dir: Path, fallback: str) -> str:
    name = images_dir.name.lower()
    if "rgb" in name:
        return "RGB"
    if "sar" in name:
        return "SAR"
    if "ir" in name or "infr" in name:
        return "IR"
    return fallback


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO_ROOT), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def append_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def append_md(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    cols = ["date", "modality", "image_count", "gt_count", "pred_count", "tp", "fp", "fn", "precision", "recall", "ap50"]
    with path.open("a", encoding="utf-8") as f:
        if not exists:
            f.write("| " + " | ".join(cols) + " |\n")
            f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |\n")


def build_result_row(
    args,
    exp_id: str,
    date: str,
    commit: str,
    images_dir: str,
    labels_dir: str,
    modality: str,
    gt_stats: dict,
    metrics: dict,
    model_class: str,
    task: str,
    last_layer_type: str,
    set_classes_ok: bool,
    pe_shape,
    score_channels_seen: set[int],
    notes: list[str],
) -> dict:
    return {
        "exp_id": exp_id,
        "date": date,
        "git_commit": commit,
        "weights": str(Path(args.weights).resolve()),
        "data_yaml": str(Path(args.data).resolve()),
        "modality": modality,
        "images_dir": images_dir,
        "labels_dir": labels_dir,
        "image_count": gt_stats["image_count"],
        "missing_label_files": gt_stats["missing_label_files"],
        "bad_label_rows": gt_stats["bad_label_rows"],
        "non_target_rows": gt_stats["non_target_rows"],
        "target_class": args.target_class,
        "target_name": args.target_name,
        "gt_count": metrics["gt_count"],
        "pred_count": metrics["prediction_count"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "precision": f"{metrics['precision']:.6f}",
        "recall": f"{metrics['recall']:.6f}",
        "ap50": f"{metrics['ap50']:.6f}",
        "imgsz": args.imgsz,
        "conf": args.conf,
        "pred_iou": args.pred_iou,
        "match_iou": args.match_iou,
        "max_det": args.max_det,
        "batch": args.batch,
        "device": args.device,
        "model_class": model_class,
        "task": task,
        "last_layer_type": last_layer_type,
        "set_classes_ok": set_classes_ok,
        "pe_shape": json.dumps(pe_shape),
        "score_channels_seen": json.dumps(sorted(score_channels_seen)),
        "native_val_called": False,
        "masks_ignored": True,
        "notes": "; ".join(notes),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Box-only YOLOESeg teacher eval for external ship detect labels.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--target-class", type=int, default=0)
    parser.add_argument("--target-name", default="ship")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--pred-iou", type=float, default=0.7)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--modality", default="unknown")
    parser.add_argument("--save-csv", default="")
    parser.add_argument("--save-md", default="")
    parser.add_argument("--strict-label", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_local_ultralytics_import()

    from ultralytics import YOLO
    import ultralytics as ultralytics_pkg

    exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commit = git_commit()
    data_yaml = Path(args.data).resolve()
    weights = Path(args.weights).resolve()

    log("[IMPORT CHECK]")
    log(f"ultralytics imported from: {getattr(ultralytics_pkg, '__file__', 'UNKNOWN')}")
    log(f"first sys.path entries: {sys.path[:3]}")
    log("")
    log("[CONFIG]")
    log(f"weights: {weights}")
    log(f"data yaml: {data_yaml}")
    log(f"modality arg: {args.modality}")
    log(f"target class: {args.target_class}")
    log(f"target name: {args.target_name}")
    log(f"imgsz={args.imgsz} conf={args.conf} pred_iou={args.pred_iou} match_iou={args.match_iou}")
    log(f"max_det={args.max_det} batch={args.batch} device={args.device} strict_label={args.strict_label}")

    val_dirs = resolve_val_dirs(data_yaml)
    log("")
    log("[DATA]")
    for d in val_dirs:
        log(f"images dir: {d}")
        log(f"labels dir: {labels_dir_from_images_dir(d)}")

    log("")
    log("[LOAD MODEL]")
    model = YOLO(str(weights))
    inner = safe_getattr(model, "model", None)
    before_head = head_state(inner)
    model_class = obj_type(inner)
    log(f"YOLO wrapper type: {obj_type(model)}")
    log(f"inner model type: {model_class}")
    log(f"task before: {json.dumps(get_task_state(model, inner), ensure_ascii=False)}")
    log(f"names before: {json.dumps(get_names_state(model, inner), ensure_ascii=False)}")
    log(f"last layer type: {before_head['last_layer_type']}")
    log(f"last layer nc before: {before_head['last_layer.nc']}")
    log(f"is YOLOESegModel: {'YOLOESegModel' in model_class}")
    log(f"is YOLOESegment: {'YOLOESegment' in before_head['last_layer_type']}")

    log("")
    log("[GLOBAL4 SET_CLASSES]")
    set_classes_ok = False
    embeddings, set_target, get_sig, set_sig = set_global4_classes(model, inner)
    set_classes_ok = True
    after_head = head_state(inner)
    pe_shape = shape_of(embeddings)
    task_after = get_task_state(model, inner)
    log(f"global4 names: {GLOBAL4_CLASS_NAMES}")
    log(f"get_text_pe signature: {get_sig}")
    log(f"set_classes target: {set_target}")
    log(f"set_classes signature: {set_sig}")
    log(f"set_classes ok: {set_classes_ok}")
    log(f"pe shape: {pe_shape}")
    log(f"names after: {json.dumps(get_names_state(model, inner), ensure_ascii=False)}")
    log(f"task after: {json.dumps(task_after, ensure_ascii=False)}")
    log(f"last layer type after: {after_head['last_layer_type']}")
    log(f"last layer nc after: {after_head['last_layer.nc']}")

    all_predictions = []
    all_gt_by_image = {}
    all_gt_stats = {"image_count": 0, "missing_label_files": 0, "bad_label_rows": 0, "non_target_rows": 0}
    all_images_dirs = []
    all_labels_dirs = []
    all_score_channels = set()
    rows = []
    notes_global = []

    if args.target_class != 0:
        notes_global.append("non-default target class requested; this task normally evaluates ship class 0 only")

    for images_dir in val_dirs:
        labels_dir = labels_dir_from_images_dir(images_dir)
        image_paths = list_images(images_dir)
        group_modality = infer_modality(images_dir, args.modality)
        log("")
        log(f"[EVAL DATASET: {group_modality}]")
        log(f"images dir: {images_dir}")
        log(f"labels dir: {labels_dir}")
        log(f"image count: {len(image_paths)}")

        gt_by_image, gt_stats = collect_ground_truth(image_paths, labels_dir, args)
        predictions, score_channels, total_boxes_all_classes = collect_predictions(inner, image_paths, args)
        metrics = evaluate_predictions(predictions, gt_by_image, args.match_iou)
        notes = list(notes_global)
        if score_channels == {3}:
            notes.append("fused head emits 3 score channels; class 3 bridge has no true output channel")
        if args.target_class >= max(score_channels or {0}):
            notes.append("target class id is outside emitted score channels; predictions will be empty")

        log(f"missing label files: {gt_stats['missing_label_files']}")
        log(f"bad label rows: {gt_stats['bad_label_rows']}")
        log(f"non-target rows: {gt_stats['non_target_rows']}")
        log(f"target class id/name: {args.target_class}/{args.target_name}")
        log(f"score channels seen: {sorted(score_channels)}")
        log(f"all-class post-NMS predictions: {total_boxes_all_classes}")
        log(f"GT count: {metrics['gt_count']}")
        log(f"prediction count: {metrics['prediction_count']}")
        log(f"TP: {metrics['tp']}  FP: {metrics['fp']}  FN: {metrics['fn']}")
        log(f"Precision: {metrics['precision']:.6f}")
        log(f"Recall: {metrics['recall']:.6f}")
        log(f"AP50: {metrics['ap50']:.6f}")

        rows.append(
            build_result_row(
                args,
                exp_id,
                date,
                commit,
                str(images_dir),
                str(labels_dir),
                group_modality,
                gt_stats,
                metrics,
                model_class,
                str(task_after.get("inner.task")),
                after_head["last_layer_type"],
                set_classes_ok,
                pe_shape,
                score_channels,
                notes,
            )
        )

        all_predictions.extend(predictions)
        all_gt_by_image.update(gt_by_image)
        for k in all_gt_stats:
            all_gt_stats[k] += gt_stats[k]
        all_images_dirs.append(str(images_dir))
        all_labels_dirs.append(str(labels_dir))
        all_score_channels.update(score_channels)

    if len(val_dirs) > 1:
        metrics = evaluate_predictions(all_predictions, all_gt_by_image, args.match_iou)
        notes = list(notes_global)
        if all_score_channels == {3}:
            notes.append("fused head emits 3 score channels; class 3 bridge has no true output channel")
        log("")
        log("[EVAL DATASET: ALL]")
        log(f"images dirs: {';'.join(all_images_dirs)}")
        log(f"labels dirs: {';'.join(all_labels_dirs)}")
        log(f"image count: {all_gt_stats['image_count']}")
        log(f"missing label files: {all_gt_stats['missing_label_files']}")
        log(f"bad label rows: {all_gt_stats['bad_label_rows']}")
        log(f"non-target rows: {all_gt_stats['non_target_rows']}")
        log(f"score channels seen: {sorted(all_score_channels)}")
        log(f"GT count: {metrics['gt_count']}")
        log(f"prediction count: {metrics['prediction_count']}")
        log(f"TP: {metrics['tp']}  FP: {metrics['fp']}  FN: {metrics['fn']}")
        log(f"Precision: {metrics['precision']:.6f}")
        log(f"Recall: {metrics['recall']:.6f}")
        log(f"AP50: {metrics['ap50']:.6f}")
        rows.append(
            build_result_row(
                args,
                exp_id,
                date,
                commit,
                ";".join(all_images_dirs),
                ";".join(all_labels_dirs),
                args.modality if args.modality != "unknown" else "ALL",
                all_gt_stats,
                metrics,
                model_class,
                str(task_after.get("inner.task")),
                after_head["last_layer_type"],
                set_classes_ok,
                pe_shape,
                all_score_channels,
                notes,
            )
        )

    if args.save_csv:
        append_csv(Path(args.save_csv), rows)
        log(f"[SAVE] CSV appended: {args.save_csv}")
    if args.save_md:
        append_md(Path(args.save_md), rows)
        log(f"[SAVE] Markdown appended: {args.save_md}")

    log("")
    log("[CONCLUSION]")
    log(f"native model.val called: False")
    log(f"masks ignored: True")
    log(f"model remains YOLOESegModel: {'YOLOESegModel' in model_class}")
    log(f"head remains YOLOESegment: {'YOLOESegment' in after_head['last_layer_type']}")
    log(f"set_classes ok: {set_classes_ok}")
    log(f"pe shape: {pe_shape}")
    log(f"score channels seen: {sorted(all_score_channels)}")
    if all_score_channels == {3}:
        log("WARNING: fused head only emits 3 score channels; class 3 bridge has no true output channel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
