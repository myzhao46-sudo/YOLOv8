# tools/eval/eval_yoloeseg_box_only.py
# -*- coding: utf-8 -*-

"""
Minimal box-only evaluation for YOLOESegModel on 5-column YOLO detect labels.

This script intentionally avoids native model.val(), because YOLOESegModel uses the
segmentation validator, which requires masks/segments. It keeps the model structure
unchanged, runs YOLOE fixed-text prompt initialization, predicts boxes, ignores masks,
and computes class-0 ship box metrics manually.

It does not train, save .pt files, edit data, edit YAML files, or replace the head.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image


REPO_ROOT = Path(r"C:\Users\DOCTOR\Documents\GitHub\YOLOv8")
PACKAGE_ROOT = REPO_ROOT / "ultralytics"
WEIGHTS_PATH = REPO_ROOT / "ultralytics" / "best.pt"

REQUESTED_DATA_YAML = REPO_ROOT / "configs" / "datasets" / "global4_eval_ship.yaml"
FALLBACK_DATA_YAML = REPO_ROOT / "configs" / "global4_eval_ship.yaml"
SHIP_SMALL_SPLIT_VAL_IMAGES = REPO_ROOT / "ultralytics" / "datasets" / "ship_small_split" / "images" / "val"

GLOBAL4_CLASS_NAMES = ["ship", "harbor", "tank", "bridge"]
TARGET_CLASS_ID = 0
TARGET_CLASS_NAME = "ship"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMG_SIZE = 640
BATCH = 1
DEVICE = "cpu"
PRED_CONF = 0.001
PRED_IOU = 0.7
MAX_DET = 300
MATCH_IOU = 0.5


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


def summarize_tensor(x: object) -> dict:
    if torch.is_tensor(x):
        return {
            "type": "torch.Tensor",
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "device": str(x.device),
        }
    return {"type": obj_type(x), "shape": shape_of(x), "repr": repr(x)[:300]}


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
    if isinstance(inner_args, dict):
        inner_args_task = inner_args.get("task")
    else:
        inner_args_task = safe_getattr(inner_args, "task", None)
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
    }


def resolve_source_yaml() -> Path | None:
    for p in [REQUESTED_DATA_YAML, FALLBACK_DATA_YAML]:
        if p.exists():
            return p.resolve()
    return None


def yaml_val_paths(data_yaml: Path | None) -> list[Path]:
    if data_yaml is None:
        return []
    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    val = data.get("val")
    if val is None:
        return []
    values = val if isinstance(val, list) else [val]
    return [Path(str(v)) for v in values]


def resolve_images_dir() -> tuple[Path, list[str]]:
    notes = []
    data_yaml = resolve_source_yaml()
    notes.append(f"requested DATA_YAML: {REQUESTED_DATA_YAML} exists={REQUESTED_DATA_YAML.exists()}")
    notes.append(f"fallback DATA_YAML: {FALLBACK_DATA_YAML} exists={FALLBACK_DATA_YAML.exists()}")

    for p in yaml_val_paths(data_yaml):
        notes.append(f"YAML val candidate: {p} exists={p.exists()}")
        if p.exists():
            return p.resolve(), notes

    if SHIP_SMALL_SPLIT_VAL_IMAGES.exists():
        notes.append(
            f"YAML val path is missing/stale; using existing ship_small_split images: {SHIP_SMALL_SPLIT_VAL_IMAGES}"
        )
        return SHIP_SMALL_SPLIT_VAL_IMAGES.resolve(), notes

    raise FileNotFoundError("No existing ship val images directory found.\n" + "\n".join(notes))


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
        return im.size  # width, height


def make_letterbox_rgb(image_path: Path, size: int = IMG_SIZE) -> tuple[np.ndarray, dict]:
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

    rgb = np.ascontiguousarray(np.asarray(canvas))
    meta = {
        "orig_w": orig_w,
        "orig_h": orig_h,
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
    }
    return rgb, meta


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


def read_gt_boxes(label_path: Path, img_w: int, img_h: int, target_cls: int) -> tuple[list[list[float]], int, int]:
    boxes = []
    skipped_non_target = 0
    skipped_bad = 0

    if not label_path.exists():
        return boxes, skipped_non_target, skipped_bad

    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            skipped_bad += 1
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:])
        except Exception:
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


def evaluate_predictions(predictions: list[dict], gt_by_image: dict[str, list[list[float]]]) -> dict:
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

        if best_iou >= MATCH_IOU and not matched[image_key][best_idx]:
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
    ap50 = compute_ap_from_pr(recalls, precisions)

    return {
        "gt_count": total_gt,
        "prediction_count": len(preds),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "ap50": ap50,
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
    """
    Run backbone/neck and manually decode a fused YOLOESegment head.

    The checkpoint inspected here has head.is_fused=True and cv3 outputs 3 fixed class channels.
    Global4 set_classes still records names/pe/nc, but the fused conv cannot emit bridge scores
    without rebuilding/unfusing the head, which this probe intentionally does not do.
    """
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
            preds = {
                "boxes": torch.cat(boxes, dim=-1),
                "scores": scores,
                "feats": feats,
            }
            dbox = m._get_decode_boxes(preds)
            raw = torch.cat((dbox, scores.sigmoid()), dim=1)
            return raw, int(scores.shape[1])

        out = m(out)
        y.append(out if m.i in inner_model.save else None)

    raise RuntimeError("YOLOESegment head was not reached.")


def collect_predictions(inner_model, image_paths: list[Path]) -> list[dict]:
    from ultralytics.utils import nms

    log("")
    log("[PREDICT]")
    log(
        f"direct raw fused YOLOESegModel inference; letterbox each image to {IMG_SIZE}x{IMG_SIZE}; "
        f"images={len(image_paths)} batch={BATCH} device={DEVICE} conf={PRED_CONF} iou={PRED_IOU} max_det={MAX_DET}"
    )

    inner_model.eval().to(DEVICE)
    metas = {}
    predictions = []
    total_boxes = 0
    total_target_boxes = 0
    score_channels_seen = set()

    with torch.no_grad():
        for start in range(0, len(image_paths), BATCH):
            batch_paths = image_paths[start : start + BATCH]
            batch_imgs = []
            for p in batch_paths:
                arr, meta = make_letterbox_rgb(p, IMG_SIZE)
                batch_imgs.append(arr)
                metas[p.name] = meta

            batch_tensor = tensor_from_rgb_images(batch_imgs).to(DEVICE)
            raw_pred, score_nc = raw_yoloeseg_fused_forward(inner_model, batch_tensor)
            score_channels_seen.add(score_nc)
            dets = nms.non_max_suppression(
                raw_pred,
                conf_thres=PRED_CONF,
                iou_thres=PRED_IOU,
                classes=None,
                agnostic=False,
                max_det=MAX_DET,
                nc=score_nc,
            )

            for image_path, det in zip(batch_paths, dets):
                if det is None or det.shape[0] == 0:
                    continue
                det_np = det.detach().cpu().numpy()
                total_boxes += int(det_np.shape[0])

                for row in det_np:
                    k = int(row[5])
                    if k != TARGET_CLASS_ID:
                        continue
                    total_target_boxes += 1
                    predictions.append(
                        {
                            "image_key": image_path.name,
                            "xyxy": unletterbox_xyxy(row[:4], metas[image_path.name]),
                            "conf": float(row[4]),
                            "cls": k,
                        }
                    )

    log(f"raw fused score channels seen: {sorted(score_channels_seen)}")
    if score_channels_seen == {3}:
        log("NOTE: fused head emits 3 class score channels; class 3 bridge has no score channel without head changes.")
    log(f"total predicted boxes all classes: {total_boxes}")
    log(f"target class predicted boxes ({TARGET_CLASS_ID}={TARGET_CLASS_NAME}): {total_target_boxes}")
    return predictions


def collect_ground_truth(image_paths: list[Path], labels_dir: Path) -> tuple[dict[str, list[list[float]]], dict]:
    gt_by_image = {}
    bad_rows = 0
    non_target_rows = 0
    missing_label_files = 0

    for image_path in image_paths:
        img_w, img_h = read_image_size(image_path)
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing_label_files += 1
        boxes, skipped_non_target, skipped_bad = read_gt_boxes(label_path, img_w, img_h, TARGET_CLASS_ID)
        gt_by_image[image_path.name] = boxes
        non_target_rows += skipped_non_target
        bad_rows += skipped_bad

    stats = {
        "images": len(image_paths),
        "missing_label_files": missing_label_files,
        "bad_or_non_5col_rows": bad_rows,
        "non_target_rows": non_target_rows,
        "target_gt_boxes": sum(len(v) for v in gt_by_image.values()),
    }
    return gt_by_image, stats


def main() -> int:
    setup_local_ultralytics_import()

    log("[IMPORT CHECK]")
    log(f"repo root: {REPO_ROOT.resolve()}")
    log(f"package root inserted first: {PACKAGE_ROOT.resolve()}")
    log(f"expected __init__.py: {PACKAGE_ROOT / 'ultralytics' / '__init__.py'}")
    log("first sys.path entries:")
    for p in sys.path[:5]:
        log(f"  {p}")

    from ultralytics import YOLO
    import ultralytics as ultralytics_pkg

    log(f"ultralytics imported from: {getattr(ultralytics_pkg, '__file__', 'UNKNOWN')}")
    log(f"YOLO class: {YOLO}")

    images_dir, data_notes = resolve_images_dir()
    labels_dir = labels_dir_from_images_dir(images_dir)
    log("")
    log("[DATA]")
    for note in data_notes:
        log(note)
    log(f"images_dir: {images_dir} exists={images_dir.exists()}")
    log(f"labels_dir: {labels_dir} exists={labels_dir.exists()}")
    image_paths = list_images(images_dir)
    log(f"image count: {len(image_paths)}")
    log(f"first image: {image_paths[0]}")

    log("")
    log("[LOAD MODEL]")
    log(f"weights: {WEIGHTS_PATH}")
    model = YOLO(str(WEIGHTS_PATH))
    inner = safe_getattr(model, "model", None)
    before_head = head_state(inner)
    log(f"YOLO wrapper type: {obj_type(model)}")
    log(f"inner model type: {obj_type(inner)}")
    log(f"task before: {json.dumps(get_task_state(model, inner), ensure_ascii=False)}")
    log(f"names before: {json.dumps(get_names_state(model, inner), ensure_ascii=False)}")
    log(f"head before: {json.dumps(before_head, ensure_ascii=False, default=str)}")

    log("")
    log("[GLOBAL4 SET_CLASSES]")
    embeddings, set_target, get_sig, set_sig = set_global4_classes(model, inner)
    after_head = head_state(inner)
    log(f"global4 names: {GLOBAL4_CLASS_NAMES}")
    log(f"get_text_pe signature: {get_sig}")
    log(f"set_classes target: {set_target}")
    log(f"set_classes signature: {set_sig}")
    log(f"embedding summary: {json.dumps(summarize_tensor(embeddings), ensure_ascii=False)}")
    log(f"names after: {json.dumps(get_names_state(model, inner), ensure_ascii=False)}")
    log(f"inner model pe shape if exists: {shape_of(safe_getattr(inner, 'pe', None))}")
    log(f"task after: {json.dumps(get_task_state(model, inner), ensure_ascii=False)}")
    log(f"head after: {json.dumps(after_head, ensure_ascii=False, default=str)}")

    log("")
    log("[GROUND TRUTH]")
    gt_by_image, gt_stats = collect_ground_truth(image_paths, labels_dir)
    log(json.dumps(gt_stats, ensure_ascii=False, indent=2))

    predictions = collect_predictions(inner, image_paths)

    log("")
    log("[BOX-ONLY METRICS]")
    metrics = evaluate_predictions(predictions, gt_by_image)
    log(f"class id: {TARGET_CLASS_ID}")
    log(f"class name: {TARGET_CLASS_NAME}")
    log(f"match IoU threshold: {MATCH_IOU}")
    log(f"GT count: {metrics['gt_count']}")
    log(f"prediction count: {metrics['prediction_count']}")
    log(f"TP: {metrics['tp']}")
    log(f"FP: {metrics['fp']}")
    log(f"FN: {metrics['fn']}")
    log(f"Precision: {metrics['precision']:.6f}")
    log(f"Recall: {metrics['recall']:.6f}")
    log(f"AP50: {metrics['ap50']:.6f}")

    log("")
    log("[CONCLUSION]")
    log(f"best.pt loaded: True")
    log(f"global4 set_classes ok: True")
    log(f"native model.val called: False")
    log(f"model remains YOLOESegModel related: {'YOLOESegModel' in obj_type(inner)}")
    log(f"head remains YOLOESegment: {'YOLOESegment' in after_head['last_layer_type']}")
    log(f"task remains segment: {get_task_state(model, inner)}")
    log(f"last_layer.nc before -> after: {before_head['last_layer.nc']} -> {after_head['last_layer.nc']}")
    log(f"masks ignored: True")
    log(f"box-only ship AP50: {metrics['ap50']:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
