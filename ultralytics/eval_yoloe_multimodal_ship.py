#!/usr/bin/env python
"""
Evaluate a YOLOE/YOLO checkpoint on a multimodal ship test dataset.

Default dataset:
  images: E:/YOLODATA/Ship/images/test
  labels: E:/YOLODATA/Ship/labels/test
  names:  {0: ship}

This script supports:
  - YOLOE text-embedding initialization via get_text_pe() or build_text_model()
  - 5-column YOLO labels: cls cx cy w h
  - 9-column polygon/DOTA-like labels: cls x1 y1 x2 y2 x3 y3 x4 y4
  - dataset class id -> model class id remapping by class name
  - optional per-modality metrics grouped by filename prefix, e.g. ir_1_4.jpg -> ir

Important:
  For reliable YOLOE evaluation, do NOT use pseudo text embeddings.
  If Strategy A/B fails, install the required text model/mobileclip or run in the training environment.

Example:
  python eval_yoloe_multimodal_ship.py ^
    --model C:/path/to/weights/best.pt ^
    --data E:/YOLODATA/Ship/ship_multimodal_test.yaml ^
    --conf 0.1 --iou 0.5 --imgsz 640 --device 0

PowerShell:
  python eval_yoloe_multimodal_ship.py `
    --model C:/path/to/weights/best.pt `
    --data E:/YOLODATA/Ship/ship_multimodal_test.yaml `
    --conf 0.1 --iou 0.5 --imgsz 640 --device 0

If your checkpoint names are wrong or stale, explicitly set:
  --class-names '["ship","tank"]'
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

# Try to make local Ultralytics import work when this script is placed in YOLOv8/ or YOLOv8/ultralytics.
_script_dir = os.path.dirname(os.path.abspath(__file__))
for _c in [_script_dir, os.path.join(_script_dir, "ultralytics"), os.path.dirname(_script_dir)]:
    if os.path.isdir(os.path.join(_c, "ultralytics")) and _c not in sys.path:
        sys.path.insert(0, _c)
        break

import yaml
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


DEFAULT_MODEL = r"C:\Users\DOCTOR\Documents\GitHub\YOLOv8\ultralytics\datasets\best.pt"
# DEFAULT_MODEL = r"C:\Users\DOCTOR\Desktop\yoloe-v8-s_distill_noreplay_300\dinotest\yoloe_linux_dino_on_freeze22_60ep.pt"
DEFAULT_DATA = r"C:\Users\DOCTOR\Documents\GitHub\YOLOv8\ultralytics\datasets\tank_optical_test.yaml" 
DEFAULT_CONF = 0.1
DEFAULT_IOU = 0.5
DEFAULT_IMGSZ = 640
DEFAULT_DEVICE = "0"


# =====================================================================
# Args
# =====================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate YOLOE on multimodal ship test set.")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Path to checkpoint .pt")
    p.add_argument("--data", default=DEFAULT_DATA, help="Path to dataset YAML")
    p.add_argument("--labels", default=None, help="Optional explicit label directory. Overrides YAML labels.")
    p.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Prediction confidence threshold")
    p.add_argument("--iou", type=float, default=DEFAULT_IOU, help="IoU threshold for NMS and AP50 matching")
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="Inference image size")
    p.add_argument("--device", default=DEFAULT_DEVICE, help="Device, e.g. cpu or 0")
    p.add_argument("--class-names", default=None, help='Optional YOLOE class names, e.g. \'["ship","tank"]\' or ship,tank')
    p.add_argument("--allow-pseudo-pe", action="store_true", help="Allow random pseudo text embeddings if real text PE fails. Not recommended.")
    p.add_argument("--no-group-by-prefix", action="store_true", help="Disable per-modality grouping by filename prefix.")
    p.add_argument("--verbose", action="store_true", help="Print additional debug details.")

    # Visualization options. These create side-by-side GT vs prediction images.
    p.add_argument("--save-vis", action="store_true", help="Save GT/prediction visualization images.")
    p.add_argument("--vis-dir", default=None, help="Directory to save visualization images. Default: <data_yaml_dir>/eval_vis")
    p.add_argument("--vis-num", type=int, default=16, help="Maximum number of evaluated images to visualize.")
    p.add_argument("--vis-cols", type=int, default=2, help="Number of image pairs per row in the summary grid.")
    p.add_argument("--vis-thumb-width", type=int, default=900, help="Width of each GT/pred side-by-side pair in the summary grid.")
    return p.parse_args()


# =====================================================================
# YAML / path / remap helpers
# =====================================================================
def load_yaml(p: str | Path):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_names(raw) -> Dict[int, str]:
    if isinstance(raw, list):
        return {i: str(n) for i, n in enumerate(raw)}
    if isinstance(raw, dict):
        return {int(k): str(v) for k, v in raw.items()}
    raise ValueError(f"Cannot parse names: {raw}")


def normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("_", " ").replace("-", " ")


def build_class_remap(dataset_names: Dict[int, str], model_names: Dict[int, str]) -> Dict[int, int]:
    """Map dataset class id to model class id by class name."""
    model_lookup = {normalize_name(n): i for i, n in model_names.items()}

    # A small alias table for common naming variants.
    aliases = {
        "boat": "ship",
        "vessel": "ship",
        "ships": "ship",
        "oil tank": "tank",
        "oiltank": "tank",
        "storage tank": "tank",
    }

    remap = {}
    for dataset_id, dataset_name in dataset_names.items():
        key = normalize_name(dataset_name)
        key = aliases.get(key, key)
        if key not in model_lookup:
            raise ValueError(
                f"Dataset class '{dataset_name}'(idx {dataset_id}) not found in model names {model_names}. "
                f"Use --class-names to force correct YOLOE names, e.g. --class-names '[\"ship\",\"tank\"]'."
            )
        remap[dataset_id] = model_lookup[key]
    return remap


def _resolve_path_from_yaml(cfg: dict, yaml_path: str | Path, key: str, must_exist: bool = True) -> Path:
    yaml_dir = Path(yaml_path).resolve().parent
    raw = cfg.get(key)
    if not raw:
        raise ValueError(f"No '{key}' in YAML")
    p = Path(raw)

    candidates = []
    if p.is_absolute():
        candidates.append(p)
    base = cfg.get("path", "")
    if base:
        b = Path(base)
        if not b.is_absolute():
            b = yaml_dir / b
        candidates.append(b / raw)
    candidates.append(yaml_dir / raw)
    candidates.append(p)

    for c in candidates:
        c = c.resolve() if not str(c).startswith(("E:", "C:", "D:")) else c
        if c.exists():
            return c

    if must_exist:
        raise FileNotFoundError(f"Cannot find path for YAML key '{key}': {raw}. Tried: {candidates}")
    return candidates[0]


def resolve_val_path(cfg: dict, yaml_path: str | Path) -> Path:
    return _resolve_path_from_yaml(cfg, yaml_path, "val", must_exist=True)


def resolve_label_path(cfg: dict, yaml_path: str | Path, image_dir: Path, explicit_labels: Optional[str] = None) -> Path:
    if explicit_labels:
        p = Path(explicit_labels)
        if p.exists():
            return p
        raise FileNotFoundError(f"Explicit labels path does not exist: {p}")

    if cfg.get("labels"):
        return _resolve_path_from_yaml(cfg, yaml_path, "labels", must_exist=True)

    # Standard YOLO layout fallback: .../images/test -> .../labels/test
    image_str = str(image_dir)
    if "images" in image_str:
        candidate = Path(image_str.replace("images", "labels", 1))
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Cannot infer label directory. Add `labels: ...` to YAML or pass --labels explicitly."
    )


def parse_class_names_arg(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    value = value.strip()
    try:
        obj = json.loads(value)
        if isinstance(obj, list):
            return [str(x) for x in obj]
    except Exception:
        pass
    try:
        obj = ast.literal_eval(value)
        if isinstance(obj, (list, tuple)):
            return [str(x) for x in obj]
    except Exception:
        pass
    return [x.strip() for x in value.split(",") if x.strip()]


# =====================================================================
# YOLOE text embedding initialization
# =====================================================================
def load_checkpoint_names(model_path: str | Path, fallback: Optional[List[str]] = None) -> List[str]:
    """Read names from checkpoint. If not available, use fallback or a safe default."""
    if fallback:
        return fallback
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    names = None
    if isinstance(ckpt, dict):
        for key in ("model", "ema"):
            m = ckpt.get(key)
            if m is not None and hasattr(m, "names"):
                names = dict(m.names)
                break
    elif hasattr(ckpt, "names"):
        names = dict(ckpt.names)
    del ckpt

    if names:
        return [names[i] for i in sorted(names)]
    return ["ship", "harbor", "tank"]


def init_yoloe(model, class_names: List[str], allow_pseudo_pe: bool = False):
    """Initialize YOLOE model with text embeddings so inference works."""
    inner = model.model
    nc = len(class_names)

    try:
        last = list(inner.model)[-1]
        embed_dim = getattr(last, "embed", 512)
    except Exception:
        embed_dim = 512

    print(f"  YOLOE init class_names={class_names}")
    print(f"  nc={nc}, embed_dim={embed_dim}")

    # Strategy A: use model's own get_text_pe().
    print(f"  Strategy A: inner.get_text_pe({class_names}) ...")
    try:
        pe = inner.get_text_pe(class_names)
        print(f"    get_text_pe -> shape={tuple(pe.shape)}, dtype={pe.dtype}")
        assert pe.ndim == 3, f"Expected 3D PE, got {pe.ndim}D"
        inner.set_classes(class_names, pe)
        print(f"    set_classes OK. Names: {dict(model.names)}")
        return model
    except Exception as e:
        print(f"    Failed: {e}")

    # Strategy B: build_text_model directly.
    print("  Strategy B: build_text_model ...")
    try:
        from ultralytics.nn.text_model import build_text_model

        device = next(inner.model.parameters()).device
        text_model = build_text_model("mobileclip:blt", device=device)
        text_token = text_model.tokenize(class_names)
        txt_feats = text_model.encode_text(text_token).detach()
        pe = txt_feats.reshape(1, nc, embed_dim)
        print(f"    pe shape={tuple(pe.shape)}")
        inner.set_classes(class_names, pe)
        print(f"    set_classes OK. Names: {dict(model.names)}")
        return model
    except Exception as e:
        print(f"    Failed: {e}")

    # Strategy C: pseudo-embeddings. Disabled by default because AP is unreliable.
    if not allow_pseudo_pe:
        raise RuntimeError(
            "Cannot initialize YOLOE with real text embeddings. Strategy A/B failed.\n"
            "Do NOT evaluate with random pseudo embeddings unless this is only a code smoke test.\n"
            "Fix mobileclip/text model in the environment, or pass --allow-pseudo-pe only for debugging."
        )

    print("  Strategy C: pseudo-embeddings (RESULTS WILL BE DEGRADED!) ...")
    rng = torch.Generator().manual_seed(42)
    pe = torch.randn(1, nc, embed_dim, generator=rng)
    pe = pe / pe.norm(dim=-1, keepdim=True)
    inner.set_classes(class_names, pe)
    print(f"    set_classes with pseudo-pe OK. Names: {dict(model.names)}")
    print("    WARNING: Random embeddings are used. AP will NOT reflect true model capability.")
    return model


# =====================================================================
# Labels / matching / AP
# =====================================================================
def read_labels(path: str | Path, remap: Optional[Dict[int, int]] = None):
    """Read labels supporting:
      - 5-col YOLO: cls cx cy w h
      - 9-col polygon/DOTA: cls x1 y1 x2 y2 x3 y3 x4 y4

    Returns tuples:
      (cls, x1_norm, y1_norm, x2_norm, y2_norm, "xyxy")
      or
      (cls, cx_norm, cy_norm, w_norm, h_norm, "xywh")
    """
    boxes = []
    if not os.path.exists(path):
        return boxes

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            p = line.strip().split()
            if len(p) < 5:
                continue
            try:
                raw = int(float(p[0]))
            except ValueError:
                print(f"  WARNING: invalid class id in {path}:{line_no}: {p[0]}")
                continue

            if remap is not None:
                if raw not in remap:
                    continue
                cls = remap[raw]
            else:
                cls = raw

            try:
                vals = list(map(float, p[1:]))
            except ValueError:
                print(f"  WARNING: invalid numeric label in {path}:{line_no}: {line.strip()}")
                continue

            if len(p) >= 9:
                coords = vals[:8]
                xs = coords[0::2]
                ys = coords[1::2]
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)
                boxes.append((cls, x1, y1, x2, y2, "xyxy"))
            else:
                cx, cy, w, h = vals[:4]
                boxes.append((cls, cx, cy, w, h, "xywh"))
    return boxes


def label_to_xyxy(box_tuple):
    if box_tuple[-1] == "xyxy":
        cls, x1, y1, x2, y2 = box_tuple[:5]
        return cls, x1, y1, x2, y2
    cls, cx, cy, w, h = box_tuple[:5]
    return cls, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def clip_xyxy(b):
    x1, y1, x2, y2 = b
    return (
        max(0.0, min(1.0, x1)),
        max(0.0, min(1.0, y1)),
        max(0.0, min(1.0, x2)),
        max(0.0, min(1.0, y2)),
    )


def iou(a, b):
    ax1, ay1, ax2, ay2 = clip_xyxy(a)
    bx1, by1, bx2, by2 = clip_xyxy(b)
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match(preds, gts, thr: float, target_class: Optional[int] = None):
    if target_class is not None:
        preds = [p for p in preds if p["c"] == target_class]
        gts = [g for g in gts if g["c"] == target_class]

    preds = sorted(preds, key=lambda x: x["s"], reverse=True)
    gt_matched = [False] * len(gts)
    tp = fp = 0
    det = []

    for p in preds:
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if gt_matched[j] or p["c"] != g["c"]:
                continue
            v = iou(p["b"], g["b"])
            if v > best_iou:
                best_iou, best_j = v, j

        if best_iou >= thr and best_j >= 0:
            tp += 1
            gt_matched[best_j] = True
            det.append((1, p["s"]))
        else:
            fp += 1
            det.append((0, p["s"]))

    fn = sum(1 for m in gt_matched if not m)
    return tp, fp, fn, det


def compute_ap(flags: Iterable[int], confs: Iterable[float], num_gt: int) -> float:
    flags = list(flags)
    confs = list(confs)
    if num_gt == 0:
        return 0.0
    if not flags:
        return 0.0

    order = np.argsort(-np.array(confs))
    t = np.array(flags)[order]
    cum_tp = np.cumsum(t)
    cum_fp = np.cumsum(1 - t)
    recall = np.concatenate(([0.0], cum_tp / num_gt, [1.0]))
    precision = np.concatenate(([1.0], cum_tp / (cum_tp + cum_fp), [0.0]))

    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    idx = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[idx + 1] - recall[idx]) * precision[idx + 1]))


def modality_from_stem(stem: str) -> str:
    """Group multimodal data by filename prefix, e.g. ir_1_4 -> ir."""
    if "_" in stem:
        return stem.split("_", 1)[0].lower()
    if "-" in stem:
        return stem.split("-", 1)[0].lower()
    return "unknown"


def init_stats():
    return {
        "flags": defaultdict(list),
        "confs": defaultdict(list),
        "gt": defaultdict(int),
        "tp": defaultdict(int),
        "fp": defaultdict(int),
        "fn": defaultdict(int),
        "images": 0,
    }


def update_stats(stats, ev_classes, preds, gts, iou_thr):
    stats["images"] += 1
    for g in gts:
        stats["gt"][g["c"]] += 1

    for cls_id in ev_classes:
        tp, fp, fn, det = match(preds, gts, iou_thr, cls_id)
        stats["tp"][cls_id] += tp
        stats["fp"][cls_id] += fp
        stats["fn"][cls_id] += fn
        for flag, conf in det:
            stats["flags"][cls_id].append(flag)
            stats["confs"][cls_id].append(conf)


def print_result_table(title: str, stats, class_names: Dict[int, str], ev_classes):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    print(f"Images: {stats['images']}")
    hdr = f"{'Class':>15}|{'GT':>6}|{'TP':>6}|{'FP':>6}|{'FN':>6}|{'P':>8}|{'R':>8}|{'F1':>8}|{'AP50':>8}"
    print(hdr)
    print("-" * len(hdr))

    total_tp = total_fp = total_fn = total_gt = 0
    aps = []

    for cls_id in sorted(ev_classes):
        name = class_names[cls_id]
        gt = stats["gt"][cls_id]
        tp = stats["tp"][cls_id]
        fp = stats["fp"][cls_id]
        fn = stats["fn"][cls_id]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        ap = compute_ap(stats["flags"][cls_id], stats["confs"][cls_id], gt)

        print(f"{name:>15}|{gt:6}|{tp:6}|{fp:6}|{fn:6}|{precision:8.4f}|{recall:8.4f}|{f1:8.4f}|{ap:8.4f}")

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_gt += gt
        aps.append(ap)

    op = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    or_ = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    of = 2 * op * or_ / (op + or_) if op + or_ else 0.0
    mean_ap = float(np.mean(aps)) if aps else 0.0

    print("-" * len(hdr))
    print(f"{'ALL':>15}|{total_gt:6}|{total_tp:6}|{total_fp:6}|{total_fn:6}|{op:8.4f}|{or_:8.4f}|{of:8.4f}|{mean_ap:8.4f}")
    print(f"\n  mAP50 = {mean_ap:.4f} ({mean_ap * 100:.2f}%)")
    return mean_ap


# =====================================================================
# Visualization
# =====================================================================
def _safe_class_name(class_names: Dict[int, str], cls_id: int) -> str:
    try:
        return str(class_names[int(cls_id)])
    except Exception:
        return str(cls_id)


def _draw_normalized_boxes(
    img: Image.Image,
    boxes: List[dict],
    class_names: Dict[int, str],
    is_pred: bool,
    title: str,
) -> Image.Image:
    """Draw normalized xyxy boxes on a copy of the image."""
    img = img.convert("RGB")
    w, h = img.size
    title_h = max(28, int(h * 0.045))
    canvas = Image.new("RGB", (w, h + title_h), "white")
    canvas.paste(img, (0, title_h))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    draw.text((8, 7), title, fill=(0, 0, 0), font=font)

    # Red for GT, blue for predictions.
    color = (220, 30, 30) if not is_pred else (20, 80, 230)
    fill_color = (255, 245, 245) if not is_pred else (235, 242, 255)
    thickness = max(2, int(round(min(w, h) / 320)))

    for item in boxes:
        cls_id = int(item.get("c", -1))
        x1, y1, x2, y2 = clip_xyxy(item["b"])
        px1 = int(round(x1 * w))
        py1 = int(round(y1 * h)) + title_h
        px2 = int(round(x2 * w))
        py2 = int(round(y2 * h)) + title_h

        if px2 <= px1 or py2 <= py1:
            continue

        for t in range(thickness):
            draw.rectangle((px1 - t, py1 - t, px2 + t, py2 + t), outline=color)

        label = _safe_class_name(class_names, cls_id)
        if is_pred and "s" in item:
            label = f"{label} {float(item['s']):.2f}"

        # Draw a small label background.
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = len(label) * 7, 12
        label_y1 = max(title_h, py1 - th - 4)
        draw.rectangle((px1, label_y1, px1 + tw + 4, label_y1 + th + 4), fill=fill_color, outline=color)
        draw.text((px1 + 2, label_y1 + 2), label, fill=color, font=font)

    return canvas


def _make_side_by_side_visualization(record: dict, class_names: Dict[int, str]) -> Image.Image:
    img = Image.open(record["image_path"]).convert("RGB")
    left = _draw_normalized_boxes(img, record["gts"], class_names, is_pred=False, title="Ground Truth Labels")
    right = _draw_normalized_boxes(img, record["preds"], class_names, is_pred=True, title="Model Predictions")

    w, h = left.size
    gap = max(8, int(w * 0.012))
    canvas = Image.new("RGB", (w * 2 + gap, h), "white")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (w + gap, 0))
    return canvas


def _resize_keep_aspect(img: Image.Image, target_w: int) -> Image.Image:
    if target_w <= 0 or img.width <= target_w:
        return img.copy()
    scale = target_w / img.width
    target_h = max(1, int(round(img.height * scale)))
    return img.resize((target_w, target_h), Image.Resampling.LANCZOS)


def save_visualizations(records: List[dict], class_names: Dict[int, str], vis_dir: Path, cols: int = 2, thumb_width: int = 900):
    """Save per-image GT/pred pairs and one summary grid."""
    if not records:
        print("  No visualization records to save.")
        return

    vis_dir.mkdir(parents=True, exist_ok=True)
    pair_paths = []

    for i, rec in enumerate(records, start=1):
        pair = _make_side_by_side_visualization(rec, class_names)
        out_path = vis_dir / f"vis_{i:03d}_{Path(rec['image_path']).stem}.jpg"
        pair.save(out_path, quality=95)
        pair_paths.append(out_path)

    # Build a compact summary grid from the saved pairs.
    cols = max(1, int(cols))
    thumbs = [_resize_keep_aspect(Image.open(p).convert("RGB"), int(thumb_width)) for p in pair_paths]
    rows = (len(thumbs) + cols - 1) // cols
    gap = 12
    cell_w = max(t.width for t in thumbs)
    cell_h = max(t.height for t in thumbs)

    grid = Image.new("RGB", (cols * cell_w + (cols - 1) * gap, rows * cell_h + (rows - 1) * gap), "white")
    for i, t in enumerate(thumbs):
        r = i // cols
        c = i % cols
        x = c * (cell_w + gap)
        y = r * (cell_h + gap)
        grid.paste(t, (x, y))

    grid_path = vis_dir / "summary_grid.jpg"
    grid.save(grid_path, quality=95)

    print(f"  Saved visualization pairs: {len(pair_paths)}")
    print(f"  Visualization dir: {vis_dir}")
    print(f"  Summary grid: {grid_path}")


# =====================================================================
# Main
# =====================================================================
def main():
    args = parse_args()

    print("=" * 90)
    print("YOLOE Multimodal Ship Evaluation")
    print("=" * 90)
    print(f"  Model:  {args.model}")
    print(f"  Data:   {args.data}")
    print(f"  Labels: {args.labels or '(from YAML or images->labels fallback)'}")
    print(f"  Conf={args.conf}  IoU={args.iou}  ImgSz={args.imgsz}  Device={args.device}")

    # Dataset.
    print("\n[1] Dataset ...")
    dcfg = load_yaml(args.data)
    dataset_names = parse_names(dcfg.get("names", {}))
    print(f"  Dataset names: {dataset_names}")

    image_dir = resolve_val_path(dcfg, args.data)
    label_dir = resolve_label_path(dcfg, args.data, image_dir, explicit_labels=args.labels)
    print(f"  Images: {image_dir}")
    print(f"  Labels: {label_dir}")

    # Model names and YOLO load.
    print("\n[2] Loading model ...")
    forced_names = parse_class_names_arg(args.class_names)
    class_list = load_checkpoint_names(args.model, fallback=forced_names)
    print(f"  Class names for YOLOE set_classes: {class_list}")

    from ultralytics import YOLO
    model = YOLO(args.model)
    print(f"  Loaded task: {model.task}, initial names: {dict(model.names)}")

    # YOLOE text PE init. If the model is non-YOLOE, this may fail; in that case, keep native model names.
    try:
        model = init_yoloe(model, class_list, allow_pseudo_pe=args.allow_pseudo_pe)
    except Exception as e:
        # If this is a standard YOLO model, get_text_pe may not exist. For YOLOE, this should be fixed.
        if hasattr(model.model, "get_text_pe") or "YOLOE" in type(model.model).__name__:
            raise
        print(f"  Non-YOLOE text init skipped: {e}")

    final_names = dict(model.names)
    print(f"  Final model names: {final_names}")

    # Remap dataset class ids to model class ids by names.
    print("\n[3] Class remap ...")
    remap = build_class_remap(dataset_names, final_names)
    print(f"  Remap: {remap}")
    for d, m in remap.items():
        print(f"    label[{d}] '{dataset_names[d]}' -> model[{m}] '{final_names[m]}'")
    eval_classes = set(remap.values())

    # Images.
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = sorted(f for f in image_dir.iterdir() if f.suffix.lower() in exts)
    print(f"\n[4] Images found: {len(images)}")
    if not images:
        print("ERROR: no images found")
        sys.exit(1)

    # Sanity check labels.
    label_indices = set()
    missing_label_count = 0
    for f in images[: min(50, len(images))]:
        lf = label_dir / (f.stem + ".txt")
        if not lf.exists():
            missing_label_count += 1
            continue
        for line in open(lf, encoding="utf-8"):
            p = line.strip().split()
            if p:
                try:
                    label_indices.add(int(float(p[0])))
                except ValueError:
                    pass
    print(f"  Label indices in first samples: {sorted(label_indices)}")
    print(f"  YAML expects: {sorted(dataset_names.keys())}")
    if missing_label_count:
        print(f"  WARNING: {missing_label_count} of first sampled images have no label txt.")
    missing_in_yaml = label_indices - set(dataset_names.keys())
    if missing_in_yaml:
        print(f"  WARNING: labels contain ids not in YAML names: {sorted(missing_in_yaml)}")

    # Eval.
    print("\n[5] Evaluating ...")
    overall = init_stats()
    by_group = defaultdict(init_stats)
    vis_records = []

    for idx, image_path in enumerate(images, start=1):
        label_path = label_dir / (image_path.stem + ".txt")
        raw_gts = read_labels(label_path, remap=remap)

        gt_boxes = []
        for box_tuple in raw_gts:
            cls, x1, y1, x2, y2 = label_to_xyxy(box_tuple)
            gt_boxes.append({"c": cls, "b": clip_xyxy((x1, y1, x2, y2))})

        try:
            results = model.predict(
                str(image_path),
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
                save=False,
            )
        except Exception as e:
            if idx == 1:
                print(f"\n  FATAL on first image: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)
            print(f"  WARNING: prediction failed on {image_path}: {e}")
            continue

        pred_boxes = []
        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                if cls_id not in eval_classes:
                    continue
                conf = float(boxes.conf[i].item())

                if hasattr(boxes, "xyxyn") and boxes.xyxyn is not None and len(boxes.xyxyn) > 0:
                    x1, y1, x2, y2 = boxes.xyxyn[i].tolist()
                else:
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    h, w = r.orig_shape
                    x1, y1, x2, y2 = x1 / w, y1 / h, x2 / w, y2 / h

                pred_boxes.append({"c": cls_id, "s": conf, "b": clip_xyxy((x1, y1, x2, y2))})

        if args.save_vis and len(vis_records) < max(0, args.vis_num):
            vis_records.append({
                "image_path": image_path,
                "gts": gt_boxes,
                "preds": pred_boxes,
            })

        update_stats(overall, eval_classes, pred_boxes, gt_boxes, args.iou)

        if not args.no_group_by_prefix:
            group = modality_from_stem(image_path.stem)
            update_stats(by_group[group], eval_classes, pred_boxes, gt_boxes, args.iou)

        if idx % 50 == 0 or idx == len(images):
            print(f"  {idx}/{len(images)} ...")

    print_result_table("OVERALL RESULTS", overall, final_names, eval_classes)

    if not args.no_group_by_prefix:
        for group in sorted(by_group):
            print_result_table(f"GROUP RESULTS: {group}", by_group[group], final_names, eval_classes)

    if args.save_vis:
        vis_dir = Path(args.vis_dir) if args.vis_dir else Path(args.data).resolve().parent / "eval_vis"
        print("\n[6] Saving visualizations ...")
        save_visualizations(
            vis_records,
            final_names,
            vis_dir=vis_dir,
            cols=args.vis_cols,
            thumb_width=args.vis_thumb_width,
        )

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)
    print(f"  Remap: {remap}")
    print(f"  NOTE: P/R/F1 are computed at conf={args.conf}. AP50 uses predictions retained after this conf threshold.")
    print("  If YOLOE Strategy C pseudo-embeddings were used, this evaluation is NOT reliable.")


if __name__ == "__main__":
    main()
