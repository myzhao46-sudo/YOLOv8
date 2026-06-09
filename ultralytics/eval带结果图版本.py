#!/usr/bin/env python
"""
Evaluate YOLOE/YOLO checkpoint and show GT-vs-pred visualization.

This version is based on the previous eval_yoloe_show_random_vis.py, with these changes:
- Visualization titles are English to avoid Chinese font square-garbled text.
- Visualization sample is selected as 2 SAR images + 1 RGB/optical image when possible.
- Visualization layout is similar to:
    Left block:  Ground Truth Labels
    Right block: Model Predictions
  Each block contains the same selected images.
- The visualization is shown with plt.show() and also saved as summary_grid.png.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

_script_dir = os.path.dirname(os.path.abspath(__file__))
for _c in [_script_dir, os.path.join(_script_dir, "ultralytics"), os.path.dirname(_script_dir)]:
    if os.path.isdir(os.path.join(_c, "ultralytics")) and _c not in sys.path:
        sys.path.insert(0, _c)
        break

import yaml
import numpy as np
import torch

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib import gridspec


DEFAULT_MODEL = r"C:\Users\DOCTOR\Desktop\yoloe-v8-s_distill_noreplay_300\dinotest\yoloe_linux_dino_on_freeze22_60ep.pt"
DEFAULT_DATA = r"C:\Users\DOCTOR\Documents\GitHub\YOLOv8\ultralytics\datasets\tank_optical_test.yaml"
DEFAULT_CONF = 0.1
DEFAULT_IOU = 0.5
DEFAULT_IMGSZ = 640
DEFAULT_DEVICE = "0"


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate YOLOE and show random 2 SAR + 1 RGB GT-vs-pred visualization.")

    p.add_argument("--model", default=DEFAULT_MODEL, help="Path to checkpoint .pt")
    p.add_argument("--data", default=DEFAULT_DATA, help="Path to dataset YAML")
    p.add_argument("--labels", default=None, help="Optional explicit label directory. Overrides YAML labels.")

    p.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Prediction confidence threshold")
    p.add_argument("--iou", type=float, default=DEFAULT_IOU, help="IoU threshold for NMS and AP50 matching")
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="Inference image size")
    p.add_argument("--device", default=DEFAULT_DEVICE, help="Device, e.g. cpu or 0")

    p.add_argument("--class-names", default=None, help='Optional YOLOE class names, e.g. \'["ship","tank"]\' or ship,tank')
    p.add_argument("--allow-pseudo-pe", action="store_true", help="Allow random pseudo text embeddings if real text PE fails. Not recommended.")

    p.add_argument("--no-group-by-prefix", action="store_true", help="Disable per-prefix group metrics.")
    p.add_argument("--verbose", action="store_true", help="Print extra debug info.")

    p.add_argument("--vis-seed", type=int, default=None, help="Optional random seed. Default None means different random images every run.")
    p.add_argument("--vis-sar-num", type=int, default=2, help="Number of SAR images to visualize.")
    p.add_argument("--vis-rgb-num", type=int, default=1, help="Number of RGB/optical images to visualize.")
    p.add_argument("--vis-dir", default=None, help="Directory to save summary_grid.png. Default: <data_yaml_dir>/eval_vis_show")
    p.add_argument("--no-show", action="store_true", help="Do not call plt.show(). The grid image will still be saved.")
    p.add_argument("--max-pred-draw", type=int, default=120, help="Max predicted boxes to draw per image.")

    return p.parse_args()


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
    model_lookup = {normalize_name(n): i for i, n in model_names.items()}

    aliases = {
        "boat": "ship",
        "vessel": "ship",
        "ships": "ship",
        "oil tank": "tank",
        "oiltank": "tank",
        "storage tank": "tank",
        "circular storage tank": "tank",
        "fuel storage tank": "tank",
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


def _safe_resolve(p: Path) -> Path:
    s = str(p)
    if len(s) >= 2 and s[1] == ":":
        return p
    return p.resolve()


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
        c2 = _safe_resolve(c)
        if c2.exists():
            return c2

    if must_exist:
        raise FileNotFoundError(f"Cannot find path for YAML key '{key}': {raw}. Tried: {candidates}")

    return _safe_resolve(candidates[0])


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

    image_str = str(image_dir)
    if "images" in image_str:
        candidate = Path(image_str.replace("images", "labels", 1))
        if candidate.exists():
            return candidate

    raise FileNotFoundError("Cannot infer label directory. Add `labels: ...` to YAML or pass --labels explicitly.")


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


def load_checkpoint_names(model_path: str | Path, fallback: Optional[List[str]] = None) -> List[str]:
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
        return [str(names[i]) for i in sorted(names)]

    return ["ship", "harbor", "tank"]


def init_yoloe(model, class_names: List[str], allow_pseudo_pe: bool = False):
    inner = model.model
    nc = len(class_names)

    try:
        last = list(inner.model)[-1]
        embed_dim = getattr(last, "embed", 512)
    except Exception:
        embed_dim = 512

    print(f"  YOLOE init class_names={class_names}")
    print(f"  nc={nc}, embed_dim={embed_dim}")

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

    if not allow_pseudo_pe:
        raise RuntimeError(
            "Cannot initialize YOLOE with real text embeddings. Strategy A/B failed.\n"
            "Do NOT evaluate with random pseudo embeddings unless this is only a smoke test.\n"
            "Fix mobileclip/text model in the environment, or pass --allow-pseudo-pe only for debugging."
        )

    print("  Strategy C: pseudo-embeddings. WARNING: AP will NOT be reliable.")
    rng = torch.Generator().manual_seed(42)
    pe = torch.randn(1, nc, embed_dim, generator=rng)
    pe = pe / pe.norm(dim=-1, keepdim=True)
    inner.set_classes(class_names, pe)
    print(f"    set_classes with pseudo-pe OK. Names: {dict(model.names)}")
    return model


def read_labels(path: str | Path, remap: Optional[Dict[int, int]] = None):
    boxes = []

    if not os.path.exists(path):
        return boxes

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            p = line.strip().split()

            if len(p) < 5:
                continue

            try:
                raw_cls = int(float(p[0]))
            except ValueError:
                print(f"  WARNING: invalid class id in {path}:{line_no}: {p[0]}")
                continue

            if remap is not None:
                if raw_cls not in remap:
                    continue
                cls = remap[raw_cls]
            else:
                cls = raw_cls

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

    tp = 0
    fp = 0
    det = []

    for p in preds:
        best_iou = 0.0
        best_j = -1

        for j, g in enumerate(gts):
            if gt_matched[j] or p["c"] != g["c"]:
                continue

            v = iou(p["b"], g["b"])
            if v > best_iou:
                best_iou = v
                best_j = j

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

    if num_gt == 0 or not flags:
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
    s = stem.lower()

    if s.startswith("sar_") or s.startswith("sar-"):
        return "sar"

    if s.startswith("rgb_") or s.startswith("rgb-") or s.startswith("rgbtank_") or s.startswith("rgbtank-"):
        return "rgb"

    if s.startswith("optical_") or s.startswith("optical-"):
        return "rgb"

    if s.startswith("ir_") or s.startswith("ir-") or s.startswith("tir_") or s.startswith("tir-"):
        return "ir"

    if "sar" in s:
        return "sar"
    if "rgb" in s or "optical" in s:
        return "rgb"
    if "infra" in s or "thermal" in s or "tir" in s:
        return "ir"

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

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt = 0
    aps = []

    for cls_id in sorted(ev_classes):
        name = class_names.get(cls_id, str(cls_id))
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


def select_vis_images(images: List[Path], sar_num: int = 2, rgb_num: int = 1, seed: Optional[int] = None) -> set[Path]:
    """
    Select visualization images using the original filename-based modality logic.

    Important:
      - This version does NOT infer modality from file suffix.
      - SAR/RGB are still recognized by modality_from_stem(), e.g. sar_, rgb_, rgbtank_, optical_, etc.
      - By default, seed=None means a different random set is selected every run.
      - Pass --vis-seed 42 only when you want reproducible visualization samples.
    """
    rng = random.Random(time.time_ns()) if seed is None else random.Random(seed)

    sar_images = [p for p in images if modality_from_stem(p.stem) == "sar"]
    rgb_images = [p for p in images if modality_from_stem(p.stem) == "rgb"]

    selected: List[Path] = []

    if len(sar_images) >= sar_num:
        selected.extend(rng.sample(sar_images, sar_num))
    else:
        selected.extend(sar_images)
        if sar_num > 0:
            print(f"[VIS] WARNING: requested {sar_num} SAR images, found {len(sar_images)}.")

    if len(rgb_images) >= rgb_num:
        selected.extend(rng.sample(rgb_images, rgb_num))
    else:
        selected.extend(rgb_images)
        if rgb_num > 0:
            print(f"[VIS] WARNING: requested {rgb_num} RGB images, found {len(rgb_images)}.")

    total_needed = sar_num + rgb_num
    remaining = [p for p in images if p not in selected]

    while len(selected) < min(total_needed, len(images)) and remaining:
        x = rng.choice(remaining)
        selected.append(x)
        remaining.remove(x)

    print("\n[VIS] Selected visualization images:")
    for p in selected:
        print(f"  - {p.name}  ({modality_from_stem(p.stem)})")

    return set(selected)

def _load_font(size: int = 14):
    candidates = [
        "arial.ttf",
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]

    for fp in candidates:
        try:
            return ImageFont.truetype(fp, size=size)
        except Exception:
            pass

    return ImageFont.load_default()


def _box_to_pixels(box, width: int, height: int):
    x1, y1, x2, y2 = clip_xyxy(box)
    return (
        int(round(x1 * width)),
        int(round(y1 * height)),
        int(round(x2 * width)),
        int(round(y2 * height)),
    )


def draw_boxes_on_image(image_path: Path, boxes: List[dict], class_names: Dict[int, str], is_prediction: bool, max_pred_draw: int = 120) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _load_font(14)

    w, h = img.size

    if is_prediction:
        box_color = (0, 90, 255)
        text_bg = (0, 90, 255)
        text_color = (255, 255, 255)
    else:
        box_color = (255, 80, 40)
        text_bg = (255, 255, 255)
        text_color = (160, 30, 20)

    draw_boxes = list(boxes)
    if is_prediction:
        draw_boxes = sorted(draw_boxes, key=lambda x: x.get("s", 0.0), reverse=True)[:max_pred_draw]

    for item in draw_boxes:
        cls_id = int(item["c"])
        name = class_names.get(cls_id, str(cls_id))
        x1, y1, x2, y2 = _box_to_pixels(item["b"], w, h)

        line_width = max(2, int(round(min(w, h) / 350)))
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=line_width)

        if is_prediction:
            score = float(item.get("s", 0.0))
            text = f"{name} {score:.2f}"
        else:
            text = name

        try:
            bbox = draw.textbbox((x1, y1), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except Exception:
            tw, th = draw.textsize(text, font=font)

        tx1 = x1
        ty1 = max(0, y1 - th - 3)
        tx2 = min(w, tx1 + tw + 4)
        ty2 = min(h, ty1 + th + 3)

        draw.rectangle([tx1, ty1, tx2, ty2], fill=text_bg)
        draw.text((tx1 + 2, ty1 + 1), text, fill=text_color, font=font)

    return img


def make_visualization_grid(records: List[dict], class_names: Dict[int, str], out_dir: Path, no_show: bool, max_pred_draw: int):
    if not records:
        print("  WARNING: no records to visualize.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    def sort_key(rec):
        stem = rec["image_path"].stem
        m = modality_from_stem(stem)
        priority = {"sar": 0, "rgb": 1, "ir": 2, "unknown": 3}
        return priority.get(m, 9), rec["image_path"].name

    records = sorted(records, key=sort_key)

    n = len(records)
    fig = plt.figure(figsize=(4.2 * n * 2, 4.8))
    outer = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.05)

    gt_grid = gridspec.GridSpecFromSubplotSpec(1, n, subplot_spec=outer[0], wspace=0.02)
    pred_grid = gridspec.GridSpecFromSubplotSpec(1, n, subplot_spec=outer[1], wspace=0.02)

    gt_axes = [fig.add_subplot(gt_grid[0, i]) for i in range(n)]
    pred_axes = [fig.add_subplot(pred_grid[0, i]) for i in range(n)]

    for col_idx, rec in enumerate(records):
        image_path = rec["image_path"]
        gt_boxes = rec["gt_boxes"]
        pred_boxes = rec["pred_boxes"]

        gt_img = draw_boxes_on_image(image_path, gt_boxes, class_names, is_prediction=False, max_pred_draw=max_pred_draw)
        pred_img = draw_boxes_on_image(image_path, pred_boxes, class_names, is_prediction=True, max_pred_draw=max_pred_draw)

        gt_axes[col_idx].imshow(gt_img)
        gt_axes[col_idx].axis("off")
        gt_axes[col_idx].set_title(image_path.name, fontsize=8)

        pred_axes[col_idx].imshow(pred_img)
        pred_axes[col_idx].axis("off")
        pred_axes[col_idx].set_title(image_path.name, fontsize=8)

    fig.text(0.255, 0.98, "Ground Truth Labels", ha="center", va="top", fontsize=16)
    fig.text(0.755, 0.98, "Model Predictions", ha="center", va="top", fontsize=16)

    plt.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.03)

    out_path = out_dir / "summary_grid.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"\n[VIS] Saved visualization grid to: {out_path}")

    if not no_show:
        print("[VIS] Calling plt.show() now.")
        try:
            plt.show()
        except Exception as e:
            print(f"[VIS] plt.show() failed: {e}")
            print("[VIS] The image has still been saved to summary_grid.png.")

    plt.close(fig)


def predict_one_image(model, image_path: Path, args, eval_classes):
    results = model.predict(
        str(image_path),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
        save=False,
    )

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

    return pred_boxes


def main():
    args = parse_args()

    print("=" * 90)
    print("YOLOE Evaluation + SAR/RGB Visualization")
    print("=" * 90)
    print(f"  Model:  {args.model}")
    print(f"  Data:   {args.data}")
    print(f"  Labels: {args.labels or '(from YAML or images->labels fallback)'}")
    print(f"  Conf={args.conf}  IoU={args.iou}  ImgSz={args.imgsz}  Device={args.device}")
    print(f"  Vis SAR num={args.vis_sar_num}  Vis RGB num={args.vis_rgb_num}  Vis seed={args.vis_seed}")

    print("\n[1] Dataset ...")
    dcfg = load_yaml(args.data)
    dataset_names = parse_names(dcfg.get("names", {}))
    print(f"  Dataset names: {dataset_names}")

    image_dir = resolve_val_path(dcfg, args.data)
    label_dir = resolve_label_path(dcfg, args.data, image_dir, explicit_labels=args.labels)

    print(f"  Images: {image_dir}")
    print(f"  Labels: {label_dir}")

    print("\n[2] Loading model ...")
    forced_names = parse_class_names_arg(args.class_names)
    class_list = load_checkpoint_names(args.model, fallback=forced_names)
    print(f"  Class names for YOLOE set_classes: {class_list}")

    from ultralytics import YOLO

    model = YOLO(args.model)
    print(f"  Loaded task: {model.task}, initial names: {dict(model.names)}")

    try:
        model = init_yoloe(model, class_list, allow_pseudo_pe=args.allow_pseudo_pe)
    except Exception as e:
        if hasattr(model.model, "get_text_pe") or "YOLOE" in type(model.model).__name__:
            raise
        print(f"  Non-YOLOE text init skipped: {e}")

    final_names = {int(k): str(v) for k, v in dict(model.names).items()}
    print(f"  Final model names: {final_names}")

    print("\n[3] Class remap ...")
    remap = build_class_remap(dataset_names, final_names)
    print(f"  Remap: {remap}")

    for d, m in remap.items():
        print(f"    label[{d}] '{dataset_names[d]}' -> model[{m}] '{final_names[m]}'")

    eval_classes = set(remap.values())

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images = sorted(f for f in image_dir.iterdir() if f.suffix.lower() in exts)

    print(f"\n[4] Images found: {len(images)}")
    if not images:
        print("ERROR: no images found.")
        sys.exit(1)

    label_indices = set()
    missing_label_count = 0

    for f in images[: min(50, len(images))]:
        lf = label_dir / (f.stem + ".txt")
        if not lf.exists():
            missing_label_count += 1
            continue

        with open(lf, "r", encoding="utf-8") as fp:
            for line in fp:
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

    vis_image_set = select_vis_images(
        images=images,
        sar_num=args.vis_sar_num,
        rgb_num=args.vis_rgb_num,
        seed=args.vis_seed,
    )
    vis_records = []

    print("\n[5] Evaluating ...")
    overall = init_stats()
    by_group = defaultdict(init_stats)

    for idx, image_path in enumerate(images, start=1):
        label_path = label_dir / (image_path.stem + ".txt")
        raw_gts = read_labels(label_path, remap=remap)

        gt_boxes = []
        for box_tuple in raw_gts:
            cls, x1, y1, x2, y2 = label_to_xyxy(box_tuple)
            gt_boxes.append({"c": int(cls), "b": clip_xyxy((x1, y1, x2, y2))})

        try:
            pred_boxes = predict_one_image(model=model, image_path=image_path, args=args, eval_classes=eval_classes)
        except Exception as e:
            if idx == 1:
                print(f"\n  FATAL on first image: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)

            print(f"  WARNING: prediction failed on {image_path}: {e}")
            continue

        update_stats(overall, eval_classes, pred_boxes, gt_boxes, args.iou)

        if not args.no_group_by_prefix:
            group = modality_from_stem(image_path.stem)
            update_stats(by_group[group], eval_classes, pred_boxes, gt_boxes, args.iou)

        if image_path in vis_image_set:
            vis_records.append({"image_path": image_path, "gt_boxes": gt_boxes, "pred_boxes": pred_boxes})

        if idx % 50 == 0 or idx == len(images):
            print(f"  {idx}/{len(images)} ...")

    print_result_table("OVERALL RESULTS", overall, final_names, eval_classes)

    if not args.no_group_by_prefix:
        for group in sorted(by_group):
            print_result_table(f"GROUP RESULTS: {group}", by_group[group], final_names, eval_classes)

    print("\n[6] Visualization ...")

    if args.vis_dir:
        out_dir = Path(args.vis_dir)
    else:
        out_dir = Path(args.data).resolve().parent / "eval_vis_show"

    make_visualization_grid(
        records=vis_records,
        class_names=final_names,
        out_dir=out_dir,
        no_show=args.no_show,
        max_pred_draw=args.max_pred_draw,
    )

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)
    print(f"  Remap: {remap}")
    print(f"  NOTE: P/R/F1 are computed at conf={args.conf}.")
    print(f"  NOTE: AP50 uses predictions retained after conf={args.conf}.")
    print("  If AP is extremely low, also run with --conf 0.001.")
    print("  Visualization is randomly sampled each run by default. Pass --vis-seed 42 for reproducible samples.")
    print("  If YOLOE Strategy C pseudo-embeddings were used, this evaluation is NOT reliable.")


if __name__ == "__main__":
    main()