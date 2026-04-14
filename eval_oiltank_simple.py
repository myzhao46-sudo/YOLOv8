#!/usr/bin/env python
"""
Simple, read-only evaluation script for one class (oiltank) on your local dataset.

What it does:
1) Try official Ultralytics `model.val(...)` once.
2) If that fails (common for seg-model + detect-label mismatch), fallback to manual bbox evaluation.

No dataset files are modified.
"""

from __future__ import annotations

import argparse
import importlib.util
import tempfile
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# Prefer local repo package over site-packages
THIS_DIR = Path(__file__).resolve().parent
LOCAL_ULTRALYTICS_ROOT = THIS_DIR / "ultralytics"
if LOCAL_ULTRALYTICS_ROOT.exists():
    import sys

    sys.path.insert(0, str(LOCAL_ULTRALYTICS_ROOT))

from ultralytics import YOLO


DEFAULT_MODEL = Path(r"C:\Users\DOCTOR\Documents\GitHub\YOLOv8\ultralytics\ultralytics\pretrain\best.pt")
DEFAULT_DATASET_ROOT = Path(r"C:\Users\DOCTOR\Documents\GitHub\YOLOv8\ultralytics\datasets\oiltank")


def print_section(title: str) -> None:
    bar = "=" * 96
    print(f"\n{bar}\n{title}\n{bar}")


def nrm_name(s: str) -> str:
    return s.lower().replace("_", "").replace("-", "").replace(" ", "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Simple oiltank eval for YOLOE/YOLO models")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Checkpoint path (.pt)")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT, help="Dataset root with images/labels")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--target-class", type=str, default="oiltank", help="Target class name")
    parser.add_argument("--model-class-id", type=int, default=None, help="Override predicted class id in model output")
    parser.add_argument("--gt-class-id", type=int, default=None, help="Override GT class id in label txt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001, help="Predict conf threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for TP match")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--skip-official-val", action="store_true", help="Skip model.val attempt")
    return parser.parse_args()


def list_images(image_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def parse_yolo_bbox_line(line: str) -> tuple[int, np.ndarray] | None:
    line = line.strip()
    if not line:
        return None
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        cls_id = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:5])
    except Exception:
        return None
    # normalized xywh -> normalized xyxy
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return cls_id, np.array([x1, y1, x2, y2], dtype=np.float32)


def load_gt_boxes_norm(label_file: Path) -> list[tuple[int, np.ndarray]]:
    if not label_file.exists():
        return []
    out: list[tuple[int, np.ndarray]] = []
    for line in label_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        parsed = parse_yolo_bbox_line(line)
        if parsed is not None:
            out.append(parsed)
    return out


def scan_label_ids(label_dir: Path) -> list[int]:
    ids: list[int] = []
    if not label_dir.exists():
        return ids
    for txt in sorted(label_dir.glob("*.txt")):
        for cls_id, _ in load_gt_boxes_norm(txt):
            ids.append(cls_id)
    return ids


def resolve_model_class_id(model_names: Any, target_class: str) -> int | None:
    if model_names is None:
        return None

    aliases = {
        nrm_name(target_class),
        "oiltank",
        "oiltanks",
        "oiltankship",
        "oiltank",
        "oiltanks",
        "tanker",
        "tank",
    }

    if isinstance(model_names, dict):
        items = sorted(model_names.items(), key=lambda x: int(x[0]))
    elif isinstance(model_names, (list, tuple)):
        items = list(enumerate(model_names))
    else:
        return None

    # exact/normalized match first
    for idx, name in items:
        nn = nrm_name(str(name))
        if nn in aliases:
            return int(idx)
    # relaxed contains rule
    for idx, name in items:
        nn = nrm_name(str(name))
        if "tank" in nn:
            return int(idx)
    return None


def resolve_gt_class_id(unique_label_ids: list[int], model_class_id: int | None, override: int | None) -> int | None:
    if override is not None:
        return int(override)
    ids = sorted(set(unique_label_ids))
    if not ids:
        return None
    if len(ids) == 1:
        return ids[0]
    if model_class_id is not None and model_class_id in ids:
        return model_class_id
    return ids[0]


def prepare_yoloe_inference_if_needed(model: YOLO) -> None:
    """
    For some YOLOE checkpoints, `pe` is not stored in ckpt.
    Then predict/val may fail unless classes are explicitly set with embeddings.

    Strategy:
    1) Try text-embedding path (requires clip/mobileclip deps).
    2) If unavailable, fallback to zero embeddings so inference can run.
       (metrics may be less reliable; we print a warning)
    """
    inner = getattr(model, "model", None)
    if inner is None:
        return
    head = getattr(inner, "model", [None])[-1] if hasattr(inner, "model") else None
    cls_name = inner.__class__.__name__.lower()
    is_yoloe = "yoloe" in cls_name or (head is not None and "yoloe" in head.__class__.__name__.lower())
    if not is_yoloe:
        return
    if hasattr(head, "lrpc"):  # prompt-free models typically already ready
        return
    if hasattr(inner, "pe"):  # already has prompt embeddings
        return

    names_map = getattr(inner, "names", {})
    if isinstance(names_map, dict):
        names = [names_map[k] for k in sorted(names_map.keys())]
    elif isinstance(names_map, (list, tuple)):
        names = list(names_map)
    else:
        names = []
    if not names:
        return

    print_section("YOLOE Inference Preparation")
    print("- Detected YOLOE model without cached `pe` embeddings.")

    has_clip = importlib.util.find_spec("clip") is not None
    if has_clip:
        print("- Trying to build text embeddings from class names...")
        try:
            tpe = inner.get_text_pe(names)
            inner.set_classes(names, tpe)
            print("- Success: set_classes with text embeddings.")
            return
        except Exception as exc:
            print(f"- Text embedding path failed: {type(exc).__name__}: {exc}")
    else:
        print("- `clip` dependency not found; skip text-embedding path.")

    # Fallback: zero embeddings
    embed_dim = getattr(head, "embed", 512) if head is not None else 512
    device = next(inner.parameters()).device
    emb = torch.zeros((1, len(names), int(embed_dim)), device=device, dtype=torch.float32)
    inner.set_classes(names, emb)
    print("- Fallback applied: set_classes with zero embeddings.")
    print("- Warning: this allows runtime evaluation but may under-estimate true YOLOE accuracy.")


def write_runtime_yaml(dataset_root: Path, label_ids: list[int], target_class: str) -> Path:
    max_id = max(label_ids) if label_ids else 0
    names = [f"class_{i}" for i in range(max_id + 1)]
    if len(names) == 1:
        names[0] = target_class
    else:
        # If only class id=1 exists, keep id=1 as target and id=0 as dummy.
        uniq = set(label_ids)
        if uniq == {1}:
            names[0] = "bg_dummy"
            names[1] = target_class
        else:
            # best-effort: mark last class as target
            names[-1] = target_class

    data = {
        "path": str(dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(names),
        "names": names,
    }
    out = Path(tempfile.gettempdir()) / "oiltank_eval_runtime.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out


def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a: [N,4], b: [M,4]
    returns IoU matrix [N,M]
    """
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)
    inter_w = np.clip(inter_x2 - inter_x1, 0.0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0.0, None)
    inter = inter_w * inter_h

    area_a = np.clip(ax2 - ax1, 0.0, None) * np.clip(ay2 - ay1, 0.0, None)
    area_b = np.clip(bx2 - bx1, 0.0, None) * np.clip(by2 - by1, 0.0, None)
    union = area_a + area_b - inter
    return inter / np.clip(union, 1e-9, None)


def ap_from_ranked_predictions(conf_and_tp: list[tuple[float, int]], n_gt: int) -> float:
    if n_gt <= 0 or not conf_and_tp:
        return float("nan")
    ranked = sorted(conf_and_tp, key=lambda x: x[0], reverse=True)
    tp = np.array([x[1] for x in ranked], dtype=np.float32)
    fp = 1.0 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / max(n_gt, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    # COCO/VOC style integral over precision envelope
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    return float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))


def try_official_val(
    model: YOLO,
    runtime_yaml: Path,
    split: str,
    imgsz: int,
    batch: int,
    device: str,
) -> tuple[bool, Any]:
    print_section("1) Official Ultralytics val() Attempt")
    print(f"- data yaml: {runtime_yaml}")
    try:
        metrics = model.val(
            data=str(runtime_yaml),
            split=split,
            task="detect",  # force detect-style validation attempt
            imgsz=imgsz,
            batch=batch,
            device=device,
            plots=False,
            save_json=False,
            verbose=False,
        )
        print("- official val() succeeded")
        box_map = getattr(getattr(metrics, "box", None), "map", None)
        box_map50 = getattr(getattr(metrics, "box", None), "map50", None)
        print(f"- box mAP50-95: {box_map}")
        print(f"- box mAP50: {box_map50}")
        return True, metrics
    except Exception as exc:
        print(f"- official val() failed: {type(exc).__name__}: {exc}")
        print("- traceback (for diagnosis):")
        print(traceback.format_exc())
        return False, None


def run_manual_eval(
    model: YOLO,
    image_paths: list[Path],
    image_dir: Path,
    label_dir: Path,
    model_class_id: int,
    gt_class_id: int,
    conf: float,
    iou_thr: float,
    imgsz: int,
    device: str,
    batch: int,
) -> dict[str, float]:
    print_section("2) Manual BBox Evaluation (Fallback / Always Available)")
    print("- metric style: single-class detection metrics on target class")
    print(f"- model class id used: {model_class_id}")
    print(f"- gt class id used: {gt_class_id}")
    print(f"- num images: {len(image_paths)}")
    print(f"- IoU threshold: {iou_thr}")
    print(f"- conf threshold: {conf}")

    tp_total = 0
    fp_total = 0
    fn_total = 0
    n_gt_total = 0
    pred_rank: list[tuple[float, int]] = []  # (confidence, is_tp)

    # Use directory source so result.path keeps original filenames.
    all_results = model.predict(
        source=str(image_dir),
        imgsz=imgsz,
        conf=conf,
        iou=0.7,
        device=device,
        batch=batch,
        verbose=False,
        stream=False,
    )

    for res in all_results:
        img_path = Path(res.path)
        h, w = res.orig_shape[:2]
        label_file = label_dir / f"{img_path.stem}.txt"
        gt_raw = load_gt_boxes_norm(label_file)
        gt_boxes = [b for cls, b in gt_raw if cls == gt_class_id]
        gt_boxes = np.array(gt_boxes, dtype=np.float32) if gt_boxes else np.zeros((0, 4), dtype=np.float32)
        n_gt_total += int(gt_boxes.shape[0])

        # convert gt normalized -> pixel xyxy
        if gt_boxes.shape[0] > 0:
            gt_boxes[:, [0, 2]] *= float(w)
            gt_boxes[:, [1, 3]] *= float(h)

        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            fn_total += int(gt_boxes.shape[0])
            continue

        pred_xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        pred_cls = boxes.cls.detach().cpu().numpy().astype(np.int64)
        pred_conf = boxes.conf.detach().cpu().numpy().astype(np.float32)
        keep = pred_cls == int(model_class_id)
        pred_xyxy = pred_xyxy[keep]
        pred_conf = pred_conf[keep]

        if pred_xyxy.shape[0] == 0:
            fn_total += int(gt_boxes.shape[0])
            continue

        order = np.argsort(-pred_conf)
        pred_xyxy = pred_xyxy[order]
        pred_conf = pred_conf[order]

        matched_gt = np.zeros(gt_boxes.shape[0], dtype=bool)
        if gt_boxes.shape[0] > 0:
            ious = box_iou_xyxy(pred_xyxy, gt_boxes)
        else:
            ious = np.zeros((pred_xyxy.shape[0], 0), dtype=np.float32)

        for pi in range(pred_xyxy.shape[0]):
            is_tp = 0
            if gt_boxes.shape[0] > 0:
                best_j = int(np.argmax(ious[pi]))
                best_iou = float(ious[pi, best_j])
                if best_iou >= iou_thr and not matched_gt[best_j]:
                    matched_gt[best_j] = True
                    is_tp = 1
            pred_rank.append((float(pred_conf[pi]), is_tp))
            if is_tp:
                tp_total += 1
            else:
                fp_total += 1

        fn_total += int((~matched_gt).sum()) if gt_boxes.shape[0] > 0 else 0

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    ap50 = ap_from_ranked_predictions(pred_rank, n_gt_total)
    acc_like = tp_total / (tp_total + fp_total + fn_total) if (tp_total + fp_total + fn_total) > 0 else 0.0

    out = {
        "images": float(len(image_paths)),
        "gt_instances": float(n_gt_total),
        "tp": float(tp_total),
        "fp": float(fp_total),
        "fn": float(fn_total),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "ap50": float(ap50) if np.isfinite(ap50) else float("nan"),
        "accuracy_like_tp_over_tp_fp_fn": float(acc_like),
    }
    return out


def main() -> None:
    args = parse_args()

    print_section("Config")
    print(f"- model: {args.model}")
    print(f"- dataset_root: {args.dataset_root}")
    print(f"- split: {args.split}")
    print(f"- target_class: {args.target_class}")
    print(f"- device: {args.device}")

    image_dir = args.dataset_root / "images" / args.split
    label_dir = args.dataset_root / "labels" / args.split
    if not image_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Label dir not found: {label_dir}")

    image_paths = list_images(image_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    label_ids = scan_label_ids(label_dir)
    unique_label_ids = sorted(set(label_ids))
    print_section("Dataset Quick Scan")
    print(f"- images found: {len(image_paths)}")
    print(f"- unique label class ids in split: {unique_label_ids if unique_label_ids else 'none'}")

    model = YOLO(str(args.model))
    model_names = getattr(model, "names", None)
    model_task = getattr(model, "task", None)

    print_section("Model Quick Scan")
    print(f"- wrapper type: {type(model).__name__}")
    print(f"- model.task: {model_task}")
    print(f"- model.names: {model_names}")
    prepare_yoloe_inference_if_needed(model)

    model_class_id = args.model_class_id if args.model_class_id is not None else resolve_model_class_id(model_names, args.target_class)
    gt_class_id = resolve_gt_class_id(unique_label_ids, model_class_id, args.gt_class_id)

    print(f"- resolved model target class id: {model_class_id}")
    print(f"- resolved GT target class id: {gt_class_id}")
    if model_class_id is None:
        raise RuntimeError(
            f"Cannot resolve model class id for target '{args.target_class}'. "
            "Try changing --target-class or check model.names."
        )
    if gt_class_id is None:
        raise RuntimeError(
            "Cannot resolve GT class id from label files. "
            "Try passing --gt-class-id explicitly."
        )

    runtime_yaml = write_runtime_yaml(args.dataset_root, label_ids, args.target_class)

    if not args.skip_official_val:
        _ok, _metrics = try_official_val(
            model=model,
            runtime_yaml=runtime_yaml,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )

    manual = run_manual_eval(
        model=model,
        image_paths=image_paths,
        image_dir=image_dir,
        label_dir=label_dir,
        model_class_id=model_class_id,
        gt_class_id=gt_class_id,
        conf=args.conf,
        iou_thr=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
    )

    print_section("Final Result (Target Class)")
    print(f"- target class: {args.target_class}")
    print(f"- split: {args.split}")
    print(f"- GT instances: {int(manual['gt_instances'])}")
    print(f"- TP / FP / FN: {int(manual['tp'])} / {int(manual['fp'])} / {int(manual['fn'])}")
    print(f"- Precision: {manual['precision']:.4f}")
    print(f"- Recall: {manual['recall']:.4f}")
    print(f"- F1: {manual['f1']:.4f}")
    print(f"- AP50 (manual): {manual['ap50']:.4f}" if np.isfinite(manual["ap50"]) else "- AP50 (manual): NaN")
    print(f"- Accuracy-like (TP/(TP+FP+FN)): {manual['accuracy_like_tp_over_tp_fp_fn']:.4f}")

    print_section("Notes")
    print("- For detection, AP50/Precision/Recall/F1 are usually more meaningful than a single 'accuracy'.")
    print("- This script is read-only: it does not modify your dataset or training pipeline.")


if __name__ == "__main__":
    main()
