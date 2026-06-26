from __future__ import annotations

import sys
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "ultralytics"))
sys.path.insert(1, str(PROJECT_ROOT))

DEFAULT_RUN = Path("runs/bridge_expert/yolov8m_total_bridge_split_img1024_ep150")
DEFAULT_TEST = "ultralytics/datasets/total_bridge_test"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MODALITIES = ("rgb", "sar", "ir")


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def list_images(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def xywhn_to_xyxy(box, w: int, h: int) -> list[float]:
    cx, cy, bw, bh = box
    return [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]


def read_gt(label: Path, image: Path) -> list[list[float]]:
    with Image.open(image) as im:
        w, h = im.size
    boxes = []
    for line_no, line in enumerate(label.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"bad label row {label}:{line_no}: {line}")
        cls = int(float(parts[0]))
        if cls != 0:
            raise ValueError(f"GT class must be 0 bridge, got {cls}: {label}:{line_no}")
        boxes.append(xywhn_to_xyxy([float(x) for x in parts[1:]], w, h))
    return boxes


def iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    area2 = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area1 + area2 - inter, 1e-9)


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if recalls.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate(preds: list[dict], gt: dict[str, list[list[float]]], iou_thr: float = 0.5) -> dict:
    gt_arrays = {k: np.asarray(v, dtype=np.float32).reshape(-1, 4) for k, v in gt.items()}
    matched = {k: np.zeros(len(v), dtype=bool) for k, v in gt_arrays.items()}
    total_gt = sum(len(v) for v in gt_arrays.values())
    preds = sorted(preds, key=lambda x: x["conf"], reverse=True)
    tp_flags = []
    fp_flags = []
    for pred in preds:
        arr = gt_arrays.get(pred["image"], np.zeros((0, 4), dtype=np.float32))
        if arr.shape[0] == 0:
            tp_flags.append(0.0)
            fp_flags.append(1.0)
            continue
        ious = iou_one_to_many(np.asarray(pred["xyxy"], dtype=np.float32), arr)
        best = int(np.argmax(ious)) if ious.size else -1
        if best >= 0 and float(ious[best]) >= iou_thr and not matched[pred["image"]][best]:
            matched[pred["image"]][best] = True
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
    recalls = tp_cum / max(total_gt, 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    return {
        "gt": int(total_gt),
        "pred": len(preds),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(total_gt, 1)),
        "ap50": compute_ap(recalls, precisions),
    }


def collect_predictions(model, images: list[Path], imgsz: int, conf: float, iou: float, device: str) -> list[dict]:
    results = model.predict(
        source=[str(p) for p in images],
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
        stream=True,
    )
    preds = []
    for result in results:
        image_key = str(Path(result.path).resolve())
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls = int(box.cls.item())
            if cls != 0:
                continue
            preds.append(
                {
                    "image": image_key,
                    "xyxy": [float(x) for x in box.xyxy[0].tolist()],
                    "conf": float(box.conf.item()),
                    "class_id": 0,
                    "class": "bridge",
                }
            )
    return preds


def write_md(path: Path, summary: dict) -> None:
    lines = [
        "# total_bridge_test Bridge Expert Eval",
        "",
        "This is an internal hold-out test split from total_bridge. It was not used for training.",
        "This result validates whether the bridge-only expert can be trained; it is not an external bridge generalization result.",
        "",
    ]
    for conf, data in summary["conf_results"].items():
        lines.append(f"## conf {conf}")
        lines.append("| split | GT | Pred | TP | FP | FN | P | R | AP50 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for key in ("RGB", "SAR", "IR", "ALL"):
            m = data[key]
            lines.append(f"| {key} | {m['gt']} | {m['pred']} | {m['tp']} | {m['fp']} | {m['fn']} | {m['precision']:.6f} | {m['recall']:.6f} | {m['ap50']:.6f} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(DEFAULT_RUN / "weights" / "best.pt"))
    parser.add_argument("--test-root", default=DEFAULT_TEST)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf-list", nargs="+", type=float, default=[0.05, 0.10, 0.20, 0.25, 0.35, 0.50])
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out-dir", default=str(DEFAULT_RUN / "test_eval"))
    args = parser.parse_args()

    from ultralytics import YOLO

    weights = resolve_path(args.weights)
    test_root = resolve_path(args.test_root)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = {m.upper(): list_images(test_root / "images" / f"test_{m}") for m in MODALITIES}
    gt_groups = {}
    for name, imgs in groups.items():
        label_dir = test_root / "labels" / f"test_{name.lower()}"
        gt_groups[name] = {str(img.resolve()): read_gt(label_dir / f"{img.stem}.txt", img) for img in imgs}

    model = YOLO(str(weights))
    summary = {
        "note": "This is total_bridge internal hold-out test. It was not used for training. Not an external bridge generalization conclusion.",
        "weights_input": args.weights,
        "weights": str(weights.resolve()),
        "test_root_input": args.test_root,
        "test_root": str(test_root.resolve()),
        "out_dir_input": args.out_dir,
        "out_dir": str(out_dir.resolve()),
        "imgsz": args.imgsz,
        "nms_iou": args.iou,
        "conf_results": {},
    }
    for conf in args.conf_list:
        conf_key = f"{conf:.2f}"
        predictions_json = []
        result = {}
        all_preds = []
        all_gt = {}
        for name, imgs in groups.items():
            preds = collect_predictions(model, imgs, args.imgsz, conf, args.iou, args.device)
            metric = evaluate(preds, gt_groups[name], 0.5)
            result[name] = metric
            all_preds.extend(preds)
            all_gt.update(gt_groups[name])
            predictions_json.extend(preds)
        result["ALL"] = evaluate(all_preds, all_gt, 0.5)
        summary["conf_results"][conf_key] = result
        pred_path = out_dir / f"test_predictions_conf{int(round(conf * 100)):03d}.json"
        pred_path.write_text(json.dumps(predictions_json, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "test_eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(out_dir / "test_eval_summary.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
