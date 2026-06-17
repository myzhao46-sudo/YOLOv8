# tools/distill/export_teacher_ship_pseudo.py
# -*- coding: utf-8 -*-

"""
Export hard teacher pseudo labels for ship only from the original YOLOE best.pt.

Teacher is used only for class 0 ship. The script does not train, does not call
model.val(), does not save checkpoints, and does not modify best.pt or the teacher head.
Output label txt files are 5-column YOLO detect labels: cls cx cy w h.
Confidence is written only to the optional CSV.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(r"C:\Users\DOCTOR\Documents\GitHub\YOLOv8")
PACKAGE_ROOT = REPO_ROOT / "ultralytics"
DEFAULT_TEACHER = REPO_ROOT / "ultralytics" / "best.pt"
DEFAULT_IMAGES = REPO_ROOT / "ultralytics" / "datasets" / "ship_small_split" / "images" / "train"
DEFAULT_OUT_LABELS = REPO_ROOT / "ultralytics" / "datasets" / "ship_teacher_pseudo_conf025" / "labels" / "train"
DEFAULT_OUT_CSV = REPO_ROOT / "ultralytics" / "datasets" / "ship_teacher_pseudo_conf025" / "pseudo_conf.csv"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def setup_paths() -> None:
    for p in [str(PACKAGE_ROOT.resolve()), str(REPO_ROOT.resolve())]:
        while p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, str(PACKAGE_ROOT.resolve()))
    sys.path.insert(1, str(REPO_ROOT.resolve()))


def parse_args():
    parser = argparse.ArgumentParser(description="Export teacher ship-only pseudo labels.")
    parser.add_argument("--teacher-weights", default=str(DEFAULT_TEACHER))
    parser.add_argument("--images", default=str(DEFAULT_IMAGES))
    parser.add_argument("--out-labels", default=str(DEFAULT_OUT_LABELS))
    parser.add_argument("--out-csv", default=str(DEFAULT_OUT_CSV))
    parser.add_argument("--teacher-names", default="ship,harbor,tank,bridge")
    parser.add_argument("--target-class", type=int, default=0)
    parser.add_argument("--target-name", default="ship")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=1)
    return parser.parse_args()


def list_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"images directory not found: {images_dir}")
    images = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if not images:
        raise FileNotFoundError(f"no images found in: {images_dir}")
    return images


def xyxy_to_xywhn(xyxy: list[float], image_path: Path) -> tuple[float, float, float, float]:
    with Image.open(image_path) as im:
        w_img, h_img = im.size
    x1, y1, x2, y2 = xyxy
    x1 = max(0.0, min(w_img, x1))
    x2 = max(0.0, min(w_img, x2))
    y1 = max(0.0, min(h_img, y1))
    y2 = max(0.0, min(h_img, y2))
    cx = ((x1 + x2) / 2.0) / w_img
    cy = ((y1 + y2) / 2.0) / h_img
    bw = max(0.0, x2 - x1) / w_img
    bh = max(0.0, y2 - y1) / h_img
    return cx, cy, bw, bh


def main() -> int:
    args = parse_args()
    setup_paths()

    from ultralytics import YOLO
    from tools.eval.eval_teacher_ship_external import (
        collect_predictions,
        get_names_state,
        get_task_state,
        head_state,
        obj_type,
        safe_getattr,
        set_global4_classes,
    )

    images_dir = Path(args.images).resolve()
    out_labels = Path(args.out_labels).resolve()
    out_csv = Path(args.out_csv).resolve() if args.out_csv else None
    image_paths = list_images(images_dir)

    print("[LOAD TEACHER]", flush=True)
    print(f"teacher weights: {Path(args.teacher_weights).resolve()}", flush=True)
    print(f"images: {images_dir}", flush=True)
    print(f"out labels: {out_labels}", flush=True)
    print(f"target: {args.target_class} {args.target_name}", flush=True)
    print(f"conf={args.conf} iou={args.iou} imgsz={args.imgsz} max_det={args.max_det}", flush=True)

    teacher = YOLO(str(Path(args.teacher_weights).resolve()))
    inner = safe_getattr(teacher, "model", None)
    print(f"teacher wrapper type: {obj_type(teacher)}", flush=True)
    print(f"teacher model class: {obj_type(inner)}", flush=True)
    print(f"task: {get_task_state(teacher, inner)}", flush=True)
    print(f"names before: {get_names_state(teacher, inner)}", flush=True)
    print(f"head before: {head_state(inner)}", flush=True)

    teacher_names = [x.strip() for x in args.teacher_names.split(",") if x.strip()]
    if teacher_names != ["ship", "harbor", "tank", "bridge"]:
        print(f"WARNING: non-default teacher names requested: {teacher_names}", flush=True)
    # Reuse the verified global4 initialization helper.
    embeddings, _, _, _ = set_global4_classes(teacher, inner)
    print(f"set_classes ok: True", flush=True)
    print(f"teacher_names used by helper: ['ship', 'harbor', 'tank', 'bridge']", flush=True)
    print(f"pe shape: {list(embeddings.shape)}", flush=True)
    print(f"names after: {get_names_state(teacher, inner)}", flush=True)
    print(f"head after: {head_state(inner)}", flush=True)

    # collect_predictions expects these Ultralytics-like attribute names.
    args.pred_iou = args.iou
    predictions, score_channels_seen, total_boxes_all_classes = collect_predictions(inner, image_paths, args)

    by_image = {str(p.resolve()): [] for p in image_paths}
    for pred in predictions:
        by_image[pred["image_key"]].append(pred)

    out_labels.mkdir(parents=True, exist_ok=True)
    csv_rows = []
    pseudo_boxes = 0
    empty_label_files = 0
    labels_written = 0

    for image_path in image_paths:
        key = str(image_path.resolve())
        label_path = out_labels / f"{image_path.stem}.txt"
        rows = []
        for pred in sorted(by_image[key], key=lambda x: x["conf"], reverse=True):
            cx, cy, bw, bh = xyxy_to_xywhn(pred["xyxy"], image_path)
            rows.append(f"{args.target_class} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")
            csv_rows.append(
                {
                    "image": str(image_path),
                    "cls": args.target_class,
                    "cx": f"{cx:.8f}",
                    "cy": f"{cy:.8f}",
                    "w": f"{bw:.8f}",
                    "h": f"{bh:.8f}",
                    "conf": f"{pred['conf']:.8f}",
                }
            )
            pseudo_boxes += 1
        label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        labels_written += 1
        if not rows:
            empty_label_files += 1

    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "cls", "cx", "cy", "w", "h", "conf"])
            writer.writeheader()
            writer.writerows(csv_rows)

    print("", flush=True)
    print("[EXPORT SUMMARY]", flush=True)
    print(f"images: {len(image_paths)}", flush=True)
    print(f"labels written: {labels_written}", flush=True)
    print(f"pseudo boxes: {pseudo_boxes}", flush=True)
    print(f"empty label files: {empty_label_files}", flush=True)
    print(f"conf threshold: {args.conf}", flush=True)
    print(f"score channels seen: {sorted(score_channels_seen)}", flush=True)
    print(f"all-class post-NMS boxes seen internally: {total_boxes_all_classes}", flush=True)
    if out_csv:
        print(f"confidence CSV: {out_csv}", flush=True)
    print("teacher frozen/no train: True", flush=True)
    print("teacher exported class ids: [0] only", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
