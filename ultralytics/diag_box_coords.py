#!/usr/bin/env python
"""Print predicted vs GT box coordinates for the first few images to debug matching."""
import sys, os, torch
from pathlib import Path

_script_dir = os.path.dirname(os.path.abspath(__file__))
for _c in [_script_dir, os.path.join(_script_dir, "ultralytics")]:
    if os.path.isdir(os.path.join(_c, "ultralytics")):
        sys.path.insert(0, _c)
        break

MODEL = r"C:\Users\DOCTOR\Documents\GitHub\YOLOv8\ultralytics\ultralytics\best.pt"
IMG_DIR = r"C:\Users\DOCTOR\Documents\GitHub\YOLOv8\ultralytics\datasets\tank_extratest\images\val"
LBL_DIR = r"C:\Users\DOCTOR\Documents\GitHub\YOLOv8\ultralytics\datasets\tank_extratest\labels\val"

from ultralytics import YOLO

# Load and init
model = YOLO(MODEL)
inner = model.model
class_names = ["ship", "harbor", "tank"]
pe = inner.get_text_pe(class_names)
inner.set_classes(class_names, pe)
print(f"Model names: {dict(model.names)}")

# Get first 3 images
img_dir = Path(IMG_DIR)
lbl_dir = Path(LBL_DIR)
ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
imgs = sorted(f for f in img_dir.iterdir() if f.suffix.lower() in ext)[:3]

for img_path in imgs:
    print(f"\n{'='*80}")
    print(f"Image: {img_path.name}")

    # Read GT
    lbl_file = lbl_dir / (img_path.stem + ".txt")
    print(f"\n  GT boxes (from label file, YOLO format: cls cx cy w h):")
    gt_boxes = []
    if lbl_file.exists():
        with open(lbl_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    gt_boxes.append((cls, cx, cy, w, h))
                    print(f"    cls={cls} cx={cx:.4f} cy={cy:.4f} w={w:.4f} h={h:.4f}")
    else:
        print(f"    (no label file)")
    print(f"  Total GT: {len(gt_boxes)}")

    # Predict
    results = model.predict(str(img_path), conf=0.1, iou=0.5, imgsz=640,
                            device="cpu", verbose=False, save=False)

    print(f"\n  Predictions:")
    for r in results:
        print(f"    orig_shape (H,W): {r.orig_shape}")
        bx = r.boxes
        if bx is None or len(bx) == 0:
            print(f"    (no detections)")
            continue

        print(f"    Total detections: {len(bx)}")
        print(f"    boxes.xyxy available: {bx.xyxy is not None}")
        print(f"    boxes.xyxyn available: {hasattr(bx, 'xyxyn') and bx.xyxyn is not None}")
        print(f"    boxes.xywhn available: {hasattr(bx, 'xywhn') and bx.xywhn is not None}")

        # Show first 5 predictions in all formats
        n_show = min(5, len(bx))
        print(f"\n    First {n_show} predictions:")
        for i in range(n_show):
            cls_id = int(bx.cls[i].item())
            conf = float(bx.conf[i].item())

            xyxy = bx.xyxy[i].tolist()
            print(f"      [{i}] cls={cls_id} conf={conf:.3f}")
            print(f"           xyxy (pixel):      [{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]")

            if hasattr(bx, "xyxyn") and bx.xyxyn is not None:
                xyxyn = bx.xyxyn[i].tolist()
                print(f"           xyxyn (normalized): [{xyxyn[0]:.4f}, {xyxyn[1]:.4f}, {xyxyn[2]:.4f}, {xyxyn[3]:.4f}]")

            if hasattr(bx, "xywhn") and bx.xywhn is not None:
                xywhn = bx.xywhn[i].tolist()
                print(f"           xywhn (normalized): [{xywhn[0]:.4f}, {xywhn[1]:.4f}, {xywhn[2]:.4f}, {xywhn[3]:.4f}]")

        # Show GT converted to xyxy (normalized) for comparison
        print(f"\n    GT converted to xyxy (normalized) for comparison:")
        for cls, cx, cy, w, h in gt_boxes[:5]:
            x1 = cx - w/2; y1 = cy - h/2; x2 = cx + w/2; y2 = cy + h/2
            print(f"      cls={cls} xyxy_norm=[{x1:.4f}, {y1:.4f}, {x2:.4f}, {y2:.4f}]")

        # Quick IoU check between first pred and first GT
        if gt_boxes and len(bx) > 0:
            # Pred in normalized coords
            if hasattr(bx, "xyxyn") and bx.xyxyn is not None:
                px1, py1, px2, py2 = bx.xyxyn[0].tolist()
            else:
                ih, iw = r.orig_shape
                px1, py1, px2, py2 = [v/s for v, s in zip(bx.xyxy[0].tolist(), [iw, ih, iw, ih])]

            # First GT in normalized
            _, cx, cy, w, h = gt_boxes[0]
            gx1, gy1, gx2, gy2 = cx-w/2, cy-h/2, cx+w/2, cy+h/2

            # IoU
            ix1 = max(px1, gx1); iy1 = max(py1, gy1)
            ix2 = min(px2, gx2); iy2 = min(py2, gy2)
            inter = max(0, ix2-ix1) * max(0, iy2-iy1)
            a1 = (px2-px1)*(py2-py1)
            a2 = (gx2-gx1)*(gy2-gy1)
            union = a1+a2-inter
            iou_val = inter/union if union > 0 else 0
            print(f"\n    IoU between pred[0] and gt[0]: {iou_val:.4f}")
            print(f"      pred[0] norm: [{px1:.4f},{py1:.4f},{px2:.4f},{py2:.4f}]")
            print(f"      gt[0]   norm: [{gx1:.4f},{gy1:.4f},{gx2:.4f},{gy2:.4f}]")

print("\nDone.")
