#!/usr/bin/env python
"""Just print raw label file content."""
from pathlib import Path

LBL_DIR = Path(r"C:\Users\DOCTOR\Documents\GitHub\YOLOv8\ultralytics\datasets\tank_extratest\labels\val")
IMG_DIR = Path(r"C:\Users\DOCTOR\Documents\GitHub\YOLOv8\ultralytics\datasets\tank_extratest\images\val")

ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
imgs = sorted(f for f in IMG_DIR.iterdir() if f.suffix.lower() in ext)[:3]

for img in imgs:
    lf = LBL_DIR / (img.stem + ".txt")
    print(f"\n{'='*60}")
    print(f"Image: {img.name}")
    print(f"Label: {lf.name} (exists={lf.exists()})")
    if lf.exists():
        lines = lf.read_text().strip().split("\n")
        print(f"Lines: {len(lines)}")
        print(f"First 5 raw lines:")
        for line in lines[:5]:
            print(f"  '{line}'")
            parts = line.strip().split()
            print(f"    fields={len(parts)}: {parts}")
