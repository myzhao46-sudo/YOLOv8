from __future__ import annotations

import sys
sys.path.insert(0, r"C:/Users/DOCTOR/Documents/GitHub/YOLOv8/ultralytics")
sys.path.insert(1, r"C:/Users/DOCTOR/Documents/GitHub/YOLOv8")

import argparse
import json
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(r"C:/Users/DOCTOR/Documents/GitHub/YOLOv8")
DEFAULT_RUN = REPO_ROOT / "runs" / "bridge_expert" / "yolov8m_total_bridge_split_img1024_ep150"
DEFAULT_TEST = REPO_ROOT / "ultralytics" / "datasets" / "total_bridge_test"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export bridge expert predictions for later late-fusion consumption.")
    parser.add_argument("--weights", default=str(DEFAULT_RUN / "weights" / "best.pt"))
    parser.add_argument(
        "--image-roots",
        nargs="+",
        default=[
            str(DEFAULT_TEST / "images" / "test_rgb"),
            str(DEFAULT_TEST / "images" / "test_sar"),
            str(DEFAULT_TEST / "images" / "test_ir"),
        ],
    )
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", default=str(DEFAULT_RUN / "bridge_predictions_total_bridge_test_conf025.json"))
    args = parser.parse_args()

    from ultralytics import YOLO

    images = []
    for root in args.image_roots:
        images.extend(list_images(Path(root)))
    model = YOLO(args.weights)
    output = []
    for result in model.predict(
        source=[str(p) for p in images],
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=False,
        stream=True,
    ):
        image_path = Path(result.path).resolve()
        with Image.open(image_path) as im:
            width, height = im.size
        boxes = []
        if result.boxes is not None:
            for box in result.boxes:
                if int(box.cls.item()) != 0:
                    continue
                boxes.append(
                    {
                        "xyxy": [float(x) for x in box.xyxy[0].tolist()],
                        "conf": float(box.conf.item()),
                        "class": "bridge",
                        "class_id": 0,
                        "source_model": Path(args.weights).parents[1].name,
                    }
                )
        output.append({"image": str(image_path), "width": width, "height": height, "boxes": boxes})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "images": len(output), "boxes": sum(len(x["boxes"]) for x in output)}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
