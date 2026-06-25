from __future__ import annotations

import sys
sys.path.insert(0, r"C:/Users/DOCTOR/Documents/GitHub/YOLOv8/ultralytics")
sys.path.insert(1, r"C:/Users/DOCTOR/Documents/GitHub/YOLOv8")

import argparse
import json
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(r"C:/Users/DOCTOR/Documents/GitHub/YOLOv8")
DEFAULT_PROJECT = REPO_ROOT / "runs" / "bridge_expert"
DEFAULT_NAME = "yolov8m_total_bridge_split_img1024_ep150"
DEFAULT_DATA = REPO_ROOT / "configs" / "datasets" / "total_bridge_trainval.yaml"
DEFAULT_TRAINVAL = REPO_ROOT / "ultralytics" / "datasets" / "total_bridge_trainval"
DEFAULT_TEST = REPO_ROOT / "ultralytics" / "datasets" / "total_bridge_test"
DEFAULT_SPLIT_CHECK = REPO_ROOT / "runs" / "bridge_expert" / "split_check" / "total_bridge_split_check.json"


def write_md(path: Path, summary: dict) -> None:
    lines = ["# Bridge Expert Train Summary", ""]
    for key in [
        "model_base",
        "data_yaml",
        "trainval_root",
        "test_root",
        "imgsz",
        "epochs",
        "batch",
        "device",
        "seed",
        "best_checkpoint",
        "last_checkpoint",
        "start_time",
        "end_time",
        "split_summary_path",
        "split_check_path",
    ]:
        lines.append(f"- {key}: `{summary.get(key)}`")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train ordinary YOLOv8 bridge-only expert on total_bridge_trainval.")
    parser.add_argument("--model", default="yolov8m.pt")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default=str(DEFAULT_PROJECT))
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--close-mosaic", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exist-ok", action="store_true", default=True)
    args = parser.parse_args()

    from ultralytics import YOLO

    start = datetime.now().isoformat()
    run_dir = Path(args.project) / args.name
    summary = {
        "purpose": "Bridge-only expert training for late fusion. This is not YOLOE and not a 2-class student.",
        "model_base": args.model,
        "data_yaml": str(Path(args.data).resolve()),
        "trainval_root": str(DEFAULT_TRAINVAL),
        "test_root": str(DEFAULT_TEST),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "seed": args.seed,
        "project": str(Path(args.project).resolve()),
        "name": args.name,
        "start_time": start,
        "split_summary_path": str(DEFAULT_TRAINVAL / "split_summary.json"),
        "split_check_path": str(DEFAULT_SPLIT_CHECK),
    }
    try:
        model = YOLO(args.model)
        results = model.train(
            data=args.data,
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=args.project,
            name=args.name,
            patience=args.patience,
            close_mosaic=args.close_mosaic,
            pretrained=True,
            seed=args.seed,
            exist_ok=args.exist_ok,
        )
        summary["train_return"] = str(results)
        summary["completed"] = True
    except Exception as exc:
        summary["completed"] = False
        summary["error"] = repr(exc)
        raise
    finally:
        end = datetime.now().isoformat()
        summary["end_time"] = end
        summary["run_dir"] = str(run_dir)
        summary["best_checkpoint"] = str(run_dir / "weights" / "best.pt")
        summary["last_checkpoint"] = str(run_dir / "weights" / "last.pt")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        write_md(run_dir / "train_summary.md", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
