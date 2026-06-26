from __future__ import annotations

import sys
import argparse
import json
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "ultralytics"))
sys.path.insert(1, str(PROJECT_ROOT))

DEFAULT_PROJECT = "runs/bridge_expert"
DEFAULT_NAME = "yolov8m_total_bridge_split_img1024_ep150"
DEFAULT_DATA = "configs/datasets/total_bridge_trainval.yaml"
DEFAULT_TRAINVAL = "ultralytics/datasets/total_bridge_trainval"
DEFAULT_TEST = "ultralytics/datasets/total_bridge_test"
DEFAULT_SPLIT_CHECK = "runs/bridge_expert/split_check/total_bridge_split_check.json"


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def project_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def make_runtime_data_yaml(data_yaml: Path, project_dir: Path) -> Path:
    try:
        import yaml

        data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Dataset YAML is not a mapping: {data_yaml}")
        source_path = Path(data.get("path", ""))
        data["path"] = str(resolve_path(source_path).resolve()) if source_path else str(data_yaml.parent.resolve())
        project_dir.mkdir(parents=True, exist_ok=True)
        runtime_yaml = project_dir / "runtime_total_bridge_trainval.yaml"
        runtime_yaml.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return runtime_yaml
    except Exception:
        # Fallback keeps the user-provided path if PyYAML is unavailable or malformed.
        return data_yaml


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
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--close-mosaic", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exist-ok", action="store_true", default=True)
    args = parser.parse_args()

    from ultralytics import YOLO

    start = datetime.now().isoformat()
    data_yaml = resolve_path(args.data)
    project = resolve_path(args.project)
    run_dir = project / args.name
    runtime_data_yaml = make_runtime_data_yaml(data_yaml, project)
    trainval_root = resolve_path(DEFAULT_TRAINVAL)
    test_root = resolve_path(DEFAULT_TEST)
    split_check = resolve_path(DEFAULT_SPLIT_CHECK)
    summary = {
        "purpose": "Bridge-only expert training for late fusion. This is not YOLOE and not a 2-class student.",
        "model_base": args.model,
        "data_yaml_input": args.data,
        "data_yaml": str(data_yaml.resolve()),
        "runtime_data_yaml": str(runtime_data_yaml.resolve()),
        "trainval_root_input": DEFAULT_TRAINVAL,
        "trainval_root": str(trainval_root.resolve()),
        "test_root_input": DEFAULT_TEST,
        "test_root": str(test_root.resolve()),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "seed": args.seed,
        "project_input": args.project,
        "project": str(project.resolve()),
        "name": args.name,
        "start_time": start,
        "split_summary_path": str(trainval_root / "split_summary.json"),
        "split_check_path": str(split_check),
    }
    try:
        model = YOLO(args.model)
        results = model.train(
            data=str(runtime_data_yaml),
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=str(project),
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
