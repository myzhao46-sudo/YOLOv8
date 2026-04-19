from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import sys

# Prefer local repo package over site-packages when running as:
#   python ultralytics/scripts/eval_incremental_model.py ...
THIS_FILE = Path(__file__).resolve()
LOCAL_ULTRALYTICS_ROOT = THIS_FILE.parents[1]  # .../ultralytics
if str(LOCAL_ULTRALYTICS_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_ULTRALYTICS_ROOT))

from ultralytics import YOLO
from ultralytics.data.sliding_window import SliceConfig, prepare_sliced_image_paths
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import LOGGER, YAML


def _restore_path_type(original: str | list[str], paths: list[str]) -> str | list[str]:
    return paths if isinstance(original, list) else paths[0]


def _prepare_eval_data_yaml(
    data_yaml: str,
    *,
    enable_slicing: bool,
    slice_cfg: SliceConfig,
    save_dir: Path,
    tag: str,
) -> str:
    if not enable_slicing:
        return data_yaml

    resolved = check_det_dataset(data_yaml, autodownload=False)
    val_slice_cfg = SliceConfig(
        enabled=True,
        slice_size=slice_cfg.slice_size,
        slice_overlap=slice_cfg.slice_overlap,
        min_box_keep_ratio=slice_cfg.min_box_keep_ratio,
        # Keep empty tiles in eval to avoid random eval-set drift.
        empty_tile_keep_prob=1.0,
        cache_slices=slice_cfg.cache_slices,
        cache_dir=slice_cfg.cache_dir,
        seed=slice_cfg.seed,
    )

    for split in ("train", "val", "test", "minival"):
        if not resolved.get(split):
            continue
        src = resolved[split]
        src_list = src if isinstance(src, list) else [src]
        sliced = prepare_sliced_image_paths(
            src_list,
            mode="val",
            cfg=val_slice_cfg,
            run_cache_dir=save_dir / "slice_cache",
            log_prefix=f"{tag}:{split}: ",
        )
        resolved[split] = _restore_path_type(src, sliced)

    out_dir = save_dir / "sliced_eval_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_yaml = out_dir / f"{Path(data_yaml).stem}_{tag}_sliced.yaml"
    out_dict = {
        "path": "",
        "train": resolved.get("train"),
        "val": resolved.get("val"),
        "test": resolved.get("test"),
        "nc": resolved["nc"],
        "names": resolved["names"],
        "channels": resolved.get("channels", 3),
    }
    YAML.save(out_yaml, out_dict)
    return str(out_yaml)


def _default_tags(data_list: list[str]) -> list[str]:
    used: dict[str, int] = {}
    tags = []
    for path in data_list:
        stem = Path(path).stem
        idx = used.get(stem, 0)
        used[stem] = idx + 1
        tags.append(stem if idx == 0 else f"{stem}_{idx}")
    return tags


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone evaluation for trained incremental detect models.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint (best.pt/last.pt).")
    parser.add_argument("--data", nargs="+", required=True, help="One or more detection data YAML files.")
    parser.add_argument("--tags", nargs="*", default=None, help="Optional tags for each data YAML (same length as --data).")
    parser.add_argument("--split", type=str, default="val", help="Dataset split for validation.")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
    parser.add_argument("--batch", type=int, default=16, help="Validation batch size.")
    parser.add_argument("--device", type=str, default="0", help="Device id, e.g. 0 or cpu.")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers.")
    parser.add_argument("--conf", type=float, default=None, help="Optional confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold for validation.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image.")
    parser.add_argument("--plots", action="store_true", help="Save validation plots.")
    parser.add_argument("--save-json", action="store_true", help="Save COCO-style prediction JSON when possible.")
    parser.add_argument("--project", type=str, default="ultralytics/logcopy", help="Output project directory.")
    parser.add_argument("--name", type=str, default="eval_incremental_model", help="Output run name prefix.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow existing output directory.")

    parser.add_argument("--enable-slicing", action="store_true", help="Enable sliding-window slicing for eval.")
    parser.add_argument("--slice-size", type=int, default=512, help="Slicing tile size.")
    parser.add_argument("--slice-overlap", type=float, default=0.25, help="Slicing overlap ratio.")
    parser.add_argument("--min-box-keep-ratio", type=float, default=0.5, help="Minimum kept box area ratio per tile.")
    parser.add_argument("--cache-slices", action="store_true", help="Cache sliced tiles.")
    parser.add_argument("--slice-cache-dir", type=str, default=None, help="Optional custom slicing cache directory.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for deterministic slicing decisions.")
    return parser.parse_args()


def _results_summary(results_dict: dict[str, Any]) -> dict[str, float]:
    wanted = (
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "fitness",
    )
    out = {}
    for k in wanted:
        if k in results_dict:
            out[k] = round(float(results_dict[k]), 5)
    return out


def main() -> None:
    args = _parse_args()

    tags = args.tags if args.tags else _default_tags(args.data)
    if len(tags) != len(args.data):
        raise ValueError(f"--tags length ({len(tags)}) must match --data length ({len(args.data)}).")

    model = YOLO(args.model)
    LOGGER.info(f"Loaded model for evaluation: {args.model}")
    LOGGER.info(f"Model names: {getattr(model, 'names', None)}")

    slice_cfg = SliceConfig(
        enabled=args.enable_slicing,
        slice_size=args.slice_size,
        slice_overlap=args.slice_overlap,
        min_box_keep_ratio=args.min_box_keep_ratio,
        empty_tile_keep_prob=1.0,
        cache_slices=args.cache_slices,
        cache_dir=args.slice_cache_dir,
        seed=args.seed,
    )

    all_summaries: dict[str, dict[str, float]] = {}
    base_save_dir = Path(args.project) / args.name
    base_save_dir.mkdir(parents=True, exist_ok=True)

    for data_yaml, tag in zip(args.data, tags):
        eval_yaml = _prepare_eval_data_yaml(
            data_yaml,
            enable_slicing=args.enable_slicing,
            slice_cfg=slice_cfg,
            save_dir=base_save_dir,
            tag=tag,
        )

        LOGGER.info(f"[{tag}] evaluating data={eval_yaml} split={args.split}")
        metrics = model.val(
            data=eval_yaml,
            split=args.split,
            task="detect",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            plots=args.plots,
            save_json=args.save_json,
            save_txt=False,
            save_conf=False,
            project=args.project,
            name=f"{args.name}_{tag}",
            exist_ok=args.exist_ok,
            verbose=True,
            compile=False,
        )
        results_dict = getattr(metrics, "results_dict", {}) or {}
        all_summaries[tag] = _results_summary(results_dict)
        LOGGER.info(f"[{tag}] summary: {all_summaries[tag]}")

    LOGGER.info("Evaluation done. Consolidated summary:")
    for tag, summary in all_summaries.items():
        LOGGER.info(f"  - {tag}: {summary}")


if __name__ == "__main__":
    main()
