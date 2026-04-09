from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sure local editable Ultralytics package is importable from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_ROOT = REPO_ROOT / "ultralytics"
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(ULTRALYTICS_ROOT) not in sys.path:
    sys.path.insert(0, str(ULTRALYTICS_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import torch

from custom_trainer import EWC_DetectionTrainer
from ewc_utils import (
    compute_fisher,
    load_ewc_artifacts,
    save_ewc_artifacts,
    save_old_params,
    select_topk_fisher,
)
from ultralytics.data.utils import IMG_FORMATS, check_det_dataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER


DEFAULTS = {
    "old_model_path": "ultralytics/runs/detect/ship_teacher/weights/best.pt",
    "old_data_yaml": "ultralytics/tmp_data/ship_teacher_abs.yaml",
    "new_data_yaml": "ultralytics/tmp_data/oiltank_new_eval_abs.yaml",
    "joint_val_yaml": "ultralytics/tmp_data/ship_oiltank_joint_abs.yaml",
    "default_cfg_path": "ultralytics/ultralytics/cfg/default.yaml",
    "project": "ultralytics/runs/ewc",
    "name": "oiltank_ewc",
    "fisher_cache_dir": "ultralytics/runs/ewc_cache",
}


def _resolve_path(p: str) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _must_exist(path: Path, desc: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{desc} not found: {path}")


def _count_images(split_paths: str | list[str] | None) -> int:
    if not split_paths:
        return 0
    paths = split_paths if isinstance(split_paths, list) else [split_paths]
    total = 0
    for p in paths:
        root = Path(p)
        if not root.exists():
            continue
        total += sum(1 for f in root.rglob("*") if f.is_file() and f.suffix[1:].lower() in IMG_FORMATS)
    return total


def _inspect_dataset(yaml_path: Path, tag: str) -> dict[str, int]:
    data = check_det_dataset(str(yaml_path), autodownload=False)
    counts = {
        "train": _count_images(data.get("train")),
        "val": _count_images(data.get("val")),
        "test": _count_images(data.get("test")),
    }
    LOGGER.info(
        f"[DATA] {tag} counts: train={counts['train']}, val={counts['val']}, test={counts['test']}"
    )
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 detector with EWC incremental regularization.")
    parser.add_argument("--old-model-path", type=str, default=DEFAULTS["old_model_path"])
    parser.add_argument("--old-data-yaml", type=str, default=DEFAULTS["old_data_yaml"])
    parser.add_argument("--new-data-yaml", type=str, default=DEFAULTS["new_data_yaml"])
    parser.add_argument("--old-val-yaml", type=str, default=None)
    parser.add_argument("--new-val-yaml", type=str, default=None)
    parser.add_argument("--joint-val-yaml", type=str, default=DEFAULTS["joint_val_yaml"])
    parser.add_argument("--default-cfg-path", type=str, default=DEFAULTS["default_cfg_path"])
    parser.add_argument("--project", type=str, default=DEFAULTS["project"])
    parser.add_argument("--name", type=str, default=DEFAULTS["name"])

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--optimizer", type=str, default="auto")
    parser.add_argument("--fraction", type=float, default=1.0)

    parser.add_argument("--lambda-ewc", type=float, default=500.0)
    parser.add_argument("--ewc-log-interval", type=int, default=20)
    parser.add_argument("--extra-eval-interval", type=int, default=10, help="Run old/new/joint eval every N epochs")
    parser.add_argument("--topk-ratio", "--fisher-topk-ratio", dest="topk_ratio", type=float, default=1.0)
    parser.add_argument("--fisher-max-batches", type=int, default=0, help="0 means full old-task dataloader")
    parser.add_argument("--fisher-cache-dir", type=str, default=DEFAULTS["fisher_cache_dir"])
    parser.add_argument("--reuse-fisher", action="store_true", help="Reuse old_params/fisher from --fisher-cache-dir")

    parser.add_argument("--val", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--smoke", action="store_true", help="Run a quick connectivity check with tiny settings")
    return parser.parse_args()


def apply_smoke_profile(args: argparse.Namespace) -> None:
    args.epochs = 1
    args.batch = min(args.batch, 4)
    args.imgsz = min(args.imgsz, 320)
    args.workers = 0
    args.fraction = min(args.fraction, 0.01)
    args.fisher_max_batches = 1
    args.val = False
    args.plots = False
    args.save = False
    args.name = f"{args.name}_smoke"


def build_fisher(args: argparse.Namespace, old_model_path: Path, old_data_yaml: Path, default_cfg_path: Path) -> tuple[dict, dict]:
    LOGGER.info("===== Step 1: Compute Fisher on old task =====")
    fisher_overrides = {
        "model": str(old_model_path),
        "data": str(old_data_yaml),
        "epochs": 1,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "workers": args.workers,
        "device": args.device,
        "project": args.project,
        "name": f"{args.name}_fisher_tmp",
        "save": False,
        "plots": False,
        "val": False,
        "fraction": args.fraction,
        "resume": False,
    }
    temp_trainer = DetectionTrainer(cfg=str(default_cfg_path), overrides=fisher_overrides)

    # Ultralytics 8.4.32 signature is _setup_train() without world_size argument.
    temp_trainer._setup_train()
    LOGGER.info(f"[DATA] old task effective train images (after trainer build): {len(temp_trainer.train_loader.dataset)}")

    old_params = save_old_params(temp_trainer.model)
    max_batches = args.fisher_max_batches if args.fisher_max_batches > 0 else None
    fisher = compute_fisher(
        model=temp_trainer.model,
        dataloader=temp_trainer.train_loader,
        trainer=temp_trainer,
        device=temp_trainer.device,
        max_batches=max_batches,
    )
    return old_params, fisher


def main() -> None:
    args = parse_args()
    if args.smoke:
        apply_smoke_profile(args)

    old_model_path = _resolve_path(args.old_model_path)
    old_data_yaml = _resolve_path(args.old_data_yaml)
    new_data_yaml = _resolve_path(args.new_data_yaml)
    old_val_yaml = _resolve_path(args.old_val_yaml) if args.old_val_yaml else old_data_yaml
    new_val_yaml = _resolve_path(args.new_val_yaml) if args.new_val_yaml else new_data_yaml
    joint_val_yaml = _resolve_path(args.joint_val_yaml) if args.joint_val_yaml else None
    default_cfg_path = _resolve_path(args.default_cfg_path)
    fisher_cache_dir = _resolve_path(args.fisher_cache_dir)

    _must_exist(old_model_path, "old_model_path")
    _must_exist(old_data_yaml, "old_data_yaml")
    _must_exist(new_data_yaml, "new_data_yaml")
    _must_exist(old_val_yaml, "old_val_yaml")
    _must_exist(new_val_yaml, "new_val_yaml")
    if joint_val_yaml:
        _must_exist(joint_val_yaml, "joint_val_yaml")
    _must_exist(default_cfg_path, "default_cfg_path")

    old_counts = _inspect_dataset(old_data_yaml, tag="old_task")
    new_counts = _inspect_dataset(new_data_yaml, tag="new_task")
    old_val_counts = _inspect_dataset(old_val_yaml, tag="old_eval")
    new_val_counts = _inspect_dataset(new_val_yaml, tag="new_eval")
    if joint_val_yaml:
        joint_val_counts = _inspect_dataset(joint_val_yaml, tag="joint_eval")
    else:
        joint_val_counts = {"val": 0}

    LOGGER.info("===== EWC config =====")
    LOGGER.info(f"old_model_path: {old_model_path}")
    LOGGER.info(f"old_data_yaml: {old_data_yaml}")
    LOGGER.info(f"new_data_yaml: {new_data_yaml}")
    LOGGER.info(f"default_cfg_path: {default_cfg_path}")
    LOGGER.info(f"project/name: {args.project}/{args.name}")
    LOGGER.info(f"device: {args.device}")
    LOGGER.info(f"old_val_yaml: {old_val_yaml}")
    LOGGER.info(f"new_val_yaml: {new_val_yaml}")
    LOGGER.info(f"joint_val_yaml: {joint_val_yaml}")
    LOGGER.info(f"lambda_ewc: {args.lambda_ewc}")
    LOGGER.info(f"topk_ratio: {args.topk_ratio}")
    LOGGER.info(f"extra_eval_interval: {args.extra_eval_interval}")
    LOGGER.info(f"old task train image count: {old_counts['train']}")
    LOGGER.info(f"new task train image count: {new_counts['train']}")
    LOGGER.info(f"new task val image count: {new_counts['val']}")
    LOGGER.info(f"old eval val image count: {old_val_counts['val']}")
    LOGGER.info(f"new eval val image count: {new_val_counts['val']}")
    LOGGER.info(f"joint eval val image count: {joint_val_counts['val']}")
    if "eval" in new_data_yaml.name.lower():
        LOGGER.warning(
            f"new_data_yaml filename contains 'eval': {new_data_yaml.name}. "
            "Checked split paths show train split exists; verify this matches your experiment protocol."
        )
    if new_counts["train"] == 0:
        raise RuntimeError("new_data_yaml has zero training images.")
    LOGGER.warning(
        "Current EWC script trains on new_data only; old/new/joint are evaluation splits, not replay training."
    )

    if args.reuse_fisher:
        old_params, fisher = load_ewc_artifacts(fisher_cache_dir)
    else:
        old_params, fisher = build_fisher(args, old_model_path, old_data_yaml, default_cfg_path)
        save_ewc_artifacts(fisher_cache_dir, old_params, fisher)

    fisher_mask = select_topk_fisher(fisher, topk_ratio=args.topk_ratio)
    mask_total = sum(mask.numel() for mask in fisher_mask.values())
    mask_kept = sum(float(mask.sum()) for mask in fisher_mask.values())
    LOGGER.info(
        f"[EWC] fisher mask generated: tensors={len(fisher_mask)}, kept_ratio={mask_kept / max(mask_total, 1):.6f}"
    )

    LOGGER.info("===== Step 2: Train new task with EWC =====")
    train_overrides = {
        "model": str(old_model_path),
        "data": str(new_data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "project": args.project,
        "name": args.name,
        "resume": False,
        "optimizer": args.optimizer,
        "fraction": args.fraction,
        "val": args.val,
        "plots": args.plots,
        "save": args.save,
    }

    trainer = EWC_DetectionTrainer(
        cfg=str(default_cfg_path),
        overrides=train_overrides,
        old_params=old_params,
        fisher=fisher,
        fisher_mask=fisher_mask,
        lambda_ewc=args.lambda_ewc,
        ewc_log_interval=args.ewc_log_interval,
        extra_eval_interval=args.extra_eval_interval,
        old_val_data=str(old_val_yaml),
        new_val_data=str(new_val_yaml),
        joint_val_data=str(joint_val_yaml) if joint_val_yaml else None,
    )
    trainer.train()


if __name__ == "__main__":
    main()
