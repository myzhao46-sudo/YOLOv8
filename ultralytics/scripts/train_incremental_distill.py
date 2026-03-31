from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from ultralytics.models.yolo.detect import IncrementalDistillTrainer
from ultralytics.utils import DEFAULT_CFG, LOGGER, YAML


def _parse_kv_overrides(items: list[str]) -> dict:
    parsed = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected KEY=VALUE format.")
        key, value = item.split("=", 1)
        parsed[key] = yaml.safe_load(value)
    return parsed


def main():
    parser = argparse.ArgumentParser(description="Train incremental detector with optional distillation/slicing/replay.")
    parser.add_argument(
        "--config",
        type=str,
        default="ultralytics/ultralytics/cfg/experiments/incremental_distill_v1.yaml",
        help="Path to experiment YAML.",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Inline overrides, e.g. --override epochs=50 experiment_mode=naive_finetune enable_distillation=False",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    cfg_overrides = YAML.load(cfg_path)
    cli_overrides = _parse_kv_overrides(args.override)
    overrides = {**cfg_overrides, **cli_overrides}

    LOGGER.info(f"Launching IncrementalDistillTrainer with config: {cfg_path}")
    trainer = IncrementalDistillTrainer(cfg=DEFAULT_CFG, overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()

