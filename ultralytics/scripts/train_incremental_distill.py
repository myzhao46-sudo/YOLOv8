from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from ultralytics.models.yolo.detect import IncrementalDistillTrainer
from ultralytics.utils import DEFAULT_CFG, LOGGER, YAML
from ultralytics.utils.torch_utils import unwrap_model


def _parse_kv_overrides(items: list[str]) -> dict:
    parsed = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected KEY=VALUE format.")
        key, value = item.split("=", 1)
        parsed[key] = yaml.safe_load(value)
    return parsed


def _resolve_layer_sequence(model) -> list:
    """Resolve model layers from model.model.model (preferred) or model.model."""
    nested = getattr(getattr(model, "model", None), "model", None)
    if nested is not None:
        try:
            return list(nested)
        except TypeError:
            pass

    direct = getattr(model, "model", None)
    if direct is not None:
        try:
            return list(direct)
        except TypeError:
            pass
    return []


def _debug_model_init_state(trainer: IncrementalDistillTrainer) -> None:
    """Print lightweight diagnostics right before training starts."""
    model = unwrap_model(trainer.model)
    layers = _resolve_layer_sequence(model)

    print("[DEBUG] Pre-train model diagnostics")
    print(f"  resolved_layers={len(layers)}")

    if len(layers) > 22:
        layer22 = layers[22]
        print("  layer22 trainable params:")
        found = False
        for name, p in layer22.named_parameters():
            if p.requires_grad:
                found = True
                mean = p.data.mean().item()
                std = p.data.std(unbiased=False).item()
                print(f"    {name}: mean={mean:.6f}, std={std:.6f}, shape={tuple(p.shape)}")
        if not found:
            print("    <none>")
    else:
        print("  layer22 trainable params: <unavailable, model has fewer than 23 layers>")

    pe = None
    if hasattr(model, "pe"):
        pe = getattr(model, "pe")
    elif hasattr(getattr(model, "model", None), "pe"):
        pe = getattr(model.model, "pe")

    if isinstance(pe, torch.Tensor):
        print(f"  pe: {tuple(pe.shape)}")
    else:
        print(f"  pe: {'None' if pe is None else type(pe).__name__}")


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
    trainer.add_callback("on_pretrain_routine_end", _debug_model_init_state)
    trainer.train()


if __name__ == "__main__":
    main()

