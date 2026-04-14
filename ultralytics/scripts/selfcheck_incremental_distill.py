from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from ultralytics.models.yolo.detect import IncrementalDistillTrainer
from ultralytics.models.yolo.detect.distill import DistillationLossWrapper
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


def _build_overrides(config_path: Path, cli_overrides: dict) -> dict:
    cfg_overrides = YAML.load(config_path)
    # Self-check defaults keep runtime small while reusing training path.
    defaults = {
        "epochs": 1,
        "workers": 0,
        "plots": False,
        "save": False,
        "val": False,
        "enable_slicing": False,
        "amp": False,
    }
    return {**cfg_overrides, **defaults, **cli_overrides}


def _pred_has_scores_feats(parsed: dict) -> tuple[bool, bool]:
    return isinstance(parsed.get("scores", None), torch.Tensor), isinstance(parsed.get("feats", None), list)


def main():
    parser = argparse.ArgumentParser(description="Self-check incremental distillation pipeline with one batch.")
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
        help="Inline overrides, e.g. --override model=/path/best.pt student_arch=yoloe yoloe_class_names=['ship','oiltank']",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    cli_overrides = _parse_kv_overrides(args.override)
    overrides = _build_overrides(cfg_path, cli_overrides)

    LOGGER.info(f"Launching IncrementalDistillTrainer self-check with config: {cfg_path}")
    trainer = IncrementalDistillTrainer(cfg=DEFAULT_CFG, overrides=overrides)

    # Reuse full setup path to verify model construction, fixed-text injection, and criterion wiring.
    trainer._setup_train()
    LOGGER.info("Self-check: setup_model/train-pipeline initialization succeeded.")

    batch = next(iter(trainer.train_loader))
    batch = trainer.preprocess_batch(batch)
    LOGGER.info(
        "Self-check: fetched one train batch "
        f"(batch_size={batch['img'].shape[0]}, image_shape={tuple(batch['img'].shape[-2:])})."
    )

    student = unwrap_model(trainer.model)

    with torch.no_grad():
        student_preds_raw = trainer.model(batch["img"])
    student_preds = DistillationLossWrapper._parse_preds(student_preds_raw)
    student_scores_ok, student_feats_ok = _pred_has_scores_feats(student_preds)
    LOGGER.info(
        "Self-check: student forward succeeded "
        f"(scores_ok={student_scores_ok}, feats_ok={student_feats_ok})."
    )

    teacher_scores_ok = False
    teacher_feats_ok = False
    if trainer.teacher_model is not None:
        t_device, t_dtype = DistillationLossWrapper._model_device_dtype(
            trainer.teacher_model,
            fallback_device=batch["img"].device,
            fallback_dtype=batch["img"].dtype,
        )
        teacher_img = batch["img"].to(device=t_device, dtype=t_dtype, non_blocking=True)
        with torch.no_grad():
            teacher_preds_raw = trainer.teacher_model(teacher_img)
        teacher_preds = DistillationLossWrapper._parse_preds(teacher_preds_raw)
        teacher_scores_ok, teacher_feats_ok = _pred_has_scores_feats(teacher_preds)
        LOGGER.info(
            "Self-check: teacher forward succeeded "
            f"(scores_ok={teacher_scores_ok}, feats_ok={teacher_feats_ok})."
        )
    else:
        LOGGER.warning("Self-check: teacher model is not configured (distillation disabled).")

    # Verify criterion path (includes distillation wrapper checks when enabled).
    with torch.no_grad():
        total_loss, loss_items = student.loss(batch, student_preds_raw)
    total_loss_value = float(total_loss.detach().mean())
    LOGGER.info(
        "Self-check: criterion/loss forward succeeded "
        f"(total_loss_mean={total_loss_value:.6f}, loss_items_shape={tuple(loss_items.shape)})."
    )

    validator = trainer.get_validator()
    LOGGER.info(f"Self-check: validator initialization succeeded ({validator.__class__.__name__}).")

    # Final strict checks
    if not (student_scores_ok and student_feats_ok):
        raise RuntimeError("Self-check failed: student predictions missing required 'scores' or 'feats'.")

    if trainer.enable_distillation:
        if trainer.teacher_model is None:
            raise RuntimeError("Self-check failed: distillation enabled but teacher model is missing.")
        if not (teacher_scores_ok and teacher_feats_ok):
            raise RuntimeError("Self-check failed: teacher predictions missing required 'scores' or 'feats'.")

    if trainer.enable_distillation:
        summary = "setup_model, fixed-text init, student+teacher forward, distill parse, validator init"
    else:
        summary = "setup_model, fixed-text init, student forward, distill-wrapper path, validator init"
    LOGGER.info(f"SELF-CHECK PASSED: {summary}.")


if __name__ == "__main__":
    main()
