from __future__ import annotations

from copy import copy, deepcopy
from typing import Any

import torch

from ultralytics.models import yolo
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.utils.torch_utils import unwrap_model


class EWC_DetectionTrainer(DetectionTrainer):
    """Detection trainer that adds EWC penalty on top of native YOLO detection loss."""

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks: dict | None = None,
        old_params: dict[str, torch.Tensor] | None = None,
        fisher: dict[str, torch.Tensor] | None = None,
        fisher_mask: dict[str, torch.Tensor] | None = None,
        lambda_ewc: float = 0.0,
        ewc_log_interval: int = 50,
        extra_eval_interval: int = 10,
        old_val_data: str | None = None,
        new_val_data: str | None = None,
        joint_val_data: str | None = None,
    ):
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        self.old_params = self._normalize_state(old_params)
        self.fisher = self._normalize_state(fisher)
        self.fisher_mask = self._normalize_state(fisher_mask)
        self.lambda_ewc = float(lambda_ewc)
        self.ewc_log_interval = max(int(ewc_log_interval), 0)

        self._ewc_hook_installed = False
        self._ewc_loss_calls = 0
        self._ewc_matched: list[str] = []
        self._ewc_shape_skipped: list[str] = []
        self._ewc_missing_skipped: list[str] = []
        self._ewc_param_refs: dict[str, torch.nn.Parameter] = {}

        self.ewc_last_base_loss = 0.0
        self.ewc_last_loss = 0.0
        self.ewc_last_total_loss = 0.0
        self.extra_eval_interval = max(int(extra_eval_interval), 0)
        self._extra_eval_data = {
            "old": old_val_data,
            "new": new_val_data,
            "joint": joint_val_data,
        }
        self._extra_metric_keys = ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)")

    @staticmethod
    def _normalize_state(state: dict[str, torch.Tensor] | None) -> dict[str, torch.Tensor]:
        if not state:
            return {}
        return {k: v.detach().cpu() for k, v in state.items()}

    def _setup_train(self):
        # 8.4.32 expects _setup_train() without world_size argument.
        super()._setup_train()
        self._prepare_ewc_param_cache()
        self._install_ewc_loss_hook()

    def _prepare_ewc_param_cache(self) -> None:
        model = unwrap_model(self.model)
        self._ewc_param_refs = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self._ewc_matched.clear()
        self._ewc_shape_skipped.clear()
        self._ewc_missing_skipped.clear()

        for name, p in self._ewc_param_refs.items():
            old = self.old_params.get(name)
            fisher = self.fisher.get(name)
            if old is None or fisher is None:
                self._ewc_missing_skipped.append(name)
                continue
            if p.shape != old.shape or p.shape != fisher.shape:
                self._ewc_shape_skipped.append(
                    f"{name}: model={tuple(p.shape)} old={tuple(old.shape)} fisher={tuple(fisher.shape)}"
                )
                continue
            self._ewc_matched.append(name)

        detect_prefix = None
        if hasattr(model, "model"):
            detect_prefix = f"model.{len(model.model) - 1}."
        detect_head_matched = [n for n in self._ewc_matched if detect_prefix and n.startswith(detect_prefix)]

        LOGGER.info(f"[EWC] matched params: {len(self._ewc_matched)}")
        if self._ewc_matched:
            LOGGER.info("[EWC] matched parameter names:\n" + "\n".join(self._ewc_matched))
        LOGGER.info(f"[EWC] shape-mismatch skipped params: {len(self._ewc_shape_skipped)}")
        if self._ewc_shape_skipped:
            LOGGER.info("[EWC] shape-mismatch details:\n" + "\n".join(self._ewc_shape_skipped))
        LOGGER.info(f"[EWC] missing-in-old-or-fisher skipped params: {len(self._ewc_missing_skipped)}")
        LOGGER.info(f"[EWC] skipped params total: {len(self._ewc_shape_skipped) + len(self._ewc_missing_skipped)}")
        LOGGER.info(
            f"[EWC] detect head matched: {bool(detect_head_matched)} (count={len(detect_head_matched)})"
        )
        if detect_head_matched:
            LOGGER.info("[EWC] detect head matched names:\n" + "\n".join(detect_head_matched))
        if self.fisher_mask:
            total = sum(mask.numel() for mask in self.fisher_mask.values())
            kept = sum(float(mask.sum()) for mask in self.fisher_mask.values())
            LOGGER.info(f"[EWC] attached fisher_mask kept_ratio={kept / max(total, 1):.6f}")

    def _install_ewc_loss_hook(self) -> None:
        if self._ewc_hook_installed:
            return
        model = unwrap_model(self.model)
        orig_loss = model.loss

        def ewc_loss_wrapper(batch, preds=None):
            base_loss, loss_items = orig_loss(batch, preds)
            ewc_loss = self.compute_ewc_loss()
            total_loss = base_loss + ewc_loss
            base_scalar = base_loss.sum() if isinstance(base_loss, torch.Tensor) and base_loss.ndim else base_loss
            total_scalar = total_loss.sum() if isinstance(total_loss, torch.Tensor) and total_loss.ndim else total_loss

            self.ewc_last_base_loss = float(base_scalar.detach())
            self.ewc_last_loss = float(ewc_loss.detach())
            self.ewc_last_total_loss = float(total_scalar.detach())
            self._ewc_loss_calls += 1

            if self.ewc_log_interval and (
                self._ewc_loss_calls == 1 or self._ewc_loss_calls % self.ewc_log_interval == 0
            ):
                LOGGER.info(
                    f"[EWC] base_det_loss={self.ewc_last_base_loss:.6f} "
                    f"ewc_loss={self.ewc_last_loss:.6f} total_loss={self.ewc_last_total_loss:.6f}"
                )

            return total_loss, loss_items

        model.loss = ewc_loss_wrapper
        self._ewc_hook_installed = True

    def _validator_args(self) -> dict[str, Any]:
        return vars(copy(self.args)).copy()

    def validate(self):
        metrics, fitness = super().validate()
        if metrics is None:
            return metrics, fitness
        extra_metrics = self._empty_extra_metrics()
        if self._should_run_extra_validations():
            extra_metrics.update(self._run_extra_validations())
        else:
            epoch = int(getattr(self, "epoch", -1)) + 1
            LOGGER.info(
                f"[EWC] skip old/new/joint extra eval at epoch {epoch}; extra_eval_interval={self.extra_eval_interval}"
            )
        metrics.update(extra_metrics)
        return metrics, fitness

    def _empty_extra_metrics(self) -> dict[str, float]:
        empty = {}
        for tag, data_yaml in self._extra_eval_data.items():
            if not data_yaml:
                continue
            for key in self._extra_metric_keys:
                empty[f"{tag}/{key}"] = float("nan")
        return empty

    def _should_run_extra_validations(self) -> bool:
        if not any(self._extra_eval_data.values()):
            return False
        epoch = int(getattr(self, "epoch", -1)) + 1
        total_epochs = int(getattr(self, "epochs", epoch))
        is_final = epoch >= total_epochs
        if is_final:
            return True
        if self.extra_eval_interval <= 0:
            return False
        return epoch % self.extra_eval_interval == 0

    def _run_extra_validations(self) -> dict[str, float]:
        model_for_eval = self.ema.ema if self.ema else unwrap_model(self.model)
        results = {}
        for tag, data_yaml in self._extra_eval_data.items():
            if not data_yaml:
                continue
            args = self._validator_args()
            args["data"] = str(data_yaml)
            args["mode"] = "val"
            args["split"] = "val"
            args["plots"] = False
            args["save_json"] = False
            args["save_txt"] = False
            args["save_conf"] = False
            args["compile"] = False

            validator = yolo.detect.DetectionValidator(
                dataloader=None,
                save_dir=self.save_dir / f"val_{tag}",
                args=args,
                _callbacks=self.callbacks,
            )
            eval_model = deepcopy(unwrap_model(model_for_eval)).eval()
            stats = validator(model=eval_model)
            if not isinstance(stats, dict):
                continue
            for k, v in stats.items():
                if k == "fitness":
                    continue
                results[f"{tag}/{k}"] = round(float(v), 5)
        return results

    def compute_ewc_loss(self) -> torch.Tensor:
        if self.lambda_ewc <= 0 or not self._ewc_matched:
            return torch.zeros((), device=self.device)

        penalty = torch.zeros((), device=self.device)
        for name in self._ewc_matched:
            p = self._ewc_param_refs[name]
            old_p = self.old_params[name].to(device=p.device, dtype=p.dtype, non_blocking=True)
            fisher = self.fisher[name].to(device=p.device, dtype=p.dtype, non_blocking=True)
            if self.fisher_mask and name in self.fisher_mask:
                mask = self.fisher_mask[name].to(device=p.device, dtype=p.dtype, non_blocking=True)
            else:
                mask = 1.0
            penalty = penalty + (fisher * (p - old_p).pow(2) * mask).sum()

        return 0.5 * self.lambda_ewc * penalty
