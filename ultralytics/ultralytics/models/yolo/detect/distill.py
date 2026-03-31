# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import LOGGER


@dataclass
class DistillationConfig:
    """Teacher-student distillation configuration."""

    enabled: bool = False
    feature_weight: float = 1.0
    cls_weight: float = 1.0
    temperature: float = 1.0
    student_old_class_index: int = 0
    teacher_old_class_index: int = 0


def load_teacher_model(weights: str, device: torch.device) -> torch.nn.Module:
    """Load teacher checkpoint as a frozen eval model."""
    model, _ = load_checkpoint(weights, device=device, inplace=True, fuse=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    LOGGER.info(f"Loaded frozen teacher model from {weights}")
    return model


class DistillationLossWrapper:
    """Wrap a base detection criterion with optional feature and classification distillation losses."""

    def __init__(self, base_criterion: Any, teacher_model: torch.nn.Module | None, cfg: DistillationConfig):
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        self.cfg = cfg

    def __getattr__(self, name: str):
        # Delegate unknown attributes to wrapped criterion (e.g., parse_output, assigner, update).
        base = self.__dict__.get("base_criterion", None)
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    def __getstate__(self) -> dict[str, Any]:
        """Return deepcopy-safe state and avoid serializing teacher model in checkpoints."""
        state = self.__dict__.copy()
        state["teacher_model"] = None
        return state

    @staticmethod
    def _parse_preds(
        preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        if isinstance(preds, tuple):
            preds = preds[1]
        if isinstance(preds, dict) and "one2many" in preds:
            preds = preds["one2many"]
        return preds if isinstance(preds, dict) else {}

    @staticmethod
    def _infer_device(preds: dict[str, Any], fallback: torch.device) -> torch.device:
        for v in preds.values():
            if isinstance(v, torch.Tensor):
                return v.device
            if isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, torch.Tensor):
                        return item.device
        return fallback

    @staticmethod
    def _model_device_dtype(
        model: torch.nn.Module | None, fallback_device: torch.device, fallback_dtype: torch.dtype
    ) -> tuple[torch.device, torch.dtype]:
        if model is None:
            return fallback_device, fallback_dtype
        for p in model.parameters():
            return p.device, p.dtype
        for b in model.buffers():
            return b.device, b.dtype
        return fallback_device, fallback_dtype

    def _feature_distill_loss(
        self,
        student_preds: dict[str, torch.Tensor],
        teacher_preds: dict[str, torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        sf = student_preds.get("feats", None)
        tf = teacher_preds.get("feats", None)
        if not isinstance(sf, list) or not isinstance(tf, list) or not sf or not tf:
            return torch.zeros((), device=self._infer_device(student_preds, fallback=device))

        loss = 0.0
        n = min(len(sf), len(tf))
        for i in range(n):
            s = sf[i]
            t = tf[i].detach()
            if s.shape[-2:] != t.shape[-2:]:
                t = F.interpolate(t, size=s.shape[-2:], mode="bilinear", align_corners=False)
            c = min(s.shape[1], t.shape[1])
            s = s[:, :c]
            t = t[:, :c]
            if t.dtype != s.dtype:
                t = t.to(dtype=s.dtype)
            loss = loss + F.mse_loss(s, t)
        return loss / max(n, 1)

    def _cls_distill_loss(
        self,
        student_preds: dict[str, torch.Tensor],
        teacher_preds: dict[str, torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        s_scores = student_preds.get("scores", None)
        t_scores = teacher_preds.get("scores", None)
        if s_scores is None or t_scores is None:
            return torch.zeros((), device=self._infer_device(student_preds, fallback=device))

        s_idx = min(max(self.cfg.student_old_class_index, 0), s_scores.shape[1] - 1)
        if t_scores.shape[1] == 1:
            t_idx = 0
        else:
            t_idx = min(max(self.cfg.teacher_old_class_index, 0), t_scores.shape[1] - 1)

        s_logit = s_scores[:, s_idx, :]
        t_logit = t_scores[:, t_idx, :]
        if s_logit.shape[-1] != t_logit.shape[-1]:
            n = min(s_logit.shape[-1], t_logit.shape[-1])
            s_logit = s_logit[..., :n]
            t_logit = t_logit[..., :n]
        if t_logit.dtype != s_logit.dtype:
            t_logit = t_logit.to(dtype=s_logit.dtype)

        t = max(self.cfg.temperature, 1e-6)
        if t != 1.0:
            s_logit = s_logit / t
            t_logit = t_logit / t
        target_prob = torch.sigmoid(t_logit).detach().to(dtype=s_logit.dtype)
        cls_loss = F.binary_cross_entropy_with_logits(s_logit, target_prob)
        if t != 1.0:
            cls_loss = cls_loss * (t * t)
        return cls_loss

    def __call__(
        self,
        preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base_total_loss, base_items = self.base_criterion(preds, batch)
        if not isinstance(base_items, torch.Tensor):
            base_items = torch.as_tensor(base_items, device=base_total_loss.device, dtype=base_total_loss.dtype)
        if base_items.ndim == 0:
            base_items = base_items.unsqueeze(0)

        distill_items = torch.zeros(2, device=base_total_loss.device, dtype=base_total_loss.dtype)
        if not self.cfg.enabled or self.teacher_model is None:
            return base_total_loss, torch.cat((base_items, distill_items))

        student_preds = self._parse_preds(preds)
        if not student_preds:
            return base_total_loss, torch.cat((base_items, distill_items))

        teacher_device, teacher_dtype = self._model_device_dtype(
            self.teacher_model,
            fallback_device=batch["img"].device,
            fallback_dtype=batch["img"].dtype,
        )
        teacher_img = batch["img"]
        if teacher_img.device != teacher_device or teacher_img.dtype != teacher_dtype:
            teacher_img = teacher_img.to(device=teacher_device, dtype=teacher_dtype, non_blocking=True)

        with torch.no_grad():
            teacher_out = self.teacher_model(teacher_img)
            teacher_preds = self._parse_preds(teacher_out)

        feat_loss = self._feature_distill_loss(student_preds, teacher_preds, device=base_total_loss.device)
        cls_loss = self._cls_distill_loss(student_preds, teacher_preds, device=base_total_loss.device)

        weighted = self.cfg.feature_weight * feat_loss + self.cfg.cls_weight * cls_loss
        distill_total_loss = weighted * batch["img"].shape[0]
        total_loss = base_total_loss + distill_total_loss

        distill_items[0] = feat_loss.detach()
        distill_items[1] = cls_loss.detach()
        return total_loss, torch.cat((base_items, distill_items))

    def update(self) -> None:
        if hasattr(self.base_criterion, "update"):
            self.base_criterion.update()
