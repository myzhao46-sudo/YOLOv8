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
    cls_only_old_classes: bool = True
    student_old_class_indices: tuple[int, ...] = ()
    teacher_old_class_indices: tuple[int, ...] = ()
    feature_mode: str = "global"  # global | old_only
    feature_old_score_thresh: float = 0.3


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
        self._teacher_head_train_hint_logged = False

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

    def _forward_teacher_raw(self, teacher_img: torch.Tensor) -> Any:
        """Run teacher forward while forcing detection/seg head to return raw train-style preds dict."""
        if self.teacher_model is None:
            return None

        head = None
        model_list = getattr(self.teacher_model, "model", None)
        if model_list is not None:
            try:
                if len(model_list):
                    head = model_list[-1]
            except Exception:
                head = None

        head_training = None
        if isinstance(head, torch.nn.Module):
            head_training = bool(getattr(head, "training", False))
            # Only flip top-level head.training flag to bypass inference concat/postprocess path.
            # Do not call head.train() to avoid recursively changing BN/dropout submodules.
            head.training = True
            if not self._teacher_head_train_hint_logged:
                LOGGER.info("Teacher forward uses head.training=True path to return raw distill predictions.")
                self._teacher_head_train_hint_logged = True

        try:
            return self.teacher_model(teacher_img)
        finally:
            if isinstance(head, torch.nn.Module) and head_training is not None:
                head.training = head_training

    @staticmethod
    def _resolve_class_indices(
        requested: tuple[int, ...] | list[int] | None,
        class_count: int,
        fallback_index: int,
    ) -> list[int]:
        if class_count <= 0:
            return []
        indices = list(requested) if requested else [fallback_index]
        resolved = []
        for idx in indices:
            idx = min(max(int(idx), 0), class_count - 1)
            if idx not in resolved:
                resolved.append(idx)
        return resolved

    def _build_teacher_old_prob_maps(
        self,
        teacher_preds: dict[str, torch.Tensor],
        feature_shapes: list[tuple[int, int]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[torch.Tensor] | None:
        t_scores = teacher_preds.get("scores", None)
        if not isinstance(t_scores, torch.Tensor) or t_scores.ndim < 3:
            return None

        old_ids = self._resolve_class_indices(
            self.cfg.teacher_old_class_indices,
            class_count=t_scores.shape[1],
            fallback_index=self.cfg.teacher_old_class_index,
        )
        if not old_ids:
            return None

        old_prob = torch.sigmoid(t_scores[:, old_ids, :].detach()).amax(dim=1, keepdim=True).to(device=device, dtype=dtype)
        expected_points = sum(h * w for h, w in feature_shapes)
        if expected_points <= 0:
            return None

        if old_prob.shape[-1] < expected_points:
            old_prob = F.pad(old_prob, (0, expected_points - old_prob.shape[-1]))

        start = 0
        maps = []
        for h, w in feature_shapes:
            points = h * w
            chunk = old_prob[..., start : start + points]
            if chunk.shape[-1] < points:
                chunk = F.pad(chunk, (0, points - chunk.shape[-1]))
            maps.append(chunk.reshape(old_prob.shape[0], 1, h, w))
            start += points
        return maps

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

        feature_mode = str(self.cfg.feature_mode).strip().lower()
        use_old_only = feature_mode == "old_only"
        loss = 0.0
        n = min(len(sf), len(tf))
        old_prob_maps = None
        if use_old_only:
            teacher_shapes = [(int(tf[i].shape[-2]), int(tf[i].shape[-1])) for i in range(n)]
            old_prob_maps = self._build_teacher_old_prob_maps(
                teacher_preds,
                teacher_shapes,
                device=self._infer_device(student_preds, fallback=device),
                dtype=sf[0].dtype,
            )
            if not old_prob_maps:
                return torch.zeros((), device=self._infer_device(student_preds, fallback=device), dtype=sf[0].dtype)

        valid_levels = 0
        for i in range(n):
            s = sf[i]
            t = tf[i].detach()
            mask = None
            if old_prob_maps is not None and i < len(old_prob_maps):
                mask = old_prob_maps[i]
            if s.shape[-2:] != t.shape[-2:]:
                t = F.interpolate(t, size=s.shape[-2:], mode="bilinear", align_corners=False)
            if mask is not None and mask.shape[-2:] != s.shape[-2:]:
                mask = F.interpolate(mask, size=s.shape[-2:], mode="nearest")
            c = min(s.shape[1], t.shape[1])
            s = s[:, :c]
            t = t[:, :c]
            if t.dtype != s.dtype:
                t = t.to(dtype=s.dtype)

            if use_old_only and mask is None:
                continue
            if mask is None:
                loss = loss + F.mse_loss(s, t)
                valid_levels += 1
                continue

            m = (mask >= float(self.cfg.feature_old_score_thresh)).to(dtype=s.dtype)
            m = m.expand(-1, c, -1, -1)
            denom = m.sum()
            if denom.item() < 1:
                continue
            loss = loss + ((s - t).square() * m).sum() / denom
            valid_levels += 1

        return loss / max(valid_levels, 1)

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

        if self.cfg.cls_only_old_classes:
            s_ids = self._resolve_class_indices(
                self.cfg.student_old_class_indices,
                class_count=s_scores.shape[1],
                fallback_index=self.cfg.student_old_class_index,
            )
            if t_scores.shape[1] == 1:
                t_ids = [0] * len(s_ids)
            else:
                t_ids = self._resolve_class_indices(
                    self.cfg.teacher_old_class_indices,
                    class_count=t_scores.shape[1],
                    fallback_index=self.cfg.teacher_old_class_index,
                )
                if len(t_ids) == 1 and len(s_ids) > 1:
                    t_ids = t_ids * len(s_ids)
            pair_count = min(len(s_ids), len(t_ids))
            if pair_count == 0:
                return torch.zeros((), device=self._infer_device(student_preds, fallback=device))
            s_logits = [s_scores[:, s_ids[i], :] for i in range(pair_count)]
            t_logits = [t_scores[:, t_ids[i], :] for i in range(pair_count)]
            s_logit = torch.cat(s_logits, dim=0)
            t_logit = torch.cat(t_logits, dim=0)
        else:
            if self.cfg.student_old_class_indices and self.cfg.teacher_old_class_indices:
                s_ids = [min(max(int(i), 0), s_scores.shape[1] - 1) for i in self.cfg.student_old_class_indices]
                t_ids = [min(max(int(i), 0), t_scores.shape[1] - 1) for i in self.cfg.teacher_old_class_indices]
                pair_count = min(len(s_ids), len(t_ids))
                if pair_count > 0:
                    s_logits = [s_scores[:, s_ids[i], :] for i in range(pair_count)]
                    t_logits = [t_scores[:, t_ids[i], :] for i in range(pair_count)]
                    s_logit = torch.cat(s_logits, dim=0)
                    t_logit = torch.cat(t_logits, dim=0)
                else:
                    return torch.zeros((), device=self._infer_device(student_preds, fallback=device))
            else:
                n_cls = min(s_scores.shape[1], t_scores.shape[1])
                if n_cls <= 0:
                    return torch.zeros((), device=self._infer_device(student_preds, fallback=device))
                s_logit = s_scores[:, :n_cls, :]
                t_logit = t_scores[:, :n_cls, :]
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
        # Validation runs under torch.no_grad(); skip teacher distillation there to avoid unstable
        # teacher inference-side paths and keep val loss consistent with base detector loss.
        if not torch.is_grad_enabled() or not self.cfg.enabled or self.teacher_model is None:
            return base_total_loss, torch.cat((base_items, distill_items))

        student_preds = self._parse_preds(preds)
        if not student_preds:
            raise RuntimeError(
                "Distillation is enabled but student predictions could not be parsed into the expected dict format."
            )

        teacher_device, teacher_dtype = self._model_device_dtype(
            self.teacher_model,
            fallback_device=batch["img"].device,
            fallback_dtype=batch["img"].dtype,
        )
        teacher_img = batch["img"]
        if teacher_img.device != teacher_device or teacher_img.dtype != teacher_dtype:
            teacher_img = teacher_img.to(device=teacher_device, dtype=teacher_dtype, non_blocking=True)

        with torch.no_grad():
            teacher_out = self._forward_teacher_raw(teacher_img)
            teacher_preds = self._parse_preds(teacher_out)
        if not teacher_preds:
            raise RuntimeError(
                "Distillation is enabled but teacher predictions could not be parsed into the expected dict format."
            )

        if self.cfg.feature_weight > 0 and (
            not isinstance(student_preds.get("feats", None), list) or not isinstance(teacher_preds.get("feats", None), list)
        ):
            raise RuntimeError(
                "Feature distillation requires both student and teacher predictions to include list-type 'feats'."
            )
        if self.cfg.feature_weight > 0 and str(self.cfg.feature_mode).strip().lower() == "old_only" and (
            student_preds.get("scores", None) is None or teacher_preds.get("scores", None) is None
        ):
            raise RuntimeError(
                "Feature distill mode='old_only' requires both student and teacher predictions to include 'scores'."
            )
        if self.cfg.cls_weight > 0 and (
            student_preds.get("scores", None) is None or teacher_preds.get("scores", None) is None
        ):
            raise RuntimeError(
                "Classification distillation requires both student and teacher predictions to include 'scores'."
            )

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
