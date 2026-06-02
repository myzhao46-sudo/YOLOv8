# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils.ops import xywh2xyxy
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
    cls_old_score_thresh: float = 0.0
    start_epoch: int = 0
    ramp_epochs: int = 0


@dataclass
class DINOFeatureKDConfig:
    """DINO feature KD configuration."""

    enabled: bool = False
    model_name: str = "timm/convnext_tiny.dinov3_lvd1689m"
    model_backend: str = "timm"
    frozen: bool = True
    no_grad: bool = True
    loss_weight: float = 0.05
    target_class_names: tuple[str, ...] = ("tank", "oiltank")
    region_source: str = "gt_box"
    region_only: bool = True
    max_regions_per_image: int = 16
    student_feature_source: str = "head_cv4"
    teacher_feature_source: str = "last_feature_map"
    projection_dim: int = 256
    loss_type: str = "cosine"
    loss_every_n_batches: int = 1
    save_teacher: bool = False
    debug_batches: int = 5


def load_teacher_model(weights: str, device: torch.device) -> torch.nn.Module:
    """Load teacher checkpoint as a frozen eval model."""
    model, _ = load_checkpoint(weights, device=device, inplace=True, fuse=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    LOGGER.info(f"Loaded frozen teacher model from {weights}")
    return model


def build_dino_feature_teacher(model_name: str, backend: str, device: torch.device) -> torch.nn.Module:
    """Build a frozen DINO feature teacher model."""
    backend_norm = str(backend).strip().lower()
    if backend_norm != "timm":
        raise ValueError(f"Unsupported dino_model_backend='{backend}'. Currently only 'timm' is supported.")
    try:
        import timm
    except Exception as e:
        raise RuntimeError("timm is required for DINO feature KD. Please run: pip install timm") from e

    timm_name = str(model_name).strip()
    if timm_name.startswith("timm/"):
        timm_name = timm_name.split("/", 1)[1]

    model = timm.create_model(timm_name, pretrained=True, features_only=True)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class DistillationLossWrapper:
    """Wrap a base detection criterion with optional feature and classification distillation losses."""

    def __init__(self, base_criterion: Any, teacher_model: torch.nn.Module | None, cfg: DistillationConfig):
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        self.cfg = cfg
        self._teacher_head_train_hint_logged = False
        self._epoch = 0

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

    def _distill_scale(self) -> float:
        """Epoch-based distill scaling: optional delayed start + linear ramp."""
        start = max(int(self.cfg.start_epoch), 0)
        ramp = max(int(self.cfg.ramp_epochs), 0)
        if self._epoch < start:
            return 0.0
        if ramp <= 0:
            return 1.0
        return min(1.0, float(self._epoch - start + 1) / float(ramp))

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
        loss = torch.zeros((), device=self._infer_device(student_preds, fallback=device), dtype=sf[0].dtype)
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

        cls_mask = None
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
            cls_thr = float(self.cfg.cls_old_score_thresh)
            if cls_thr > 0:
                cls_mask = (torch.sigmoid(t_logit.detach()) >= cls_thr).to(dtype=s_logit.dtype)
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
        if cls_mask is None:
            cls_loss = F.binary_cross_entropy_with_logits(s_logit, target_prob)
        else:
            denom = cls_mask.sum()
            if denom.item() < 1:
                return torch.zeros((), device=self._infer_device(student_preds, fallback=device), dtype=s_logit.dtype)
            cls_raw = F.binary_cross_entropy_with_logits(s_logit, target_prob, reduction="none")
            cls_loss = (cls_raw * cls_mask).sum() / denom
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

        distill_scale = self._distill_scale()
        if distill_scale <= 0.0:
            return base_total_loss, torch.cat((base_items, distill_items))

        feat_loss = self._feature_distill_loss(student_preds, teacher_preds, device=base_total_loss.device)
        cls_loss = self._cls_distill_loss(student_preds, teacher_preds, device=base_total_loss.device)

        weighted = distill_scale * (self.cfg.feature_weight * feat_loss + self.cfg.cls_weight * cls_loss)
        distill_total_loss = weighted * batch["img"].shape[0]
        total_loss = base_total_loss + distill_total_loss

        distill_items[0] = feat_loss.detach()
        distill_items[1] = cls_loss.detach()
        return total_loss, torch.cat((base_items, distill_items))

    def update(self) -> None:
        self._epoch += 1
        if hasattr(self.base_criterion, "update"):
            self.base_criterion.update()


class DINOFeatureKDLossWrapper:
    """Wrap criterion with optional DINO region feature KD."""

    def __init__(
        self,
        base_criterion: Any,
        dino_teacher: nn.Module | None,
        cfg: DINOFeatureKDConfig,
        *,
        student_model: nn.Module | None = None,
        target_class_ids: tuple[int, ...] = (),
        class_names: tuple[str, ...] = (),
    ):
        self.base_criterion = base_criterion
        self.dino_teacher = dino_teacher
        self.cfg = cfg
        self.target_class_ids = tuple(int(x) for x in target_class_ids)
        self.class_names = tuple(str(x) for x in class_names)
        self._batch_calls = 0
        self._student_hook_features: list[torch.Tensor | None] = []
        self._hook_handles: list[Any] = []
        self._last_student_shape: tuple[int, ...] | None = None
        self._last_teacher_shape: tuple[int, ...] | None = None

        proj_dim = max(int(self.cfg.projection_dim), 1)
        self.student_projector = nn.LazyLinear(proj_dim, bias=True)
        self.teacher_projector = nn.LazyLinear(proj_dim, bias=True)
        self.student_projector.train()
        self.teacher_projector.train()
        self._register_student_feature_hooks(student_model)

    def __getattr__(self, name: str):
        base = self.__dict__.get("base_criterion", None)
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    def __getstate__(self) -> dict[str, Any]:
        """Keep checkpoints clean: no DINO teacher and no transient hook handles."""
        state = self.__dict__.copy()
        state["dino_teacher"] = None
        state["_hook_handles"] = []
        state["_student_hook_features"] = []
        return state

    def _register_student_feature_hooks(self, student_model: nn.Module | None) -> None:
        if student_model is None:
            return
        source = str(self.cfg.student_feature_source).strip().lower()
        if source != "head_cv4":
            return
        head = None
        model_list = getattr(student_model, "model", None)
        if model_list is not None:
            try:
                if len(model_list):
                    head = model_list[-1]
            except Exception:
                head = None
        cv4 = getattr(head, "cv4", None) if head is not None else None
        if not isinstance(cv4, nn.ModuleList) or len(cv4) == 0:
            LOGGER.warning(
                "DINO student feature source 'head_cv4' requested but YOLO head has no cv4 module list; "
                "falling back to neck_last."
            )
            self.cfg.student_feature_source = "neck_last"
            return
        self._student_hook_features = [None for _ in range(len(cv4))]

        for i, module in enumerate(cv4):
            def _pre_hook(_, inputs, idx=i):
                if not inputs:
                    return
                feat = inputs[0]
                if isinstance(feat, torch.Tensor):
                    self._student_hook_features[idx] = feat

            self._hook_handles.append(module.register_forward_pre_hook(_pre_hook))

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
    def _extract_last_feature_map(obj: Any) -> torch.Tensor | None:
        if isinstance(obj, torch.Tensor) and obj.ndim == 4:
            return obj
        if isinstance(obj, dict):
            keys = list(obj.keys())
            for k in reversed(keys):
                hit = DINOFeatureKDLossWrapper._extract_last_feature_map(obj[k])
                if hit is not None:
                    return hit
            return None
        if isinstance(obj, (list, tuple)):
            for item in reversed(obj):
                hit = DINOFeatureKDLossWrapper._extract_last_feature_map(item)
                if hit is not None:
                    return hit
        return None

    def _get_student_feature(self, student_preds: dict[str, torch.Tensor]) -> torch.Tensor | None:
        source = str(self.cfg.student_feature_source).strip().lower()
        if source == "head_cv4":
            for feat in reversed(self._student_hook_features):
                if isinstance(feat, torch.Tensor):
                    return feat
        feats = student_preds.get("feats", None)
        if not isinstance(feats, list) or not feats:
            return None
        if source == "neck_p3":
            return feats[0]
        return feats[-1]

    def _get_teacher_feature(self, images: torch.Tensor) -> torch.Tensor | None:
        if self.dino_teacher is None:
            return None
        if self.cfg.no_grad:
            with torch.no_grad():
                out = self.dino_teacher(images)
        else:
            out = self.dino_teacher(images)
        return self._extract_last_feature_map(out)

    @staticmethod
    def _collect_image_boxes_xyxy(
        batch: dict[str, torch.Tensor],
        target_class_ids: tuple[int, ...],
        image_h: int,
        image_w: int,
        device: torch.device,
    ) -> dict[int, torch.Tensor]:
        if not target_class_ids:
            return {}
        cls = batch["cls"].view(-1).to(device=device).long()
        batch_idx = batch["batch_idx"].view(-1).to(device=device).long()
        bboxes = batch["bboxes"].view(-1, 4).to(device=device)
        class_mask = torch.zeros_like(cls, dtype=torch.bool)
        for cid in target_class_ids:
            class_mask |= cls == int(cid)
        if not class_mask.any():
            return {}
        selected_boxes = bboxes[class_mask]
        selected_batch_idx = batch_idx[class_mask]
        scale = torch.tensor([image_w, image_h, image_w, image_h], device=device, dtype=selected_boxes.dtype)
        boxes_xyxy = xywh2xyxy(selected_boxes * scale)
        boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clamp_(0, float(image_w))
        boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clamp_(0, float(image_h))

        out: dict[int, list[torch.Tensor]] = {}
        for i in range(boxes_xyxy.shape[0]):
            bi = int(selected_batch_idx[i].item())
            x1, y1, x2, y2 = boxes_xyxy[i]
            if (x2 - x1) < 1 or (y2 - y1) < 1:
                continue
            out.setdefault(bi, []).append(boxes_xyxy[i])
        return {k: torch.stack(v, dim=0) for k, v in out.items() if v}

    @staticmethod
    def _region_pool_from_boxes(
        feat_map: torch.Tensor,
        boxes_by_image: dict[int, torch.Tensor],
        image_h: int,
        image_w: int,
        max_regions_per_image: int,
    ) -> torch.Tensor | None:
        if feat_map.ndim != 4:
            return None
        _, _, fh, fw = feat_map.shape
        pooled = []
        scale_x = fw / max(float(image_w), 1.0)
        scale_y = fh / max(float(image_h), 1.0)
        for bi, boxes in boxes_by_image.items():
            if bi >= feat_map.shape[0]:
                continue
            limit = min(int(boxes.shape[0]), max(int(max_regions_per_image), 1))
            for box in boxes[:limit]:
                x1, y1, x2, y2 = box
                fx1 = int(torch.floor(x1 * scale_x).item())
                fy1 = int(torch.floor(y1 * scale_y).item())
                fx2 = int(torch.ceil(x2 * scale_x).item())
                fy2 = int(torch.ceil(y2 * scale_y).item())
                fx1 = min(max(fx1, 0), fw - 1)
                fy1 = min(max(fy1, 0), fh - 1)
                fx2 = min(max(fx2, fx1 + 1), fw)
                fy2 = min(max(fy2, fy1 + 1), fh)
                patch = feat_map[bi : bi + 1, :, fy1:fy2, fx1:fx2]
                if patch.numel() == 0:
                    continue
                pooled.append(patch.mean(dim=(2, 3)).squeeze(0))
        if not pooled:
            return None
        return torch.stack(pooled, dim=0)

    @staticmethod
    def _any_nonzero_grad(model: nn.Module | None) -> bool:
        if model is None:
            return False
        for p in model.parameters():
            if p.grad is not None and torch.count_nonzero(p.grad).item() > 0:
                return True
        return False

    def _dino_feature_loss(
        self,
        student_preds: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, int, tuple[int, ...] | None, tuple[int, ...] | None]:
        student_feat = self._get_student_feature(student_preds)
        if student_feat is None:
            raise RuntimeError(
                f"DINO feature KD enabled but failed to resolve student feature source '{self.cfg.student_feature_source}'."
            )
        teacher_feat = self._get_teacher_feature(batch["img"])
        if teacher_feat is None:
            raise RuntimeError(
                f"DINO feature KD enabled but failed to resolve teacher feature source '{self.cfg.teacher_feature_source}'."
            )
        self._last_student_shape = tuple(student_feat.shape)
        self._last_teacher_shape = tuple(teacher_feat.shape)

        image_h, image_w = int(batch["img"].shape[-2]), int(batch["img"].shape[-1])
        boxes_by_image = self._collect_image_boxes_xyxy(
            batch,
            self.target_class_ids,
            image_h=image_h,
            image_w=image_w,
            device=device,
        )
        if not boxes_by_image:
            return torch.zeros((), device=device, dtype=dtype), 0, None, None

        student_regions = self._region_pool_from_boxes(
            student_feat,
            boxes_by_image,
            image_h=image_h,
            image_w=image_w,
            max_regions_per_image=self.cfg.max_regions_per_image,
        )
        teacher_regions = self._region_pool_from_boxes(
            teacher_feat,
            boxes_by_image,
            image_h=image_h,
            image_w=image_w,
            max_regions_per_image=self.cfg.max_regions_per_image,
        )
        if student_regions is None or teacher_regions is None:
            return torch.zeros((), device=device, dtype=dtype), 0, None, None

        n = min(student_regions.shape[0], teacher_regions.shape[0])
        if n <= 0:
            return torch.zeros((), device=device, dtype=dtype), 0, None, None
        student_regions = student_regions[:n]
        teacher_regions = teacher_regions[:n].detach()

        self.student_projector = self.student_projector.to(device=device)
        self.teacher_projector = self.teacher_projector.to(device=device)
        s_proj = self.student_projector(student_regions)
        t_proj = self.teacher_projector(teacher_regions)
        s_norm = F.normalize(s_proj, dim=-1)
        t_norm = F.normalize(t_proj, dim=-1)

        loss_type = str(self.cfg.loss_type).strip().lower()
        if loss_type == "mse":
            loss_raw = F.mse_loss(s_norm, t_norm)
        else:
            loss_raw = 1.0 - F.cosine_similarity(s_norm, t_norm, dim=-1).mean()
        return loss_raw, int(n), tuple(student_regions.shape), tuple(teacher_regions.shape)

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

        dino_item = torch.zeros(1, device=base_total_loss.device, dtype=base_total_loss.dtype)
        if not torch.is_grad_enabled() or not self.cfg.enabled or self.dino_teacher is None:
            return base_total_loss, torch.cat((base_items, dino_item))

        if self.cfg.frozen and self._any_nonzero_grad(self.dino_teacher):
            raise RuntimeError("DINO teacher has non-zero gradients, but dino_frozen=True requires no teacher grad.")

        self._batch_calls += 1
        every_n = max(int(self.cfg.loss_every_n_batches), 1)
        if self._batch_calls % every_n != 0:
            return base_total_loss, torch.cat((base_items, dino_item))

        student_preds = self._parse_preds(preds)
        if not student_preds:
            raise RuntimeError("DINO feature KD enabled but failed to parse student prediction dict.")

        dino_raw, region_count, student_region_shape, teacher_region_shape = self._dino_feature_loss(
            student_preds,
            batch,
            device=base_total_loss.device,
            dtype=base_total_loss.dtype,
        )
        if region_count == 0:
            dino_item[0] = 0.0
            return base_total_loss, torch.cat((base_items, dino_item))

        if not torch.isfinite(dino_raw).all():
            raise RuntimeError(f"DINO region loss produced invalid value: {dino_raw}")

        dino_total = float(self.cfg.loss_weight) * dino_raw * batch["img"].shape[0]
        total_loss = base_total_loss + dino_total
        dino_item[0] = dino_raw.detach()

        if self._batch_calls <= max(int(self.cfg.debug_batches), 0):
            det_for_log = base_total_loss.detach()
            if det_for_log.numel() > 1:
                det_for_log = det_for_log.mean()
            LOGGER.info(
                "DINO KD batch debug: "
                f"loss_det={float(det_for_log):.6f}, "
                f"loss_dino={float(dino_raw.detach()):.6f}, "
                f"dino_loss_weight={float(self.cfg.loss_weight):.4f}, "
                f"num_dino_regions={region_count}, "
                f"student_feature_shape={self._last_student_shape}, "
                f"dino_feature_shape={self._last_teacher_shape}, "
                f"student_region_feature_shape={student_region_shape}, "
                f"dino_region_feature_shape={teacher_region_shape}, "
                f"dino_teacher_has_grad={self._any_nonzero_grad(self.dino_teacher)}"
            )

        return total_loss, torch.cat((base_items, dino_item))

    def update(self) -> None:
        if hasattr(self.base_criterion, "update"):
            self.base_criterion.update()
