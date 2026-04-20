# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import importlib.util
import re
from copy import deepcopy
from copy import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from ultralytics.data.utils import check_det_dataset
from ultralytics.models import yolo
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.distill import DistillationConfig, DistillationLossWrapper, load_teacher_model
from ultralytics.data.sliding_window import SliceConfig, prepare_sliced_image_paths
from ultralytics.nn.tasks import YOLOEModel, yaml_model_load
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, YAML
from ultralytics.utils.checks import check_yaml
from ultralytics.utils.torch_utils import unwrap_model


class IncrementalDistillTrainer(DetectionTrainer):
    """Incremental detection trainer with optional replay, slicing, and teacher-student distillation."""

    _CUSTOM_OVERRIDE_KEYS = {
        "experiment_mode",
        "enable_distillation",
        "enable_replay",
        "enable_slicing",
        "teacher_weights",
        "distill_feature_weight",
        "distill_cls_weight",
        "distill_temperature",
        "distill_only_old_classes",
        "distill_auto_class_align",
        "distill_old_class_ids",
        "distill_teacher_old_class_ids",
        "distill_feature_mode",
        "distill_feature_old_score_thr",
        "distill_cls_old_score_thr",
        "distill_start_epoch",
        "distill_ramp_epochs",
        "distill_relax_single_class_teacher",
        "distill_single_class_teacher_scale",
        "distill_student_old_class",
        "distill_teacher_old_class",
        "old_val_data",
        "new_val_data",
        "joint_val_data",
        "replay_data",
        "slice_size",
        "slice_overlap",
        "min_box_keep_ratio",
        "empty_tile_keep_prob",
        "cache_slices",
        "slice_cache_dir",
        "student_arch",
        "yoloe_class_names",
        "yoloe_prompt_alias_names",
        "yoloe_prompt_mode",
        "yoloe_allow_seg_init",
        "yoloe_zero_embedding_fallback",
        "distill_student_old_class_name",
        "distill_teacher_old_class_name",
    }
    _DISTILL_MODES = {"distill_only", "replay_distill"}
    _REPLAY_MODES = {"replay_only", "replay_distill"}
    _SUPPORTED_MODES = {"naive_finetune", "distill_only", "replay_only", "replay_distill"}
    _SUPPORTED_STUDENT_ARCH = {"auto", "yolov8", "yoloe"}

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        overrides = (overrides or {}).copy()
        custom = {k: overrides.pop(k) for k in list(overrides) if k in self._CUSTOM_OVERRIDE_KEYS}
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        for k, v in custom.items():
            setattr(self.args, k, v)

        mode = str(getattr(self.args, "experiment_mode", "naive_finetune"))
        if mode not in self._SUPPORTED_MODES:
            LOGGER.warning(
                f"Unsupported experiment_mode='{mode}', falling back to 'naive_finetune'. "
                f"Supported: {sorted(self._SUPPORTED_MODES)}"
            )
            mode = "naive_finetune"
        self.experiment_mode = mode

        self.enable_distillation = self._get_bool(
            "enable_distillation", default=self.experiment_mode in self._DISTILL_MODES
        )
        self.enable_replay = self._get_bool("enable_replay", default=self.experiment_mode in self._REPLAY_MODES)
        self.enable_slicing = self._get_bool("enable_slicing", default=False)
        self.student_arch = self._resolve_student_arch()
        self.yoloe_prompt_mode = str(self._get_arg("yoloe_prompt_mode", "fixed_text")).strip().lower()
        if self.student_arch == "yoloe" and self.yoloe_prompt_mode != "fixed_text":
            raise ValueError(
                f"Unsupported yoloe_prompt_mode='{self.yoloe_prompt_mode}'. Only 'fixed_text' is currently supported."
            )
        self.slice_cfg = SliceConfig(
            enabled=self.enable_slicing,
            slice_size=int(self._get_arg("slice_size", 512)),
            slice_overlap=float(self._get_arg("slice_overlap", 0.25)),
            min_box_keep_ratio=float(self._get_arg("min_box_keep_ratio", 0.5)),
            empty_tile_keep_prob=float(self._get_arg("empty_tile_keep_prob", 0.3)),
            cache_slices=self._get_bool("cache_slices", default=True),
            cache_dir=self._get_arg("slice_cache_dir", None),
            seed=int(self._get_arg("seed", 0)),
        )

        self.teacher_model = None
        self._sliced_eval_yaml: dict[str, str] = {}

        if RANK in {-1, 0}:
            LOGGER.info(
                "Incremental trainer config: "
                f"mode={self.experiment_mode}, "
                f"student_arch={self.student_arch}, "
                f"slicing={self.enable_slicing}, "
                f"distill={self.enable_distillation}, "
                f"replay={self.enable_replay}"
            )

    def _get_arg(self, key: str, default: Any = None) -> Any:
        value = getattr(self.args, key, default)
        return default if value is None else value

    def _get_bool(self, key: str, default: bool = False) -> bool:
        value = getattr(self.args, key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def _resolve_student_arch(self) -> str:
        arch = str(self._get_arg("student_arch", "auto")).strip().lower()
        if arch not in self._SUPPORTED_STUDENT_ARCH:
            raise ValueError(
                f"Unsupported student_arch='{arch}'. Supported: {sorted(self._SUPPORTED_STUDENT_ARCH)}"
            )

        if arch == "auto":
            model_stem = Path(str(getattr(self.args, "model", ""))).stem.lower()
            return "yoloe" if "yoloe" in model_stem else "yolov8"
        return arch

    @staticmethod
    def _normalize_names(names: Any) -> list[str]:
        if isinstance(names, dict):
            return [str(v) for _, v in sorted(names.items(), key=lambda x: int(x[0]))]
        if isinstance(names, (list, tuple)):
            return [str(v) for v in names]
        return []

    @staticmethod
    def _parse_class_names(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [x.strip() for x in value.split(",") if x.strip()]
        if isinstance(value, (list, tuple)):
            return [str(x).strip() for x in value if str(x).strip()]
        raise TypeError(
            f"Expected yoloe_class_names to be None, comma-separated string, or list/tuple, but got {type(value)}."
        )

    @staticmethod
    def _parse_class_indices(value: Any) -> list[int]:
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("[") and text.endswith("]"):
                text = text[1:-1]
            tokens = [x.strip() for x in text.split(",") if x.strip()]
            values = tokens
        elif isinstance(value, (list, tuple)):
            values = list(value)
        else:
            values = [value]

        indices = []
        for v in values:
            idx = int(v)
            if idx not in indices:
                indices.append(idx)
        return indices

    @staticmethod
    def _resolve_class_index(
        names: list[str],
        class_name: Any,
        fallback_index: int,
        *,
        scope: str,
    ) -> int:
        if class_name is None:
            return int(fallback_index)
        if not isinstance(class_name, str):
            return int(class_name)
        if not names:
            raise ValueError(f"Cannot resolve {scope} class name '{class_name}' because class names are unavailable.")
        lookup = {str(name): i for i, name in enumerate(names)}
        if class_name not in lookup:
            raise ValueError(
                f"{scope} class name '{class_name}' not found in available names: {names}. "
                "Please fix the class name or use explicit index."
            )
        return int(lookup[class_name])

    @staticmethod
    def _names_for_indices(names: list[str], indices: list[int]) -> list[str]:
        resolved = []
        for idx in indices:
            if 0 <= idx < len(names):
                resolved.append(names[idx])
            else:
                resolved.append(f"<out_of_range:{idx}>")
        return resolved

    @staticmethod
    def _infer_yoloe_head(weights: Any) -> tuple[str, bool]:
        """Return (head_name, has_prompt_free_lrpc) from a loaded checkpoint model."""
        head_name = ""
        has_lrpc = False
        with_head = getattr(weights, "model", None)
        if isinstance(with_head, (list, tuple)) and with_head:
            head = with_head[-1]
            head_name = head.__class__.__name__.lower()
            has_lrpc = hasattr(head, "lrpc")
        elif hasattr(with_head, "__getitem__"):
            try:
                head = with_head[-1]
                head_name = head.__class__.__name__.lower()
                has_lrpc = hasattr(head, "lrpc")
            except Exception:
                head_name = ""
                has_lrpc = False
        return head_name, has_lrpc

    @staticmethod
    def _seg_source_to_detect_yaml(source: str | None) -> str:
        """Convert a YOLOE seg/seg-pf source reference to its detect YAML reference."""
        if not source:
            raise ValueError("Unable to infer detect YAML for YOLOE: source reference is empty.")
        src = str(source).strip()
        src_path = Path(src)
        detect_stem = src_path.stem.replace("-seg-pf", "").replace("-seg", "")
        if not detect_stem:
            raise ValueError(f"Unable to derive detect YAML from source '{source}'.")
        return str(src_path.with_name(f"{detect_stem}.yaml"))

    @staticmethod
    def _extract_yoloe_scale(stem: str) -> str:
        """Extract YOLOE scale char from a model stem, e.g. yoloe-26n-seg -> n."""
        stem = stem.lower()
        stem = stem.replace("-seg-pf", "").replace("-seg-det", "").replace("-seg", "")
        m = re.match(r"^yoloe-(?:v8|11|26)([nslmx])$", stem)
        return m.group(1) if m else ""

    @staticmethod
    def _family_yoloe_yaml_from_stem(stem: str) -> str:
        """Map YOLOE stem to existing family YAML names in this repository."""
        stem = stem.lower()
        if "yoloe-v8" in stem:
            return "yoloe-v8.yaml"
        if "yoloe-11" in stem:
            return "yoloe-11.yaml"
        if "yoloe-26" in stem:
            return "yoloe-26.yaml"
        return ""

    @staticmethod
    def _yaml_module_name(entry: Any) -> str:
        if isinstance(entry, (list, tuple)) and len(entry) >= 3:
            return str(entry[2]).lower()
        return ""

    @classmethod
    def _family_yoloe_yaml_from_cfg_dict(cls, cfg_dict: dict[str, Any] | None) -> str:
        """Infer YOLOE family detect YAML from parsed cfg dictionary when filename hints are unavailable."""
        if not isinstance(cfg_dict, dict):
            return ""
        head = cfg_dict.get("head", []) or []
        backbone = cfg_dict.get("backbone", []) or []
        head_mods = [cls._yaml_module_name(x) for x in head]
        backbone_mods = [cls._yaml_module_name(x) for x in backbone]

        if head:
            last = head[-1]
            if isinstance(last, (list, tuple)) and len(last) >= 1:
                from_idx = last[0]
                if from_idx == [15, 18, 21]:
                    return "yoloe-v8.yaml"

        if any("yoloesegment26" in m for m in head_mods):
            return "yoloe-26.yaml"
        if any("c2f" in m for m in backbone_mods + head_mods):
            return "yoloe-v8.yaml"
        if any("c3k2" in m for m in backbone_mods + head_mods):
            if (
                bool(cfg_dict.get("end2end", False))
                or str(cfg_dict.get("text_model", "")).startswith("mobileclip2")
                or int(cfg_dict.get("reg_max", 16)) == 1
            ):
                return "yoloe-26.yaml"
            return "yoloe-11.yaml"
        return ""

    @staticmethod
    def _resolve_existing_yaml(candidates: list[str]) -> str:
        seen = set()
        for cand in candidates:
            if not cand or cand in seen:
                continue
            seen.add(cand)
            resolved = check_yaml(cand, hard=False)
            if resolved:
                return str(resolved)
        return ""

    def _resolve_detect_yaml_from_sources(
        self,
        sources: list[str],
        *,
        weights: Any = None,
        cfg: dict[str, Any] | None = None,
    ) -> str:
        """Resolve a real detect YAML path from source hints and loaded YOLOE metadata."""
        candidates = []
        for src in sources:
            if not src:
                continue
            stem = Path(str(src)).stem.lower()
            direct = self._seg_source_to_detect_yaml(stem)
            candidates.append(Path(direct).name)
            family = self._family_yoloe_yaml_from_stem(stem)
            if family:
                candidates.append(family)

        cfg_family = self._family_yoloe_yaml_from_cfg_dict(cfg)
        if cfg_family:
            candidates.append(cfg_family)

        if weights is not None:
            w_yaml = getattr(weights, "yaml", None)
            w_family = self._family_yoloe_yaml_from_cfg_dict(w_yaml if isinstance(w_yaml, dict) else None)
            if w_family:
                candidates.append(w_family)

        deduped = [x for i, x in enumerate(candidates) if x and x not in candidates[:i]]
        resolved = self._resolve_existing_yaml(deduped)
        if resolved:
            return resolved

        raise ValueError(
            "YOLOE segmentation source detected, but failed to resolve matching detect YAML from candidates: "
            f"{deduped}. Existing YOLOE detect YAML families are: ['yoloe-v8.yaml', 'yoloe-11.yaml', 'yoloe-26.yaml']."
        )

    @staticmethod
    def _is_seg_source_stem(stem: str) -> bool:
        """Return True for segmentation sources, excluding explicit '-seg-det' detect-converted checkpoints."""
        stem = stem.lower()
        return "-seg" in stem and "-seg-det" not in stem

    def _validate_yoloe_student_source(self, cfg: str | dict | None, weights: Any) -> str | dict[str, Any]:
        """Validate YOLOE source and return cfg (str or dict) used to build YOLOEModel."""
        allow_seg_init = self._get_bool("yoloe_allow_seg_init", default=True)

        model_ref = str(getattr(self.args, "model", "") or "")
        model_stem = Path(model_ref).stem.lower()
        cfg_ref = cfg.get("yaml_file", None) if isinstance(cfg, dict) else cfg
        cfg_stem = Path(str(cfg_ref or "")).stem.lower()
        stems = [s for s in (model_stem, cfg_stem) if s]

        seg_pf_source = any("-seg-pf" in stem for stem in stems)
        seg_source = any(self._is_seg_source_stem(stem) for stem in stems)

        head_name = ""
        if weights is not None:
            w_task = getattr(weights, "task", None)
            head_name, has_lrpc = self._infer_yoloe_head(weights)
            seg_source = seg_source or w_task == "segment" or "segment" in head_name
            seg_pf_source = seg_pf_source or has_lrpc
            if w_task not in {None, "detect", "segment"}:
                raise ValueError(
                    f"Loaded YOLOE checkpoint task='{w_task}' is incompatible with this incremental detect pipeline."
                )
            if RANK in {-1, 0}:
                LOGGER.info(
                    "YOLOE source probe: "
                    f"model='{model_ref}', cfg_ref='{cfg_ref}', task='{w_task}', head='{head_name or 'unknown'}', "
                    f"seg_source={seg_source}, prompt_free={seg_pf_source}"
                )

        if seg_pf_source:
            raise ValueError(
                "YOLOE prompt-free segmentation checkpoints ('-seg-pf' / LRPC head) are not supported in this "
                "incremental detect-distill pipeline. Please use a YOLOE detect checkpoint, or a non-prompt-free "
                "YOLOE '-seg.pt' checkpoint for detect-initialization."
            )

        cfg_for_build: str | dict[str, Any] = cfg if isinstance(cfg, dict) else (cfg_ref or model_ref)
        if not cfg_for_build:
            raise ValueError("Unable to determine YOLOE cfg for student construction.")

        if seg_source:
            if not allow_seg_init:
                raise ValueError(
                    "YOLOE segmentation source detected but yoloe_allow_seg_init=False. "
                    "Provide a detect checkpoint/YAML, or set yoloe_allow_seg_init=True."
                )
            detect_yaml = self._resolve_detect_yaml_from_sources(
                [str(cfg_ref or ""), model_ref],
                weights=weights,
                cfg=cfg if isinstance(cfg, dict) else None,
            )
            scale = self._extract_yoloe_scale(model_stem) or self._extract_yoloe_scale(cfg_stem) or (
                str(cfg.get("scale", "")) if isinstance(cfg, dict) else ""
            )
            try:
                detect_cfg = yaml_model_load(detect_yaml)
            except Exception as e:
                raise ValueError(
                    "YOLOE segmentation source detected, but failed to resolve matching detect YAML "
                    f"('{detect_yaml}'). Please provide a valid YOLOE detect YAML/checkpoint."
                ) from e
            # Keep the original scale (n/s/m/l/x) from checkpoint stem when YAML uses family naming (e.g. yoloe-26.yaml).
            if scale:
                detect_cfg["scale"] = scale
            cfg_for_build = detect_cfg
            if RANK in {-1, 0}:
                LOGGER.warning(
                    "YOLOE segmentation source detected; constructing detect student from "
                    f"'{detect_yaml}' (scale='{detect_cfg.get('scale', '')}') and loading intersected pretrained weights."
                )
        elif RANK in {-1, 0} and head_name:
            LOGGER.info(f"YOLOE student source head='{head_name}' is compatible with detect pipeline.")

        return cfg_for_build

    @staticmethod
    def _build_zero_text_embeddings(model: YOLOEModel, class_count: int) -> torch.Tensor:
        head = model.model[-1] if hasattr(model, "model") and len(model.model) else None
        embed_dim = int(getattr(head, "embed", 512))
        device = next(model.parameters()).device
        return torch.zeros((1, class_count, embed_dim), device=device, dtype=torch.float32)

    def _configure_yoloe_student_model(self, model: YOLOEModel) -> None:
        if self.yoloe_prompt_mode != "fixed_text":
            raise ValueError(
                f"Unsupported yoloe_prompt_mode='{self.yoloe_prompt_mode}'. Only 'fixed_text' is currently supported."
            )

        dataset_names = self._normalize_names(self.data.get("names", {}))
        canonical_names = self._parse_class_names(self._get_arg("yoloe_class_names", None)) or dataset_names
        if not canonical_names:
            raise ValueError("Unable to determine YOLOE class names. Please provide yoloe_class_names explicitly.")
        if len(canonical_names) != int(self.data["nc"]):
            raise ValueError(
                f"YOLOE class count mismatch: got {len(canonical_names)} names {canonical_names}, but data.nc={self.data['nc']}."
            )
        if dataset_names and canonical_names != dataset_names:
            raise ValueError(
                "For fixed closed-set incremental experiments, yoloe_class_names must match data names order. "
                f"Got yoloe_class_names={canonical_names}, data names={dataset_names}."
            )
        prompt_names = self._parse_class_names(self._get_arg("yoloe_prompt_alias_names", None)) or canonical_names
        if len(prompt_names) != len(canonical_names):
            raise ValueError(
                "yoloe_prompt_alias_names must have the same length as canonical class names. "
                f"Got alias={prompt_names}, canonical={canonical_names}."
            )

        allow_zero_fallback = self._get_bool("yoloe_zero_embedding_fallback", default=True)
        used_fallback = False
        try:
            has_clip = importlib.util.find_spec("clip") is not None
            if not has_clip and allow_zero_fallback:
                raise ModuleNotFoundError("clip")
            text_embeddings = model.get_text_pe(prompt_names)
        except Exception as e:
            if not allow_zero_fallback:
                raise RuntimeError(
                    "YOLOE fixed-text embedding injection failed while configuring student model. "
                    "Please verify checkpoint compatibility and text model dependencies."
                ) from e
            text_embeddings = self._build_zero_text_embeddings(model, class_count=len(canonical_names))
            used_fallback = True
            if RANK in {-1, 0}:
                LOGGER.warning(
                    "YOLOE fixed-text embedding injection failed; using zero-embedding fallback "
                    f"(class_count={len(canonical_names)}, error={type(e).__name__}: {e}). "
                    "This keeps pipeline runnable but may degrade YOLOE prompt quality."
                )

        model.set_classes(canonical_names, text_embeddings)
        configured_names = self._normalize_names(getattr(model, "names", None))
        if configured_names != canonical_names:
            raise RuntimeError(
                "YOLOE fixed-text class injection mismatch after set_classes. "
                f"expected={canonical_names}, got={configured_names}"
            )
        has_pe = hasattr(model, "pe")
        pe_shape = tuple(getattr(model, "pe").shape) if has_pe else None

        if RANK in {-1, 0}:
            LOGGER.info(
                "YOLOE fixed-text class setup: "
                f"canonical_names={canonical_names} | prompt_names={prompt_names} | "
                f"embedding_mode={'zero_fallback' if used_fallback else 'text'} | "
                f"has_pe={has_pe} | pe_shape={pe_shape}"
            )

    @staticmethod
    def _to_list(path_or_paths: str | list[str]) -> list[str]:
        return path_or_paths if isinstance(path_or_paths, list) else [path_or_paths]

    @staticmethod
    def _restore_path_type(original: str | list[str], paths: list[str]) -> str | list[str]:
        return paths if isinstance(original, list) else paths[0]

    def _merge_replay_paths(self, img_path: str | list[str], mode: str) -> str | list[str]:
        if mode != "train" or not self.enable_replay:
            return img_path

        replay_data = self._get_arg("replay_data", None)
        if not replay_data:
            return img_path

        merged = self._to_list(img_path) + self._to_list(replay_data)
        if RANK in {-1, 0}:
            LOGGER.info(f"Replay enabled: merged {len(merged)} train image roots.")
        return merged if isinstance(img_path, list) else merged[0] if len(merged) == 1 else merged

    def _apply_slicing(self, img_path: str | list[str], mode: str) -> str | list[str]:
        if not self.enable_slicing:
            return img_path
        sliced = prepare_sliced_image_paths(
            img_path,
            mode=mode,
            cfg=self.slice_cfg,
            run_cache_dir=self.save_dir / "slice_cache",
            log_prefix=f"{mode}: ",
        )
        return self._restore_path_type(img_path, sliced)

    @staticmethod
    def _layers_from_model(model: Any) -> list[Any]:
        """Resolve YOLO layer sequence from either model.model.model or model.model."""
        nested = getattr(getattr(model, "model", None), "model", None)
        if isinstance(nested, (list, tuple)):
            return list(nested)
        if nested is not None:
            try:
                return list(nested)
            except TypeError:
                pass

        direct = getattr(model, "model", None)
        if isinstance(direct, (list, tuple)):
            return list(direct)
        if direct is not None:
            try:
                return list(direct)
            except TypeError:
                pass
        return []

    def _apply_layer_freeze(self, model: Any) -> None:
        """Freeze layers [0, freeze) after weights are loaded and before optimizer is built."""
        freeze_arg = self._get_arg("freeze", 0)
        try:
            freeze_n = int(freeze_arg)
        except (TypeError, ValueError):
            return
        if freeze_n <= 0:
            return

        model_layers = self._layers_from_model(model)
        if not model_layers:
            if RANK in {-1, 0}:
                LOGGER.warning("freeze is set but model layer sequence was not found; skip incremental layer freeze.")
            return

        freeze_n = min(freeze_n, len(model_layers))
        for i, layer in enumerate(model_layers):
            if i >= freeze_n:
                break
            for param in layer.parameters():
                param.requires_grad = False

        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        if RANK in {-1, 0}:
            LOGGER.info(f"Froze layers 0..{freeze_n - 1}: {frozen:,}/{total:,} ({100 * frozen / max(total, 1):.1f}%)")

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        img_path = self._merge_replay_paths(img_path, mode)
        img_path = self._apply_slicing(img_path, mode)
        return super().build_dataset(img_path=img_path, mode=mode, batch=batch)

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        if self.student_arch != "yoloe":
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            self._apply_layer_freeze(model)
            return model

        cfg_for_build = self._validate_yoloe_student_source(cfg=cfg, weights=weights)
        model = YOLOEModel(
            cfg_for_build,
            ch=self.data["channels"],
            nc=self.data["nc"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        self._configure_yoloe_student_model(model)
        self._apply_layer_freeze(model)
        return model

    def get_validator(self):
        """Return DetectionValidator with extended loss columns."""
        if self.enable_distillation:
            self.loss_names = ("box_loss", "cls_loss", "dfl_loss", "distill_feat_loss", "distill_cls_loss")
        else:
            self.loss_names = ("box_loss", "cls_loss", "dfl_loss")
        return yolo.detect.DetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=self._validator_args(),
            _callbacks=self.callbacks,
        )

    def _validator_args(self) -> dict[str, Any]:
        """Build validator-safe args by removing custom incremental keys."""
        args = vars(copy(self.args)).copy()
        for key in self._CUSTOM_OVERRIDE_KEYS:
            args.pop(key, None)
        return args

    def _setup_train(self):
        super()._setup_train()
        if not self.enable_distillation:
            if RANK in {-1, 0}:
                LOGGER.info("Distillation disabled: keep base detection criterion without distillation wrapper.")
            return
        self._attach_distillation_criterion()

    def _ensure_loss_model_attrs(self, model: Any) -> None:
        """Ensure model has runtime attributes required by detection loss initialization."""
        if model is None:
            return
        # v8DetectionLoss reads model.args (box/cls/dfl and related hparams) via attribute access.
        model.args = self.args
        if getattr(model, "nc", None) is None:
            model.nc = self.data.get("nc", None)
        if getattr(model, "names", None) in (None, {}, []):
            model.names = self.data.get("names", None)

    def _attach_distillation_criterion(self) -> None:
        student = unwrap_model(self.model)
        self._ensure_loss_model_attrs(student)
        if getattr(student, "criterion", None) is None:
            student.criterion = student.init_criterion()

        teacher_weights = self._get_arg("teacher_weights", None)
        if self.enable_distillation:
            if not teacher_weights:
                raise ValueError("Distillation is enabled but 'teacher_weights' is not set.")
            self.teacher_model = load_teacher_model(teacher_weights, device=self.device)
        else:
            self.teacher_model = None

        student_names = self._normalize_names(getattr(student, "names", None))
        teacher_names = self._normalize_names(getattr(self.teacher_model, "names", None))
        student_old_idx = self._resolve_class_index(
            names=student_names,
            class_name=self._get_arg("distill_student_old_class_name", None),
            fallback_index=int(self._get_arg("distill_student_old_class", 0)),
            scope="Student old class",
        )
        teacher_old_idx = self._resolve_class_index(
            names=teacher_names,
            class_name=self._get_arg("distill_teacher_old_class_name", None),
            fallback_index=int(self._get_arg("distill_teacher_old_class", 0)),
            scope="Teacher old class",
        )
        cls_only_old = self._get_bool("distill_only_old_classes", default=True)
        auto_class_align = self._get_bool("distill_auto_class_align", default=not cls_only_old)
        raw_student_ids = self._parse_class_indices(self._get_arg("distill_old_class_ids", None))
        raw_teacher_ids = self._parse_class_indices(self._get_arg("distill_teacher_old_class_ids", None))
        student_old_ids = raw_student_ids.copy()
        teacher_old_ids = raw_teacher_ids.copy()
        if not cls_only_old and auto_class_align and student_names and teacher_names:
            student_lookup = {str(n): i for i, n in enumerate(student_names)}
            teacher_lookup = {str(n): i for i, n in enumerate(teacher_names)}
            shared_names = [n for n in student_names if n in teacher_lookup]
            if shared_names:
                student_old_ids = [student_lookup[n] for n in shared_names]
                teacher_old_ids = [teacher_lookup[n] for n in shared_names]
                if RANK in {-1, 0}:
                    if raw_student_ids or raw_teacher_ids:
                        LOGGER.warning(
                            "distill_auto_class_align=True: ignoring distill_old_class_ids / "
                            "distill_teacher_old_class_ids and using class-name alignment."
                        )
                    LOGGER.info(
                        "Distill cls all-overlap class-name alignment enabled: "
                        f"shared_names={shared_names}, student_ids={student_old_ids}, teacher_ids={teacher_old_ids}"
                    )
            elif RANK in {-1, 0}:
                LOGGER.warning(
                    "distill_auto_class_align=True but no shared class names found between student and teacher; "
                    "falling back to explicit/default class indices."
                )
        if not student_old_ids:
            student_old_ids = [student_old_idx]
        if not teacher_old_ids:
            if len(teacher_names) == 1:
                teacher_old_ids = [0 for _ in student_old_ids]
            else:
                teacher_old_ids = [teacher_old_idx]
        if len(student_old_ids) > 1 and len(teacher_old_ids) == 1:
            teacher_old_ids = teacher_old_ids * len(student_old_ids)
        elif len(teacher_old_ids) > 1 and len(student_old_ids) == 1:
            student_old_ids = student_old_ids * len(teacher_old_ids)
        elif len(student_old_ids) != len(teacher_old_ids):
            n = min(len(student_old_ids), len(teacher_old_ids))
            if RANK in {-1, 0}:
                LOGGER.warning(
                    "distill_old_class_ids and distill_teacher_old_class_ids lengths differ. "
                    f"Using first {n} pairs: student={student_old_ids}, teacher={teacher_old_ids}."
                )
            student_old_ids = student_old_ids[:n]
            teacher_old_ids = teacher_old_ids[:n]

        feature_mode = str(self._get_arg("distill_feature_mode", "global")).strip().lower()
        if feature_mode not in {"global", "old_only"}:
            raise ValueError(
                f"Unsupported distill_feature_mode='{feature_mode}'. Supported: ['global', 'old_only']."
            )

        feature_weight = float(self._get_arg("distill_feature_weight", 0.25))
        cls_weight = float(self._get_arg("distill_cls_weight", 0.5))
        relax_single_teacher = self._get_bool("distill_relax_single_class_teacher", default=True)
        relax_scale = min(max(float(self._get_arg("distill_single_class_teacher_scale", 0.35)), 0.0), 1.0)
        if (
            relax_single_teacher
            and self.enable_distillation
            and len(teacher_names) == 1
            and len(student_names) > 1
            and (feature_weight > 0.0 or cls_weight > 0.0)
        ):
            feature_weight *= relax_scale
            cls_weight *= relax_scale
            if RANK in {-1, 0}:
                LOGGER.info(
                    "Single-class teacher detected with multi-class student; relaxed distillation weights for new-class "
                    f"plasticity: scale={relax_scale}, feature_w={feature_weight}, cls_w={cls_weight}"
                )

        distill_cfg = DistillationConfig(
            enabled=self.enable_distillation and self.teacher_model is not None,
            feature_weight=feature_weight,
            cls_weight=cls_weight,
            temperature=float(self._get_arg("distill_temperature", 2.0)),
            student_old_class_index=student_old_idx,
            teacher_old_class_index=teacher_old_idx,
            cls_only_old_classes=cls_only_old,
            student_old_class_indices=tuple(student_old_ids),
            teacher_old_class_indices=tuple(teacher_old_ids),
            feature_mode=feature_mode,
            feature_old_score_thresh=float(self._get_arg("distill_feature_old_score_thr", 0.3)),
            cls_old_score_thresh=float(self._get_arg("distill_cls_old_score_thr", 0.0)),
            start_epoch=int(self._get_arg("distill_start_epoch", 0)),
            ramp_epochs=int(self._get_arg("distill_ramp_epochs", 0)),
        )
        student.criterion = DistillationLossWrapper(
            base_criterion=student.criterion,
            teacher_model=self.teacher_model,
            cfg=distill_cfg,
        )
        if self.ema and getattr(self.ema, "ema", None) is not None:
            ema_model = unwrap_model(self.ema.ema)
            self._ensure_loss_model_attrs(ema_model)
            if getattr(ema_model, "criterion", None) is None:
                ema_model.criterion = ema_model.init_criterion()
            ema_model.criterion = DistillationLossWrapper(
                base_criterion=ema_model.criterion,
                teacher_model=self.teacher_model,
                cfg=distill_cfg,
            )

        if RANK in {-1, 0}:
            student_head = getattr(getattr(student, "model", None), "__getitem__", lambda x: None)(-1)
            student_old_names = self._names_for_indices(student_names, list(distill_cfg.student_old_class_indices))
            teacher_old_names = self._names_for_indices(teacher_names, list(distill_cfg.teacher_old_class_indices))
            LOGGER.info(
                "Distillation setup complete: "
                f"enabled={distill_cfg.enabled}, "
                f"feature_w={distill_cfg.feature_weight}, "
                f"cls_w={distill_cfg.cls_weight}, "
                f"temperature={distill_cfg.temperature}, "
                f"cls_scope={'old_only' if distill_cfg.cls_only_old_classes else 'all_overlap'}, "
                f"auto_class_align={auto_class_align}, "
                f"feature_mode={distill_cfg.feature_mode}, "
                f"feature_old_thr={distill_cfg.feature_old_score_thresh}, "
                f"cls_old_thr={distill_cfg.cls_old_score_thresh}, "
                f"start_epoch={distill_cfg.start_epoch}, "
                f"ramp_epochs={distill_cfg.ramp_epochs}, "
                f"student_old_idx={distill_cfg.student_old_class_index}, "
                f"teacher_old_idx={distill_cfg.teacher_old_class_index}, "
                f"student_head={student_head.__class__.__name__.lower() if student_head is not None else 'unknown'}"
            )
            LOGGER.info(
                "Distillation class mapping: "
                f"student_names={student_names}, "
                f"teacher_names={teacher_names}, "
                f"student_old_ids={list(distill_cfg.student_old_class_indices)} ({student_old_names}), "
                f"teacher_old_ids={list(distill_cfg.teacher_old_class_indices)} ({teacher_old_names})"
            )

    def validate(self):
        metrics, fitness = super().validate()
        if metrics is None or RANK not in {-1, 0}:
            return metrics, fitness

        extra_metrics = self._run_extra_validations()
        if extra_metrics:
            metrics.update(extra_metrics)
        return metrics, fitness

    def _run_extra_validations(self) -> dict[str, float]:
        model_for_eval = self.ema.ema if self.ema else unwrap_model(self.model)
        eval_specs = {
            "old": self._get_arg("old_val_data", None),
            "new": self._get_arg("new_val_data", None),
            "joint": self._get_arg("joint_val_data", None),
        }

        results = {}
        for tag, data_yaml in eval_specs.items():
            if not data_yaml:
                continue
            eval_data_yaml = self._prepare_eval_data_yaml(str(data_yaml))
            args = self._validator_args()
            args["data"] = str(eval_data_yaml)
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

    def _prepare_eval_data_yaml(self, data_yaml: str) -> str:
        if not self.enable_slicing:
            return data_yaml
        if data_yaml in self._sliced_eval_yaml:
            return self._sliced_eval_yaml[data_yaml]

        resolved = check_det_dataset(data_yaml, autodownload=False)
        val_slice_cfg = replace(self.slice_cfg, empty_tile_keep_prob=1.0)
        for split in ("train", "val", "test", "minival"):
            if not resolved.get(split):
                continue
            sliced = prepare_sliced_image_paths(
                resolved[split],
                mode="val",
                cfg=val_slice_cfg,
                run_cache_dir=self.save_dir / "slice_cache",
                log_prefix=f"eval-{split}: ",
            )
            resolved[split] = sliced if isinstance(resolved[split], list) else sliced[0]

        save_dir = self.save_dir / "sliced_eval_data"
        save_dir.mkdir(parents=True, exist_ok=True)
        out_yaml = save_dir / f"{Path(data_yaml).stem}_sliced.yaml"
        out_dict = {
            "path": "",
            "train": resolved["train"],
            "val": resolved["val"],
            "test": resolved.get("test", None),
            "nc": resolved["nc"],
            "names": resolved["names"],
            "channels": resolved.get("channels", 3),
        }
        YAML.save(out_yaml, out_dict)
        self._sliced_eval_yaml[data_yaml] = str(out_yaml)
        return str(out_yaml)
