# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import deepcopy
from copy import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

from ultralytics.data.utils import check_det_dataset
from ultralytics.models import yolo
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.data.sliding_window import SliceConfig, prepare_sliced_image_paths
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, YAML
from ultralytics.utils.torch_utils import unwrap_model


class IncrementalDistillTrainer(DetectionTrainer):
    """Incremental detection trainer with optional replay and slicing."""

    _CUSTOM_OVERRIDE_KEYS = {
        "experiment_mode",
        "enable_distillation",
        "enable_replay",
        "enable_slicing",
        "teacher_weights",
        "distill_feature_weight",
        "distill_cls_weight",
        "distill_temperature",
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
    }
    _DISTILL_ARG_KEYS = {
        "enable_distillation",
        "teacher_weights",
        "distill_feature_weight",
        "distill_cls_weight",
        "distill_temperature",
        "distill_student_old_class",
        "distill_teacher_old_class",
    }
    _MODE_ALIASES = {"distill_only": "naive_finetune", "replay_distill": "replay_only"}
    _SUPPORTED_MODES = {"naive_finetune", "replay_only"}

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        overrides = (overrides or {}).copy()
        custom = {k: overrides.pop(k) for k in list(overrides) if k in self._CUSTOM_OVERRIDE_KEYS}
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        for k, v in custom.items():
            setattr(self.args, k, v)

        mode = str(getattr(self.args, "experiment_mode", "naive_finetune"))
        if mode in self._MODE_ALIASES:
            mapped_mode = self._MODE_ALIASES[mode]
            LOGGER.warning(
                f"experiment_mode='{mode}' is deprecated because distillation is removed. Falling back to '{mapped_mode}'."
            )
            mode = mapped_mode
        if mode not in self._SUPPORTED_MODES:
            LOGGER.warning(
                f"Unsupported experiment_mode='{mode}', falling back to 'naive_finetune'. "
                f"Supported: {sorted(self._SUPPORTED_MODES)}"
            )
            mode = "naive_finetune"
        self.experiment_mode = mode

        self.enable_replay = self._get_bool("enable_replay", default=self.experiment_mode == "replay_only")
        self.enable_slicing = self._get_bool("enable_slicing", default=False)
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

        self._warn_ignored_distillation_args()
        self._sliced_eval_yaml: dict[str, str] = {}

        if RANK in {-1, 0}:
            LOGGER.info(
                "Incremental trainer config: "
                f"mode={self.experiment_mode}, "
                f"slicing={self.enable_slicing}, "
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

    def _warn_ignored_distillation_args(self) -> None:
        ignored = {}
        for key in sorted(self._DISTILL_ARG_KEYS):
            value = getattr(self.args, key, None)
            if value not in (None, False, "", 0, 0.0):
                ignored[key] = value
        if ignored:
            LOGGER.warning(
                "Distillation options are ignored because distillation support has been removed: "
                + ", ".join(f"{k}={v}" for k, v in ignored.items())
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

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        img_path = self._merge_replay_paths(img_path, mode)
        img_path = self._apply_slicing(img_path, mode)
        return super().build_dataset(img_path=img_path, mode=mode, batch=batch)

    def get_validator(self):
        """Return DetectionValidator with standard detection loss columns."""
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
