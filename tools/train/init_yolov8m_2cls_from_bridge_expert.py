from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "ultralytics"
STUDENT_NAMES = {0: "ship", 1: "bridge"}
DEFAULT_BRIDGE_WEIGHTS = (
    "runs/bridge_expert/yolov8m_total_bridge_split_img1024_ep150/weights/best.pt"
)
DEFAULT_BASE_MODEL = "yolov8m.pt"
DEFAULT_OUT = "runs/student_2cls/init/yolov8m_2cls_from_bridge_expert.pt"


def setup_local_ultralytics_import() -> None:
    """Prefer this checkout over any separately installed Ultralytics package."""
    for value in (str(PACKAGE_ROOT.resolve()), str(PROJECT_ROOT.resolve())):
        while value in sys.path:
            sys.path.remove(value)
    sys.path.insert(0, str(PACKAGE_ROOT.resolve()))
    sys.path.insert(1, str(PROJECT_ROOT.resolve()))


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def default_summary_path(out: Path) -> Path:
    return out.with_name(f"{out.stem}_summary.json")


def qualified_name(obj: object) -> str:
    cls = type(obj)
    return f"{cls.__module__}.{cls.__name__}"


def normalize_names(names: Any) -> dict[int, str]:
    if isinstance(names, (list, tuple)):
        return {i: str(name) for i, name in enumerate(names)}
    if isinstance(names, dict):
        try:
            return {int(key): str(value) for key, value in names.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Class names must use integer keys, got {names!r}") from exc
    raise TypeError(f"Unsupported class names value: {names!r}")


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")


def require_ordinary_detect(inner: object, description: str, expected_nc: int | None = None):
    from ultralytics.nn.modules.head import Detect
    from ultralytics.nn.tasks import DetectionModel

    if type(inner) is not DetectionModel:
        raise TypeError(
            f"{description} must be an ordinary DetectionModel, got {qualified_name(inner)}. "
            "YOLOE/segment checkpoints are not accepted."
        )
    layers = getattr(inner, "model", None)
    if layers is None or len(layers) == 0:
        raise TypeError(f"{description} has no model layers")
    head = layers[-1]
    if type(head) is not Detect:
        raise TypeError(
            f"{description} must end in an ordinary Detect head, got {qualified_name(head)}. "
            "YOLOEDetect, Segment, and YOLOESegment are not accepted."
        )
    task = getattr(inner, "task", "detect")
    if task != "detect":
        raise ValueError(f"{description} task must be 'detect', got {task!r}")
    if expected_nc is not None and int(head.nc) != expected_nc:
        raise ValueError(f"{description} Detect.nc must be {expected_nc}, got {head.nc}")
    return head


def load_shape_compatible(source: object, destination: object) -> dict[str, Any]:
    """Load state tensors whose key and shape match, returning an auditable report."""
    source_state = source.state_dict()
    destination_state = destination.state_dict()
    compatible = {}
    skipped = []
    for key, tensor in source_state.items():
        target = destination_state.get(key)
        if target is None:
            skipped.append(
                {"key": key, "reason": "missing_in_student", "source_shape": list(tensor.shape)}
            )
        elif tuple(tensor.shape) != tuple(target.shape):
            skipped.append(
                {
                    "key": key,
                    "reason": "shape_mismatch",
                    "source_shape": list(tensor.shape),
                    "student_shape": list(target.shape),
                }
            )
        else:
            compatible[key] = tensor.detach().to(dtype=target.dtype)

    result = destination.load_state_dict(compatible, strict=False)
    return {
        "source_state_tensors": len(source_state),
        "student_state_tensors": len(destination_state),
        "num_shape_compatible_loaded": len(compatible),
        "num_skipped": len(skipped),
        "loaded_key_examples": sorted(compatible)[:20],
        "skipped": skipped,
        "load_missing_keys": list(result.missing_keys)[:20],
        "load_unexpected_keys": list(result.unexpected_keys)[:20],
    }


def _classification_final_convs(head: object, description: str):
    import torch.nn as nn

    branches = getattr(head, "cv3", None)
    if branches is None or len(branches) != int(head.nl):
        raise TypeError(
            f"{description} Detect.cv3 must contain one branch per detection scale; "
            f"got cv3={type(branches).__name__}, nl={head.nl}"
        )
    final_convs = []
    for scale, branch in enumerate(branches):
        if not hasattr(branch, "__len__") or len(branch) == 0:
            raise TypeError(f"{description} cv3[{scale}] is not a non-empty classification branch")
        final = branch[-1]
        if not isinstance(final, nn.Conv2d):
            raise TypeError(
                f"{description} cv3[{scale}] final layer must be Conv2d, got {qualified_name(final)}"
            )
        final_convs.append(final)
    return final_convs


def classification_final_state_keys(model: object, head: object, description: str) -> set[str]:
    """Return exact state-dict keys for every final classification Conv2d."""
    final_convs = _classification_final_convs(head, description)
    module_names = {id(module): name for name, module in model.named_modules()}
    keys = set()
    for scale, conv in enumerate(final_convs):
        name = module_names.get(id(conv))
        if not name:
            raise RuntimeError(f"Could not locate {description} cv3[{scale}] in named_modules()")
        keys.add(f"{name}.weight")
        if conv.bias is not None:
            keys.add(f"{name}.bias")
    return keys


def require_only_classifier_skips(
    migration: dict[str, Any], expected_keys: set[str], description: str
) -> None:
    skipped_keys = {item["key"] for item in migration["skipped"]}
    if skipped_keys != expected_keys:
        unexpected = sorted(skipped_keys - expected_keys)
        absent = sorted(expected_keys - skipped_keys)
        raise RuntimeError(
            f"{description} architecture is not fully compatible apart from final class rows. "
            f"Unexpected skipped keys={unexpected}; expected-but-not-skipped keys={absent}"
        )


def copy_bridge_class_row(old_head: object, new_head: object) -> dict[str, Any]:
    """Copy old class 0 (bridge) to new class 1 while preserving new class 0 (ship)."""
    import torch

    old_convs = _classification_final_convs(old_head, "bridge expert")
    new_convs = _classification_final_convs(new_head, "2-class student")
    if len(old_convs) != len(new_convs):
        raise ValueError(
            f"Classification scale count differs: bridge={len(old_convs)}, student={len(new_convs)}"
        )

    copied_layers = []
    with torch.no_grad():
        for scale, (old_conv, new_conv) in enumerate(zip(old_convs, new_convs)):
            if old_conv.out_channels != 1 or new_conv.out_channels != 2:
                raise ValueError(
                    f"cv3[{scale}] class channels must be old=1/new=2, got "
                    f"old={old_conv.out_channels}, new={new_conv.out_channels}"
                )
            if tuple(old_conv.weight.shape[1:]) != tuple(new_conv.weight.shape[1:]):
                raise ValueError(
                    f"cv3[{scale}] bridge/student classifier inputs differ: "
                    f"{tuple(old_conv.weight.shape)} vs {tuple(new_conv.weight.shape)}"
                )
            if old_conv.bias is None or new_conv.bias is None:
                raise ValueError(f"cv3[{scale}] final classifier must have a bias")

            ship_weight_before = new_conv.weight[0].detach().clone()
            ship_bias_before = new_conv.bias[0].detach().clone()
            new_conv.weight[1].copy_(old_conv.weight[0])
            new_conv.bias[1].copy_(old_conv.bias[0])

            bridge_weight_exact = torch.equal(new_conv.weight[1], old_conv.weight[0])
            bridge_bias_exact = torch.equal(new_conv.bias[1], old_conv.bias[0])
            ship_row_preserved = torch.equal(new_conv.weight[0], ship_weight_before) and torch.equal(
                new_conv.bias[0], ship_bias_before
            )
            if not bridge_weight_exact or not bridge_bias_exact or not ship_row_preserved:
                raise RuntimeError(f"cv3[{scale}] class-row copy verification failed")
            copied_layers.append(
                {
                    "scale": scale,
                    "source": f"Detect.cv3[{scale}] class 0",
                    "destination": f"Detect.cv3[{scale}] class 1",
                    "weight_shape": list(old_conv.weight[0].shape),
                    "bias_shape": list(old_conv.bias[0].shape),
                    "bridge_weight_exact_before_save": bridge_weight_exact,
                    "bridge_bias_exact_before_save": bridge_bias_exact,
                    "ship_row_preserved": ship_row_preserved,
                }
            )
    return {"bridge_class_row_copied": True, "copied_layers": copied_layers}


def save_init_checkpoint(model: object, out: Path, metadata: dict[str, Any]) -> None:
    import torch
    from ultralytics import __version__

    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": -1,
        "best_fitness": None,
        "model": deepcopy(model).half(),
        "ema": None,
        "updates": None,
        "optimizer": None,
        "train_args": {"task": "detect", "model": str(out), "pretrained": False},
        "date": datetime.now().isoformat(),
        "version": __version__,
        "license": "AGPL-3.0 License (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
        "bridge_init": metadata,
    }
    torch.save(checkpoint, out)


def verify_saved_checkpoint(out: Path, source_head: object) -> dict[str, Any]:
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    reloaded = YOLO(str(out), verbose=False)
    head = require_ordinary_detect(reloaded.model, "saved student checkpoint", expected_nc=2)
    names = normalize_names(reloaded.model.names)
    if names != STUDENT_NAMES:
        raise ValueError(f"Saved student names must be {STUDENT_NAMES}, got {names}")

    source_convs = _classification_final_convs(source_head, "bridge expert")
    saved_convs = _classification_final_convs(head, "saved 2-class student")
    layer_checks = []
    for scale, (source, saved) in enumerate(zip(source_convs, saved_convs)):
        weight_delta = float((saved.weight[1].float() - source.weight[0].float()).abs().max().item())
        bias_delta = float((saved.bias[1].float() - source.bias[0].float()).abs().max().item())
        weight_ok = torch.allclose(saved.weight[1].float(), source.weight[0].float(), rtol=1e-3, atol=1e-4)
        bias_ok = torch.allclose(saved.bias[1].float(), source.bias[0].float(), rtol=1e-3, atol=1e-4)
        if not weight_ok or not bias_ok:
            raise RuntimeError(
                f"Saved cv3[{scale}] bridge row differs after FP16 checkpoint serialization: "
                f"weight_max_abs_diff={weight_delta}, bias_max_abs_diff={bias_delta}"
            )
        layer_checks.append(
            {
                "scale": scale,
                "saved_bridge_weight_allclose": weight_ok,
                "saved_bridge_bias_allclose": bias_ok,
                "weight_max_abs_diff": weight_delta,
                "bias_max_abs_diff": bias_delta,
            }
        )

    # Model.train() constructs a fresh DetectionModel from model.yaml and calls
    # DetectionModel.load(weights). Reproduce that path here so a later trainer
    # rebuild cannot silently discard either two-class classifier row.
    reconstructed = DetectionModel(deepcopy(reloaded.model.yaml), ch=3, nc=2, verbose=False)
    reconstructed.load(reloaded.model)
    reconstructed_head = require_ordinary_detect(
        reconstructed, "trainer-reconstructed student", expected_nc=2
    )
    reconstructed_convs = _classification_final_convs(
        reconstructed_head, "trainer-reconstructed student"
    )
    reconstruction_checks = []
    for scale, (saved, rebuilt) in enumerate(zip(saved_convs, reconstructed_convs)):
        weight_exact = torch.equal(saved.weight, rebuilt.weight)
        bias_exact = torch.equal(saved.bias, rebuilt.bias)
        if not weight_exact or not bias_exact:
            raise RuntimeError(
                f"Trainer-style reconstruction did not preserve both class rows at cv3[{scale}]"
            )
        reconstruction_checks.append(
            {"scale": scale, "all_class_weights_exact": weight_exact, "all_class_biases_exact": bias_exact}
        )
    return {
        "checkpoint_reloaded": True,
        "model_class": qualified_name(reloaded.model),
        "head_class": qualified_name(head),
        "task": reloaded.task,
        "nc": int(head.nc),
        "names": names,
        "layer_checks": layer_checks,
        "trainer_reconstruction_preserved_classifier": True,
        "trainer_reconstruction_checks": reconstruction_checks,
    }


def initialize_student(
    bridge_weights_value: str | Path,
    base_model_value: str | Path,
    out_value: str | Path,
    summary_value: str | Path | None = None,
) -> dict[str, Any]:
    setup_local_ultralytics_import()
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    bridge_weights = resolve_path(bridge_weights_value)
    base_model = resolve_path(base_model_value)
    out = resolve_path(out_value)
    summary_path = resolve_path(summary_value) if summary_value else default_summary_path(out)
    summary: dict[str, Any] = {
        "task": "initialize ordinary YOLOv8m 2-class student from bridge-only expert",
        "project_root": str(PROJECT_ROOT),
        "bridge_weights": str(bridge_weights),
        "base_model": str(base_model),
        "out": str(out),
        "summary": str(summary_path),
        "old_class_mapping": {"0": "bridge"},
        "new_class_mapping": {"0": "ship", "1": "bridge"},
        "bridge_class_mapping": "old class 0 -> new class 1",
        "started_at": datetime.now().isoformat(),
        "status": "running",
    }
    try:
        require_file(bridge_weights, "Bridge expert weights")
        require_file(base_model, "Base YOLOv8m model")

        bridge_wrapper = YOLO(str(bridge_weights), verbose=False)
        bridge_head = require_ordinary_detect(
            bridge_wrapper.model, "bridge expert", expected_nc=1
        )
        bridge_names = normalize_names(bridge_wrapper.model.names)
        if bridge_names != {0: "bridge"}:
            raise ValueError(
                f"Bridge expert names must be {{0: 'bridge'}}, got {bridge_names}. "
                "Refusing to infer the source class mapping."
            )

        base_wrapper = YOLO(str(base_model), verbose=False)
        require_ordinary_detect(base_wrapper.model, "base model")

        student_yaml = deepcopy(base_wrapper.model.yaml)
        student = DetectionModel(student_yaml, ch=3, nc=2, verbose=False)
        student.task = "detect"
        student.names = dict(STUDENT_NAMES)
        student_head = require_ordinary_detect(student, "new student", expected_nc=2)

        # Establish an official YOLOv8m initialization first, then overwrite all compatible
        # tensors with the trained bridge expert. The new two-channel classifier finals are
        # intentionally excluded by shape and handled explicitly below.
        base_migration = load_shape_compatible(base_wrapper.model, student)
        bridge_migration = load_shape_compatible(bridge_wrapper.model, student)
        require_only_classifier_skips(
            base_migration,
            classification_final_state_keys(base_wrapper.model, base_wrapper.model.model[-1], "base model"),
            "Base model",
        )
        require_only_classifier_skips(
            bridge_migration,
            classification_final_state_keys(bridge_wrapper.model, bridge_head, "bridge expert"),
            "Bridge expert",
        )
        class_copy = copy_bridge_class_row(bridge_head, student_head)

        names = normalize_names(student.names)
        if names != STUDENT_NAMES:
            raise RuntimeError(f"New student names must be {STUDENT_NAMES}, got {names}")

        summary.update(
            {
                "ultralytics_version": ultralytics.__version__,
                "torch_version": torch.__version__,
                "bridge_model_class": qualified_name(bridge_wrapper.model),
                "bridge_head_class": qualified_name(bridge_head),
                "base_model_class": qualified_name(base_wrapper.model),
                "student_model_class": qualified_name(student),
                "student_head_class": qualified_name(student_head),
                "old_nc": int(bridge_head.nc),
                "new_nc": int(student_head.nc),
                "names": names,
                "base_initialization": base_migration,
                "bridge_migration": bridge_migration,
                "num_shape_compatible_loaded": bridge_migration["num_shape_compatible_loaded"],
                "num_skipped": bridge_migration["num_skipped"],
                **class_copy,
            }
        )

        checkpoint_metadata = {
            "bridge_weights": str(bridge_weights),
            "base_model": str(base_model),
            "old_nc": 1,
            "new_nc": 2,
            "names": dict(STUDENT_NAMES),
            "bridge_class_mapping": "old class 0 -> new class 1",
            "num_shape_compatible_loaded": bridge_migration["num_shape_compatible_loaded"],
            "num_skipped": bridge_migration["num_skipped"],
            **class_copy,
        }
        save_init_checkpoint(student, out, checkpoint_metadata)
        summary["saved_checkpoint_verification"] = verify_saved_checkpoint(out, bridge_head)
        summary["status"] = "completed"
        summary["completed_at"] = datetime.now().isoformat()
        return summary
    except Exception as exc:
        summary["status"] = "failed"
        summary["failed_at"] = datetime.now().isoformat()
        summary["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        console_summary = {
            "status": summary["status"],
            "bridge_weights": summary["bridge_weights"],
            "base_model": summary["base_model"],
            "out": summary["out"],
            "summary": summary["summary"],
            "old_nc": summary.get("old_nc"),
            "new_nc": summary.get("new_nc"),
            "names": summary.get("names"),
            "num_shape_compatible_loaded": summary.get("num_shape_compatible_loaded"),
            "num_skipped": summary.get("num_skipped"),
            "bridge_class_row_copied": summary.get("bridge_class_row_copied"),
            "copied_layers": summary.get("copied_layers"),
            "saved_checkpoint_verification": summary.get("saved_checkpoint_verification"),
            "error": summary.get("error"),
        }
        print(json.dumps(console_summary, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an ordinary YOLOv8m nc=2 student and explicitly copy bridge expert "
            "class 0 into student class 1."
        )
    )
    parser.add_argument("--bridge-weights", default=DEFAULT_BRIDGE_WEIGHTS)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--summary",
        default=None,
        help="Defaults to <out stem>_summary.json next to the initialized checkpoint.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    initialize_student(args.bridge_weights, args.base_model, args.out, args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
