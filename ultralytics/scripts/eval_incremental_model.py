from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import sys
import torch

# Prefer local repo package over site-packages when running as:
#   python ultralytics/scripts/eval_incremental_model.py ...
THIS_FILE = Path(__file__).resolve()
LOCAL_ULTRALYTICS_ROOT = THIS_FILE.parents[1]  # .../ultralytics
if str(LOCAL_ULTRALYTICS_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_ULTRALYTICS_ROOT))

from ultralytics import YOLO
from ultralytics.data.sliding_window import SliceConfig, prepare_sliced_image_paths
from ultralytics.data.utils import check_det_dataset
from ultralytics.nn.tasks import YOLOEModel, yaml_model_load
from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.checks import check_yaml


def _restore_path_type(original: str | list[str], paths: list[str]) -> str | list[str]:
    return paths if isinstance(original, list) else paths[0]


def _prepare_eval_data_yaml(
    data_yaml: str,
    *,
    enable_slicing: bool,
    slice_cfg: SliceConfig,
    save_dir: Path,
    tag: str,
) -> str:
    if not enable_slicing:
        return data_yaml

    resolved = check_det_dataset(data_yaml, autodownload=False)
    val_slice_cfg = SliceConfig(
        enabled=True,
        slice_size=slice_cfg.slice_size,
        slice_overlap=slice_cfg.slice_overlap,
        min_box_keep_ratio=slice_cfg.min_box_keep_ratio,
        # Keep empty tiles in eval to avoid random eval-set drift.
        empty_tile_keep_prob=1.0,
        cache_slices=slice_cfg.cache_slices,
        cache_dir=slice_cfg.cache_dir,
        seed=slice_cfg.seed,
    )

    for split in ("train", "val", "test", "minival"):
        if not resolved.get(split):
            continue
        src = resolved[split]
        src_list = src if isinstance(src, list) else [src]
        sliced = prepare_sliced_image_paths(
            src_list,
            mode="val",
            cfg=val_slice_cfg,
            run_cache_dir=save_dir / "slice_cache",
            log_prefix=f"{tag}:{split}: ",
        )
        resolved[split] = _restore_path_type(src, sliced)

    out_dir = save_dir / "sliced_eval_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_yaml = out_dir / f"{Path(data_yaml).stem}_{tag}_sliced.yaml"
    out_dict = {
        "path": "",
        "train": resolved.get("train"),
        "val": resolved.get("val"),
        "test": resolved.get("test"),
        "nc": resolved["nc"],
        "names": resolved["names"],
        "channels": resolved.get("channels", 3),
    }
    YAML.save(out_yaml, out_dict)
    return str(out_yaml)


def _default_tags(data_list: list[str]) -> list[str]:
    used: dict[str, int] = {}
    tags = []
    for path in data_list:
        stem = Path(path).stem
        idx = used.get(stem, 0)
        used[stem] = idx + 1
        tags.append(stem if idx == 0 else f"{stem}_{idx}")
    return tags


def _normalize_names(names: Any) -> list[str]:
    if isinstance(names, dict):
        return [str(v) for _, v in sorted(names.items(), key=lambda x: int(x[0]))]
    if isinstance(names, (list, tuple)):
        return [str(v) for v in names]
    return []


def _family_yoloe_yaml_from_stem(stem: str) -> str:
    stem = stem.lower()
    if "yoloe-v8" in stem:
        return "yoloe-v8.yaml"
    if "yoloe-11" in stem:
        return "yoloe-11.yaml"
    if "yoloe-26" in stem:
        return "yoloe-26.yaml"
    return ""


def _yaml_module_name(entry: Any) -> str:
    if isinstance(entry, (list, tuple)) and len(entry) >= 3:
        return str(entry[2]).lower()
    return ""


def _family_yoloe_yaml_from_cfg_dict(cfg_dict: dict[str, Any] | None) -> str:
    if not isinstance(cfg_dict, dict):
        return ""
    head = cfg_dict.get("head", []) or []
    backbone = cfg_dict.get("backbone", []) or []
    head_mods = [_yaml_module_name(x) for x in head]
    backbone_mods = [_yaml_module_name(x) for x in backbone]

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


def _infer_detect_yaml_for_yoloe_seg(model_ref: str, loaded: YOLO) -> str:
    candidates: list[str] = []
    sources = [model_ref]
    inner = getattr(loaded, "model", None)
    yaml_file = getattr(inner, "yaml_file", None)
    if yaml_file:
        sources.append(str(yaml_file))
    yaml_dict = getattr(inner, "yaml", None) if inner is not None else None
    family_from_cfg = _family_yoloe_yaml_from_cfg_dict(yaml_dict if isinstance(yaml_dict, dict) else None)
    if family_from_cfg:
        candidates.append(family_from_cfg)

    for src in sources:
        stem = Path(str(src)).stem.lower()
        detect_stem = stem.replace("-seg-pf", "").replace("-seg", "")
        if detect_stem and detect_stem not in {"best", "last"}:
            candidates.append(f"{detect_stem}.yaml")
            candidates.append(str(Path(str(src)).with_name(f"{detect_stem}.yaml")))
        fam = _family_yoloe_yaml_from_stem(stem)
        if fam:
            candidates.append(fam)
    # Hard fallback for generic ckpt names like best.pt/last.pt.
    candidates.extend(["yoloe-v8.yaml", "yoloe-11.yaml", "yoloe-26.yaml"])

    seen = set()
    for cand in candidates:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        resolved = check_yaml(cand, hard=False)
        if resolved:
            return str(resolved)
    raise FileNotFoundError(
        "Failed to resolve detect YAML for YOLOE seg checkpoint. "
        f"Candidates tried: {candidates}. Please ensure yoloe-v8/11/26 detect YAML exists in this repo."
    )


def _build_zero_text_embeddings(model: YOLOEModel, class_count: int) -> torch.Tensor:
    head = model.model[-1] if hasattr(model, "model") and len(model.model) else None
    embed_dim = int(getattr(head, "embed", 512))
    device = next(model.parameters()).device
    return torch.zeros((1, class_count, embed_dim), device=device, dtype=torch.float32)


def _adapt_yoloe_seg_checkpoint_for_detect_eval(
    loaded: YOLO,
    model_ref: str,
    class_names: list[str],
    *,
    detect_yaml_override: str | None = None,
    scale_override: str | None = None,
    zero_embedding_fallback: bool = True,
) -> YOLO:
    if detect_yaml_override:
        detect_yaml = str(check_yaml(detect_yaml_override, hard=True))
    else:
        detect_yaml = _infer_detect_yaml_for_yoloe_seg(model_ref, loaded)
    detect_cfg = yaml_model_load(detect_yaml)
    if scale_override:
        detect_cfg["scale"] = str(scale_override)
    model = YOLOEModel(detect_cfg, ch=3, nc=len(class_names), verbose=False)
    model.load(model_ref)  # intersect load

    try:
        text_embeddings = model.get_text_pe(class_names)
        emb_mode = "text"
    except Exception as e:
        if not zero_embedding_fallback:
            raise RuntimeError("Failed to build YOLOE text embeddings for eval adaptation.") from e
        text_embeddings = _build_zero_text_embeddings(model, class_count=len(class_names))
        emb_mode = "zero_fallback"
        LOGGER.warning(
            "YOLOE detect-eval adaptation text embedding failed; using zero fallback. "
            f"error={type(e).__name__}: {e}"
        )

    model.set_classes(class_names, text_embeddings)
    loaded.model = model
    loaded.task = "detect"
    if getattr(loaded, "overrides", None) is None:
        loaded.overrides = {}
    loaded.overrides["task"] = "detect"
    loaded.overrides["nc"] = len(class_names)
    loaded.overrides["names"] = class_names
    LOGGER.warning(
        "Adapted YOLOE seg checkpoint to detect model for evaluation: "
        f"detect_yaml={detect_yaml}, classes={class_names}, embedding_mode={emb_mode}"
    )
    return loaded


def _maybe_adapt_model_for_eval(
    loaded: YOLO,
    model_ref: str,
    class_names: list[str],
    *,
    detect_yaml_override: str | None,
    scale_override: str | None,
    zero_embedding_fallback: bool,
) -> YOLO:
    inner = getattr(loaded, "model", None)
    task = str(getattr(inner, "task", getattr(loaded, "task", "")) or "").lower()
    head_name = ""
    try:
        head_name = inner.model[-1].__class__.__name__.lower()
    except Exception:
        pass
    cls_name = inner.__class__.__name__.lower() if inner is not None else ""
    is_yoloe = "yoloe" in cls_name or "yoloe" in head_name
    if task == "segment" and is_yoloe:
        return _adapt_yoloe_seg_checkpoint_for_detect_eval(
            loaded,
            model_ref,
            class_names,
            detect_yaml_override=detect_yaml_override,
            scale_override=scale_override,
            zero_embedding_fallback=zero_embedding_fallback,
        )
    return loaded


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone evaluation for trained incremental detect models.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint (best.pt/last.pt).")
    parser.add_argument("--data", nargs="+", required=True, help="One or more detection data YAML files.")
    parser.add_argument("--tags", nargs="*", default=None, help="Optional tags for each data YAML (same length as --data).")
    parser.add_argument("--split", type=str, default="val", help="Dataset split for validation.")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
    parser.add_argument("--batch", type=int, default=16, help="Validation batch size.")
    parser.add_argument("--device", type=str, default="0", help="Device id, e.g. 0 or cpu.")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers.")
    parser.add_argument("--conf", type=float, default=None, help="Optional confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold for validation.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image.")
    parser.add_argument("--plots", action="store_true", help="Save validation plots.")
    parser.add_argument("--save-json", action="store_true", help="Save COCO-style prediction JSON when possible.")
    parser.add_argument("--project", type=str, default="ultralytics/logcopy", help="Output project directory.")
    parser.add_argument("--name", type=str, default="eval_incremental_model", help="Output run name prefix.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow existing output directory.")

    parser.add_argument("--enable-slicing", action="store_true", help="Enable sliding-window slicing for eval.")
    parser.add_argument("--slice-size", type=int, default=512, help="Slicing tile size.")
    parser.add_argument("--slice-overlap", type=float, default=0.25, help="Slicing overlap ratio.")
    parser.add_argument("--min-box-keep-ratio", type=float, default=0.5, help="Minimum kept box area ratio per tile.")
    parser.add_argument("--cache-slices", action="store_true", help="Cache sliced tiles.")
    parser.add_argument("--slice-cache-dir", type=str, default=None, help="Optional custom slicing cache directory.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for deterministic slicing decisions.")
    parser.add_argument(
        "--disable-yoloe-zero-embedding-fallback",
        action="store_true",
        help="Disable zero-text-embedding fallback when adapting YOLOE seg checkpoint to detect eval.",
    )
    parser.add_argument(
        "--yoloe-detect-yaml",
        type=str,
        default=None,
        help="Optional explicit YOLOE detect YAML for seg->detect eval adaptation (e.g. yoloe-v8.yaml).",
    )
    parser.add_argument(
        "--yoloe-scale",
        type=str,
        default=None,
        help="Optional YOLOE scale override when using family detect YAML (n/s/m/l/x).",
    )
    return parser.parse_args()


def _results_summary(results_dict: dict[str, Any]) -> dict[str, float]:
    wanted = (
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "fitness",
    )
    out = {}
    for k in wanted:
        if k in results_dict:
            out[k] = round(float(results_dict[k]), 5)
    return out


def main() -> None:
    args = _parse_args()

    tags = args.tags if args.tags else _default_tags(args.data)
    if len(tags) != len(args.data):
        raise ValueError(f"--tags length ({len(tags)}) must match --data length ({len(args.data)}).")

    primary_data = check_det_dataset(args.data[0], autodownload=False)
    primary_names = _normalize_names(primary_data.get("names", {}))
    if not primary_names:
        raise ValueError(f"Unable to read class names from primary data yaml: {args.data[0]}")

    model = YOLO(args.model)
    model = _maybe_adapt_model_for_eval(
        model,
        args.model,
        primary_names,
        detect_yaml_override=args.yoloe_detect_yaml,
        scale_override=args.yoloe_scale,
        zero_embedding_fallback=not args.disable_yoloe_zero_embedding_fallback,
    )
    LOGGER.info(f"Loaded model for evaluation: {args.model}")
    LOGGER.info(f"Model names: {getattr(model, 'names', None)}")

    slice_cfg = SliceConfig(
        enabled=args.enable_slicing,
        slice_size=args.slice_size,
        slice_overlap=args.slice_overlap,
        min_box_keep_ratio=args.min_box_keep_ratio,
        empty_tile_keep_prob=1.0,
        cache_slices=args.cache_slices,
        cache_dir=args.slice_cache_dir,
        seed=args.seed,
    )

    all_summaries: dict[str, dict[str, float]] = {}
    base_save_dir = Path(args.project) / args.name
    base_save_dir.mkdir(parents=True, exist_ok=True)

    for data_yaml, tag in zip(args.data, tags):
        eval_yaml = _prepare_eval_data_yaml(
            data_yaml,
            enable_slicing=args.enable_slicing,
            slice_cfg=slice_cfg,
            save_dir=base_save_dir,
            tag=tag,
        )

        LOGGER.info(f"[{tag}] evaluating data={eval_yaml} split={args.split}")
        metrics = model.val(
            data=eval_yaml,
            split=args.split,
            task="detect",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            plots=args.plots,
            save_json=args.save_json,
            save_txt=False,
            save_conf=False,
            project=args.project,
            name=f"{args.name}_{tag}",
            exist_ok=args.exist_ok,
            verbose=True,
            compile=False,
        )
        results_dict = getattr(metrics, "results_dict", {}) or {}
        all_summaries[tag] = _results_summary(results_dict)
        LOGGER.info(f"[{tag}] summary: {all_summaries[tag]}")

    LOGGER.info("Evaluation done. Consolidated summary:")
    for tag, summary in all_summaries.items():
        LOGGER.info(f"  - {tag}: {summary}")


if __name__ == "__main__":
    main()
