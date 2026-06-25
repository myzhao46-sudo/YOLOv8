from __future__ import annotations

import sys
sys.path.insert(0, r"C:/Users/DOCTOR/Documents/GitHub/YOLOv8/ultralytics")
sys.path.insert(1, r"C:/Users/DOCTOR/Documents/GitHub/YOLOv8")

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from tools.eval.eval_teacher_ship_external import (
    collect_ground_truth,
    collect_predictions,
    evaluate_predictions,
    labels_dir_from_images_dir,
    make_letterbox_rgb,
    obj_type,
    raw_yoloeseg_fused_forward,
    shape_of,
    tensor_from_rgb_images,
    to_plain_names,
)


REPO_ROOT = Path(r"C:/Users/DOCTOR/Documents/GitHub/YOLOv8")
PACKAGE_ROOT = REPO_ROOT / "ultralytics"
SOURCE_DATASET = PACKAGE_ROOT / "datasets" / "bridge_ship_distill_2cls"
GLOBAL4_DATASET = PACKAGE_ROOT / "datasets" / "bridge_ship_distill_global4"
SHIP_EXTERNAL = PACKAGE_ROOT / "datasets" / "ship_extratest_no_overlap"
BRIDGE_EXTERNAL = PACKAGE_ROOT / "datasets" / "extracttest_bridge"
INIT_CHECKPOINT = (
    REPO_ROOT
    / "runs"
    / "probe"
    / "yoloeseg_global4_expanded_init"
    / "weights"
    / "global4_expanded_bestpt.pt"
)
RUN_DIR = REPO_ROOT / "runs" / "train" / "yoloeseg_global4_boxonly_100ep_bridge_row_only"
SUMMARY_PATH = REPO_ROOT / "runs" / "train" / "yoloeseg_global4_boxonly_100ep_summary.json"

NAMES4 = ["ship", "harbor", "tank", "bridge"]
MODALITIES = {"RGB": "extratest_rgb", "SAR": "extratest_sar", "IR": "extratest_ir"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EXPECTED_TRAINABLE = [
    f"model.22.cv3.{scale}.2.{kind}" for scale in range(3) for kind in ("weight", "bias")
]


def log(message: object = "") -> None:
    print(message, flush=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(file_sha256(path).encode("ascii"))
    return digest.hexdigest()


def list_images(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMG_EXTS)


def safe_overwrite_output(root: Path) -> None:
    resolved = root.resolve()
    allowed_parent = (PACKAGE_ROOT / "datasets").resolve()
    if resolved != GLOBAL4_DATASET.resolve() or resolved.parent != allowed_parent:
        raise RuntimeError(f"Refusing to overwrite unexpected directory: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def create_global4_dataset() -> dict:
    source_digest_before = directory_digest(SOURCE_DATASET)
    safe_overwrite_output(GLOBAL4_DATASET)
    stats = {
        "source_local2_dataset": str(SOURCE_DATASET),
        "global4_dataset": str(GLOBAL4_DATASET),
        "splits": {},
        "class_histogram": {},
        "bad_rows": 0,
        "label_rows": 0,
        "source_digest_before": source_digest_before,
    }
    for split in ("train", "val"):
        src_images = SOURCE_DATASET / "images" / split
        src_labels = SOURCE_DATASET / "labels" / split
        dst_images = GLOBAL4_DATASET / "images" / split
        dst_labels = GLOBAL4_DATASET / "labels" / split
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)
        images = list_images(src_images)
        missing_labels = 0
        split_rows = 0
        split_histogram: dict[int, int] = {}
        for image in images:
            shutil.copy2(image, dst_images / image.name)
            source_label = src_labels / f"{image.stem}.txt"
            target_label = dst_labels / source_label.name
            if not source_label.exists():
                missing_labels += 1
                target_label.write_text("", encoding="utf-8")
                continue
            output_rows = []
            for line_number, line in enumerate(source_label.read_text(encoding="utf-8").splitlines(), 1):
                parts = line.split()
                if not parts:
                    continue
                if len(parts) != 5:
                    raise ValueError(f"Non-5-column source label: {source_label}:{line_number}: {line}")
                cls = int(float(parts[0]))
                if cls not in {0, 1}:
                    raise ValueError(f"Unexpected local2 class {cls}: {source_label}:{line_number}")
                coords = [float(value) for value in parts[1:]]
                target_cls = 0 if cls == 0 else 3
                output_rows.append(" ".join([str(target_cls), *[f"{value:.8g}" for value in coords]]))
                split_rows += 1
                split_histogram[target_cls] = split_histogram.get(target_cls, 0) + 1
            target_label.write_text("\n".join(output_rows) + ("\n" if output_rows else ""), encoding="utf-8")
        stats["splits"][split] = {
            "images_dir": str(dst_images),
            "labels_dir": str(dst_labels),
            "image_count": len(images),
            "missing_source_label_files": missing_labels,
            "label_rows": split_rows,
            "class_histogram": split_histogram,
        }
        stats["label_rows"] += split_rows
        for cls, count in split_histogram.items():
            stats["class_histogram"][cls] = stats["class_histogram"].get(cls, 0) + count

    data = {
        "path": GLOBAL4_DATASET.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "nc": 4,
        "names": {0: "ship", 1: "harbor", 2: "tank", 3: "bridge"},
    }
    (GLOBAL4_DATASET / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    source_manifest = SOURCE_DATASET / "manifest.csv"
    if source_manifest.exists():
        shutil.copy2(source_manifest, GLOBAL4_DATASET / "source_manifest.csv")
    stats["data_yaml"] = str(GLOBAL4_DATASET / "data.yaml")
    stats["source_digest_after"] = directory_digest(SOURCE_DATASET)
    stats["source_local2_modified"] = stats["source_digest_before"] != stats["source_digest_after"]
    return stats


def check_detect_labels(root: Path, allowed_classes: set[int], expected_names: list[str] | None = None) -> dict:
    report = {"root": str(root), "splits": {}, "class_histogram": {}, "label_rows": 0, "bad_rows": 0}
    for split in ("train", "val"):
        images_dir = root / "images" / split
        labels_dir = root / "labels" / split
        images = list_images(images_dir)
        missing = 0
        histogram: dict[int, int] = {}
        rows = bad = 0
        for image in images:
            label = labels_dir / f"{image.stem}.txt"
            if not label.exists():
                missing += 1
                continue
            for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                parts = line.split()
                if not parts:
                    continue
                if len(parts) != 5:
                    bad += 1
                    continue
                try:
                    cls = int(float(parts[0]))
                    coords = [float(value) for value in parts[1:]]
                except ValueError:
                    bad += 1
                    continue
                if cls not in allowed_classes or not all(math.isfinite(value) for value in coords):
                    bad += 1
                    continue
                rows += 1
                histogram[cls] = histogram.get(cls, 0) + 1
        report["splits"][split] = {
            "images": len(images),
            "label_rows": rows,
            "missing_label_files": missing,
            "bad_rows": bad,
            "class_histogram": histogram,
        }
        report["label_rows"] += rows
        report["bad_rows"] += bad
        for cls, count in histogram.items():
            report["class_histogram"][cls] = report["class_histogram"].get(cls, 0) + count
    if expected_names is not None:
        data = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
        yaml_names = data.get("names", {})
        normalized_names = [yaml_names.get(i, yaml_names.get(str(i))) for i in range(len(expected_names))]
        report["yaml"] = {"nc": data.get("nc"), "names": normalized_names}
        if data.get("nc") != len(expected_names) or normalized_names != expected_names:
            raise ValueError(f"Unexpected data.yaml class configuration: {report['yaml']}")
    report["only_allowed_classes"] = set(report["class_histogram"]) <= allowed_classes
    if report["bad_rows"] or not report["only_allowed_classes"]:
        raise ValueError(f"Dataset label validation failed: {report}")
    return report


def check_external(root: Path, expected_class: int) -> tuple[dict, dict[str, list[Path]]]:
    report = {"root": str(root), "modalities": {}, "class_histogram": {}, "label_rows": 0, "bad_rows": 0}
    groups = {}
    for modality, split in MODALITIES.items():
        images_dir = root / "images" / split
        labels_dir = root / "labels" / split
        images = list_images(images_dir)
        groups[modality] = images
        histogram: dict[int, int] = {}
        rows = bad = missing = 0
        for image in images:
            label = labels_dir / f"{image.stem}.txt"
            if not label.exists():
                missing += 1
                continue
            for line in label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if not parts:
                    continue
                if len(parts) != 5:
                    bad += 1
                    continue
                try:
                    cls = int(float(parts[0]))
                    [float(value) for value in parts[1:]]
                except ValueError:
                    bad += 1
                    continue
                rows += 1
                histogram[cls] = histogram.get(cls, 0) + 1
        report["modalities"][modality] = {
            "images_dir": str(images_dir),
            "labels_dir": str(labels_dir),
            "image_count": len(images),
            "missing_label_files": missing,
            "label_rows": rows,
            "bad_rows": bad,
            "class_histogram": histogram,
        }
        report["label_rows"] += rows
        report["bad_rows"] += bad
        for cls, count in histogram.items():
            report["class_histogram"][cls] = report["class_histogram"].get(cls, 0) + count
    report["only_expected_class"] = set(report["class_histogram"]) <= {expected_class}
    if report["bad_rows"] or not report["only_expected_class"]:
        raise ValueError(f"External dataset validation failed: {report}")
    return report, groups


def indexed_hashes(paths: list[Path]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    hashes: dict[str, list[str]] = {}
    stems: dict[str, list[str]] = {}
    for path in paths:
        resolved = str(path.resolve())
        hashes.setdefault(file_sha256(path), []).append(resolved)
        stems.setdefault(path.stem.lower(), []).append(resolved)
    return hashes, stems


def overlap_report(left: list[Path], right: list[Path]) -> dict:
    left_hashes, left_stems = indexed_hashes(left)
    right_hashes, right_stems = indexed_hashes(right)
    common_hashes = sorted(set(left_hashes) & set(right_hashes))
    common_stems = sorted(set(left_stems) & set(right_stems))
    return {
        "hash_overlap_count": len(common_hashes),
        "stem_overlap_count": len(common_stems),
        "hash_overlap_examples": [
            {"sha256": key, "training": left_hashes[key], "external": right_hashes[key]} for key in common_hashes[:20]
        ],
        "stem_overlap_examples": [
            {"stem": key, "training": left_stems[key], "external": right_stems[key]} for key in common_stems[:20]
        ],
    }


def leakage_check(train_images: list[Path], val_images: list[Path], ship_images: list[Path], bridge_images: list[Path]) -> dict:
    training = train_images + val_images
    ship = overlap_report(training, ship_images)
    bridge = overlap_report(training, bridge_images)
    return {
        "train_image_count": len(train_images),
        "val_image_count": len(val_images),
        "external_ship_image_count": len(ship_images),
        "external_bridge_image_count": len(bridge_images),
        "train_val_vs_ship": ship,
        "train_val_vs_bridge": bridge,
        "leakage_detected": ship["hash_overlap_count"] > 0 or bridge["hash_overlap_count"] > 0,
        "possible_name_overlap": ship["stem_overlap_count"] > 0 or bridge["stem_overlap_count"] > 0,
    }


class DetectLabelDataset(Dataset):
    def __init__(self, images_dir: Path, labels_dir: Path, imgsz: int):
        self.images = list_images(images_dir)
        self.labels_dir = labels_dir
        self.imgsz = imgsz

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict:
        image_path = self.images[index]
        rgb, meta = make_letterbox_rgb(image_path, self.imgsz)
        image = tensor_from_rgb_images([rgb])[0]
        classes = []
        boxes = []
        label_path = self.labels_dir / f"{image_path.stem}.txt"
        for line in label_path.read_text(encoding="utf-8").splitlines() if label_path.exists() else []:
            parts = line.split()
            if not parts:
                continue
            cls = int(float(parts[0]))
            cx, cy, width, height = [float(value) for value in parts[1:]]
            new_cx = (cx * meta["orig_w"] * meta["scale"] + meta["pad_x"]) / self.imgsz
            new_cy = (cy * meta["orig_h"] * meta["scale"] + meta["pad_y"]) / self.imgsz
            new_w = width * meta["orig_w"] * meta["scale"] / self.imgsz
            new_h = height * meta["orig_h"] * meta["scale"] / self.imgsz
            classes.append([float(cls)])
            boxes.append([new_cx, new_cy, new_w, new_h])
        return {
            "img": image,
            "cls": torch.tensor(classes, dtype=torch.float32).reshape(-1, 1),
            "bboxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "path": str(image_path),
        }


def collate_detect(batch: list[dict]) -> dict:
    classes = []
    boxes = []
    batch_indices = []
    for index, sample in enumerate(batch):
        count = sample["cls"].shape[0]
        if count:
            classes.append(sample["cls"])
            boxes.append(sample["bboxes"])
            batch_indices.append(torch.full((count,), index, dtype=torch.long))
    return {
        "img": torch.stack([sample["img"] for sample in batch]),
        "cls": torch.cat(classes) if classes else torch.zeros((0, 1), dtype=torch.float32),
        "bboxes": torch.cat(boxes) if boxes else torch.zeros((0, 4), dtype=torch.float32),
        "batch_idx": torch.cat(batch_indices) if batch_indices else torch.zeros((0,), dtype=torch.long),
        "paths": [sample["path"] for sample in batch],
    }


def set_global4(model) -> torch.Tensor:
    embeddings = model.get_text_pe(NAMES4)
    model.set_classes(NAMES4, embeddings)
    model.model.set_classes(NAMES4, embeddings)
    return embeddings


def get_head(model):
    return model.model.model[-1]


def task_state(model) -> dict:
    args = getattr(model.model, "args", {})
    return {
        "wrapper.task": getattr(model, "task", None),
        "inner.task": getattr(model.model, "task", None),
        "inner.args.task": args.get("task") if isinstance(args, dict) else getattr(args, "task", None),
    }


def probe_model(model, embeddings: torch.Tensor, device: str, imgsz: int) -> dict:
    inner = model.model.eval().to(device)
    head = get_head(model)
    with torch.no_grad():
        _, channels = raw_yoloeseg_fused_forward(inner, torch.zeros(1, 3, imgsz, imgsz, device=device))
    return {
        "wrapper_type": obj_type(model),
        "model_class": obj_type(inner),
        "task": task_state(model),
        "head": obj_type(head),
        "nc": int(head.nc),
        "no": int(head.no),
        "is_fused": bool(head.is_fused),
        "pe_shape": shape_of(embeddings),
        "score_channels_seen": [int(channels)],
        "names": to_plain_names(inner.names),
        "total_params": sum(parameter.numel() for parameter in inner.parameters()),
        "trainable_params": sum(parameter.numel() for parameter in inner.parameters() if parameter.requires_grad),
    }


def raw_detection_forward(inner: nn.Module, image: torch.Tensor) -> dict[str, torch.Tensor]:
    from ultralytics.nn.modules.head import YOLOESegment

    saved = []
    output = image
    head = inner.model[-1]
    for layer in inner.model:
        if layer.f != -1:
            output = saved[layer.f] if isinstance(layer.f, int) else [output if j == -1 else saved[j] for j in layer.f]
        if layer is head:
            if not isinstance(head, YOLOESegment) or not head.is_fused:
                raise TypeError(f"Expected fused YOLOESegment, got {obj_type(head)}")
            features = output
            batch_size = features[0].shape[0]
            boxes = []
            scores = []
            for scale in range(head.nl):
                boxes.append(head.cv2[scale](features[scale]).view(batch_size, 4 * head.reg_max, -1))
                cls_features = head.cv3[scale](features[scale])
                score = head.cv4[scale](cls_features, None)
                scores.append(score.reshape(batch_size, score.shape[1], -1))
            return {"boxes": torch.cat(boxes, 2), "scores": torch.cat(scores, 2), "feats": features}
        output = layer(output)
        saved.append(output if layer.i in inner.save else None)
    raise RuntimeError("YOLOESegment head was not reached.")


def configure_bridge_row_only(model) -> tuple[torch.optim.Optimizer, dict]:
    inner = model.model
    inner.eval()
    for parameter in inner.parameters():
        parameter.requires_grad_(False)
    head = get_head(model)
    trainable_names = []
    trainable_parameters = []
    old_rows = {}
    hook_handles = []
    for scale, branch in enumerate(head.cv3):
        conv = branch[-1]
        if not isinstance(conv, nn.Conv2d) or conv.out_channels != 4:
            raise TypeError(f"Unexpected fused score conv at scale {scale}: {conv}")
        for kind in ("weight", "bias"):
            parameter = getattr(conv, kind)
            name = f"model.22.cv3.{scale}.2.{kind}"
            parameter.requires_grad_(True)
            old_rows[name] = parameter.detach()[:3].clone()

            def keep_bridge_row_only(gradient, rows=3):
                gradient = gradient.clone()
                gradient[:rows].zero_()
                return gradient

            hook_handles.append(parameter.register_hook(keep_bridge_row_only))
            trainable_names.append(name)
            trainable_parameters.append(parameter)
    optimizer = torch.optim.AdamW(trainable_parameters, lr=0.001, weight_decay=0.0)
    return optimizer, {
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_tensors": len(trainable_parameters),
        "trainable_scalar_params": sum(parameter.numel() for parameter in trainable_parameters),
        "old_rows": old_rows,
        "hook_handles": hook_handles,
        "bn_frozen": all(not module.training for module in inner.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)),
        "frozen_parameter_scalar_count": sum(parameter.numel() for parameter in inner.parameters() if not parameter.requires_grad),
    }


def snapshot_state(inner: nn.Module) -> dict[str, torch.Tensor]:
    return {key: tensor.detach().clone() for key, tensor in inner.state_dict().items()}


def restore_old_rows(inner: nn.Module, initial: dict[str, torch.Tensor]) -> None:
    state = inner.state_dict()
    with torch.no_grad():
        for key in EXPECTED_TRAINABLE:
            state[key][:3].copy_(initial[key][:3])


def verify_preservation(inner: nn.Module, initial: dict[str, torch.Tensor]) -> tuple[bool, bool, list[str]]:
    current = inner.state_dict()
    old_rows_unchanged = all(torch.equal(current[key][:3], initial[key][:3]) for key in EXPECTED_TRAINABLE)
    changed_unrelated = [
        key for key in current if key not in EXPECTED_TRAINABLE and not torch.equal(current[key], initial[key])
    ]
    return old_rows_unchanged, not changed_unrelated, changed_unrelated[:20]


def grad_norm(parameters: list[nn.Parameter]) -> float:
    squared = sum(float(parameter.grad.detach().float().pow(2).sum()) for parameter in parameters if parameter.grad is not None)
    return math.sqrt(squared)


def train_bridge_row_only(model, args, summary: dict) -> list[dict]:
    from ultralytics.cfg import DEFAULT_CFG
    from ultralytics.utils.loss import v8DetectionLoss

    inner = model.model.to(args.device).eval()
    optimizer, config = configure_bridge_row_only(model)
    optimizer.param_groups[0]["lr"] = args.lr
    summary["training"]["parameter_config"] = {
        key: value for key, value in config.items() if key not in {"old_rows", "hook_handles"}
    }
    summary["training"]["optimizer"] = "AdamW"
    summary["training"]["weight_decay"] = 0.0
    initial_state = snapshot_state(inner)
    criterion = v8DetectionLoss(inner)
    if isinstance(criterion.hyp, dict):
        merged_hyp = dict(vars(DEFAULT_CFG))
        merged_hyp.update(criterion.hyp)
        criterion.hyp = SimpleNamespace(**merged_hyp)
    summary["training"]["loss_gains"] = {
        "box": float(criterion.hyp.box),
        "cls": float(criterion.hyp.cls),
        "dfl": float(criterion.hyp.dfl),
    }
    dataset = DetectLabelDataset(GLOBAL4_DATASET / "images/train", GLOBAL4_DATASET / "labels/train", args.imgsz)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_detect,
        pin_memory=args.device.startswith("cuda"),
        generator=generator,
    )
    trainable_parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    history = []
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        sums = torch.zeros(3, dtype=torch.float64)
        total_loss_sum = 0.0
        max_grad_norm = 0.0
        bridge_nonzero_grad_batches = 0
        for batch in loader:
            image = batch["img"].to(args.device, non_blocking=True)
            loss_batch = {
                "batch_idx": batch["batch_idx"].to(args.device),
                "cls": batch["cls"].to(args.device),
                "bboxes": batch["bboxes"].to(args.device),
            }
            optimizer.zero_grad(set_to_none=True)
            predictions = raw_detection_forward(inner, image)
            loss_items, detached = criterion(predictions, loss_batch)
            total_loss = loss_items.sum()
            total_loss.backward()
            current_grad_norm = grad_norm(trainable_parameters)
            max_grad_norm = max(max_grad_norm, current_grad_norm)
            bridge_nonzero_grad_batches += int(current_grad_norm > 0)
            optimizer.step()
            restore_old_rows(inner, initial_state)
            total_loss_sum += float(total_loss.detach())
            sums += detached.detach().double().cpu()
        old_rows_ok, unrelated_ok, changed_examples = verify_preservation(inner, initial_state)
        row = {
            "epoch": epoch,
            "loss_total": total_loss_sum / len(loader),
            "loss_box": float(sums[0] / len(loader)),
            "loss_cls": float(sums[1] / len(loader)),
            "loss_dfl": float(sums[2] / len(loader)),
            "trainable_grad_norm_max": max_grad_norm,
            "batches_with_nonzero_bridge_grad": bridge_nonzero_grad_batches,
            "batches": len(loader),
            "old_rows_unchanged": old_rows_ok,
            "unrelated_params_unchanged": unrelated_ok,
            "changed_unrelated_examples": changed_examples,
        }
        history.append(row)
        log(
            f"epoch {epoch:03d}/{args.epochs}: total={row['loss_total']:.6f} box={row['loss_box']:.6f} "
            f"cls={row['loss_cls']:.6f} dfl={row['loss_dfl']:.6f} grad={max_grad_norm:.6f} "
            f"old_rows={old_rows_ok} unrelated={unrelated_ok}"
        )
        if not old_rows_ok or not unrelated_ok:
            raise RuntimeError(f"Parameter preservation failed at epoch {epoch}: {row}")
        summary["training"]["history"] = history
        save_summary(summary, Path(args.summary))
    final_old_rows, final_unrelated, changed = verify_preservation(inner, initial_state)
    summary["training"].update(
        {
            "old_rows_unchanged_after_training": final_old_rows,
            "unrelated_params_unchanged_after_training": final_unrelated,
            "changed_unrelated_examples": changed,
            "loss_terms": {
                "box_loss_computed": True,
                "cls_loss_computed": True,
                "dfl_loss_computed": True,
                "box_loss_updates_trainable_parameters": False,
                "dfl_loss_updates_trainable_parameters": False,
                "cls_loss_updates_bridge_rows": True,
            },
        }
    )
    for handle in config["hook_handles"]:
        handle.remove()
    return history


def eval_target(model, groups: dict[str, list[Path]], pred_internal_class: int, gt_class: int, args) -> dict:
    report = {}
    all_predictions = []
    all_ground_truth = {}
    all_channels = set()
    totals = {"image_count": 0, "missing_label_files": 0, "bad_label_rows": 0, "non_target_rows": 0}
    pred_args = SimpleNamespace(
        device=args.device,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        pred_iou=args.pred_iou,
        max_det=args.max_det,
        target_class=pred_internal_class,
        strict_label=True,
    )
    gt_args = SimpleNamespace(target_class=gt_class, strict_label=True)
    for modality, images in groups.items():
        labels_dir = labels_dir_from_images_dir(images[0].parent)
        predictions, channels, nms_count = collect_predictions(model.model, images, pred_args)
        ground_truth, gt_stats = collect_ground_truth(images, labels_dir, gt_args)
        metrics = evaluate_predictions(predictions, ground_truth, args.match_iou)
        report[modality] = {
            **metrics,
            **gt_stats,
            "nms_boxes_all_classes": nms_count,
            "score_channels_seen": sorted(channels),
            "pred_internal_class": pred_internal_class,
            "gt_class": gt_class,
        }
        log(
            f"{modality}: GT={metrics['gt_count']} pred={metrics['prediction_count']} TP/FP/FN="
            f"{metrics['tp']}/{metrics['fp']}/{metrics['fn']} P={metrics['precision']:.6f} "
            f"R={metrics['recall']:.6f} AP50={metrics['ap50']:.6f}"
        )
        all_predictions.extend(predictions)
        all_ground_truth.update(ground_truth)
        all_channels.update(channels)
        for key in totals:
            totals[key] += gt_stats[key]
    all_metrics = evaluate_predictions(all_predictions, all_ground_truth, args.match_iou)
    report["ALL"] = {**all_metrics, **totals, "score_channels_seen": sorted(all_channels)}
    log(
        f"ALL: GT={all_metrics['gt_count']} pred={all_metrics['prediction_count']} TP/FP/FN="
        f"{all_metrics['tp']}/{all_metrics['fp']}/{all_metrics['fn']} P={all_metrics['precision']:.6f} "
        f"R={all_metrics['recall']:.6f} AP50={all_metrics['ap50']:.6f}"
    )
    return report


def combined_metrics(ship: dict, bridge: dict) -> dict:
    ship_all = ship["ALL"]
    bridge_all = bridge["ALL"]
    tp = ship_all["tp"] + bridge_all["tp"]
    fp = ship_all["fp"] + bridge_all["fp"]
    fn = ship_all["fn"] + bridge_all["fn"]
    return {
        "definition": "macro AP50 over output classes 0 ship and 1 bridge; micro P/R from summed TP/FP/FN",
        "all_ap50": (ship_all["ap50"] + bridge_all["ap50"]) / 2,
        "precision_micro": tp / max(tp + fp, 1),
        "recall_micro": tp / max(tp + fn, 1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ship": ship_all,
        "bridge": bridge_all,
        "class_mapping": {"internal_0": "output_0_ship", "internal_3": "output_1_bridge"},
    }


def save_checkpoints(model, run_dir: Path) -> dict:
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    last = weights_dir / "last.pt"
    best = weights_dir / "best.pt"
    model.save(last)
    shutil.copy2(last, best)
    return {"last": str(last), "best": str(best), "last_exists": last.exists(), "best_exists": best.exists()}


def save_summary(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="100-epoch bridge-row-only box training for global4 fused YOLOESegment.")
    parser.add_argument("--init", default=str(INIT_CHECKPOINT))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--pred-iou", type=float, default=0.7)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--summary", default=str(SUMMARY_PATH))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if str(args.device).isdigit():
        args.device = f"cuda:{args.device}"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    from ultralytics.models.yolo.model import YOLOE
    import ultralytics

    summary = {
        "time": datetime.now().isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "ultralytics_imported_from": str(Path(ultralytics.__file__).resolve()),
        "settings": vars(args),
        "native_segmentation_trainer_called": False,
        "native_val_called": False,
        "masks_used": False,
        "strategy": "bridge_row_only",
        "training": {
            "completed": False,
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "device": args.device,
            "lr": args.lr,
            "weight_decay": 0.0,
            "history": [],
        },
    }
    summary_path = Path(args.summary).resolve()
    try:
        log("[CREATE GLOBAL4 DATASET]")
        summary["dataset_creation"] = create_global4_dataset()
        summary["training_data_check"] = check_detect_labels(GLOBAL4_DATASET, {0, 3}, NAMES4)
        if summary["dataset_creation"]["source_local2_modified"]:
            raise RuntimeError("Source local2 dataset changed while creating the global4 copy.")
        log(json.dumps(summary["training_data_check"], ensure_ascii=False, indent=2))

        log("\n[EXTERNAL DATA CHECK]")
        summary["ship_external"], ship_groups = check_external(SHIP_EXTERNAL, 0)
        summary["bridge_external"], bridge_groups = check_external(BRIDGE_EXTERNAL, 1)
        log(json.dumps({"ship": summary["ship_external"], "bridge": summary["bridge_external"]}, ensure_ascii=False, indent=2))

        train_images = list_images(GLOBAL4_DATASET / "images/train")
        val_images = list_images(GLOBAL4_DATASET / "images/val")
        ship_images = [path for group in ship_groups.values() for path in group]
        bridge_images = [path for group in bridge_groups.values() for path in group]
        summary["leakage_check"] = leakage_check(train_images, val_images, ship_images, bridge_images)
        log("\n[LEAKAGE CHECK]")
        log(json.dumps(summary["leakage_check"], ensure_ascii=False, indent=2))

        log("\n[LOAD AND PRE-TRAIN PROBE]")
        model = YOLOE(str(Path(args.init).resolve()), task="segment")
        embeddings = set_global4(model)
        summary["pre_train_probe"] = probe_model(model, embeddings, args.device, args.imgsz)
        log(json.dumps(summary["pre_train_probe"], ensure_ascii=False, indent=2))
        probe = summary["pre_train_probe"]
        if not (
            probe["model_class"].endswith("YOLOESegModel")
            and probe["head"].endswith("YOLOESegment")
            and probe["task"]["inner.task"] == "segment"
            and probe["nc"] == 4
            and probe["score_channels_seen"] == [4]
            and probe["pe_shape"] == [1, 4, 512]
        ):
            raise RuntimeError(f"Pre-train structural probe failed: {probe}")

        log("\n[EPOCH 0 SHIP EXTERNAL]")
        summary["epoch0_ship_eval"] = eval_target(model, ship_groups, 0, 0, args)
        save_summary(summary, summary_path)

        log("\n[100-EPOCH BOX-ONLY BRIDGE-ROW TRAINING]")
        train_bridge_row_only(model, args, summary)
        summary["training"]["completed"] = True
        summary["checkpoints"] = save_checkpoints(model, RUN_DIR)
        save_summary(summary, summary_path)

        log("\n[RELOAD FINAL CHECKPOINT AND PROBE]")
        final_model = YOLOE(summary["checkpoints"]["last"], task="segment")
        final_embeddings = set_global4(final_model)
        summary["post_train_probe"] = probe_model(final_model, final_embeddings, args.device, args.imgsz)
        log(json.dumps(summary["post_train_probe"], ensure_ascii=False, indent=2))
        if summary["post_train_probe"]["score_channels_seen"] != [4]:
            raise RuntimeError("Final checkpoint does not emit four score channels.")

        log("\n[FINAL SHIP EXTERNAL]")
        summary["final_ship_eval"] = eval_target(final_model, ship_groups, 0, 0, args)
        log("\n[FINAL BRIDGE EXTERNAL: GT 1 <- INTERNAL PRED 3]")
        summary["final_bridge_eval"] = eval_target(final_model, bridge_groups, 3, 1, args)
        summary["final_combined_eval"] = combined_metrics(summary["final_ship_eval"], summary["final_bridge_eval"])

        epoch0_ship = summary["epoch0_ship_eval"]["ALL"]["ap50"]
        final_ship = summary["final_ship_eval"]["ALL"]["ap50"]
        final_bridge = summary["final_bridge_eval"]["ALL"]["ap50"]
        summary["conclusion"] = {
            "epoch0_ship_ap50": epoch0_ship,
            "final_ship_ap50": final_ship,
            "ship_ap50_change": final_ship - epoch0_ship,
            "final_bridge_ap50": final_bridge,
            "ship_preserved_strict": final_ship >= 0.90,
            "ship_preserved_loose": final_ship >= 0.85,
            "bridge_has_ap": final_bridge > 0,
            "leakage_detected": summary["leakage_check"]["leakage_detected"],
            "formal_result_valid": not summary["leakage_check"]["leakage_detected"],
        }
    except Exception as error:
        summary["error"] = repr(error)
        summary["traceback"] = traceback.format_exc()
        log(summary["traceback"])
    finally:
        summary["finished_at"] = datetime.now().isoformat()
        save_summary(summary, summary_path)
        log(f"\n[SUMMARY] {summary_path}")
    return 1 if "error" in summary else 0


if __name__ == "__main__":
    raise SystemExit(main())
