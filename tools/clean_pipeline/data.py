"""Prepare isolated, reproducible real-data subsets without modifying external assets."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image


MODALITIES = ("rgb", "sar", "ir")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty source manifest: {path}")
    return rows


def _basename(value: str) -> str:
    return value.replace("\\", "/").rsplit("/", 1)[-1]


def _inventory(root: Path, rows: list[dict], *, evaluation: bool) -> list[dict]:
    """Resolve recorded filenames against the actual machine's asset root."""
    records = []
    for row in rows:
        split = row["split"]
        if split not in ({"test_ship", "test_bridge"} if evaluation else {"train", "val"}):
            continue
        modality = row["modality"].lower()
        if modality not in MODALITIES:
            raise ValueError(f"Unsupported modality in {root}: {modality}")
        name = _basename(row["target_image" if evaluation else "image_path_new"])
        folder = split if evaluation else f"{split}_{modality}"
        image = root / "images" / folder / name
        label = root / "labels" / folder / Path(name).with_suffix(".txt").name
        if not image.is_file() or not label.is_file():
            raise FileNotFoundError(f"Missing manifest asset: {image} or {label}")
        records.append({
            "image": image, "label": label, "split": split, "modality": modality,
            "sha256": row["hash_sha256" if evaluation else "sha256"],
            "class_id": 1 if split == "test_bridge" else 0,
            "source_manifest": root / "manifest.csv", "source_record": row,
        })
    # Check the manifests describe the current directories, not just historical counts.
    folders = {record["image"].parent for record in records}
    for folder in folders:
        actual = {item.name for item in folder.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES}
        recorded = [record["image"].name for record in records if record["image"].parent == folder]
        if actual != set(recorded) or len(recorded) != len(set(recorded)):
            raise ValueError(f"Manifest/directory inventory mismatch or duplicate rows: {folder}")
    return records


def _select(records: list[dict], split: str, count: int, rng: random.Random) -> list[dict]:
    if count < len(MODALITIES):
        raise ValueError(f"{split} requires at least {len(MODALITIES)} images to cover RGB/SAR/IR")
    selected = []
    for index, modality in enumerate(MODALITIES):
        candidates = sorted(
            (record for record in records if record["split"] == split and record["modality"] == modality),
            key=lambda record: record["image"].name,
        )
        required = count // len(MODALITIES) + int(index < count % len(MODALITIES))
        if len(candidates) < required:
            raise ValueError(f"Insufficient {split}/{modality}: need {required}, found {len(candidates)}")
        rng.shuffle(candidates)
        selected.extend(candidates[:required])
    return selected


def _validate(record: dict) -> dict:
    image, label = record["image"], record["label"]
    actual_hash = _sha256(image)
    if actual_hash != record["sha256"]:
        raise ValueError(f"Image content differs from source manifest: {image}")
    with Image.open(image) as opened:
        opened.load()
        width, height = opened.size
        opened.convert("RGB").load()
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {image}")
    boxes = []
    for number, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            fields = [float(value) for value in line.split()]
            if len(fields) != 5 or not all(math.isfinite(value) for value in fields):
                raise ValueError("need five finite values")
            class_id, x, y, box_width, box_height = fields
            if class_id != record["class_id"]:
                raise ValueError(f"expected class {record['class_id']}, got {class_id}")
            if any(value < 0 or value > 1 for value in fields[1:]) or box_width <= 0 or box_height <= 0:
                raise ValueError("invalid normalized coordinates")
            boxes.append([int(class_id), x, y, box_width, box_height])
        except ValueError as exc:
            raise ValueError(f"Invalid label {label}:{number}: {exc}") from exc
    if not boxes:
        raise ValueError(f"Empty label: {label}")
    return {"width": width, "height": height, "labels": boxes, "label_count": len(boxes),
            "sha256": actual_hash, "label_sha256": _sha256(label)}


def prepare_data(
    asset_root: Path,
    out: Path,
    seed: int = 42,
    train_count: int = 12,
    val_count: int = 6,
    eval_per_group: int = 3,
) -> dict:
    """Copy balanced subsets from original splits; return JSON-serializable paths.

    ``eval_per_group`` is the total for each of test_ship and test_bridge, with
    at least one sample per RGB/SAR/IR modality. Training uses class 0 bridge;
    evaluation preserves class 0 ship / class 1 bridge. A fresh ``out/data`` is
    required. Returned paths are absolute; manifest output paths are relative
    to ``out``. No original labels, caches, images or manifests are written.
    """
    asset_root, out = Path(asset_root).resolve(), Path(out).resolve()
    project_root = Path(__file__).resolve().parents[2]
    if not out.is_relative_to(project_root) or out == project_root:
        raise ValueError(f"Smoke output must be inside this worktree: {project_root}")
    if out.is_relative_to(asset_root):
        raise ValueError("Smoke output must not be inside the external asset repository")
    data_root = out / "data"
    if data_root.exists():
        raise FileExistsError(f"Refusing to overwrite an existing data subset: {data_root}")
    datasets = asset_root / "ultralytics" / "datasets"
    trainval_root = datasets / "total_bridge_trainval"
    eval_source = datasets / "bridge_ship_distill_2cls_v2"
    trainval_rows = _read_manifest(trainval_root / "manifest.csv")
    eval_rows = _read_manifest(eval_source / "manifest.csv")
    heldout_rows = _read_manifest(datasets / "total_bridge_test" / "manifest.csv")
    training = _inventory(trainval_root, trainval_rows, evaluation=False)
    evaluation = _inventory(eval_source, eval_rows, evaluation=True)
    rng = random.Random(seed)
    selected = (
        _select(training, "train", train_count, rng)
        + _select(training, "val", val_count, rng)
        + _select(evaluation, "test_ship", eval_per_group, rng)
        + _select(evaluation, "test_bridge", eval_per_group, rng)
    )
    source_hashes = {
        "train": {row["sha256"] for row in trainval_rows if row["split"] == "train"},
        "val": {row["sha256"] for row in trainval_rows if row["split"] == "val"},
        "test": {row["sha256"] for row in heldout_rows}
                | {row["hash_sha256"] for row in eval_rows if row["split"] in {"test_ship", "test_bridge"}},
    }
    selected_hashes = set()
    for record in selected:
        record["validated"] = _validate(record)
        digest = record["validated"]["sha256"]
        if digest in selected_hashes:
            raise ValueError(f"Duplicate image content among selected samples: {record['image']}")
        selected_hashes.add(digest)
        category = record["split"] if record["split"] in {"train", "val"} else "test"
        for other, hashes in source_hashes.items():
            if other != category and digest in hashes:
                raise ValueError(f"Selected {category} content appears in original {other} manifest: {record['image']}")

    # Validate everything first; labels and images are copied, so caches and any
    # data-loader side effects are confined to the new output directory.
    data_root.mkdir(parents=True, exist_ok=False)
    expert_root, eval_root = data_root / "expert", data_root / "evaluation"
    manifest_records = []
    for record in selected:
        destination = expert_root if record["split"] in {"train", "val"} else eval_root
        image = destination / "images" / record["split"] / record["image"].name
        label = destination / "labels" / record["split"] / record["label"].name
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        if image.exists() or label.exists():
            raise FileExistsError(f"Output filename collision: {image}")
        shutil.copyfile(record["image"], image)
        shutil.copyfile(record["label"], label)
        if _sha256(image) != record["sha256"] or _sha256(label) != record["validated"]["label_sha256"]:
            raise RuntimeError(f"Copied data checksum mismatch: {image}")
        manifest_records.append({
            "image": image.relative_to(out).as_posix(), "label": label.relative_to(out).as_posix(),
            "source_image": record["image"].as_posix(), "source_label": record["label"].as_posix(),
            "source_manifest": record["source_manifest"].as_posix(), "source_record": record["source_record"],
            "split": record["split"], "modality": record["modality"], "class_id": record["class_id"],
            **record["validated"],
        })
    train_yaml = expert_root / "data.yaml"
    train_yaml.write_text(yaml.safe_dump({
        "path": expert_root.as_posix(), "train": "images/train", "val": "images/val",
        "nc": 1, "names": {0: "bridge"},
    }, sort_keys=False), encoding="utf-8")
    # Preserve the existing late-fusion evaluator's modality lookup convention.
    with (eval_root / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "target_image", "target_label", "modality", "hash_sha256"])
        writer.writeheader()
        for row in manifest_records:
            if row["split"] not in {"test_ship", "test_bridge"}:
                continue
            writer.writerow({
                "split": row["split"], "target_image": Path(row["image"]).relative_to("data/evaluation").as_posix(),
                "target_label": Path(row["label"]).relative_to("data/evaluation").as_posix(),
                "modality": row["modality"], "hash_sha256": row["sha256"],
            })
    counts = dict(Counter(row["split"] for row in manifest_records))
    manifest = data_root / "manifest.json"
    manifest.write_text(json.dumps({
        "seed": seed, "asset_root": asset_root.as_posix(), "output_paths_relative_to": "parent of data/",
        "selection": "Sorted original filenames, seeded shuffle per split/modality; no resplitting",
        "train_class_names": {0: "bridge"}, "eval_class_names": {0: "ship", 1: "bridge"},
        "counts": counts,
        "source_counts": dict(Counter(f"{row['split']}/{row['modality']}" for row in training + evaluation)),
        "selected_counts": dict(Counter(f"{row['split']}/{row['modality']}" for row in manifest_records)),
        "leakage_check": {
            "selected_content_hashes_recomputed": True, "selected_content_duplicates": 0,
            "selected_vs_original_other_split_manifest_hashes": "no overlap",
            "scope": "All selected image bytes verified against manifests; unselected files not rehashed",
        },
        "records": manifest_records,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"train_yaml": train_yaml.as_posix(), "eval_root": eval_root.as_posix(),
            "manifest": manifest.as_posix(), "data_root": data_root.as_posix(), "counts": counts, "seed": seed}


def validate_prepared(data: dict) -> None:
    """Recheck the exact copied fixtures before reusing a migration reference."""
    manifest = Path(data["manifest"])
    content = json.loads(manifest.read_text(encoding="utf-8"))
    base = manifest.parent.parent
    for row in content["records"]:
        for field, checksum in (("image", "sha256"), ("label", "label_sha256")):
            path = (base / row[field]).resolve()
            if not path.is_relative_to(base) or _sha256(path) != row[checksum]:
                raise ValueError(f"Fixture changed: {path}")
    expected = yaml.safe_load(Path(data["train_yaml"]).read_text(encoding="utf-8"))
    assert Path(expected["path"]).resolve() == (base / "data/expert").resolve()
    assert expected["train"] == "images/train" and expected["val"] == "images/val"
    assert expected["nc"] == 1 and expected["names"] == {0: "bridge"}


def isolate_training_data(data: dict, destination: Path) -> dict:
    """Mirror declared image directories and copy labels so training caches stay local.

    Existing splits and labels are preserved exactly. Images are verified before
    hardlinking; JPEGs without their end marker are rejected so a loader cannot
    repair an external image through a hardlink. No resplitting is performed.
    """
    source_root = Path(data["path"]).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    isolated = dict(data, path=str(destination.resolve()))
    provenance = []
    for split in ("train", "val", "test"):
        if not data.get(split):
            continue
        values = data[split] if isinstance(data[split], list) else [data[split]]
        folders = []
        for index, value in enumerate(values):
            source = Path(value)
            source = source if source.is_absolute() else source_root / source
            if not source.is_dir():
                raise ValueError(f"Expected an existing image directory in {split}: {source}")
            images = sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
            if not images:
                raise ValueError(f"Empty training/validation directory: {source}")
            folder = f"{split}_{index}"
            image_dir, label_dir = destination / "images" / folder, destination / "labels" / folder
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            for number, image in enumerate(images):
                parts = list(image.parts)
                indices = [i for i,p in enumerate(parts) if p.lower() == "images"]
                if not indices:
                    raise ValueError(f"Cannot derive label for {image}")
                parts[indices[-1]] = "labels"
                label = Path(*parts).with_suffix(".txt")
                if not label.is_file():
                    raise FileNotFoundError(label)
                with Image.open(image) as opened:
                    opened.verify()
                if image.suffix.lower() in {".jpg", ".jpeg"}:
                    with image.open("rb") as handle:
                        handle.seek(-2, 2)
                        if handle.read() != b"\xff\xd9":
                            raise ValueError(f"External JPEG requires repair before training: {image}")
                target = image_dir / f"{number:07d}_{image.name}"
                try:
                    os.link(image, target)
                except OSError:
                    shutil.copy2(image, target)
                shutil.copy2(label, label_dir / target.with_suffix(".txt").name)
                provenance.append({"split":split,"image":str(image),"label":str(label),"local_image":str(target)})
            folders.append(f"images/{folder}")
        isolated[split] = folders
    (destination / "source_manifest.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    return isolated
