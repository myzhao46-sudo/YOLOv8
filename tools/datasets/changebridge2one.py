# -*- coding: utf-8 -*-
"""
Copy bridge_small dataset to bridge_small_labels1 and remap bridge labels from class 3 to class 1.

This version supports nested multimodal dataset structure, for example:

bridge_small/
    DIOR_bridge_rgb30/
        images/...
        labels/...
    MSAR_bridge_sar30/
        images/...
        labels/...
    MassMIND_bridge_infr30/
        images/...
        labels/...

It does NOT modify the original dataset.
It does NOT train.
It only copies the dataset and rewrites YOLO label first column in the copied dataset.

Purpose:
    Prepare local 2-class student dataset:
        0: ship
        1: bridge

Original bridge labels:
        3 cx cy w h

New copied labels:
        1 cx cy w h
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from datetime import datetime
from typing import Iterable

try:
    import yaml
except Exception:
    yaml = None


DEFAULT_SRC = Path(
    r"C:\Users\DOCTOR\Documents\GitHub\YOLOv8\ultralytics\datasets\bridge_small"
)
DEFAULT_DST = Path(
    r"C:\Users\DOCTOR\Documents\GitHub\YOLOv8\ultralytics\datasets\bridge_small_labels1"
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def log(msg: object = "") -> None:
    print(msg, flush=True)


def has_path_part(path: Path, target: str) -> bool:
    target = target.lower()
    return any(part.lower() == target for part in path.parts)


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMG_EXTS


def copy_dataset(src: Path, dst: Path, overwrite: bool = False) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source dataset does not exist: {src}")

    if dst.exists():
        if not overwrite:
            raise FileExistsError(
                f"Destination already exists: {dst}\n"
                f"Use --overwrite to delete and recreate it."
            )
        log(f"[COPY] Removing existing destination: {dst}")
        shutil.rmtree(dst)

    log(f"[COPY] {src}")
    log(f"   -> {dst}")
    shutil.copytree(src, dst)


def remove_yolo_cache(dst: Path) -> int:
    removed = 0
    for p in dst.rglob("*.cache"):
        try:
            p.unlink()
            removed += 1
        except Exception as e:
            log(f"[WARN] Failed to remove cache file {p}: {e}")
    return removed


def find_label_files(dst: Path) -> list[Path]:
    """
    Find YOLO label txt files under any nested labels directory.

    We only process .txt files whose path contains a folder named 'labels'.
    This prevents accidentally modifying README/classes files outside label dirs.
    """
    label_files = []
    for p in sorted(dst.rglob("*.txt")):
        if p.is_file() and has_path_part(p.parent, "labels"):
            label_files.append(p)
    return label_files


def parse_yolo_row(line: str, label_path: Path, line_no: int) -> tuple[int, list[str]]:
    parts = line.strip().split()
    if len(parts) != 5:
        raise ValueError(
            f"Bad label row, expected 5 columns: {label_path}:{line_no}: {line}"
        )

    try:
        cls = int(float(parts[0]))
        coords = [float(x) for x in parts[1:]]
    except Exception:
        raise ValueError(
            f"Bad numeric value in label row: {label_path}:{line_no}: {line}"
        )

    for v in coords:
        if not (-1e-6 <= v <= 1.000001):
            raise ValueError(
                f"YOLO coordinate out of [0,1] range: {label_path}:{line_no}: {line}"
            )

    return cls, parts


def remap_label_file(
    label_path: Path,
    src_class: int,
    dst_class: int,
    strict: bool = True,
    allow_already_dst: bool = True,
) -> dict:
    """
    Remap one YOLO label file:
        src_class cx cy w h -> dst_class cx cy w h

    Empty label files are allowed.
    """
    text = label_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    new_rows: list[str] = []

    stats = {
        "converted": 0,
        "already_dst": 0,
        "empty_rows": 0,
        "bad_rows": 0,
        "unexpected_rows": 0,
        "total_rows": 0,
    }

    for line_no, line in enumerate(lines, start=1):
        s = line.strip()
        if not s:
            stats["empty_rows"] += 1
            continue

        try:
            cls, parts = parse_yolo_row(s, label_path, line_no)
        except Exception as e:
            stats["bad_rows"] += 1
            if strict:
                raise
            log(f"[WARN] Skip bad row: {e}")
            continue

        stats["total_rows"] += 1

        if cls == src_class:
            parts[0] = str(dst_class)
            stats["converted"] += 1
        elif allow_already_dst and cls == dst_class:
            stats["already_dst"] += 1
        else:
            stats["unexpected_rows"] += 1
            msg = (
                f"Unexpected class id {cls} in {label_path}:{line_no}. "
                f"Expected {src_class}"
                + (f" or already {dst_class}" if allow_already_dst else "")
                + f". Row: {line}"
            )
            if strict:
                raise ValueError(msg)
            log(f"[WARN] {msg}")

        new_rows.append(" ".join(parts))

    # Preserve valid empty label files as empty files.
    if new_rows:
        label_path.write_text("\n".join(new_rows) + "\n", encoding="utf-8")
    else:
        label_path.write_text("", encoding="utf-8")

    return stats


def remap_all_labels(
    dst: Path,
    src_class: int,
    dst_class: int,
    strict: bool = True,
) -> dict:
    label_files = find_label_files(dst)

    if not label_files:
        raise FileNotFoundError(
            f"No label .txt files found under nested labels directories in: {dst}\n"
            f"Expected structure like:\n"
            f"  bridge_small/DIOR_bridge_rgb30/labels/...\n"
            f"  bridge_small/MSAR_bridge_sar30/labels/...\n"
            f"Please inspect with:\n"
            f"  Get-ChildItem {dst} -Recurse -Filter *.txt | Select-Object -First 30 FullName"
        )

    total = {
        "label_files": 0,
        "converted": 0,
        "already_dst": 0,
        "empty_rows": 0,
        "bad_rows": 0,
        "unexpected_rows": 0,
        "total_rows": 0,
    }

    for label_path in label_files:
        stats = remap_label_file(
            label_path=label_path,
            src_class=src_class,
            dst_class=dst_class,
            strict=strict,
            allow_already_dst=True,
        )

        total["label_files"] += 1
        for k, v in stats.items():
            total[k] += v

    return total


def verify_classes_in_labels(dst: Path, src_class: int, dst_class: int) -> dict:
    label_files = find_label_files(dst)

    remaining_src_rows = 0
    dst_rows = 0
    other_rows = 0
    bad_rows = 0
    empty_files = 0

    class_hist: dict[int, int] = {}

    for label_path in label_files:
        lines = [x.strip() for x in label_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        if not lines:
            empty_files += 1
            continue

        for line_no, line in enumerate(lines, start=1):
            try:
                cls, _ = parse_yolo_row(line, label_path, line_no)
            except Exception:
                bad_rows += 1
                continue

            class_hist[cls] = class_hist.get(cls, 0) + 1

            if cls == src_class:
                remaining_src_rows += 1
            elif cls == dst_class:
                dst_rows += 1
            else:
                other_rows += 1

    return {
        "label_files": len(label_files),
        "empty_label_files": empty_files,
        "remaining_src_class_rows": remaining_src_rows,
        "dst_class_rows": dst_rows,
        "other_class_rows": other_rows,
        "bad_rows": bad_rows,
        "class_hist": class_hist,
    }


def count_images(dst: Path) -> int:
    return sum(1 for p in dst.rglob("*") if is_image_file(p))


def count_label_files(dst: Path) -> int:
    return len(find_label_files(dst))


def rel_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def dir_has_images(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return any(is_image_file(p) for p in path.iterdir())


def find_image_split_dirs(dst: Path, split: str) -> list[str]:
    """
    Find nested image directories for a split.

    Supports:
        xxx/images/train
        xxx/images/val
        xxx/images/test
        xxx/images/extratest_rgb
        etc.

    For train/val yaml, we mainly want exact train/val.
    """
    out: list[str] = []

    for images_dir in sorted(p for p in dst.rglob("images") if p.is_dir()):
        candidate = images_dir / split
        if dir_has_images(candidate):
            out.append(rel_posix(candidate, dst))

    return out


def find_direct_image_dirs_under_images(dst: Path) -> list[str]:
    """
    Fallback: find image directories directly under any folder named images.
    Example:
        DIOR_bridge_rgb30/images
        DIOR_bridge_rgb30/images/train
    """
    out: list[str] = []

    for images_dir in sorted(p for p in dst.rglob("images") if p.is_dir()):
        if dir_has_images(images_dir):
            out.append(rel_posix(images_dir, dst))

        for child in sorted(p for p in images_dir.iterdir() if p.is_dir()):
            if dir_has_images(child):
                out.append(rel_posix(child, dst))

    # Deduplicate while preserving order.
    seen = set()
    dedup = []
    for x in out:
        if x not in seen:
            seen.add(x)
            dedup.append(x)
    return dedup


def write_local2_yaml(dst: Path, yaml_name: str = "data_local2.yaml") -> Path:
    """
    Write a local 2-class yaml.

    If train/val dirs exist under nested images folders, use them.
    Otherwise use all direct image dirs as train and val fallback.
    """
    train_dirs = find_image_split_dirs(dst, "train")
    val_dirs = find_image_split_dirs(dst, "val")

    fallback_dirs = find_direct_image_dirs_under_images(dst)

    if not train_dirs:
        train_dirs = fallback_dirs

    if not val_dirs:
        val_dirs = fallback_dirs

    data = {
        "path": dst.as_posix(),
        "train": train_dirs[0] if len(train_dirs) == 1 else train_dirs,
        "val": val_dirs[0] if len(val_dirs) == 1 else val_dirs,
        "nc": 2,
        "names": {
            0: "ship",
            1: "bridge",
        },
    }

    out = dst / yaml_name

    if yaml is not None:
        with out.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    else:
        # Minimal YAML writer fallback.
        lines = []
        lines.append(f"path: {data['path']}")
        for key in ["train", "val"]:
            value = data[key]
            if isinstance(value, list):
                lines.append(f"{key}:")
                for item in value:
                    lines.append(f"  - {item}")
            else:
                lines.append(f"{key}: {value}")
        lines.append("nc: 2")
        lines.append("names:")
        lines.append("  0: ship")
        lines.append("  1: bridge")
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return out


def print_sample_labels(dst: Path, max_files: int = 5) -> None:
    label_files = find_label_files(dst)
    log("")
    log("[SAMPLE LABELS AFTER REMAP]")

    shown = 0
    for label_path in label_files:
        lines = [x.strip() for x in label_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        if not lines:
            continue

        log(f"{label_path}")
        for line in lines[:3]:
            log(f"  {line}")
        shown += 1

        if shown >= max_files:
            break

    if shown == 0:
        log("No non-empty label files to show.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Copy bridge_small to bridge_small_labels1 and remap nested bridge labels "
            "from class 3 to class 1 for local 2-class student training."
        )
    )
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help="Source bridge_small path.")
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST, help="Destination bridge_small_labels1 path.")
    parser.add_argument("--src-class", type=int, default=3, help="Original bridge class id.")
    parser.add_argument("--dst-class", type=int, default=1, help="New bridge class id.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite destination if it exists.")
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Do not raise on bad/unexpected rows; count warnings instead.",
    )
    parser.add_argument(
        "--no-yaml",
        action="store_true",
        help="Do not write data_local2.yaml.",
    )
    args = parser.parse_args()

    src = args.src.resolve()
    dst = args.dst.resolve()
    strict = not args.non_strict

    log("=" * 80)
    log("Bridge label remap for 2-class student")
    log("=" * 80)
    log(f"Time:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Source:      {src}")
    log(f"Destination: {dst}")
    log(f"Remap:       class {args.src_class} -> class {args.dst_class}")
    log(f"Strict:      {strict}")
    log("")

    copy_dataset(src, dst, overwrite=args.overwrite)

    removed_cache = remove_yolo_cache(dst)
    if removed_cache:
        log(f"[CACHE] Removed {removed_cache} cache file(s).")

    label_files = find_label_files(dst)

    log("")
    log("[FOUND LABEL FILES]")
    log(f"Nested label files found: {len(label_files)}")
    for p in label_files[:10]:
        log(f"  {p}")
    if len(label_files) > 10:
        log(f"  ... {len(label_files) - 10} more")

    log("")
    log("[REMAP LABELS]")
    stats = remap_all_labels(
        dst=dst,
        src_class=args.src_class,
        dst_class=args.dst_class,
        strict=strict,
    )

    verify = verify_classes_in_labels(dst, args.src_class, args.dst_class)

    yaml_path = None
    if not args.no_yaml:
        yaml_path = write_local2_yaml(dst)

    image_count = count_images(dst)
    label_file_count = count_label_files(dst)

    log("")
    log("[SUMMARY]")
    log(f"Images:                {image_count}")
    log(f"Label files:           {label_file_count}")
    log(f"Label files touched:   {stats['label_files']}")
    log(f"Total non-empty rows:  {stats['total_rows']}")
    log(f"Rows converted:        {stats['converted']}")
    log(f"Rows already {args.dst_class}:        {stats['already_dst']}")
    log(f"Empty rows skipped:    {stats['empty_rows']}")
    log(f"Bad rows:              {stats['bad_rows']}")
    log(f"Unexpected rows:       {stats['unexpected_rows']}")

    log("")
    log("[VERIFY]")
    log(f"Class histogram:                 {verify['class_hist']}")
    log(f"Remaining class {args.src_class} rows:       {verify['remaining_src_class_rows']}")
    log(f"Class {args.dst_class} rows:                 {verify['dst_class_rows']}")
    log(f"Other class rows:                {verify['other_class_rows']}")
    log(f"Bad rows after remap:            {verify['bad_rows']}")
    log(f"Empty label files:               {verify['empty_label_files']}")

    if yaml_path is not None:
        log(f"New local-2class yaml:           {yaml_path}")

    print_sample_labels(dst)

    if verify["remaining_src_class_rows"] > 0:
        raise RuntimeError(
            f"Remap incomplete: still found {verify['remaining_src_class_rows']} "
            f"rows with class {args.src_class}."
        )

    if strict and verify["other_class_rows"] > 0:
        raise RuntimeError(
            f"Strict verification failed: found {verify['other_class_rows']} rows "
            f"with classes other than {args.dst_class}."
        )

    log("")
    log("[DONE]")
    log("Original bridge_small was not modified.")
    log("New dataset is ready for local 2-class student training:")
    log("  0: ship")
    log("  1: bridge")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())