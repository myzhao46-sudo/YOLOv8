# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from ultralytics.data.utils import IMG_FORMATS
from ultralytics.utils import LOGGER, TQDM


@dataclass
class SliceConfig:
    """Sliding-window slicing configuration."""

    enabled: bool = False
    slice_size: int = 512
    slice_overlap: float = 0.25
    min_box_keep_ratio: float = 0.5
    empty_tile_keep_prob: float = 0.3
    cache_slices: bool = True
    cache_dir: str | None = None
    seed: int = 0


def prepare_sliced_image_paths(
    img_paths: str | list[str],
    mode: str,
    cfg: SliceConfig,
    run_cache_dir: Path | None = None,
    log_prefix: str = "",
) -> list[str]:
    """Slice one or more YOLO image split roots and return sliced image roots."""
    if not cfg.enabled:
        return img_paths if isinstance(img_paths, list) else [img_paths]

    paths = img_paths if isinstance(img_paths, list) else [img_paths]
    output = []
    for p in paths:
        output.append(_prepare_single_split(Path(p), mode=mode, cfg=cfg, run_cache_dir=run_cache_dir, log_prefix=log_prefix))
    return output


def _prepare_single_split(img_dir: Path, mode: str, cfg: SliceConfig, run_cache_dir: Path | None, log_prefix: str) -> str:
    if not img_dir.exists():
        raise FileNotFoundError(f"Slicing input image directory not found: {img_dir}")

    label_dir = _infer_label_dir(img_dir)
    split_name = img_dir.name
    cache_root = Path(cfg.cache_dir) if cfg.cache_dir else Path(run_cache_dir or (img_dir.parent.parent / ".slice_cache"))
    cache_root.mkdir(parents=True, exist_ok=True)

    split_token = f"{img_dir.resolve()}|{split_name}|{mode}"
    split_hash = hashlib.md5(split_token.encode("utf-8")).hexdigest()[:10]
    target_root = cache_root / f"{img_dir.parent.parent.name}_{split_name}_{split_hash}"
    out_img_dir = target_root / "images" / split_name
    out_lb_dir = target_root / "labels" / split_name
    meta_path = target_root / "meta.json"

    src_images = sorted([p for p in img_dir.glob("*") if p.suffix[1:].lower() in IMG_FORMATS])
    fingerprint = _build_fingerprint(src_images, label_dir, cfg, mode)
    if (
        cfg.cache_slices
        and meta_path.exists()
        and out_img_dir.exists()
        and any(out_img_dir.glob("*"))
        and _meta_matches(meta_path, fingerprint)
    ):
        LOGGER.info(f"{log_prefix}Using cached slices: {out_img_dir}")
        return str(out_img_dir.resolve())

    if target_root.exists():
        shutil.rmtree(target_root)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lb_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(cfg.seed)
    total_tiles, total_labels = 0, 0
    pbar = TQDM(src_images, desc=f"{log_prefix}Sliding-window slicing", total=len(src_images))
    for im_file in pbar:
        image = cv2.imread(str(im_file))
        if image is None:
            continue
        h, w = image.shape[:2]
        classes, boxes = _read_yolo_labels(_label_file_for_image(im_file, label_dir), h=h, w=w)
        windows = _generate_windows(w=w, h=h, size=cfg.slice_size, overlap=cfg.slice_overlap)
        for idx, (x1, y1, x2, y2) in enumerate(windows):
            tile = image[y1:y2, x1:x2]
            if tile.size == 0:
                continue

            tile_boxes, tile_cls = _clip_boxes_to_window(
                classes=classes,
                boxes=boxes,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                min_keep_ratio=cfg.min_box_keep_ratio,
            )

            if len(tile_boxes) == 0:
                keep_empty = mode != "train" or (rng.random() < cfg.empty_tile_keep_prob)
                if not keep_empty:
                    continue

            stem = f"{im_file.stem}__{x1}_{y1}_{idx}"
            tile_im_file = out_img_dir / f"{stem}.jpg"
            tile_lb_file = out_lb_dir / f"{stem}.txt"
            cv2.imwrite(str(tile_im_file), tile)
            _write_yolo_labels(tile_lb_file, tile_cls, tile_boxes)
            total_tiles += 1
            total_labels += len(tile_boxes)
        pbar.set_description(f"{log_prefix}Sliding-window slicing ({total_tiles} tiles)")
    pbar.close()

    if total_tiles == 0:
        raise RuntimeError(f"No sliced tiles were generated from {img_dir}")

    meta = {
        "fingerprint": fingerprint,
        "mode": mode,
        "slice_config": asdict(cfg),
        "source_image_dir": str(img_dir.resolve()),
        "source_label_dir": str(label_dir.resolve()),
        "tile_count": total_tiles,
        "label_count": total_labels,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    LOGGER.info(f"{log_prefix}Created sliced cache: {out_img_dir} ({total_tiles} tiles)")
    return str(out_img_dir.resolve())


def _infer_label_dir(img_dir: Path) -> Path:
    s = str(img_dir.resolve())
    token = f"{os.sep}images{os.sep}"
    if token in s:
        return Path(s.replace(token, f"{os.sep}labels{os.sep}", 1))
    return img_dir.parent.parent / "labels" / img_dir.name


def _label_file_for_image(im_file: Path, label_dir: Path) -> Path:
    return label_dir / f"{im_file.stem}.txt"


def _build_fingerprint(src_images: list[Path], label_dir: Path, cfg: SliceConfig, mode: str) -> str:
    h = hashlib.md5()
    h.update(json.dumps(asdict(cfg), sort_keys=True).encode("utf-8"))
    h.update(mode.encode("utf-8"))
    for im_file in src_images:
        st = im_file.stat()
        h.update(str(im_file).encode("utf-8"))
        h.update(f"{st.st_mtime_ns}:{st.st_size}".encode("utf-8"))
        lb_file = _label_file_for_image(im_file, label_dir)
        if lb_file.exists():
            st_lb = lb_file.stat()
            h.update(str(lb_file).encode("utf-8"))
            h.update(f"{st_lb.st_mtime_ns}:{st_lb.st_size}".encode("utf-8"))
    return h.hexdigest()


def _meta_matches(meta_path: Path, fingerprint: str) -> bool:
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("fingerprint") == fingerprint
    except Exception:
        return False


def _axis_starts(length: int, size: int, overlap: float) -> list[int]:
    if length <= size:
        return [0]
    step = max(int(round(size * (1.0 - overlap))), 1)
    end = length - size
    starts = list(range(0, end + 1, step))
    if starts[-1] != end:
        starts.append(end)
    return sorted(set(starts))


def _generate_windows(w: int, h: int, size: int, overlap: float) -> list[tuple[int, int, int, int]]:
    xs = _axis_starts(w, size, overlap)
    ys = _axis_starts(h, size, overlap)
    windows = []
    for y in ys:
        for x in xs:
            windows.append((x, y, min(x + size, w), min(y + size, h)))
    return windows


def _read_yolo_labels(label_file: Path, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    if not label_file.exists():
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

    rows = []
    with open(label_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            rows.append([float(x) for x in parts[:5]])
    if not rows:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

    arr = np.asarray(rows, dtype=np.float32)
    cls = arr[:, 0].astype(np.int64)
    x, y, bw, bh = arr[:, 1] * w, arr[:, 2] * h, arr[:, 3] * w, arr[:, 4] * h
    x1 = x - bw / 2
    y1 = y - bh / 2
    x2 = x + bw / 2
    y2 = y + bh / 2
    boxes = np.stack((x1, y1, x2, y2), axis=1).astype(np.float32)
    return cls, boxes


def _clip_boxes_to_window(
    classes: np.ndarray,
    boxes: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    min_keep_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    tile_w, tile_h = max(x2 - x1, 1), max(y2 - y1, 1)
    keep_boxes = []
    keep_cls = []
    for cls_id, (bx1, by1, bx2, by2) in zip(classes, boxes):
        ix1 = max(float(x1), float(bx1))
        iy1 = max(float(y1), float(by1))
        ix2 = min(float(x2), float(bx2))
        iy2 = min(float(y2), float(by2))
        iw = max(ix2 - ix1, 0.0)
        ih = max(iy2 - iy1, 0.0)
        inter_area = iw * ih
        if inter_area <= 0:
            continue
        orig_area = max((bx2 - bx1) * (by2 - by1), 1e-6)
        keep_ratio = inter_area / orig_area
        if keep_ratio < min_keep_ratio:
            continue

        lx1 = ix1 - x1
        ly1 = iy1 - y1
        lx2 = ix2 - x1
        ly2 = iy2 - y1
        cx = ((lx1 + lx2) * 0.5) / tile_w
        cy = ((ly1 + ly2) * 0.5) / tile_h
        bw = (lx2 - lx1) / tile_w
        bh = (ly2 - ly1) / tile_h
        if bw <= 0 or bh <= 0:
            continue
        keep_boxes.append((cx, cy, bw, bh))
        keep_cls.append(int(cls_id))

    if not keep_boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.asarray(keep_boxes, dtype=np.float32), np.asarray(keep_cls, dtype=np.int64)


def _write_yolo_labels(label_file: Path, classes: np.ndarray, boxes_xywh: np.ndarray) -> None:
    with open(label_file, "w", encoding="utf-8") as f:
        for cls_id, (x, y, w, h) in zip(classes.tolist(), boxes_xywh.tolist()):
            f.write(f"{int(cls_id)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

