import argparse
import json
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(images_dir: Path):
    files = []
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return sorted(files)


def find_image_by_stem(images_dir: Path, stem: str):
    for ext in IMG_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    candidates = list(images_dir.rglob(f"{stem}.*"))
    for p in candidates:
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            return p
    return None


def read_img_shape(img_path: Path):
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    h, w = img.shape[:2]
    return h, w


def parse_yolo_objects(txt_path: Path):
    objs = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_id = int(float(parts[0]))
            xc = float(parts[1])
            yc = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])
            objs.append({
                "class_id": cls_id,
                "xc": xc,
                "yc": yc,
                "bw": bw,
                "bh": bh,
            })
    return objs


def gaussian_kernel(size=9, sigma=2.0):
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


def conv2d_same(img, kernel):
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    out = np.zeros_like(img, dtype=np.float32)

    H, W = img.shape
    for y in range(H):
        for x in range(W):
            patch = padded[y:y+kh, x:x+kw]
            out[y, x] = np.sum(patch * kernel)
    return out


def build_position_prior(
    oil_images_dir: Path,
    oil_labels_dir: Path,
    class_id: int = 1,
    heatmap_size: int = 32,
    smooth_kernel_size: int = 7,
    smooth_sigma: float = 1.5,
):
    image_files = list_images(oil_images_dir)

    centers = []
    widths = []
    heights = []
    areas = []
    aspect_ratios = []
    per_image_counts = []

    valid_images = 0
    total_boxes = 0

    for img_path in image_files:
        stem = img_path.stem
        txt_path = oil_labels_dir / f"{stem}.txt"
        if not txt_path.exists():
            continue

        h, w = read_img_shape(img_path)
        objs = parse_yolo_objects(txt_path)

        selected = [o for o in objs if o["class_id"] == class_id]
        if len(selected) == 0:
            continue

        valid_images += 1
        per_image_counts.append(len(selected))

        for o in selected:
            xc = o["xc"]
            yc = o["yc"]
            bw = o["bw"]
            bh = o["bh"]

            centers.append([xc, yc])
            widths.append(bw)
            heights.append(bh)
            areas.append(bw * bh)
            aspect_ratios.append(bw / max(bh, 1e-6))
            total_boxes += 1

    if total_boxes == 0:
        raise RuntimeError("No oil tank boxes found.")

    centers = np.array(centers, dtype=np.float32)

    # 2D histogram over normalized center positions
    heatmap = np.zeros((heatmap_size, heatmap_size), dtype=np.float32)
    for xc, yc in centers:
        x_idx = min(heatmap_size - 1, max(0, int(xc * heatmap_size)))
        y_idx = min(heatmap_size - 1, max(0, int(yc * heatmap_size)))
        heatmap[y_idx, x_idx] += 1.0

    # Smooth it
    kernel = gaussian_kernel(size=smooth_kernel_size, sigma=smooth_sigma)
    heatmap = conv2d_same(heatmap, kernel)

    # Avoid all-zero
    heatmap = heatmap + 1e-8
    prob = heatmap / heatmap.sum()

    stats = {
        "valid_images": int(valid_images),
        "total_boxes": int(total_boxes),
        "heatmap_size": int(heatmap_size),
        "center_mean": centers.mean(axis=0).tolist(),
        "center_std": centers.std(axis=0).tolist(),
        "width_mean": float(np.mean(widths)),
        "width_std": float(np.std(widths)),
        "height_mean": float(np.mean(heights)),
        "height_std": float(np.std(heights)),
        "area_mean": float(np.mean(areas)),
        "area_std": float(np.std(areas)),
        "aspect_ratio_mean": float(np.mean(aspect_ratios)),
        "aspect_ratio_std": float(np.std(aspect_ratios)),
        "per_image_count_mean": float(np.mean(per_image_counts)),
        "per_image_count_std": float(np.std(per_image_counts)),
        "heatmap": prob.tolist(),
    }

    return stats, prob


def save_heatmap_png(prob_map: np.ndarray, out_path: Path, upscale: int = 16):
    vis = prob_map.copy()
    vis = vis / vis.max()
    vis = (vis * 255).astype(np.uint8)
    vis = cv2.resize(
        vis,
        (vis.shape[1] * upscale, vis.shape[0] * upscale),
        interpolation=cv2.INTER_NEAREST
    )
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    cv2.imwrite(str(out_path), vis)


def sample_center_from_heatmap(prob_map: np.ndarray, rng: np.random.Generator):
    H, W = prob_map.shape
    flat = prob_map.reshape(-1)
    idx = rng.choice(len(flat), p=flat)
    y = idx // W
    x = idx % W

    # sample inside the chosen cell
    x_norm = (x + rng.random()) / W
    y_norm = (y + rng.random()) / H
    return x_norm, y_norm


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def expand_box(box, margin, img_w, img_h):
    x1, y1, x2, y2 = box
    return (
        max(0, x1 - margin),
        max(0, y1 - margin),
        min(img_w, x2 + margin),
        min(img_h, y2 + margin),
    )


def is_valid_position(new_box, existing_boxes, img_w, img_h, margin):
    x1, y1, x2, y2 = new_box
    if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
        return False

    for eb in existing_boxes:
        eb_box = (eb["xmin"], eb["ymin"], eb["xmax"], eb["ymax"])
        if compute_iou(new_box, eb_box) > 0.0:
            return False

    for eb in existing_boxes:
        eb_box = (eb["xmin"], eb["ymin"], eb["xmax"], eb["ymax"])
        eb_expand = expand_box(eb_box, margin, img_w, img_h)
        if compute_iou(new_box, eb_expand) > 0.0:
            return False

    return True


def sample_position_from_prior(
    prob_map: np.ndarray,
    patch_w: int,
    patch_h: int,
    img_w: int,
    img_h: int,
    existing_boxes,
    margin: int = 12,
    jitter_norm_std: float = 0.03,
    max_trials: int = 200,
    seed: int = 42,
):
    """
    返回左上角 (x, y)，如果失败返回 None
    """
    rng = np.random.default_rng(seed)

    for _ in range(max_trials):
        xc_norm, yc_norm = sample_center_from_heatmap(prob_map, rng)

        # 加一点扰动
        xc_norm = xc_norm + rng.normal(0, jitter_norm_std)
        yc_norm = yc_norm + rng.normal(0, jitter_norm_std)

        # clip 到 [0,1]
        xc_norm = float(np.clip(xc_norm, 0.0, 1.0))
        yc_norm = float(np.clip(yc_norm, 0.0, 1.0))

        xc = xc_norm * img_w
        yc = yc_norm * img_h

        x1 = int(round(xc - patch_w / 2))
        y1 = int(round(yc - patch_h / 2))
        x2 = x1 + patch_w
        y2 = y1 + patch_h

        new_box = (x1, y1, x2, y2)

        if is_valid_position(new_box, existing_boxes, img_w, img_h, margin):
            return x1, y1

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oil_images_dir",
        type=str,
        default=r"C:\Users\DOCTOR\Desktop\daliy\DrZG\oiltank\images\train",
    )
    parser.add_argument(
        "--oil_labels_dir",
        type=str,
        default=r"C:\Users\DOCTOR\Desktop\daliy\DrZG\oiltank\labels\train",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"C:\Users\DOCTOR\Desktop\daliy\DrZG\oiltank_position_prior",
    )
    parser.add_argument("--class_id", type=int, default=1)
    parser.add_argument("--heatmap_size", type=int, default=32)
    parser.add_argument("--smooth_kernel_size", type=int, default=7)
    parser.add_argument("--smooth_sigma", type=float, default=1.5)

    args = parser.parse_args()

    oil_images_dir = Path(args.oil_images_dir)
    oil_labels_dir = Path(args.oil_labels_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats, prob = build_position_prior(
        oil_images_dir=oil_images_dir,
        oil_labels_dir=oil_labels_dir,
        class_id=args.class_id,
        heatmap_size=args.heatmap_size,
        smooth_kernel_size=args.smooth_kernel_size,
        smooth_sigma=args.smooth_sigma,
    )

    with open(output_dir / "position_prior.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    save_heatmap_png(prob, output_dir / "position_heatmap.png")

    print("========== DONE ==========")
    print(f"valid_images: {stats['valid_images']}")
    print(f"total_boxes:  {stats['total_boxes']}")
    print(f"center_mean:  {stats['center_mean']}")
    print(f"center_std:   {stats['center_std']}")
    print(f"width_mean:   {stats['width_mean']:.6f}")
    print(f"height_mean:  {stats['height_mean']:.6f}")
    print(f"saved json:   {output_dir / 'position_prior.json'}")
    print(f"saved heatmap:{output_dir / 'position_heatmap.png'}")


if __name__ == "__main__":
    main()