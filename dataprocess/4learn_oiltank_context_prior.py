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


def read_gray_image(img_path: Path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    return img


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


def yolo_to_xyxy(obj, img_w, img_h):
    xc = obj["xc"] * img_w
    yc = obj["yc"] * img_h
    bw = obj["bw"] * img_w
    bh = obj["bh"] * img_h

    x1 = int(round(xc - bw / 2))
    y1 = int(round(yc - bh / 2))
    x2 = int(round(xc + bw / 2))
    y2 = int(round(yc + bh / 2))

    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(1, min(img_w, x2))
    y2 = max(1, min(img_h, y2))
    return x1, y1, x2, y2


def expand_box_xyxy(box, scale, img_w, img_h):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    new_w = bw * scale
    new_h = bh * scale

    nx1 = int(np.floor(cx - new_w / 2))
    ny1 = int(np.floor(cy - new_h / 2))
    nx2 = int(np.ceil(cx + new_w / 2))
    ny2 = int(np.ceil(cy + new_h / 2))

    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    nx2 = min(img_w, nx2)
    ny2 = min(img_h, ny2)

    return nx1, ny1, nx2, ny2


def safe_crop(img, box):
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def compute_grad_mag(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return mag


def extract_context_ring_features(gray_img, box, outer_scale=2.0, bright_percentile=90.0):
    """
    ring = expanded_box - inner_box
    只对 ring 区域提取特征
    """
    H, W = gray_img.shape[:2]
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return None

    outer_box = expand_box_xyxy(box, scale=outer_scale, img_w=W, img_h=H)
    ox1, oy1, ox2, oy2 = outer_box

    outer = gray_img[oy1:oy2, ox1:ox2]
    if outer is None or outer.size == 0:
        return None

    mask = np.ones((oy2 - oy1, ox2 - ox1), dtype=np.uint8)
    ix1 = x1 - ox1
    iy1 = y1 - oy1
    ix2 = x2 - ox1
    iy2 = y2 - oy1
    mask[iy1:iy2, ix1:ix2] = 0

    ring_pixels = outer[mask > 0]
    if ring_pixels.size < 16:
        return None

    ring_pixels = ring_pixels.astype(np.float32)

    grad_mag = compute_grad_mag(outer)
    ring_grad = grad_mag[mask > 0].astype(np.float32)

    # 用高斯模糊后的局部方差表示粗糙度
    outer_f = outer.astype(np.float32)
    blur = cv2.GaussianBlur(outer_f, (5, 5), 0)
    sq_blur = cv2.GaussianBlur(outer_f ** 2, (5, 5), 0)
    local_var_map = np.maximum(sq_blur - blur ** 2, 0.0)
    ring_local_var = local_var_map[mask > 0].astype(np.float32)

    # 亮像素比例：用 ring 自身高分位阈值
    bright_thr = np.percentile(ring_pixels, bright_percentile)
    bright_ratio = float(np.mean(ring_pixels >= bright_thr))

    inner_patch = gray_img[y1:y2, x1:x2].astype(np.float32)
    if inner_patch.size == 0:
        return None

    features = {
        "mean": float(np.mean(ring_pixels)),
        "std": float(np.std(ring_pixels)),
        "p10": float(np.percentile(ring_pixels, 10)),
        "p50": float(np.percentile(ring_pixels, 50)),
        "p90": float(np.percentile(ring_pixels, 90)),
        "grad_mean": float(np.mean(ring_grad)),
        "grad_std": float(np.std(ring_grad)),
        "bright_ratio": bright_ratio,
        "local_var_mean": float(np.mean(ring_local_var)),
        "inner_mean": float(np.mean(inner_patch)),
        "inner_std": float(np.std(inner_patch)),
        "inner_outer_mean_diff": float(np.mean(inner_patch) - np.mean(ring_pixels)),
        "inner_outer_std_diff": float(np.std(inner_patch) - np.std(ring_pixels)),
        "box_w": float(x2 - x1),
        "box_h": float(y2 - y1),
        "box_area": float((x2 - x1) * (y2 - y1)),
        "box_aspect": float((x2 - x1) / max((y2 - y1), 1e-6)),
        "ring_area": float(ring_pixels.size),
        "outer_scale": float(outer_scale),
    }

    return features, outer_box


def summarize_feature_dicts(feature_dicts):
    keys = sorted(feature_dicts[0].keys())
    summary = {}
    for k in keys:
        vals = np.array([fd[k] for fd in feature_dicts], dtype=np.float32)
        summary[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals) + 1e-6),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "p10": float(np.percentile(vals, 10)),
            "p50": float(np.percentile(vals, 50)),
            "p90": float(np.percentile(vals, 90)),
        }
    return summary


def normalize_to_uint8(img):
    img = img.astype(np.float32)
    mn = img.min()
    mx = img.max()
    if mx - mn < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    out = (img - mn) / (mx - mn)
    out = (out * 255).clip(0, 255).astype(np.uint8)
    return out


def save_debug_visualization(gray_img, box, outer_box, save_path: Path):
    vis = cv2.cvtColor(normalize_to_uint8(gray_img), cv2.COLOR_GRAY2BGR)
    x1, y1, x2, y2 = box
    ox1, oy1, ox2, oy2 = outer_box

    cv2.rectangle(vis, (ox1, oy1), (ox2, oy2), (255, 0, 0), 1)   # outer ring
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)       # inner box

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), vis)


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
        default=r"C:\Users\DOCTOR\Desktop\daliy\DrZG\oiltank_context_prior",
    )
    parser.add_argument("--class_id", type=int, default=1)
    parser.add_argument("--outer_scale", type=float, default=2.0)
    parser.add_argument("--save_debug_vis", action="store_true")
    parser.add_argument("--max_debug_vis", type=int, default=12)

    args = parser.parse_args()

    oil_images_dir = Path(args.oil_images_dir)
    oil_labels_dir = Path(args.oil_labels_dir)
    output_dir = Path(args.output_dir)
    debug_dir = output_dir / "debug_vis"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = list_images(oil_images_dir)
    all_features = []
    used_samples = []
    valid_images = 0
    total_boxes = 0
    debug_saved = 0

    for img_path in image_files:
        txt_path = oil_labels_dir / f"{img_path.stem}.txt"
        if not txt_path.exists():
            continue

        gray = read_gray_image(img_path)
        H, W = gray.shape[:2]
        objs = parse_yolo_objects(txt_path)
        selected = [o for o in objs if o["class_id"] == args.class_id]
        if len(selected) == 0:
            continue

        valid_images += 1

        for obj in selected:
            total_boxes += 1
            box = yolo_to_xyxy(obj, W, H)
            result = extract_context_ring_features(
                gray_img=gray,
                box=box,
                outer_scale=args.outer_scale,
            )
            if result is None:
                continue

            feat, outer_box = result
            all_features.append(feat)
            used_samples.append({
                "image": str(img_path),
                "label": str(txt_path),
                "box_xyxy": [int(x) for x in box],
                "outer_box_xyxy": [int(x) for x in outer_box],
            })

            if args.save_debug_vis and debug_saved < args.max_debug_vis:
                save_debug_visualization(
                    gray_img=gray,
                    box=box,
                    outer_box=outer_box,
                    save_path=debug_dir / f"{img_path.stem}_debug.jpg",
                )
                debug_saved += 1

    if len(all_features) == 0:
        raise RuntimeError("No valid oil tank context-ring samples found.")

    feature_summary = summarize_feature_dicts(all_features)

    prior = {
        "version": "context_ring_prior_v1",
        "description": "Prior learned from oil tank context ring statistics in SAR patches.",
        "oil_images_dir": str(oil_images_dir),
        "oil_labels_dir": str(oil_labels_dir),
        "class_id": int(args.class_id),
        "outer_scale": float(args.outer_scale),
        "valid_images": int(valid_images),
        "total_boxes": int(total_boxes),
        "used_samples": int(len(all_features)),
        "feature_names": sorted(list(all_features[0].keys())),
        "feature_summary": feature_summary,
        "sample_examples": used_samples[:20],
    }

    out_json = output_dir / "context_prior.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(prior, f, ensure_ascii=False, indent=2)

    print("========== DONE ==========")
    print(f"valid_images:   {valid_images}")
    print(f"total_boxes:    {total_boxes}")
    print(f"used_samples:   {len(all_features)}")
    print(f"outer_scale:    {args.outer_scale}")
    print(f"saved_json:     {out_json}")
    if args.save_debug_vis:
        print(f"debug_vis_dir:  {debug_dir}")


if __name__ == "__main__":
    main()