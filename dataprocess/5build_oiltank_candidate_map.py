import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np


def read_gray_image(img_path: Path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    return img


def parse_voc(xml_path: Path):
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    size_node = root.find("size")
    w = int(size_node.find("width").text)
    h = int(size_node.find("height").text)

    objs = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        bnd = obj.find("bndbox")
        xmin = int(float(bnd.find("xmin").text))
        ymin = int(float(bnd.find("ymin").text))
        xmax = int(float(bnd.find("xmax").text))
        ymax = int(float(bnd.find("ymax").text))
        objs.append({
            "name": name,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        })

    return tree, root, w, h, objs


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


def box_intersects(box1, box2):
    return compute_iou(box1, box2) > 0.0


def min_edge_distance(box1, box2):
    """
    两个 axis-aligned box 的边缘距离；若相交则为 0
    """
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2

    dx = max(bx1 - ax2, ax1 - bx2, 0)
    dy = max(by1 - ay2, ay1 - by2, 0)
    return float(np.sqrt(dx * dx + dy * dy))


def compute_grad_mag(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return mag


def extract_context_ring_features(gray_img, box, outer_scale=2.0, bright_percentile=90.0):
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

    outer_f = outer.astype(np.float32)
    blur = cv2.GaussianBlur(outer_f, (5, 5), 0)
    sq_blur = cv2.GaussianBlur(outer_f ** 2, (5, 5), 0)
    local_var_map = np.maximum(sq_blur - blur ** 2, 0.0)
    ring_local_var = local_var_map[mask > 0].astype(np.float32)

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

    return features


def load_context_prior(prior_json: Path):
    with open(prior_json, "r", encoding="utf-8") as f:
        prior = json.load(f)
    return prior


def feature_distance_to_prior(feat, prior, used_feature_names=None):
    summary = prior["feature_summary"]

    if used_feature_names is None:
        used_feature_names = [
            "mean",
            "std",
            "p10",
            "p50",
            "p90",
            "grad_mean",
            "bright_ratio",
            "local_var_mean",
            "inner_outer_mean_diff",
        ]

    z_list = []
    detail = {}

    for k in used_feature_names:
        if k not in feat or k not in summary:
            continue
        mu = summary[k]["mean"]
        sigma = max(summary[k]["std"], 1e-6)
        z = abs(feat[k] - mu) / sigma
        z_list.append(z)
        detail[k] = float(z)

    if len(z_list) == 0:
        return 1e6, 0.0, detail

    z_arr = np.array(z_list, dtype=np.float32)
    mean_z = float(np.mean(z_arr))
    # 转为 0~1，相似度越高越接近 1
    sim = float(np.exp(-0.5 * mean_z))
    return mean_z, sim, detail


def normalize_map(arr):
    arr = arr.astype(np.float32)
    finite_mask = np.isfinite(arr)
    if finite_mask.sum() == 0:
        return np.zeros_like(arr, dtype=np.float32)

    vals = arr[finite_mask]
    mn = np.min(vals)
    mx = np.max(vals)
    if mx - mn < 1e-6:
        out = np.zeros_like(arr, dtype=np.float32)
        out[finite_mask] = 1.0
        return out

    out = np.zeros_like(arr, dtype=np.float32)
    out[finite_mask] = (arr[finite_mask] - mn) / (mx - mn)
    return out


def save_heatmap(score_map, out_path: Path):
    norm = normalize_map(score_map)
    vis = (norm * 255).clip(0, 255).astype(np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def save_overlay(gray_img, score_map, out_path: Path, alpha=0.45):
    gray_u8 = gray_img.copy()
    if gray_u8.dtype != np.uint8:
        gray_u8 = np.clip(gray_u8, 0, 255).astype(np.uint8)
    bg = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)

    norm = normalize_map(score_map)
    heat = (norm * 255).clip(0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(bg, 1.0 - alpha, heat, alpha, 0)
    cv2.imwrite(str(out_path), overlay)


def draw_boxes_on_image(gray_img, boxes, out_path: Path):
    bg = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    for b in boxes:
        x1, y1, x2, y2 = b["xmin"], b["ymin"], b["xmax"], b["ymax"]
        cv2.rectangle(bg, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(
            bg,
            b["name"],
            (x1, max(12, y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(out_path), bg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--msar_img",
        type=str,
        default=r"C:\Users\DOCTOR\Desktop\daliy\DrZG\MSAR-1.0 dataset\JPEGImages\01.jpg",
    )
    parser.add_argument(
        "--msar_xml",
        type=str,
        default=r"C:\Users\DOCTOR\Desktop\daliy\DrZG\MSAR-1.0 dataset\Annotations\01.xml",
    )
    parser.add_argument(
        "--prior_json",
        type=str,
        default=r"C:\Users\DOCTOR\Desktop\daliy\DrZG\oiltank_context_prior\context_prior.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"C:\Users\DOCTOR\Desktop\daliy\DrZG\candidate_region_demo",
    )
    parser.add_argument(
        "--patch_w",
        type=int,
        default=29,
        help="如果不给 source patch，先用 prior 平均宽度附近的像素值",
    )
    parser.add_argument(
        "--patch_h",
        type=int,
        default=26,
        help="如果不给 source patch，先用 prior 平均高度附近的像素值",
    )
    parser.add_argument("--outer_scale", type=float, default=2.0)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--margin", type=int, default=12)
    parser.add_argument("--min_ship_distance", type=float, default=12.0)
    parser.add_argument("--topk", type=int, default=50)

    args = parser.parse_args()

    msar_img = Path(args.msar_img)
    msar_xml = Path(args.msar_xml)
    prior_json = Path(args.prior_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gray = read_gray_image(msar_img)
    _, _, img_w, img_h, objs = parse_voc(msar_xml)
    prior = load_context_prior(prior_json)

    # 若 prior 中有 box 尺寸统计，则直接覆盖默认 patch_w/h
    feat_summary = prior["feature_summary"]
    if "box_w" in feat_summary:
        args.patch_w = int(round(feat_summary["box_w"]["p50"]))
    if "box_h" in feat_summary:
        args.patch_h = int(round(feat_summary["box_h"]["p50"]))

    patch_w = args.patch_w
    patch_h = args.patch_h

    print("[INFO] image size:", img_w, img_h)
    print("[INFO] num objects:", len(objs))
    print("[INFO] patch size:", patch_w, patch_h)

    H, W = gray.shape[:2]

    raw_score_map = np.full((H, W), np.nan, dtype=np.float32)
    legality_map = np.zeros((H, W), dtype=np.float32)
    similarity_map = np.zeros((H, W), dtype=np.float32)
    proximity_penalty_map = np.zeros((H, W), dtype=np.float32)

    candidates = []

    existing_boxes = []
    for o in objs:
        existing_boxes.append((o["xmin"], o["ymin"], o["xmax"], o["ymax"], o["name"]))

    for cy in range(0, H, args.stride):
        for cx in range(0, W, args.stride):
            x1 = int(round(cx - patch_w / 2))
            y1 = int(round(cy - patch_h / 2))
            x2 = x1 + patch_w
            y2 = y1 + patch_h
            box = (x1, y1, x2, y2)

            # 1) 出界直接非法
            if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
                continue

            # 2) 与已有目标 IoU 必须为 0
            illegal = False
            min_dist = 1e9

            for eb in existing_boxes:
                ebox = eb[:4]
                if box_intersects(box, ebox):
                    illegal = True
                    break

                dist = min_edge_distance(box, ebox)
                min_dist = min(min_dist, dist)

                # margin 扩张区不能碰
                ex1 = max(0, ebox[0] - args.margin)
                ey1 = max(0, ebox[1] - args.margin)
                ex2 = min(W, ebox[2] + args.margin)
                ey2 = min(H, ebox[3] + args.margin)
                expanded_box = (ex1, ey1, ex2, ey2)
                if box_intersects(box, expanded_box):
                    illegal = True
                    break

            if illegal:
                continue

            # 3) ring 特征
            feat = extract_context_ring_features(
                gray_img=gray,
                box=box,
                outer_scale=args.outer_scale,
            )
            if feat is None:
                continue

            mean_z, bg_sim, z_detail = feature_distance_to_prior(feat, prior)

            # 4) ship proximity penalty
            # 离目标越近，惩罚越大；足够远则近似 1
            proximity_score = float(np.clip(min_dist / max(args.min_ship_distance, 1e-6), 0.0, 1.0))

            # 5) clutter penalty
            # 用 grad_mean 和 local_var_mean 偏离 prior 的程度做轻惩罚
            clutter_keys = ["grad_mean", "local_var_mean"]
            clutter_z = []
            for ck in clutter_keys:
                mu = prior["feature_summary"][ck]["mean"]
                sigma = max(prior["feature_summary"][ck]["std"], 1e-6)
                clutter_z.append(abs(feat[ck] - mu) / sigma)
            clutter_z = float(np.mean(clutter_z))
            clutter_score = float(np.exp(-0.35 * clutter_z))

            final_score = float(bg_sim * proximity_score * clutter_score)

            raw_score_map[cy, cx] = final_score
            legality_map[cy, cx] = 1.0
            similarity_map[cy, cx] = bg_sim
            proximity_penalty_map[cy, cx] = proximity_score

            candidates.append({
                "center_xy": [int(cx), int(cy)],
                "box_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                "score": final_score,
                "background_similarity_score": float(bg_sim),
                "ship_proximity_score": float(proximity_score),
                "clutter_score": float(clutter_score),
                "mean_z": float(mean_z),
                "min_distance_to_existing": float(min_dist),
                "z_detail": z_detail,
            })

    if len(candidates) == 0:
        raise RuntimeError("No legal candidate positions found on this image.")

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    topk = candidates[:args.topk]

    # 热图保存
    score_map_for_vis = raw_score_map.copy()
    invalid_mask = ~np.isfinite(score_map_for_vis)
    if np.all(invalid_mask):
        score_map_for_vis[:] = 0
    else:
        valid_vals = score_map_for_vis[np.isfinite(score_map_for_vis)]
        fill_val = np.min(valid_vals)
        score_map_for_vis[invalid_mask] = fill_val

    save_heatmap(score_map_for_vis, output_dir / "placement_score_map.png")
    save_overlay(gray, score_map_for_vis, output_dir / "placement_score_overlay.png", alpha=0.45)
    save_heatmap(similarity_map, output_dir / "background_similarity_map.png")
    save_heatmap(proximity_penalty_map, output_dir / "ship_proximity_map.png")
    draw_boxes_on_image(gray, objs, output_dir / "background_with_existing_boxes.jpg")

    # 画 top-k 候选框中心
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for idx, item in enumerate(topk):
        cx, cy = item["center_xy"]
        x1, y1, x2, y2 = item["box_xyxy"]
        cv2.circle(vis, (cx, cy), 2, (0, 0, 255), -1)
        if idx < 10:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(
                vis,
                f"{idx+1}:{item['score']:.3f}",
                (x1, max(12, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

    cv2.imwrite(str(output_dir / "topk_candidates.jpg"), vis)

    result = {
        "msar_img": str(msar_img),
        "msar_xml": str(msar_xml),
        "prior_json": str(prior_json),
        "image_size": [int(W), int(H)],
        "patch_size": [int(patch_w), int(patch_h)],
        "outer_scale": float(args.outer_scale),
        "stride": int(args.stride),
        "margin": int(args.margin),
        "num_existing_objects": int(len(objs)),
        "num_legal_candidates": int(len(candidates)),
        "top_candidates": topk,
    }

    with open(output_dir / "candidate_scores.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("========== DONE ==========")
    print(f"saved: {output_dir / 'placement_score_map.png'}")
    print(f"saved: {output_dir / 'placement_score_overlay.png'}")
    print(f"saved: {output_dir / 'background_similarity_map.png'}")
    print(f"saved: {output_dir / 'ship_proximity_map.png'}")
    print(f"saved: {output_dir / 'topk_candidates.jpg'}")
    print(f"saved: {output_dir / 'candidate_scores.json'}")
    print(f"num_legal_candidates: {len(candidates)}")
    print("top-5 candidates:")
    for i, c in enumerate(topk[:5], 1):
        print(
            f"  #{i} center={c['center_xy']} score={c['score']:.4f} "
            f"bg_sim={c['background_similarity_score']:.4f} "
            f"ship_prox={c['ship_proximity_score']:.4f} "
            f"clutter={c['clutter_score']:.4f}"
        )


if __name__ == "__main__":
    main()