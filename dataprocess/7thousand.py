import argparse
import copy
import json
import random
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# =========================
# 基础 IO
# =========================

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


def read_image_any(img_path: Path):
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    return img


def save_image(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise RuntimeError(f"Failed to save image: {path}")


def normalize_to_uint8(img):
    img = img.astype(np.float32)
    mn = float(img.min())
    mx = float(img.max())
    if mx - mn < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    out = (img - mn) / (mx - mn)
    out = (out * 255.0).clip(0, 255).astype(np.uint8)
    return out


# =========================
# VOC / YOLO 解析
# =========================

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


def parse_yolo_first_target_box(txt_path: Path, img_w: int, img_h: int, target_class_id=1):
    objs = parse_yolo_objects(txt_path)
    selected = [o for o in objs if o["class_id"] == target_class_id]
    if len(selected) == 0:
        raise ValueError(f"No class_id={target_class_id} found in {txt_path}")
    return yolo_to_xyxy(selected[0], img_w, img_h)


def add_object_to_voc(root, obj_name, xmin, ymin, xmax, ymax):
    obj = ET.SubElement(root, "object")
    ET.SubElement(obj, "name").text = obj_name
    ET.SubElement(obj, "pose").text = "Unspecified"
    ET.SubElement(obj, "truncated").text = "0"
    ET.SubElement(obj, "difficult").text = "0"

    bndbox = ET.SubElement(obj, "bndbox")
    ET.SubElement(bndbox, "xmin").text = str(int(xmin))
    ET.SubElement(bndbox, "ymin").text = str(int(ymin))
    ET.SubElement(bndbox, "xmax").text = str(int(xmax))
    ET.SubElement(bndbox, "ymax").text = str(int(ymax))


def update_filename_and_path(root, out_img_path: Path):
    filename_node = root.find("filename")
    if filename_node is not None:
        filename_node.text = out_img_path.name

    path_node = root.find("path")
    if path_node is not None:
        path_node.text = str(out_img_path.resolve())


# =========================
# 几何与候选评分
# =========================

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
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2
    dx = max(bx1 - ax2, ax1 - bx2, 0)
    dy = max(by1 - ay2, ay1 - by2, 0)
    return float(np.sqrt(dx * dx + dy * dy))


def center_distance(c1, c2):
    dx = float(c1[0] - c2[0])
    dy = float(c1[1] - c2[1])
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
    bg = cv2.cvtColor(normalize_to_uint8(gray_img), cv2.COLOR_GRAY2BGR)
    for b in boxes:
        x1, y1, x2, y2 = b["xmin"], b["ymin"], b["xmax"], b["ymax"]
        cv2.rectangle(bg, (x1, y1), (x2, y2), (0, 255, 0), 1)
        show_name = b["name"]
        try:
            show_name.encode("ascii")
        except Exception:
            show_name = "obj"
        cv2.putText(
            bg,
            show_name,
            (x1, max(12, y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(out_path), bg)


def build_candidate_list_for_background(
    gray,
    objs,
    prior,
    patch_w,
    patch_h,
    outer_scale=2.0,
    stride=4,
    margin=12,
    min_ship_distance=12.0,
):
    H, W = gray.shape[:2]

    raw_score_map = np.full((H, W), np.nan, dtype=np.float32)
    similarity_map = np.zeros((H, W), dtype=np.float32)
    proximity_map = np.zeros((H, W), dtype=np.float32)

    candidates = []
    existing_boxes = []
    for o in objs:
        existing_boxes.append((o["xmin"], o["ymin"], o["xmax"], o["ymax"], o["name"]))

    for cy in range(0, H, stride):
        for cx in range(0, W, stride):
            x1 = int(round(cx - patch_w / 2))
            y1 = int(round(cy - patch_h / 2))
            x2 = x1 + patch_w
            y2 = y1 + patch_h
            box = (x1, y1, x2, y2)

            if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
                continue

            illegal = False
            min_dist = 1e9

            for eb in existing_boxes:
                ebox = eb[:4]

                if box_intersects(box, ebox):
                    illegal = True
                    break

                dist = min_edge_distance(box, ebox)
                min_dist = min(min_dist, dist)

                ex1 = max(0, ebox[0] - margin)
                ey1 = max(0, ebox[1] - margin)
                ex2 = min(W, ebox[2] + margin)
                ey2 = min(H, ebox[3] + margin)
                expanded_box = (ex1, ey1, ex2, ey2)
                if box_intersects(box, expanded_box):
                    illegal = True
                    break

            if illegal:
                continue

            feat = extract_context_ring_features(
                gray_img=gray,
                box=box,
                outer_scale=outer_scale,
            )
            if feat is None:
                continue

            mean_z, bg_sim, z_detail = feature_distance_to_prior(feat, prior)

            proximity_score = float(np.clip(min_dist / max(min_ship_distance, 1e-6), 0.0, 1.0))

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
            similarity_map[cy, cx] = bg_sim
            proximity_map[cy, cx] = proximity_score

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

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    return candidates, raw_score_map, similarity_map, proximity_map


def deduplicate_candidates(candidates, min_center_dist=20.0):
    selected = []
    for cand in candidates:
        c = cand["center_xy"]
        keep = True
        for s in selected:
            if center_distance(c, s["center_xy"]) < min_center_dist:
                keep = False
                break
        if keep:
            selected.append(cand)
    return selected


def choose_candidate(unique_candidates, rng):
    if len(unique_candidates) == 0:
        return None
    if len(unique_candidates) < 3:
        return unique_candidates[0]
    top3 = unique_candidates[:3]
    return rng.choice(top3)


# =========================
# patch 提取与粘贴
# =========================

def extract_oiltank_patch(oil_img_path: Path, oil_txt_path: Path, oil_class_id=1):
    oil_img = read_image_any(oil_img_path)
    if oil_img.ndim == 3:
        oil_gray = cv2.cvtColor(oil_img, cv2.COLOR_BGR2GRAY)
    else:
        oil_gray = oil_img

    H, W = oil_gray.shape[:2]
    x1, y1, x2, y2 = parse_yolo_first_target_box(
        oil_txt_path,
        img_w=W,
        img_h=H,
        target_class_id=oil_class_id,
    )

    patch = oil_gray[y1:y2, x1:x2].copy()
    if patch.size == 0:
        raise RuntimeError(f"Extracted patch is empty: {oil_img_path}")

    return patch, (x1, y1, x2, y2)


def feather_blend_patch(bg_gray, patch_gray, paste_x, paste_y, feather=3, match_brightness=False):
    ph, pw = patch_gray.shape[:2]
    out = bg_gray.copy()

    roi = out[paste_y:paste_y + ph, paste_x:paste_x + pw].astype(np.float32)
    patch_f = patch_gray.astype(np.float32)

    if match_brightness:
        roi_mean = float(np.mean(roi))
        patch_mean = float(np.mean(patch_f))
        patch_f = patch_f + (roi_mean - patch_mean)
        patch_f = np.clip(patch_f, 0, 255)

    if feather <= 0:
        out[paste_y:paste_y + ph, paste_x:paste_x + pw] = patch_f.astype(np.uint8)
        return out

    mask = np.zeros((ph, pw), dtype=np.float32)
    mask[:] = 1.0
    k = max(1, int(feather) * 2 + 1)
    mask = cv2.GaussianBlur(mask, (k, k), feather)

    blended = patch_f * mask + roi * (1.0 - mask)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    out[paste_y:paste_y + ph, paste_x:paste_x + pw] = blended
    return out


def draw_debug_boxes(gray_img, bg_objs, new_box, out_path: Path):
    vis = cv2.cvtColor(normalize_to_uint8(gray_img), cv2.COLOR_GRAY2BGR)

    for b in bg_objs:
        cv2.rectangle(vis, (b["xmin"], b["ymin"]), (b["xmax"], b["ymax"]), (0, 255, 0), 1)

    if new_box is not None:
        x1, y1, x2, y2 = new_box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)

    save_image(out_path, vis)


# =========================
# 数据集筛选与批量生成
# =========================

def get_xml_for_image(img_path: Path, annotations_dir: Path):
    xml_path = annotations_dir / f"{img_path.stem}.xml"
    if xml_path.exists():
        return xml_path
    return None


def select_ship_backgrounds(
    ship_images_dir: Path,
    ship_annotations_dir: Path,
    require_single_object=True,
    limit=None,
):
    ship_images = list_images(ship_images_dir)
    selected = []

    for img_path in ship_images:
        xml_path = get_xml_for_image(img_path, ship_annotations_dir)
        if xml_path is None:
            continue

        try:
            _, _, _, _, objs = parse_voc(xml_path)
        except Exception:
            continue

        if require_single_object and len(objs) != 1:
            continue

        selected.append((img_path, xml_path, len(objs)))

    selected = sorted(selected, key=lambda x: x[0].stem)

    if limit is not None:
        selected = selected[:limit]

    return selected


def select_oiltank_sources(
    oil_images_dir: Path,
    oil_labels_dir: Path,
    oil_class_id=1,
    limit=None,
):
    oil_images = list_images(oil_images_dir)
    selected = []

    for img_path in oil_images:
        txt_path = oil_labels_dir / f"{img_path.stem}.txt"
        if not txt_path.exists():
            continue
        try:
            gray = read_gray_image(img_path)
            H, W = gray.shape[:2]
            _ = parse_yolo_first_target_box(txt_path, W, H, target_class_id=oil_class_id)
            selected.append((img_path, txt_path))
        except Exception:
            continue

    selected = sorted(selected, key=lambda x: x[0].stem)

    if limit is not None:
        selected = selected[:limit]

    return selected


def save_candidate_visuals(
    gray,
    bg_objs,
    raw_score_map,
    similarity_map,
    proximity_map,
    unique_candidates,
    chosen_candidate,
    out_dir: Path,
    basename: str,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    score_map_for_vis = raw_score_map.copy()
    invalid_mask = ~np.isfinite(score_map_for_vis)
    if np.all(invalid_mask):
        score_map_for_vis[:] = 0
    else:
        valid_vals = score_map_for_vis[np.isfinite(score_map_for_vis)]
        fill_val = np.min(valid_vals)
        score_map_for_vis[invalid_mask] = fill_val

    save_heatmap(score_map_for_vis, out_dir / f"{basename}_placement_score_map.png")
    save_overlay(gray, score_map_for_vis, out_dir / f"{basename}_placement_score_overlay.png", alpha=0.45)
    save_heatmap(similarity_map, out_dir / f"{basename}_background_similarity_map.png")
    save_heatmap(proximity_map, out_dir / f"{basename}_ship_proximity_map.png")
    draw_boxes_on_image(gray, bg_objs, out_dir / f"{basename}_background_with_existing_boxes.jpg")

    vis = cv2.cvtColor(normalize_to_uint8(gray), cv2.COLOR_GRAY2BGR)
    for idx, item in enumerate(unique_candidates[:10]):
        cx, cy = item["center_xy"]
        x1, y1, x2, y2 = item["box_xyxy"]
        cv2.circle(vis, (cx, cy), 2, (0, 0, 255), -1)
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

    if chosen_candidate is not None:
        x1, y1, x2, y2 = chosen_candidate["box_xyxy"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(
            vis,
            "chosen",
            (x1, min(vis.shape[0] - 5, y2 + 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

    save_image(out_dir / f"{basename}_topk_candidates.jpg", vis)


def main():
    parser = argparse.ArgumentParser()

    # oil tank source
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

    # ship / MSAR background
    parser.add_argument(
        "--ship_images_dir",
        type=str,
        default=r"C:\Users\DOCTOR\Desktop\daliy\DrZG\MSAR-1.0 dataset\JPEGImages",
    )
    parser.add_argument(
        "--ship_annotations_dir",
        type=str,
        default=r"C:\Users\DOCTOR\Desktop\daliy\DrZG\MSAR-1.0 dataset\Annotations",
    )

    # prior
    parser.add_argument(
        "--prior_json",
        type=str,
        default=r"C:\Users\DOCTOR\Desktop\daliy\DrZG\oiltank_context_prior\context_prior.json",
    )

    # output
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"C:\Users\DOCTOR\Desktop\daliy\DrZG\oiltank_ship_aug1000",
    )

    parser.add_argument("--oil_class_id", type=int, default=1)
    parser.add_argument("--oil_class_name", type=str, default="油罐")

    # 这里直接设成 1000
    parser.add_argument("--num_pairs", type=int, default=1000)
    parser.add_argument("--require_single_object_bg", action="store_true", default=True)

    parser.add_argument("--outer_scale", type=float, default=2.0)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--margin", type=int, default=12)
    parser.add_argument("--min_ship_distance", type=float, default=12.0)

    parser.add_argument("--candidate_min_center_dist", type=float, default=20.0)
    parser.add_argument("--save_candidate_debug", action="store_true", default=True)

    parser.add_argument("--feather", type=int, default=0)
    parser.add_argument("--match_brightness", action="store_true", default=False)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    rng = random.Random(args.seed)

    oil_images_dir = Path(args.oil_images_dir)
    oil_labels_dir = Path(args.oil_labels_dir)
    ship_images_dir = Path(args.ship_images_dir)
    ship_annotations_dir = Path(args.ship_annotations_dir)
    prior_json = Path(args.prior_json)
    output_dir = Path(args.output_dir)

    out_img_dir = output_dir / "JPEGImages"
    out_xml_dir = output_dir / "Annotations"
    out_debug_dir = output_dir / "debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_xml_dir.mkdir(parents=True, exist_ok=True)
    if args.save_candidate_debug:
        out_debug_dir.mkdir(parents=True, exist_ok=True)

    prior = load_context_prior(prior_json)

    # oil 不截断，后面循环复用
    oil_sources = select_oiltank_sources(
        oil_images_dir=oil_images_dir,
        oil_labels_dir=oil_labels_dir,
        oil_class_id=args.oil_class_id,
        limit=None,
    )

    # ship 取前 num_pairs 张
    ship_backgrounds = select_ship_backgrounds(
        ship_images_dir=ship_images_dir,
        ship_annotations_dir=ship_annotations_dir,
        require_single_object=args.require_single_object_bg,
        limit=args.num_pairs,
    )

    if len(oil_sources) == 0:
        raise RuntimeError("No valid oil tank samples found.")
    if len(ship_backgrounds) < args.num_pairs:
        raise RuntimeError(f"Not enough ship backgrounds. need={args.num_pairs}, got={len(ship_backgrounds)}")

    summary = {
        "config": vars(args),
        "num_pairs_requested": args.num_pairs,
        "num_pairs_done": 0,
        "num_oil_sources": len(oil_sources),
        "results": [],
        "failed": [],
    }

    for idx in range(args.num_pairs):
        oil_idx = idx % len(oil_sources)
        oil_img_path, oil_txt_path = oil_sources[oil_idx]
        ship_img_path, ship_xml_path, ship_obj_count = ship_backgrounds[idx]

        pair_name = f"{oil_img_path.stem}_{ship_img_path.stem}"
        out_img_path = out_img_dir / f"{pair_name}.jpg"
        out_xml_path = out_xml_dir / f"{pair_name}.xml"

        print(f"\n========== [{idx+1}/{args.num_pairs}] {pair_name} ==========")

        try:
            bg_gray = read_gray_image(ship_img_path)
            bg_tree, bg_root, bg_w, bg_h, bg_objs = parse_voc(ship_xml_path)

            patch, src_box = extract_oiltank_patch(
                oil_img_path=oil_img_path,
                oil_txt_path=oil_txt_path,
                oil_class_id=args.oil_class_id,
            )
            ph, pw = patch.shape[:2]

            candidates, raw_score_map, similarity_map, proximity_map = build_candidate_list_for_background(
                gray=bg_gray,
                objs=bg_objs,
                prior=prior,
                patch_w=pw,
                patch_h=ph,
                outer_scale=args.outer_scale,
                stride=args.stride,
                margin=args.margin,
                min_ship_distance=args.min_ship_distance,
            )

            if len(candidates) == 0:
                raise RuntimeError("No legal candidates found.")

            unique_candidates = deduplicate_candidates(
                candidates,
                min_center_dist=args.candidate_min_center_dist,
            )

            if len(unique_candidates) == 0:
                raise RuntimeError("No unique candidates found after dedup.")

            chosen = choose_candidate(unique_candidates, rng)
            if chosen is None:
                raise RuntimeError("Failed to choose candidate.")

            x1, y1, x2, y2 = chosen["box_xyxy"]
            if x2 - x1 != pw or y2 - y1 != ph:
                raise RuntimeError("Chosen box size mismatch with patch size.")

            out_img = feather_blend_patch(
                bg_gray=bg_gray,
                patch_gray=patch,
                paste_x=x1,
                paste_y=y1,
                feather=args.feather,
                match_brightness=args.match_brightness,
            )

            out_root = copy.deepcopy(bg_root)
            add_object_to_voc(
                out_root,
                args.oil_class_name,
                x1, y1, x2, y2,
            )
            update_filename_and_path(out_root, out_img_path)

            save_image(out_img_path, out_img)
            ET.ElementTree(out_root).write(str(out_xml_path), encoding="utf-8", xml_declaration=False)

            if args.save_candidate_debug:
                save_candidate_visuals(
                    gray=bg_gray,
                    bg_objs=bg_objs,
                    raw_score_map=raw_score_map,
                    similarity_map=similarity_map,
                    proximity_map=proximity_map,
                    unique_candidates=unique_candidates,
                    chosen_candidate=chosen,
                    out_dir=out_debug_dir,
                    basename=pair_name,
                )

                draw_debug_boxes(
                    gray_img=out_img,
                    bg_objs=bg_objs,
                    new_box=(x1, y1, x2, y2),
                    out_path=out_debug_dir / f"{pair_name}_final_boxes.jpg",
                )

            record = {
                "pair_index": idx + 1,
                "oil_index": oil_idx,
                "pair_name": pair_name,
                "oil_img": str(oil_img_path),
                "oil_txt": str(oil_txt_path),
                "ship_img": str(ship_img_path),
                "ship_xml": str(ship_xml_path),
                "ship_obj_count": int(ship_obj_count),
                "patch_size": [int(pw), int(ph)],
                "source_box_xyxy": [int(v) for v in src_box],
                "chosen_center_xy": chosen["center_xy"],
                "chosen_box_xyxy": chosen["box_xyxy"],
                "chosen_score": float(chosen["score"]),
                "candidate_count": int(len(candidates)),
                "unique_candidate_count": int(len(unique_candidates)),
                "output_image": str(out_img_path),
                "output_xml": str(out_xml_path),
            }
            summary["results"].append(record)
            summary["num_pairs_done"] += 1

            print(f"[OK] saved image: {out_img_path}")
            print(f"[OK] saved xml:   {out_xml_path}")
            print(f"[INFO] oil_idx:     {oil_idx}")
            print(f"[INFO] patch size:  {pw}x{ph}")
            print(f"[INFO] chosen box:  {chosen['box_xyxy']}")
            print(f"[INFO] score:       {chosen['score']:.4f}")
            print(f"[INFO] candidates:  all={len(candidates)}, unique={len(unique_candidates)}")

        except Exception as e:
            fail = {
                "pair_index": idx + 1,
                "oil_index": oil_idx,
                "pair_name": pair_name,
                "oil_img": str(oil_img_path),
                "ship_img": str(ship_img_path),
                "error": repr(e),
            }
            summary["failed"].append(fail)
            print(f"[FAIL] {pair_name}: {repr(e)}")

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n========== ALL DONE ==========")
    print(f"requested:        {args.num_pairs}")
    print(f"done:             {summary['num_pairs_done']}")
    print(f"failed:           {len(summary['failed'])}")
    print(f"num_oil_sources:  {len(oil_sources)}")
    print(f"output:           {output_dir}")
    print(f"summary:          {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()