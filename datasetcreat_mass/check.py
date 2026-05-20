from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw


# =========================
# Config
# =========================
IMG_DIR = Path(r"E:\Mass\Images")
SEMANTIC_DIR = Path(r"E:\Mass\Segmentation_Masks")
INSTANCE_DIR = Path(r"E:\Mass\InstanceSegmentation_Masks")

OUT_ROOT = Path(r"E:\Mass\MassMIND_bridge_yolo")
OUT_IMG_DIR = OUT_ROOT / "images" / "train"
OUT_LABEL_DIR = OUT_ROOT / "labels" / "train"
OUT_VIS_DIR = OUT_ROOT / "visual_check"

OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_LABEL_DIR.mkdir(parents=True, exist_ok=True)
OUT_VIS_DIR.mkdir(parents=True, exist_ok=True)

# MassMIND semantic class id
MASSMIND_BRIDGE_ID = 2

# YOLO single-class dataset id
YOLO_CLASS_ID = 0

# Filter very small bridge regions
MIN_AREA = 100

# Recommended:
# True  = use instance mask to separate bridge instances when possible
# False = use connected components only on semantic bridge mask
USE_INSTANCE_MASK = True

# Save visualization images with red boxes
SAVE_VIS = True

# Supported image extensions
IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


# =========================
# Helper functions
# =========================
def load_mask(mask_path: Path) -> np.ndarray:
    """
    Load semantic or instance mask as an integer ID array.
    """
    mask_img = Image.open(mask_path)
    mask = np.array(mask_img)

    if mask.ndim == 3:
        if np.all(mask[..., 0] == mask[..., 1]) and np.all(mask[..., 1] == mask[..., 2]):
            mask = mask[..., 0]
        else:
            raise ValueError(f"Mask appears to be an RGB image, not an ID mask: {mask_path}")

    return mask


def find_image_by_stem(img_dir: Path, stem: str) -> Path | None:
    """
    Find original LWIR image by file stem.
    """
    for ext in IMAGE_EXTS:
        candidate = img_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def bbox_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    """
    Convert pixel bbox [xmin, ymin, xmax, ymax] to YOLO normalized xywh.
    """
    bbox_w = xmax - xmin + 1
    bbox_h = ymax - ymin + 1

    x_center = xmin + bbox_w / 2
    y_center = ymin + bbox_h / 2

    x_center_norm = x_center / img_w
    y_center_norm = y_center / img_h
    bbox_w_norm = bbox_w / img_w
    bbox_h_norm = bbox_h / img_h

    return x_center_norm, y_center_norm, bbox_w_norm, bbox_h_norm


def clip_yolo_values(xc, yc, w, h):
    """
    Clip YOLO values to valid range [0, 1].
    """
    xc = min(max(xc, 0.0), 1.0)
    yc = min(max(yc, 0.0), 1.0)
    w = min(max(w, 0.0), 1.0)
    h = min(max(h, 0.0), 1.0)
    return xc, yc, w, h


def get_bboxes_from_semantic_connected_components(bridge_binary: np.ndarray):
    """
    Generate bboxes by connected components on bridge binary mask.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bridge_binary.astype(np.uint8),
        connectivity=8
    )

    bboxes = []

    for comp_id in range(1, num_labels):  # 0 is background
        x, y, w, h, area = stats[comp_id]

        if area < MIN_AREA:
            continue

        xmin = int(x)
        ymin = int(y)
        xmax = int(x + w - 1)
        ymax = int(y + h - 1)

        bboxes.append((xmin, ymin, xmax, ymax, int(area)))

    return bboxes


def get_bboxes_from_instance_mask(semantic_mask: np.ndarray, instance_mask: np.ndarray):
    """
    Generate bboxes by grouping bridge pixels with instance ids.
    Only pixels where semantic_mask == MASSMIND_BRIDGE_ID are used.
    """
    bridge_binary = semantic_mask == MASSMIND_BRIDGE_ID
    bridge_instance_values = np.unique(instance_mask[bridge_binary])

    bboxes = []

    for inst_id in bridge_instance_values:
        inst_bridge_mask = bridge_binary & (instance_mask == inst_id)
        area = int(inst_bridge_mask.sum())

        if area < MIN_AREA:
            continue

        ys, xs = np.where(inst_bridge_mask)

        if len(xs) == 0 or len(ys) == 0:
            continue

        xmin = int(xs.min())
        xmax = int(xs.max())
        ymin = int(ys.min())
        ymax = int(ys.max())

        bboxes.append((xmin, ymin, xmax, ymax, area))

    return bboxes


def draw_bboxes_on_image(img_path: Path, bboxes, out_path: Path):
    """
    Draw red bboxes for visual checking.
    """
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for i, (xmin, ymin, xmax, ymax, area) in enumerate(bboxes, start=1):
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
        draw.text((xmin, max(0, ymin - 15)), f"bridge_{i}", fill="red")

    img.save(out_path)


# =========================
# Main conversion
# =========================
semantic_paths = sorted(SEMANTIC_DIR.glob("*.png"))

print(f"Found semantic masks: {len(semantic_paths)}")
print(f"Output root: {OUT_ROOT}")

total_masks = 0
total_bridge_images = 0
total_bboxes = 0
missing_images = []
failed_files = []

for semantic_path in semantic_paths:
    total_masks += 1
    stem = semantic_path.stem

    try:
        semantic_mask = load_mask(semantic_path)
        H, W = semantic_mask.shape[:2]

        bridge_binary = semantic_mask == MASSMIND_BRIDGE_ID
        bridge_pixel_count = int(bridge_binary.sum())

        # Skip images without bridge
        if bridge_pixel_count == 0:
            continue

        img_path = find_image_by_stem(IMG_DIR, stem)
        if img_path is None:
            missing_images.append(stem)
            print(f"[Missing image] {stem}")
            continue

        # Get bridge bboxes
        bboxes = []

        if USE_INSTANCE_MASK:
            instance_path = INSTANCE_DIR / semantic_path.name

            if instance_path.exists():
                instance_mask = load_mask(instance_path)

                if instance_mask.shape[:2] != semantic_mask.shape[:2]:
                    print(f"[Shape mismatch] {stem}, fallback to connected components.")
                    bridge_binary_uint8 = bridge_binary.astype(np.uint8)
                    bboxes = get_bboxes_from_semantic_connected_components(bridge_binary_uint8)
                else:
                    bboxes = get_bboxes_from_instance_mask(semantic_mask, instance_mask)
            else:
                print(f"[Missing instance mask] {stem}, fallback to connected components.")
                bridge_binary_uint8 = bridge_binary.astype(np.uint8)
                bboxes = get_bboxes_from_semantic_connected_components(bridge_binary_uint8)
        else:
            bridge_binary_uint8 = bridge_binary.astype(np.uint8)
            bboxes = get_bboxes_from_semantic_connected_components(bridge_binary_uint8)

        # If all bridge regions are filtered out by MIN_AREA, skip this image
        if len(bboxes) == 0:
            continue

        # Write YOLO label
        yolo_lines = []

        for xmin, ymin, xmax, ymax, area in bboxes:
            xcn, ycn, wn, hn = bbox_to_yolo(xmin, ymin, xmax, ymax, W, H)
            xcn, ycn, wn, hn = clip_yolo_values(xcn, ycn, wn, hn)

            yolo_line = f"{YOLO_CLASS_ID} {xcn:.6f} {ycn:.6f} {wn:.6f} {hn:.6f}"
            yolo_lines.append(yolo_line)

        label_out_path = OUT_LABEL_DIR / f"{stem}.txt"
        label_out_path.write_text("\n".join(yolo_lines), encoding="utf-8")

        # Copy image
        img_out_path = OUT_IMG_DIR / img_path.name
        shutil.copy2(img_path, img_out_path)

        # Save visual check image
        if SAVE_VIS:
            vis_out_path = OUT_VIS_DIR / f"{stem}_bbox_check.png"
            draw_bboxes_on_image(img_path, bboxes, vis_out_path)

        total_bridge_images += 1
        total_bboxes += len(bboxes)

        print(
            f"[OK] {stem}: bridge_pixels={bridge_pixel_count}, "
            f"bboxes={len(bboxes)}, label={label_out_path.name}"
        )

    except Exception as e:
        failed_files.append((stem, str(e)))
        print(f"[Failed] {stem}: {e}")


# =========================
# Write data.yaml
# =========================
data_yaml = OUT_ROOT / "data.yaml"
data_yaml.write_text(
    f"""path: {OUT_ROOT.as_posix()}
train: images/train
val: images/train

names:
  0: bridge
""",
    encoding="utf-8"
)


# =========================
# Summary
# =========================
print("\n=========================")
print("Conversion finished")
print("=========================")
print(f"Total semantic masks scanned: {total_masks}")
print(f"Bridge images kept: {total_bridge_images}")
print(f"Total YOLO bboxes: {total_bboxes}")
print(f"Missing original images: {len(missing_images)}")
print(f"Failed files: {len(failed_files)}")
print(f"Output dataset: {OUT_ROOT}")
print(f"data.yaml: {data_yaml}")

if missing_images:
    missing_txt = OUT_ROOT / "missing_images.txt"
    missing_txt.write_text("\n".join(missing_images), encoding="utf-8")
    print(f"Missing image list saved to: {missing_txt}")

if failed_files:
    failed_txt = OUT_ROOT / "failed_files.txt"
    failed_txt.write_text(
        "\n".join([f"{stem}: {err}" for stem, err in failed_files]),
        encoding="utf-8"
    )
    print(f"Failed file list saved to: {failed_txt}")