from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# =========================
# Paths
# =========================
semantic_path = Path(r"E:\Mass\Segmentation_Masks\a00295732.png")
instance_path = Path(r"E:\Mass\InstanceSegmentation_Masks\a00295732.png")

output_dir = Path(r"E:\Mass\visual_check")
output_dir.mkdir(parents=True, exist_ok=True)

semantic_out = output_dir / "a00068686_semantic_color.png"
instance_out = output_dir / "a00068686_instance_color.png"


# =========================
# Helper functions
# =========================
def load_mask_as_id_array(mask_path: Path) -> np.ndarray:
    """
    Load a mask image as an integer ID array.

    For MassMIND semantic masks:
        pixel value = class ID

    For MassMIND instance masks:
        pixel value = instance ID
    """
    mask_img = Image.open(mask_path)
    mask = np.array(mask_img)

    print(f"\nFile: {mask_path}")
    print(f"PIL mode: {mask_img.mode}")
    print(f"Array shape: {mask.shape}")
    print(f"Array dtype: {mask.dtype}")

    # If the mask is saved as RGB but each channel is identical, use one channel.
    if mask.ndim == 3:
        if np.all(mask[..., 0] == mask[..., 1]) and np.all(mask[..., 1] == mask[..., 2]):
            mask = mask[..., 0]
            print("Converted RGB-like mask to single-channel ID mask.")
        else:
            raise ValueError(
                "This mask looks like a real RGB image, not an ID mask. "
                "Please inspect it manually."
            )

    unique_values = np.unique(mask)
    print(f"Unique values: {unique_values}")
    return mask


def colorize_semantic_mask(sem: np.ndarray) -> np.ndarray:
    """
    Colorize MassMIND semantic mask.

    MassMIND class IDs:
        0: Sky
        1: Water
        2: Bridge
        3: Obstacle
        4: Living Obstacle
        5: Background
        6: Self
    """
    colors = {
        0: (110, 190, 235),  # Sky
        1: (40, 110, 190),   # Water
        2: (245, 205, 60),   # Bridge
        3: (220, 80, 45),    # Obstacle
        4: (120, 190, 80),   # Living Obstacle
        5: (130, 70, 165),   # Background
        6: (235, 45, 45),    # Self
    }

    h, w = sem.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in colors.items():
        color_img[sem == class_id] = color

    return color_img


def colorize_instance_mask(inst: np.ndarray) -> np.ndarray:
    """
    Colorize instance mask.

    Each instance ID gets a random but fixed color.
    ID 0 is kept black.
    """
    h, w = inst.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)

    rng = np.random.default_rng(42)

    for inst_id in np.unique(inst):
        if inst_id == 0:
            continue

        color = rng.integers(40, 255, size=3, dtype=np.uint8)
        color_img[inst == inst_id] = color

    return color_img


# =========================
# Load masks
# =========================
semantic_mask = load_mask_as_id_array(semantic_path)
instance_mask = load_mask_as_id_array(instance_path)


# =========================
# Check bridge class
# =========================
BRIDGE_ID = 2

bridge_pixels = int((semantic_mask == BRIDGE_ID).sum())
print(f"\nBridge pixel count in semantic mask: {bridge_pixels}")

if bridge_pixels > 0:
    print("This image contains Bridge class pixels.")
else:
    print("This image does NOT contain Bridge class pixels.")


# =========================
# Colorize and save
# =========================
semantic_color = colorize_semantic_mask(semantic_mask)
instance_color = colorize_instance_mask(instance_mask)

Image.fromarray(semantic_color).save(semantic_out)
Image.fromarray(instance_color).save(instance_out)

print(f"\nSaved semantic color mask to: {semantic_out}")
print(f"Saved instance color mask to: {instance_out}")


# =========================
# Show results
# =========================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(semantic_color)
plt.title("Semantic Segmentation Color Map")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(instance_color)
plt.title("Instance Segmentation Color Map")
plt.axis("off")

plt.tight_layout()
plt.show()