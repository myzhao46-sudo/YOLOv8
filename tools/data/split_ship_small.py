from pathlib import Path
import argparse
import random
import shutil
from collections import defaultdict

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def infer_modality(path: Path) -> str:
    """
    Infer modality from filename.

    Expected examples:
      rgb_564.jpg       -> rgb
      sar_0031427.jpg   -> sar
      ir_1_4.jpg        -> ir

    Adjust this function if your filename rule is different.
    """
    stem = path.stem.lower()

    if stem.startswith("rgb"):
        return "rgb"

    if stem.startswith("sar"):
        return "sar"

    if stem.startswith("ir") or stem.startswith("infr") or stem.startswith("infra") or stem.startswith("thermal"):
        return "ir"

    return "unknown"


def find_images(images_dir: Path):
    return sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def convert_label_to_yolo5(src_label: Path, dst_label: Path, force_class: int = 0):
    """
    Convert label to standard YOLO detect 5-col format:

      cls cx cy w h

    Supports:
      5-col YOLO:
        cls cx cy w h

      9-col polygon / DOTA-like:
        cls x1 y1 x2 y2 x3 y3 x4 y4

    Output class id is forced to force_class.
    For ship in global4, force_class = 0.
    """
    dst_label.parent.mkdir(parents=True, exist_ok=True)

    if not src_label.exists():
        print(f"[WARN] Missing label: {src_label}")
        dst_label.write_text("", encoding="utf-8")
        return 0, 1

    out_lines = []
    kept = 0
    skipped = 0

    with open(src_label, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue

            try:
                if len(parts) == 5:
                    # cls cx cy w h
                    cx, cy, w, h = map(float, parts[1:5])

                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2

                elif len(parts) >= 9:
                    # cls x1 y1 x2 y2 x3 y3 x4 y4
                    coords = list(map(float, parts[1:9]))
                    xs = coords[0::2]
                    ys = coords[1::2]

                    x1 = min(xs)
                    x2 = max(xs)
                    y1 = min(ys)
                    y2 = max(ys)

                else:
                    print(f"[WARN] Bad label line: {src_label} line {line_no}: {line.strip()}")
                    skipped += 1
                    continue

                x1 = clip01(x1)
                y1 = clip01(y1)
                x2 = clip01(x2)
                y2 = clip01(y2)

                w = x2 - x1
                h = y2 - y1

                if w <= 0 or h <= 0:
                    skipped += 1
                    continue

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                out_lines.append(
                    f"{force_class} {cx:.8f} {cy:.8f} {w:.8f} {h:.8f}\n"
                )
                kept += 1

            except Exception as e:
                print(f"[WARN] Failed parsing {src_label} line {line_no}: {line.strip()} | {e}")
                skipped += 1

    dst_label.write_text("".join(out_lines), encoding="utf-8")
    return kept, skipped


def copy_items(items, split_name: str, src_labels_dir: Path, dst_root: Path, force_class: int):
    dst_img_dir = dst_root / "images" / split_name
    dst_lbl_dir = dst_root / "labels" / split_name

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    total_boxes = 0
    total_skipped = 0

    for img_path in items:
        shutil.copy2(img_path, dst_img_dir / img_path.name)

        src_label = src_labels_dir / f"{img_path.stem}.txt"
        dst_label = dst_lbl_dir / f"{img_path.stem}.txt"

        kept, skipped = convert_label_to_yolo5(
            src_label=src_label,
            dst_label=dst_label,
            force_class=force_class,
        )

        total_boxes += kept
        total_skipped += skipped

    return total_boxes, total_skipped


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src-root", required=True)
    parser.add_argument("--dst-root", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-class", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)

    src_images_dir = src_root / "images"
    src_labels_dir = src_root / "labels"

    if not src_images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {src_images_dir}")

    if not src_labels_dir.exists():
        raise FileNotFoundError(f"Labels dir not found: {src_labels_dir}")

    if dst_root.exists():
        if args.overwrite:
            print(f"[INFO] Removing existing dst-root: {dst_root}")
            shutil.rmtree(dst_root)
        else:
            raise FileExistsError(
                f"Output dir already exists: {dst_root}\n"
                f"Use --overwrite if you want to recreate it."
            )

    images = find_images(src_images_dir)
    if not images:
        raise RuntimeError(f"No images found in {src_images_dir}")

    groups = defaultdict(list)
    for img in images:
        modality = infer_modality(img)
        groups[modality].append(img)

    if "unknown" in groups:
        print("\n[ERROR] Some images have unknown modality:")
        for p in groups["unknown"][:30]:
            print(f"  {p.name}")
        raise RuntimeError(
            "Unknown modality filenames found. "
            "Please rename files or adjust infer_modality()."
        )

    rng = random.Random(args.seed)

    total_train = []
    total_val = []

    print("=" * 80)
    print("Stratified split ship_small by modality")
    print("=" * 80)
    print(f"src_root     = {src_root}")
    print(f"dst_root     = {dst_root}")
    print(f"train_ratio  = {args.train_ratio}")
    print(f"seed         = {args.seed}")
    print(f"force_class  = {args.force_class}")
    print("=" * 80)

    for modality in sorted(groups.keys()):
        items = groups[modality]
        rng.shuffle(items)

        n_train = int(round(len(items) * args.train_ratio))
        train_items = items[:n_train]
        val_items = items[n_train:]

        total_train.extend(train_items)
        total_val.extend(val_items)

        print(
            f"[{modality}] total={len(items)}, "
            f"train={len(train_items)}, val={len(val_items)}"
        )

    print("=" * 80)
    print(f"[ALL] train={len(total_train)}, val={len(total_val)}")
    print("=" * 80)

    train_boxes, train_skipped = copy_items(
        items=total_train,
        split_name="train",
        src_labels_dir=src_labels_dir,
        dst_root=dst_root,
        force_class=args.force_class,
    )

    val_boxes, val_skipped = copy_items(
        items=total_val,
        split_name="val",
        src_labels_dir=src_labels_dir,
        dst_root=dst_root,
        force_class=args.force_class,
    )

    print("=" * 80)
    print("Done.")
    print(f"train boxes={train_boxes}, skipped={train_skipped}")
    print(f"val boxes={val_boxes}, skipped={val_skipped}")
    print("Output:")
    print(f"  {dst_root / 'images' / 'train'}")
    print(f"  {dst_root / 'images' / 'val'}")
    print(f"  {dst_root / 'labels' / 'train'}")
    print(f"  {dst_root / 'labels' / 'val'}")
    print("=" * 80)


if __name__ == "__main__":
    main()