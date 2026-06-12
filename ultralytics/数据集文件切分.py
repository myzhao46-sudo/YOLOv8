from pathlib import Path
import argparse
import random
import shutil

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def find_images(images_dir: Path):
    images = []
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            images.append(p)
    return sorted(images)


def rewrite_label_to_class0(src_label: Path, dst_label: Path, force_class: int = 0):
    """
    Copy YOLO label and force first column to force_class.
    Expected format:
      cls cx cy w h

    If a line has more columns, it still rewrites first column and keeps the rest.
    """
    if not src_label.exists():
        print(f"[WARN] Missing label: {src_label}")
        dst_label.write_text("", encoding="utf-8")
        return 0, 1

    out_lines = []
    kept = 0
    bad = 0

    with open(src_label, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue

            if len(parts) < 5:
                print(f"[WARN] Bad label line: {src_label} line {line_no}: {line.strip()}")
                bad += 1
                continue

            # force ship class id = 0
            parts[0] = str(force_class)
            out_lines.append(" ".join(parts) + "\n")
            kept += 1

    dst_label.write_text("".join(out_lines), encoding="utf-8")
    return kept, bad


def copy_split(items, split_name, src_labels_dir: Path, dst_root: Path, force_class: int):
    dst_img_dir = dst_root / "images" / split_name
    dst_lbl_dir = dst_root / "labels" / split_name

    ensure_dir(dst_img_dir)
    ensure_dir(dst_lbl_dir)

    total_labels = 0
    missing_or_bad = 0

    for img_path in items:
        dst_img = dst_img_dir / img_path.name
        shutil.copy2(img_path, dst_img)

        src_label = src_labels_dir / f"{img_path.stem}.txt"
        dst_label = dst_lbl_dir / f"{img_path.stem}.txt"

        kept, bad = rewrite_label_to_class0(
            src_label=src_label,
            dst_label=dst_label,
            force_class=force_class,
        )

        total_labels += kept
        missing_or_bad += bad

    print(
        f"[{split_name}] images={len(items)}, "
        f"boxes={total_labels}, bad_or_missing={missing_or_bad}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src-root",
        required=True,
        help="Original ship_small root, containing images/ and labels/",
    )

    parser.add_argument(
        "--dst-root",
        required=True,
        help="Output root, e.g. ship_small_split",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train ratio. Default 0.8 means train:val = 4:1",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible split",
    )

    parser.add_argument(
        "--force-class",
        type=int,
        default=0,
        help="Force output class id. For ship in global4, use 0.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing dst-root.",
    )

    args = parser.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)

    src_images_dir = src_root / "images"
    src_labels_dir = src_root / "labels"

    if not src_images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {src_images_dir}")

    if not src_labels_dir.exists():
        raise FileNotFoundError(f"Labels dir not found: {src_labels_dir}")

    if dst_root.exists() and not args.overwrite:
        raise FileExistsError(
            f"dst-root already exists: {dst_root}\n"
            f"Use --overwrite if you really want to write into it."
        )

    images = find_images(src_images_dir)

    if not images:
        raise RuntimeError(f"No images found in: {src_images_dir}")

    random.seed(args.seed)
    random.shuffle(images)

    n_train = int(round(len(images) * args.train_ratio))
    train_items = images[:n_train]
    val_items = images[n_train:]

    print("=" * 80)
    print("Split ship_small into YOLO train/val structure")
    print("=" * 80)
    print(f"src_root     = {src_root}")
    print(f"dst_root     = {dst_root}")
    print(f"total images = {len(images)}")
    print(f"train        = {len(train_items)}")
    print(f"val          = {len(val_items)}")
    print(f"force_class  = {args.force_class}")
    print(f"seed         = {args.seed}")
    print("=" * 80)

    copy_split(
        items=train_items,
        split_name="train",
        src_labels_dir=src_labels_dir,
        dst_root=dst_root,
        force_class=args.force_class,
    )

    copy_split(
        items=val_items,
        split_name="val",
        src_labels_dir=src_labels_dir,
        dst_root=dst_root,
        force_class=args.force_class,
    )

    print("=" * 80)
    print("Done.")
    print("Output structure:")
    print(f"  {dst_root / 'images' / 'train'}")
    print(f"  {dst_root / 'images' / 'val'}")
    print(f"  {dst_root / 'labels' / 'train'}")
    print(f"  {dst_root / 'labels' / 'val'}")


if __name__ == "__main__":
    main()