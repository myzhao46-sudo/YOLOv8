from pathlib import Path

# Root labels directory
LABEL_ROOT = Path(r"E:\YOLODATA\tank_optical_copy\labels")

# Subfolders to process
SPLITS = ["train", "val", "test"]

# Target class id
NEW_CLASS_ID = "1"


def convert_label_file(txt_path: Path):
    """
    Replace the first column of each valid YOLO label line with NEW_CLASS_ID.
    Keep bbox coordinates unchanged.
    """
    lines = txt_path.read_text(encoding="utf-8").splitlines()

    new_lines = []
    changed = 0
    skipped = 0

    for line in lines:
        stripped = line.strip()

        # Keep empty lines as empty
        if not stripped:
            new_lines.append("")
            continue

        parts = stripped.split()

        # YOLO detect label should have 5 columns:
        # class_id x_center y_center width height
        if len(parts) != 5:
            print(f"[WARN] Skip invalid line in {txt_path}: {line}")
            new_lines.append(line)
            skipped += 1
            continue

        old_cls = parts[0]
        parts[0] = NEW_CLASS_ID

        if old_cls != NEW_CLASS_ID:
            changed += 1

        new_lines.append(" ".join(parts))

    txt_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    return changed, skipped


def main():
    if not LABEL_ROOT.exists():
        raise FileNotFoundError(f"Label root not found: {LABEL_ROOT}")

    total_files = 0
    total_changed_lines = 0
    total_skipped_lines = 0

    for split in SPLITS:
        split_dir = LABEL_ROOT / split

        if not split_dir.exists():
            print(f"[WARN] Split folder not found, skip: {split_dir}")
            continue

        txt_files = sorted(split_dir.glob("*.txt"))
        print(f"\nProcessing {split}: {len(txt_files)} txt files")

        for txt_path in txt_files:
            changed, skipped = convert_label_file(txt_path)

            total_files += 1
            total_changed_lines += changed
            total_skipped_lines += skipped

    print("\nDone.")
    print(f"Processed files: {total_files}")
    print(f"Changed label lines: {total_changed_lines}")
    print(f"Skipped invalid lines: {total_skipped_lines}")
    print(f"All valid labels are now class id = {NEW_CLASS_ID}")


if __name__ == "__main__":
    main()