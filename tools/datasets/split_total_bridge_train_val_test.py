from __future__ import annotations

import sys
sys.path.insert(0, r"C:/Users/DOCTOR/Documents/GitHub/YOLOv8/ultralytics")
sys.path.insert(1, r"C:/Users/DOCTOR/Documents/GitHub/YOLOv8")

import argparse
import csv
import hashlib
import json
import random
import shutil
from pathlib import Path


REPO_ROOT = Path(r"C:/Users/DOCTOR/Documents/GitHub/YOLOv8")
DEFAULT_SRC = REPO_ROOT / "ultralytics" / "datasets" / "total_bridge"
DEFAULT_TRAINVAL = REPO_ROOT / "ultralytics" / "datasets" / "total_bridge_trainval"
DEFAULT_TEST = REPO_ROOT / "ultralytics" / "datasets" / "total_bridge_test"
DEFAULT_CONFIGS = REPO_ROOT / "configs" / "datasets"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MODALITIES = ("rgb", "sar", "ir")


def log(msg: object = "") -> None:
    print(msg, flush=True)


def norm(path: Path) -> str:
    return path.resolve().as_posix()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def list_images(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def read_label_count(label: Path) -> tuple[int, int, dict[int, int]]:
    rows = bad = 0
    hist: dict[int, int] = {}
    if not label.exists():
        return rows, 1, hist
    for line in label.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            if len(parts) != 5:
                raise ValueError
            cls = int(float(parts[0]))
            vals = [float(v) for v in parts[1:]]
            if cls != 0 or any(v < 0 or v > 1 for v in vals) or vals[2] <= 0 or vals[3] <= 0:
                raise ValueError
            rows += 1
            hist[cls] = hist.get(cls, 0) + 1
        except Exception:
            bad += 1
    return rows, bad, hist


def collect_records(src_root: Path) -> tuple[dict[str, list[dict]], dict]:
    records: dict[str, list[dict]] = {}
    summary = {"per_modality": {}, "bad_label_rows": 0, "missing_labels": 0, "class_histogram": {}, "boxes": 0}
    for modality in MODALITIES:
        image_dir = src_root / "images" / f"train_{modality}"
        label_dir = src_root / "labels" / f"train_{modality}"
        items = []
        for image in list_images(image_dir):
            label = label_dir / f"{image.stem}.txt"
            count, bad, hist = read_label_count(label)
            if not label.exists():
                summary["missing_labels"] += 1
            summary["bad_label_rows"] += bad
            summary["boxes"] += count
            for cls, n in hist.items():
                summary["class_histogram"][cls] = summary["class_histogram"].get(cls, 0) + n
            items.append(
                {
                    "image": image,
                    "label": label,
                    "modality": modality,
                    "stem": image.stem,
                    "sha256": sha256(image),
                    "label_count": count,
                    "bad_label_rows": bad,
                }
            )
        records[modality] = items
        summary["per_modality"][modality] = {"images": len(items), "boxes": sum(x["label_count"] for x in items)}
    return records, summary


def split_records(records: dict[str, list[dict]], train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")
    rng = random.Random(seed)
    out = {"train": [], "val": [], "test": []}
    per_modality = {}
    for modality, items in records.items():
        shuffled = list(items)
        rng.shuffle(shuffled)
        n = len(shuffled)
        val_n = round(n * val_ratio)
        test_n = round(n * test_ratio)
        train_n = n - val_n - test_n
        train_items = shuffled[:train_n]
        val_items = shuffled[train_n : train_n + val_n]
        test_items = shuffled[train_n + val_n :]
        for split, split_items in (("train", train_items), ("val", val_items), ("test", test_items)):
            for item in split_items:
                item = dict(item)
                item["split"] = split
                out[split].append(item)
        per_modality[modality] = {
            "train": {"images": len(train_items), "boxes": sum(x["label_count"] for x in train_items)},
            "val": {"images": len(val_items), "boxes": sum(x["label_count"] for x in val_items)},
            "test": {"images": len(test_items), "boxes": sum(x["label_count"] for x in test_items)},
        }
    return out, per_modality


def safe_recreate(path: Path) -> None:
    resolved = path.resolve()
    if path.exists():
        datasets_root = (REPO_ROOT / "ultralytics" / "datasets").resolve()
        if resolved.parent != datasets_root:
            raise RuntimeError(f"Refusing to remove unexpected path: {resolved}")
        shutil.rmtree(resolved)
    path.mkdir(parents=True, exist_ok=True)


def copy_split(split_items: list[dict], trainval_root: Path, test_root: Path) -> tuple[list[dict], list[dict]]:
    trainval_rows = []
    test_rows = []
    for item in split_items:
        split = item["split"]
        modality = item["modality"]
        if split == "test":
            root = test_root
            out_split = f"test_{modality}"
            rows = test_rows
        else:
            root = trainval_root
            out_split = f"{split}_{modality}"
            rows = trainval_rows
        dst_image = root / "images" / out_split / item["image"].name
        dst_label = root / "labels" / out_split / item["label"].name
        dst_image.parent.mkdir(parents=True, exist_ok=True)
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item["image"], dst_image)
        shutil.copy2(item["label"], dst_label)
        row = {
            "image_path_original": str(item["image"]),
            "label_path_original": str(item["label"]),
            "image_path_new": str(dst_image),
            "label_path_new": str(dst_label),
            "modality": modality,
            "split": split,
            "sha256": item["sha256"],
            "stem": item["stem"],
            "label_count": item["label_count"],
        }
        rows.append(row)
    return trainval_rows, test_rows


def write_manifest(path: Path, rows: list[dict]) -> None:
    fields = [
        "image_path_original",
        "label_path_original",
        "image_path_new",
        "label_path_new",
        "modality",
        "split",
        "sha256",
        "stem",
        "label_count",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def overlap(a: list[dict], b: list[dict], key: str) -> dict:
    aa = {}
    bb = {}
    for x in a:
        aa.setdefault(x[key], []).append(x["image_path_new"])
    for x in b:
        bb.setdefault(x[key], []).append(x["image_path_new"])
    common = sorted(set(aa) & set(bb))
    return {"count": len(common), "examples": [{"key": k, "a": aa[k], "b": bb[k]} for k in common[:20]]}


def write_yamls(trainval_root: Path, test_root: Path, configs_dir: Path) -> dict:
    trainval_text = f"""path: {norm(trainval_root)}

train:
  - images/train_rgb
  - images/train_sar
  - images/train_ir

val:
  - images/val_rgb
  - images/val_sar
  - images/val_ir

nc: 1

names:
  0: bridge
"""
    test_text = f"""path: {norm(test_root)}

val:
  - images/test_rgb
  - images/test_sar
  - images/test_ir

test:
  - images/test_rgb
  - images/test_sar
  - images/test_ir

nc: 1

names:
  0: bridge
"""
    trainval_yaml = trainval_root / "data.yaml"
    test_yaml = test_root / "data.yaml"
    config_trainval = configs_dir / "total_bridge_trainval.yaml"
    config_test = configs_dir / "total_bridge_test.yaml"
    configs_dir.mkdir(parents=True, exist_ok=True)
    trainval_yaml.write_text(trainval_text, encoding="utf-8")
    test_yaml.write_text(test_text, encoding="utf-8")
    config_trainval.write_text(trainval_text, encoding="utf-8")
    config_test.write_text(test_text, encoding="utf-8")
    return {
        "trainval_yaml": str(trainval_yaml),
        "test_yaml": str(test_yaml),
        "config_trainval_yaml": str(config_trainval),
        "config_test_yaml": str(config_test),
    }


def summarize_rows(rows: list[dict]) -> dict:
    out = {"images": len(rows), "boxes": sum(int(x["label_count"]) for x in rows), "modalities": {}}
    for modality in MODALITIES:
        xs = [x for x in rows if x["modality"] == modality]
        out["modalities"][modality] = {"images": len(xs), "boxes": sum(int(x["label_count"]) for x in xs)}
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", default=str(DEFAULT_SRC))
    parser.add_argument("--trainval-root", default=str(DEFAULT_TRAINVAL))
    parser.add_argument("--test-root", default=str(DEFAULT_TEST))
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--configs-dir", default=str(DEFAULT_CONFIGS))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src_root = Path(args.src_root)
    trainval_root = Path(args.trainval_root)
    test_root = Path(args.test_root)
    configs_dir = Path(args.configs_dir)

    records, source_summary = collect_records(src_root)
    splits, per_modality = split_records(records, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)

    if (trainval_root.exists() or test_root.exists()) and not args.overwrite:
        raise FileExistsError("Output split dirs exist. Re-run with --overwrite to rebuild generated split dirs.")
    safe_recreate(trainval_root)
    safe_recreate(test_root)

    trainval_rows, test_rows = copy_split(splits["train"] + splits["val"] + splits["test"], trainval_root, test_root)
    write_manifest(trainval_root / "manifest.csv", trainval_rows)
    write_manifest(test_root / "manifest.csv", test_rows)
    yaml_paths = write_yamls(trainval_root, test_root, configs_dir)

    train_rows = [x for x in trainval_rows if x["split"] == "train"]
    val_rows = [x for x in trainval_rows if x["split"] == "val"]
    summary = {
        "source_root": str(src_root),
        "trainval_root": str(trainval_root),
        "test_root": str(test_root),
        "seed": args.seed,
        "split_ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
        "source_summary": source_summary,
        "per_modality_split": per_modality,
        "all_counts": {
            "train": summarize_rows(train_rows),
            "val": summarize_rows(val_rows),
            "test": summarize_rows(test_rows),
        },
        "hash_overlap": {
            "train_val": overlap(train_rows, val_rows, "sha256"),
            "train_test": overlap(train_rows, test_rows, "sha256"),
            "val_test": overlap(val_rows, test_rows, "sha256"),
        },
        "stem_overlap": {
            "train_val": overlap(train_rows, val_rows, "stem"),
            "train_test": overlap(train_rows, test_rows, "stem"),
            "val_test": overlap(val_rows, test_rows, "stem"),
        },
        "bad_label_rows": source_summary["bad_label_rows"],
        "missing_labels": source_summary["missing_labels"],
        "class_histogram": source_summary["class_histogram"],
        "yaml_paths": yaml_paths,
    }
    summary["split_leakage_detected"] = any(v["count"] > 0 for v in summary["hash_overlap"].values())
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    (trainval_root / "split_summary.json").write_text(text, encoding="utf-8")
    (test_root / "split_summary.json").write_text(text, encoding="utf-8")
    log(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
