from __future__ import annotations

import sys
sys.path.insert(0, r"C:/Users/DOCTOR/Documents/GitHub/YOLOv8/ultralytics")
sys.path.insert(1, r"C:/Users/DOCTOR/Documents/GitHub/YOLOv8")

import argparse
import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(r"C:/Users/DOCTOR/Documents/GitHub/YOLOv8")
DEFAULT_TRAINVAL = REPO_ROOT / "ultralytics" / "datasets" / "total_bridge_trainval"
DEFAULT_TEST = REPO_ROOT / "ultralytics" / "datasets" / "total_bridge_test"
DEFAULT_OUT = REPO_ROOT / "runs" / "bridge_expert" / "split_check"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = {
    "train": ("train_rgb", "train_sar", "train_ir"),
    "val": ("val_rgb", "val_sar", "val_ir"),
    "test": ("test_rgb", "test_sar", "test_ir"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def images(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS) if path.exists() else []


def check_label(path: Path) -> tuple[int, int, dict[int, int], list[str]]:
    rows = bad = 0
    hist: dict[int, int] = {}
    examples = []
    if not path.exists():
        return rows, 1, hist, [f"missing label: {path}"]
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        try:
            if len(parts) != 5:
                raise ValueError(f"expected 5 columns, got {len(parts)}")
            cls = int(float(parts[0]))
            vals = [float(v) for v in parts[1:]]
            if cls != 0:
                raise ValueError(f"class id is {cls}, expected 0")
            if any(v < 0 or v > 1 for v in vals):
                raise ValueError("coords outside [0,1]")
            if vals[2] <= 0 or vals[3] <= 0:
                raise ValueError("non-positive width/height")
            rows += 1
            hist[cls] = hist.get(cls, 0) + 1
        except Exception as exc:
            bad += 1
            if len(examples) < 20:
                examples.append(f"{path}:{line_no}: {line} | {exc}")
    return rows, bad, hist, examples


def scan_split(root: Path, split: str, dirs: tuple[str, ...]) -> tuple[list[dict], dict]:
    records = []
    stats = {"images": 0, "labels": 0, "boxes": 0, "bad_rows": 0, "missing_labels": 0, "modalities": {}, "class_histogram": {}, "examples": []}
    for dirname in dirs:
        modality = dirname.split("_", 1)[1]
        image_dir = root / "images" / dirname
        label_dir = root / "labels" / dirname
        modality_stats = {"images": 0, "labels": 0, "boxes": 0, "bad_rows": 0, "missing_labels": 0}
        for image in images(image_dir):
            label = label_dir / f"{image.stem}.txt"
            count, bad, hist, examples = check_label(label)
            record = {"split": split, "modality": modality, "image": str(image), "label": str(label), "sha256": sha256(image), "stem": image.stem, "boxes": count}
            records.append(record)
            modality_stats["images"] += 1
            modality_stats["labels"] += int(label.exists())
            modality_stats["boxes"] += count
            modality_stats["bad_rows"] += bad
            modality_stats["missing_labels"] += int(not label.exists())
            stats["boxes"] += count
            stats["bad_rows"] += bad
            stats["missing_labels"] += int(not label.exists())
            stats["examples"].extend(examples[: max(0, 20 - len(stats["examples"]))])
            for cls, n in hist.items():
                stats["class_histogram"][cls] = stats["class_histogram"].get(cls, 0) + n
        stats["modalities"][modality] = modality_stats
        stats["images"] += modality_stats["images"]
        stats["labels"] += modality_stats["labels"]
    return records, stats


def overlap(a: list[dict], b: list[dict], key: str) -> dict:
    aa = {}
    bb = {}
    for x in a:
        aa.setdefault(x[key], []).append(x["image"])
    for x in b:
        bb.setdefault(x[key], []).append(x["image"])
    common = sorted(set(aa) & set(bb))
    return {"count": len(common), "examples": [{"key": k, "a": aa[k], "b": bb[k]} for k in common[:20]]}


def write_md(path: Path, summary: dict) -> None:
    lines = ["# total_bridge Split Check", ""]
    lines.append(f"Severe issues: `{summary['severe_issues']}`")
    lines.append(f"Hash leakage detected: `{summary['hash_leakage_detected']}`")
    lines.append("")
    lines.append("| split | images | labels | boxes | bad rows | missing labels |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for split, stats in summary["splits"].items():
        lines.append(f"| {split} | {stats['images']} | {stats['labels']} | {stats['boxes']} | {stats['bad_rows']} | {stats['missing_labels']} |")
    lines.append("")
    lines.append("## Hash Overlap")
    for k, v in summary["hash_overlap"].items():
        lines.append(f"- {k}: {v['count']}")
    lines.append("")
    lines.append("## Stem Overlap")
    for k, v in summary["stem_overlap"].items():
        lines.append(f"- {k}: {v['count']}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainval-root", default=str(DEFAULT_TRAINVAL))
    parser.add_argument("--test-root", default=str(DEFAULT_TEST))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    trainval = Path(args.trainval_root)
    test = Path(args.test_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_records, train_stats = scan_split(trainval, "train", SPLITS["train"])
    val_records, val_stats = scan_split(trainval, "val", SPLITS["val"])
    test_records, test_stats = scan_split(test, "test", SPLITS["test"])
    summary = {
        "trainval_root": str(trainval),
        "test_root": str(test),
        "splits": {"train": train_stats, "val": val_stats, "test": test_stats},
        "hash_overlap": {
            "train_val": overlap(train_records, val_records, "sha256"),
            "train_test": overlap(train_records, test_records, "sha256"),
            "val_test": overlap(val_records, test_records, "sha256"),
        },
        "stem_overlap": {
            "train_val": overlap(train_records, val_records, "stem"),
            "train_test": overlap(train_records, test_records, "stem"),
            "val_test": overlap(val_records, test_records, "stem"),
        },
    }
    summary["hash_leakage_detected"] = any(x["count"] > 0 for x in summary["hash_overlap"].values())
    summary["severe_issues"] = summary["hash_leakage_detected"] or any(
        s["bad_rows"] or s["missing_labels"] or s["images"] != s["labels"] for s in summary["splits"].values()
    )
    json_path = out_dir / "total_bridge_split_check.json"
    md_path = out_dir / "total_bridge_split_check.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(md_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
