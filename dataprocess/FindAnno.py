import argparse
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(images_dir: Path):
    images = []
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            images.append(p)
    return sorted(images)


def find_xml_for_image(image_path: Path, annotations_dir: Path):
    stem = image_path.stem
    xml_path = annotations_dir / f"{stem}.xml"
    if xml_path.exists():
        return xml_path
    return None


def parse_voc_xml(xml_path: Path):
    """
    返回:
    {
        "ok": bool,
        "object_count": int,
        "class_names": list[str],
        "error": str | None
    }
    """
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()

        objects = root.findall("object")
        class_names = []

        for obj in objects:
            name_node = obj.find("name")
            if name_node is not None and name_node.text is not None:
                class_names.append(name_node.text.strip())
            else:
                class_names.append("UNKNOWN")

        return {
            "ok": True,
            "object_count": len(objects),
            "class_names": class_names,
            "error": None,
        }

    except Exception as e:
        return {
            "ok": False,
            "object_count": None,
            "class_names": [],
            "error": repr(e),
        }


def write_list(file_path: Path, items):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(str(x) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Check empty-background images in MSAR-1.0 VOC dataset")
    parser.add_argument(
        "--images_dir",
        type=str,
        default=r"C:\Users\DOCTOR\Desktop\daliy\DrZG\MSAR-1.0 dataset\JPEGImages",
        help="Path to image directory"
    )
    parser.add_argument(
        "--annotations_dir",
        type=str,
        default=r"C:\Users\DOCTOR\Desktop\daliy\DrZG\MSAR-1.0 dataset\Annotations",
        help="Path to annotation XML directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"C:\Users\DOCTOR\Desktop\daliy\DrZG\MSAR-1.0 dataset\empty_bg_check_output",
        help="Directory to save reports"
    )
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    annotations_dir = Path(args.annotations_dir)
    output_dir = Path(args.output_dir)

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Images dir:      {images_dir}")
    print(f"[INFO] Annotations dir: {annotations_dir}")
    print(f"[INFO] Output dir:      {output_dir}")
    print("[INFO] Scanning images ...")

    images = list_images(images_dir)

    total_images = len(images)
    if total_images == 0:
        print("[WARN] No images found.")
        return

    print(f"[INFO] Found {total_images} images.")

    empty_background_images = []
    non_empty_images = []
    missing_xml_images = []
    xml_parse_failed = []

    class_counter = Counter()
    object_count_counter = Counter()

    detailed_rows = []

    for idx, img_path in enumerate(images, 1):
        xml_path = find_xml_for_image(img_path, annotations_dir)

        if xml_path is None:
            missing_xml_images.append(str(img_path))
            detailed_rows.append({
                "image": str(img_path),
                "xml": None,
                "status": "missing_xml",
                "object_count": None,
                "class_names": [],
            })
        else:
            result = parse_voc_xml(xml_path)

            if not result["ok"]:
                xml_parse_failed.append({
                    "image": str(img_path),
                    "xml": str(xml_path),
                    "error": result["error"],
                })
                detailed_rows.append({
                    "image": str(img_path),
                    "xml": str(xml_path),
                    "status": "xml_parse_failed",
                    "object_count": None,
                    "class_names": [],
                    "error": result["error"],
                })
            else:
                obj_count = result["object_count"]
                class_names = result["class_names"]

                object_count_counter[obj_count] += 1
                for c in class_names:
                    class_counter[c] += 1

                if obj_count == 0:
                    empty_background_images.append(str(img_path))
                    status = "empty_background"
                else:
                    non_empty_images.append(str(img_path))
                    status = "non_empty"

                detailed_rows.append({
                    "image": str(img_path),
                    "xml": str(xml_path),
                    "status": status,
                    "object_count": obj_count,
                    "class_names": class_names,
                })

        if idx % 500 == 0 or idx == total_images:
            print(f"[INFO] Processed {idx}/{total_images}")

    # 统计分桶
    histogram_buckets = {
        "0": 0,
        "1": 0,
        "2": 0,
        "3+": 0
    }
    for k, v in object_count_counter.items():
        if k == 0:
            histogram_buckets["0"] += v
        elif k == 1:
            histogram_buckets["1"] += v
        elif k == 2:
            histogram_buckets["2"] += v
        elif k >= 3:
            histogram_buckets["3+"] += v

    # 写文本列表
    write_list(output_dir / "empty_background_images.txt", empty_background_images)
    write_list(output_dir / "non_empty_images.txt", non_empty_images)
    write_list(output_dir / "missing_xml_images.txt", missing_xml_images)

    with open(output_dir / "xml_parse_failed.txt", "w", encoding="utf-8") as f:
        for item in xml_parse_failed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 写 JSON
    with open(output_dir / "object_count_histogram.json", "w", encoding="utf-8") as f:
        json.dump({
            "exact_object_count_distribution": dict(sorted(object_count_counter.items(), key=lambda x: x[0])),
            "bucketed_distribution": histogram_buckets,
        }, f, ensure_ascii=False, indent=2)

    with open(output_dir / "class_frequency.json", "w", encoding="utf-8") as f:
        json.dump(dict(class_counter.most_common()), f, ensure_ascii=False, indent=2)

    summary = {
        "images_dir": str(images_dir),
        "annotations_dir": str(annotations_dir),
        "total_images": total_images,
        "images_with_xml": total_images - len(missing_xml_images),
        "empty_background_images": len(empty_background_images),
        "non_empty_images": len(non_empty_images),
        "missing_xml_images": len(missing_xml_images),
        "xml_parse_failed": len(xml_parse_failed),
        "object_count_histogram_bucketed": histogram_buckets,
        "class_frequency": dict(class_counter.most_common()),
        "has_any_empty_background": len(empty_background_images) > 0,
        "looks_like_positive_only_dataset": (
            len(empty_background_images) == 0 and
            len(non_empty_images) > 0
        ),
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 控制台输出
    print("\n========== SUMMARY ==========")
    print(f"Total images:              {summary['total_images']}")
    print(f"Images with XML:           {summary['images_with_xml']}")
    print(f"Empty-background images:   {summary['empty_background_images']}")
    print(f"Non-empty images:          {summary['non_empty_images']}")
    print(f"Missing XML images:        {summary['missing_xml_images']}")
    print(f"XML parse failed:          {summary['xml_parse_failed']}")
    print("\nObject count histogram:")
    print(f"  0 objects:               {histogram_buckets['0']}")
    print(f"  1 object:                {histogram_buckets['1']}")
    print(f"  2 objects:               {histogram_buckets['2']}")
    print(f"  3+ objects:              {histogram_buckets['3+']}")

    print("\nTop classes:")
    if class_counter:
        for cls_name, freq in class_counter.most_common(20):
            print(f"  {cls_name}: {freq}")
    else:
        print("  No classes found.")

    if summary["has_any_empty_background"]:
        print("\n[RESULT] Found empty-background annotations.")
        print(f"[RESULT] See: {output_dir / 'empty_background_images.txt'}")
    else:
        print("\n[RESULT] No empty-background annotations found.")
        print("[RESULT] This dataset may contain only positive samples.")

    print(f"\n[INFO] Summary saved to: {output_dir / 'summary.json'}")
    print(f"[INFO] Detailed files saved under: {output_dir}")


if __name__ == "__main__":
    main()