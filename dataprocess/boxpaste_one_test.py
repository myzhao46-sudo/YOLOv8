import copy
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2


# ========= 你改这里 =========
MSAR_IMG = Path(r"C:\Users\DOCTOR\Desktop\daliy\DrZG\MSAR-1.0 dataset\JPEGImages\01.jpg")
MSAR_XML = Path(r"C:\Users\DOCTOR\Desktop\daliy\DrZG\MSAR-1.0 dataset\Annotations\01.xml")

OILTANK_IMG = Path(r"C:\Users\DOCTOR\Desktop\daliy\DrZG\oiltank\images\train\01620.jpg")   # 改成你实际存在的一张
OILTANK_TXT = Path(r"C:\Users\DOCTOR\Desktop\daliy\DrZG\oiltank\labels\train\01620.txt")   # 改成对应标签

OUT_DIR = Path(r"C:\Users\DOCTOR\Desktop\daliy\DrZG\boxpaste_one_test")
OUT_IMG = OUT_DIR / "test_boxpaste.jpg"
OUT_XML = OUT_DIR / "test_boxpaste.xml"

OIL_CLASS_ID = 1
OIL_CLASS_NAME = "油罐"

# 指定粘贴左上角位置。先手工给，方便验证
PASTE_X = 30
PASTE_Y = 30
# ==========================


def read_img(img_path: Path):
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
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


def parse_yolo_one(txt_path: Path, img_w: int, img_h: int, target_class_id=1):
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f.readlines() if x.strip()]

    if not lines:
        raise ValueError(f"Empty label file: {txt_path}")

    # 只取第一条符合 class_id 的框
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue

        cls_id = int(float(parts[0]))
        if cls_id != target_class_id:
            continue

        xc = float(parts[1])
        yc = float(parts[2])
        bw = float(parts[3])
        bh = float(parts[4])

        x_center = xc * img_w
        y_center = yc * img_h
        box_w = bw * img_w
        box_h = bh * img_h

        xmin = int(round(x_center - box_w / 2))
        ymin = int(round(y_center - box_h / 2))
        xmax = int(round(x_center + box_w / 2))
        ymax = int(round(y_center + box_h / 2))

        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(img_w, xmax)
        ymax = min(img_h, ymax)

        if xmax <= xmin or ymax <= ymin:
            continue

        return {
            "class_id": cls_id,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
        }

    raise ValueError(f"No class_id={target_class_id} found in {txt_path}")


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


def draw_boxes(img, boxes, color=255, thickness=1):
    out = img.copy()
    if len(out.shape) == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    for b in boxes:
        cv2.rectangle(out, (b["xmin"], b["ymin"]), (b["xmax"], b["ymax"]), (0, 255, 0), thickness)
        cv2.putText(
            out,
            b["name"],
            (b["xmin"], max(12, b["ymin"] - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bg_img = read_img(MSAR_IMG)
    oil_img = read_img(OILTANK_IMG)

    bg_tree, bg_root, bg_w, bg_h, bg_objs = parse_voc(MSAR_XML)
    oil_h, oil_w = oil_img.shape[:2]
    oil_obj = parse_yolo_one(OILTANK_TXT, oil_w, oil_h, target_class_id=OIL_CLASS_ID)

    print("[INFO] Background size:", bg_w, bg_h)
    print("[INFO] Existing background objects:", len(bg_objs))
    print("[INFO] Oil source size:", oil_w, oil_h)
    print("[INFO] Oil bbox:", oil_obj)

    patch = oil_img[oil_obj["ymin"]:oil_obj["ymax"], oil_obj["xmin"]:oil_obj["xmax"]].copy()
    if patch.size == 0:
        raise RuntimeError("Patch is empty")

    ph, pw = patch.shape[:2]
    print("[INFO] Patch size:", pw, ph)

    if PASTE_X + pw > bg_w or PASTE_Y + ph > bg_h:
        raise ValueError("Paste position out of image boundary")

    out_img = bg_img.copy()
    out_img[PASTE_Y:PASTE_Y+ph, PASTE_X:PASTE_X+pw] = patch

    out_root = copy.deepcopy(bg_root)
    add_object_to_voc(
        out_root,
        OIL_CLASS_NAME,
        PASTE_X,
        PASTE_Y,
        PASTE_X + pw,
        PASTE_Y + ph
    )
    update_filename_and_path(out_root, OUT_IMG)

    ok = cv2.imwrite(str(OUT_IMG), out_img)
    if not ok:
        raise RuntimeError(f"Failed to save image: {OUT_IMG}")

    ET.ElementTree(out_root).write(str(OUT_XML), encoding="utf-8", xml_declaration=False)

    # 再额外生成一张“带框可视化图”，方便你直接看
    vis_boxes = copy.deepcopy(bg_objs)
    vis_boxes.append({
        "name": OIL_CLASS_NAME,
        "xmin": PASTE_X,
        "ymin": PASTE_Y,
        "xmax": PASTE_X + pw,
        "ymax": PASTE_Y + ph,
    })
    vis_img = draw_boxes(out_img, vis_boxes)
    vis_path = OUT_DIR / "test_boxpaste_vis.jpg"
    cv2.imwrite(str(vis_path), vis_img)

    print("\n[OK] Saved:")
    print(" image:", OUT_IMG)
    print(" xml:  ", OUT_XML)
    print(" vis:  ", vis_path)
    print("[INFO] New oil tank box:", [PASTE_X, PASTE_Y, PASTE_X + pw, PASTE_Y + ph])


if __name__ == "__main__":
    main()