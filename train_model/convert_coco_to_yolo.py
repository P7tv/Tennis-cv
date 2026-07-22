"""แปลง COCO annotation (เช่น racket-4/ ที่โหลดจาก Roboflow ด้วย format=coco)
เป็น YOLO format (images/ + labels/ + data.yaml) เพื่อให้
train_model/train_yolo.py ใช้เทรนต่อได้ตรงๆ โดยไม่ต้องแก้ script เทรนเลย

รัน: python train_model/convert_coco_to_yolo.py --src racket-4 --out racket-4-yolo
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# Windows console บาง terminal default เป็น cp1252 ไม่รองรับตัวอักษรไทยบางตัว
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass


def convert_split(coco_dir: Path, split: str, out_dir: Path, class_map: dict[int, int]) -> int:
    coco_json_path = coco_dir / split / "_annotations.coco.json"
    data = json.loads(coco_json_path.read_text(encoding="utf-8"))

    images_by_id = {img["id"]: img for img in data["images"]}
    anns_by_image: dict[int, list] = {}
    for ann in data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    out_img_dir = out_dir / split / "images"
    out_lbl_dir = out_dir / split / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for img_id, img in images_by_id.items():
        src_img = coco_dir / split / img["file_name"]
        if not src_img.exists():
            print(f"  WARNING: missing image {src_img}, skip")
            continue
        shutil.copy(src_img, out_img_dir / img["file_name"])

        w, h = img["width"], img["height"]
        lines = []
        for ann in anns_by_image.get(img_id, []):
            cat_id = ann["category_id"]
            if cat_id not in class_map:
                continue  # class อื่นที่ไม่ได้ระบุให้แปลง (เช่น "points" ที่ไม่มี annotation จริงอยู่แล้ว)
            cls = class_map[cat_id]
            x, y, bw, bh = ann["bbox"]  # COCO: top-left x,y + width,height (พิกเซล)
            xc = (x + bw / 2.0) / w
            yc = (y + bh / 2.0) / h
            nw = bw / w
            nh = bh / h
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

        label_path = out_lbl_dir / (Path(img["file_name"]).stem + ".txt")
        label_path.write_text("\n".join(lines), encoding="utf-8")
        n_written += 1

    return n_written


def main():
    parser = argparse.ArgumentParser(description="แปลง Roboflow COCO export -> YOLO format dataset")
    parser.add_argument("--src", default="racket-4", help="โฟลเดอร์ COCO ต้นทาง (มี train/valid/test/_annotations.coco.json)")
    parser.add_argument("--out", default="racket-4-yolo", help="โฟลเดอร์ปลายทาง YOLO format")
    parser.add_argument("--class-name", default="tennis-racket",
                         help="ชื่อ category ใน COCO ที่จะแปลง (จะได้ class_id=0 ใน YOLO) — "
                              "category อื่นที่ไม่ตรง (เช่น 'points') จะถูกข้าม")
    args = parser.parse_args()

    coco_dir = Path(args.src)
    out_dir = Path(args.out)

    train_json_path = coco_dir / "train" / "_annotations.coco.json"
    if not train_json_path.exists():
        raise SystemExit(f"ไม่เจอ {train_json_path}")
    train_json = json.loads(train_json_path.read_text(encoding="utf-8"))

    cat_id = next((c["id"] for c in train_json["categories"] if c["name"] == args.class_name), None)
    if cat_id is None:
        available = [c["name"] for c in train_json["categories"]]
        raise SystemExit(f"ไม่เจอ category ชื่อ '{args.class_name}' — ที่มีคือ {available}")
    class_map = {cat_id: 0}
    print(f"map COCO category_id={cat_id} ('{args.class_name}') -> YOLO class 0")

    counts = {}
    for split in ("train", "valid", "test"):
        if not (coco_dir / split / "_annotations.coco.json").exists():
            continue
        n = convert_split(coco_dir, split, out_dir, class_map)
        counts[split] = n
        print(f"  {split}: แปลงแล้ว {n} รูป")

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        "train: ../train/images\n"
        "val: ../valid/images\n"
        "test: ../test/images\n\n"
        "nc: 1\n"
        f"names: ['{args.class_name}']\n",
        encoding="utf-8",
    )
    print(f"\nเขียนแล้ว: {data_yaml}")
    print(f"เทรนได้ด้วย: python train_model/train_yolo.py  (แล้วป้อน path: {data_yaml})")


if __name__ == "__main__":
    main()
