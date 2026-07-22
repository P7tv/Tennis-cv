"""รวมหลาย YOLO-format dataset (แต่ละอันมี data.yaml + train/valid/test/{images,labels})
เป็น dataset เดียว multi-class — เพื่อเทรนโมเดลเดียวที่ detect ได้ทุก class
พร้อมกัน (ตรงกับที่ webui/yolo_track.py คาดหวัง: "single_model_mode" ต้องการ
custom model ตัวเดียวที่มีทั้ง ball + racket)

class id จะถูก remap ใหม่ตามลำดับที่ dataset ระบุใน --dataset (ชื่อ class ที่
ซ้ำกันข้าม dataset จะถูกรวมเป็น global id เดียวกัน)

รัน:
  python train_model/merge_yolo_datasets.py \\
      --dataset Tennis-Ball-detection-1 \\
      --dataset racket-4-yolo \\
      --out tennis-ball-racket-yolo
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import yaml

# Windows console บาง terminal default เป็น cp1252 ไม่รองรับตัวอักษรไทยบางตัว
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy(src, dst)


def merge_split(dataset_dir: Path, split: str, local_to_global: dict[int, int],
                 out_dir: Path, prefix: str) -> int:
    img_dir = dataset_dir / split / "images"
    lbl_dir = dataset_dir / split / "labels"
    if not img_dir.exists():
        return 0

    out_img_dir = out_dir / split / "images"
    out_lbl_dir = out_dir / split / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for img_path in img_dir.iterdir():
        if not img_path.is_file():
            continue
        # prefix กันชื่อไฟล์ชนกันข้าม dataset (เช่นทั้งคู่มี frame_0001.jpg)
        out_name = f"{prefix}_{img_path.name}"
        _link_or_copy(img_path, out_img_dir / out_name)

        src_lbl = lbl_dir / (img_path.stem + ".txt")
        out_lbl = out_lbl_dir / (Path(out_name).stem + ".txt")
        if not src_lbl.exists():
            out_lbl.write_text("", encoding="utf-8")
            n += 1
            continue

        lines_out = []
        for line in src_lbl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            local_cls = int(parts[0])
            global_cls = local_to_global.get(local_cls)
            if global_cls is None:
                continue  # class ที่ไม่ได้อยู่ใน mapping (ไม่ควรเกิดถ้า data.yaml ถูกต้อง)
            lines_out.append(" ".join([str(global_cls)] + parts[1:]))
        out_lbl.write_text("\n".join(lines_out), encoding="utf-8")
        n += 1
    return n


def main():
    parser = argparse.ArgumentParser(description="รวมหลาย YOLO dataset เป็น multi-class dataset เดียว")
    parser.add_argument("--dataset", action="append", required=True,
                         help="path ของ YOLO dataset ที่มี data.yaml (ระบุซ้ำได้หลายครั้ง)")
    parser.add_argument("--out", required=True, help="โฟลเดอร์ปลายทาง")
    args = parser.parse_args()

    out_dir = Path(args.out)
    global_names: list[str] = []
    name_to_global: dict[str, int] = {}

    dataset_specs = []
    for ds_path in args.dataset:
        ds_dir = Path(ds_path)
        data_yaml = yaml.safe_load((ds_dir / "data.yaml").read_text(encoding="utf-8"))
        local_names = data_yaml["names"]
        local_to_global = {}
        for local_id, name in enumerate(local_names):
            if name not in name_to_global:
                name_to_global[name] = len(global_names)
                global_names.append(name)
            local_to_global[local_id] = name_to_global[name]
        dataset_specs.append((ds_dir, local_to_global))
        print(f"{ds_dir}: local classes {local_names} -> global ids "
              f"{[local_to_global[i] for i in range(len(local_names))]}")

    print(f"\nรวม global class list: {global_names}")

    for ds_dir, local_to_global in dataset_specs:
        prefix = ds_dir.name.replace(" ", "_")
        for split in ("train", "valid", "test"):
            n = merge_split(ds_dir, split, local_to_global, out_dir, prefix)
            if n:
                print(f"  [{ds_dir.name}] {split}: รวมแล้ว {n} รูป")

    data_yaml_out = out_dir / "data.yaml"
    names_str = "[" + ", ".join(f"'{n}'" for n in global_names) + "]"
    data_yaml_out.write_text(
        "train: ../train/images\n"
        "val: ../valid/images\n"
        "test: ../test/images\n\n"
        f"nc: {len(global_names)}\n"
        f"names: {names_str}\n",
        encoding="utf-8",
    )
    print(f"\nเขียนแล้ว: {data_yaml_out}")
    print(f"เทรนได้ด้วย: python train_model/train_yolo.py  (แล้วป้อน path: {data_yaml_out})")


if __name__ == "__main__":
    main()
