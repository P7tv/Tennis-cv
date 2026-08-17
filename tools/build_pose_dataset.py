"""สร้าง YOLO-Pose dataset เดียวที่รวม ball (bbox เท่านั้น) + racket (bbox + 4
keypoint) จาก 2 แหล่งข้อมูลคนละ format:
  - ball:   Tennis-Ball-detection-1/  (YOLO detect format, class 0 อยู่แล้ว)
  - racket: racket-4/                 (COCO format, มี keypoints ต่อ annotation)

label แต่ละบรรทัดใน output เป็น pose format:
  class xc yc w h  kp1x kp1y kp1v  kp2x kp2y kp2v  kp3x kp3y kp3v  kp4x kp4y kp4v
ball ไม่มี keypoint จริง -> เติม 0 0 0 ต่อจุด (v=0 = not labeled ตามธรรมเนียม COCO)

รัน: python train_model/build_pose_dataset.py \\
        --ball Tennis-Ball-detection-1 --racket racket-4 \\
        --out tennis-ball-racket-pose
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Windows console บาง terminal default เป็น cp1252 ไม่รองรับตัวอักษรไทยบางตัว
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

N_KEYPOINTS = 4
BALL_CLASS = 0
RACKET_CLASS = 1
DUMMY_KP = " 0 0 0" * N_KEYPOINTS


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy(src, dst)


def convert_ball_split(ball_dir: Path, split: str, out_dir: Path) -> int:
    """YOLO detect (class 0, 5 token/บรรทัด) -> pose format (เติม dummy keypoint)"""
    img_dir = ball_dir / split / "images"
    lbl_dir = ball_dir / split / "labels"
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
        out_name = f"ball_{img_path.name}"
        _link_or_copy(img_path, out_img_dir / out_name)

        src_lbl = lbl_dir / (img_path.stem + ".txt")
        out_lbl = out_lbl_dir / (Path(out_name).stem + ".txt")
        lines_out = []
        if src_lbl.exists():
            for line in src_lbl.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                # class เดิมของ Tennis-Ball-detection-1 คือ 0 ('ball') อยู่แล้ว
                lines_out.append(" ".join([str(BALL_CLASS)] + parts[1:]) + DUMMY_KP)
        out_lbl.write_text("\n".join(lines_out), encoding="utf-8")
        n += 1
    return n


def convert_racket_split(racket_dir: Path, split: str, out_dir: Path) -> int:
    """COCO (bbox + keypoints) -> pose format"""
    coco_json_path = racket_dir / split / "_annotations.coco.json"
    if not coco_json_path.exists():
        return 0
    data = json.loads(coco_json_path.read_text(encoding="utf-8"))

    images_by_id = {img["id"]: img for img in data["images"]}
    anns_by_image: dict[int, list] = {}
    for ann in data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    out_img_dir = out_dir / split / "images"
    out_lbl_dir = out_dir / split / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for img_id, img in images_by_id.items():
        src_img = racket_dir / split / img["file_name"]
        if not src_img.exists():
            print(f"  WARNING: missing image {src_img}, skip")
            continue
        out_name = f"racket_{img['file_name']}"
        _link_or_copy(src_img, out_img_dir / out_name)

        w, h = img["width"], img["height"]
        lines = []
        for ann in anns_by_image.get(img_id, []):
            x, y, bw, bh = ann["bbox"]
            xc, yc = (x + bw / 2.0) / w, (y + bh / 2.0) / h
            nw, nh = bw / w, bh / h

            kps = ann.get("keypoints", [])
            kp_tokens = []
            for i in range(N_KEYPOINTS):
                if i * 3 + 2 < len(kps):
                    kx, ky, kv = kps[i * 3], kps[i * 3 + 1], kps[i * 3 + 2]
                    kp_tokens.append(f"{kx / w:.6f} {ky / h:.6f} {int(kv)}")
                else:
                    kp_tokens.append("0 0 0")  # annotation นี้ไม่มีครบ 4 จุด

            lines.append(
                f"{RACKET_CLASS} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f} " + " ".join(kp_tokens)
            )

        label_path = out_lbl_dir / (Path(out_name).stem + ".txt")
        label_path.write_text("\n".join(lines), encoding="utf-8")
        n += 1
    return n


def main():
    parser = argparse.ArgumentParser(
        description="รวม ball (YOLO detect) + racket (COCO+keypoints) เป็น YOLO-Pose dataset เดียว")
    parser.add_argument("--ball", default="Tennis-Ball-detection-1")
    parser.add_argument("--racket", default="racket-4")
    parser.add_argument("--out", default="tennis-ball-racket-pose")
    args = parser.parse_args()

    ball_dir = Path(args.ball)
    racket_dir = Path(args.racket)
    out_dir = Path(args.out)

    for split in ("train", "valid", "test"):
        n_ball = convert_ball_split(ball_dir, split, out_dir)
        n_racket = convert_racket_split(racket_dir, split, out_dir)
        if n_ball or n_racket:
            print(f"{split}: ball={n_ball} racket={n_racket} รวม={n_ball + n_racket}")

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        "train: ../train/images\n"
        "val: ../valid/images\n"
        "test: ../test/images\n\n"
        "nc: 2\n"
        "names: ['ball', 'tennis-racket']\n\n"
        f"kpt_shape: [{N_KEYPOINTS}, 3]  # x, y, visibility ต่อจุด\n",
        encoding="utf-8",
    )
    print(f"\nเขียนแล้ว: {data_yaml}")
    print("หมายเหตุ: ตอนเทรนต้องปิด horizontal flip (fliplr=0.0) เพราะไม่รู้ mapping "
          "ซ้าย-ขวาของ keypoint แต่ละจุดแน่ชัด (ดู train_model/train_yolo.py)")


if __name__ == "__main__":
    main()
