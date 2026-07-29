# Fine-tune Guide — ball+racket pose model + classifier

โมเดลที่ส่งมาด้วย (`checkpoints/model_v0.1.pt`) เทรนผ่านหลายขั้นตอนที่กระจาย
อยู่ใน `train_model/` (แต่ละสคริปต์แยกกัน ไม่มีที่ไหนร้อยเรียงเป็นเส้นเดียวมา
ก่อน — เอกสารนี้คือการร้อยเรียงนั้น) แบ่งเป็น 2 ส่วน: **(A)** ball+racket
pose detector (YOLO) และ **(B)** stroke/hit classifier (RandomForest)

## A. Ball+racket pose detector

### ภาพรวม pipeline (ทำครั้งเดียว → ได้ base model)

```
1. ดาวน์โหลด dataset ball (YOLO format) + racket (COCO format) จาก Roboflow
2. convert_coco_to_yolo.py     racket-4/ (COCO) → racket-4-yolo/ (YOLO)
3. build_pose_dataset.py       รวม ball + racket-4-yolo → tennis-ball-racket-pose/
                                (YOLO-pose format, 4 keypoint ต่อไม้)
4. train_model/train_yolo.py   เทรน base model จาก tennis-ball-racket-pose/
5. finetune_racket_pose.py     fine-tune บนเฟรมจริงจากคลิปลูกค้า → checkpoints/model_v0.1.pt
6. ต่อเข้า webui/yolo_track.py::track_players_with_yolo(model_path=...)
```

### คำสั่งต่อสคริปต์

```bash
# 2. COCO → YOLO (racket dataset)
python train_model/convert_coco_to_yolo.py --src racket-4 --out racket-4-yolo

# 3. รวม ball + racket เป็น pose dataset เดียว (4-point keypoint ต่อไม้)
python train_model/build_pose_dataset.py \
    --ball Tennis-Ball-detection-1 --racket racket-4 \
    --out tennis-ball-racket-pose

# 4. เทรน base model (yolo11m-pose) — ดู hyperparameter เต็มในไฟล์ (ไม่มี CLI args)
python train_model/train_yolo.py

# 5. Fine-tune บนเฟรมจริงลูกค้า — hyperparameter ที่ใช้จริงอยู่ใน configs/train.yaml
python scripts/train.py --config configs/train.yaml
```

`train_model/merge_yolo_datasets.py` เป็น utility รวมหลาย YOLO dataset เป็น
multi-class เดียว (ใช้กรณีอยากรวม dataset เพิ่มนอกจาก ball+racket) ไม่ได้อยู่
ใน pipeline หลักด้านบน:
```bash
python train_model/merge_yolo_datasets.py \
    --dataset Tennis-Ball-detection-1 --dataset racket-4-yolo \
    --out tennis-ball-racket-yolo
```

### ⚠️ ไฟล์ไหนคือตัวจริง

มีสคริปต์ชื่อ `train_yolo.py` **สองไฟล์** ในโปรเจกต์ — ใช้ `train_model/train_yolo.py`
(yolo11m, ตามที่ README หลักอ้างอิง) **ไม่ใช่** `scripts/train_yolo.py` (yolo11n,
ดาวน์โหลด dataset คนละชุดจาก Roboflow) — ไฟล์หลังไม่ได้อยู่ใน package นี้
เพราะโค้ดเดิมมี API key ฝังอยู่ในไฟล์ (แก้เป็นอ่านจาก env var `ROBOFLOW_API_KEY`
แล้ว แต่ key เดิมอยู่ใน git history เก่า — **ควร rotate key นั้นทิ้ง**)

### Hardware + เหตุผลที่ตั้งค่าแต่ละตัวแบบนี้

- **`freeze=10`** — freeze backbone ส่วนใหญ่ตอน fine-tune กัน catastrophic
  forgetting บน ball detection (val mAP เดิมเสถียรอยู่แล้ว ไม่อยากให้เสียไป
  ตอนสอนเรื่อง racket เพิ่ม)
- **`fliplr=0.0`** — skeleton racket เป็น 4-point ไม่มี `flip_idx` ที่ชัดเจนพอ
  จะ mirror ซ้าย-ขวาให้ถูก จึงปิด horizontal flip augmentation ไว้ทั้งตอนเทรน
  base model และ fine-tune
- **`batch=4, workers=0` (บนเครื่อง local GTX 1660, 6GB VRAM)** — batch สูงกว่านี้
  OOM ทันที ส่วน `workers=0` แก้ `WinError 1114` (Windows multiprocessing DLL
  bug กับ PyTorch DataLoader)
- **`amp=False`** — GTX 1660 รุ่นนี้เจอบั๊ก mixed-precision training ทำ
  `loss=NaN` หรือ `mAP=0` บ่อย ปิด AMP แก้ปัญหานี้ได้
- **Fine-tune จริงต้องรันบน cloud GPU** — GTX 1660 (6GB) OOM ตั้งแต่ `batch=4`
  แล้วสำหรับ imgsz/batch ที่ใช้จริงใน `configs/train.yaml` (`imgsz=1024,
  batch=32`) รันสำเร็จบน **Lightning.ai L4 (23GB VRAM)**
- **ผลจริงหลัง fine-tune**: racket detection บนเฟรมจริงของลูกค้า (conf≥0.15,
  นอกชุด 12 ภาพที่ label เอง) ไปจาก **0/92 (0%) → 42/92 (45.7%)** — ดูตัวเลข
  เพิ่มเติมใน `docs/BENCHMARK.md`

## B. Stroke/hit classifier (RandomForest)

### ภาพรวม

```
build_training_data.py   session labels (dataset/labels/*.json) + วิดีโอ
                          → dataset/training/{hit_candidates,stroke_features}.csv
train_hit_classifier.py     hit_candidates.csv    → hit_classifier.pkl
train_stroke_classifier.py  stroke_features.csv   → stroke_classifier.pkl
```

```bash
python scripts/preprocess.py   # = train_model/build_training_data.py
python train_model/train_hit_classifier.py
python train_model/train_stroke_classifier.py
```

โมเดลทั้งสอง `classifier.py`/`hit_detection.py` โหลดแบบ lazy จาก **working
directory ปัจจุบัน** (`hit_classifier.pkl`, `stroke_classifier.pkl` ที่ root) —
ต้องรันคำสั่งข้างบนจาก root ของ package ไม่งั้นเซฟไฟล์ผิดที่

### `MIN_CLASS_SAMPLES = 5` — ทำไมต้องกัน class ที่ตัวอย่างน้อย

`train_model/train_stroke_classifier.py` กัน stroke type ที่มีตัวอย่างเทรนน้อย
กว่า 5 ไม่ให้ ML มีสิทธิ์ override ผล rule-based เลย เหตุผล (พบจริงตอนทดสอบ):
บน held-out test ของ set2 ตัวอย่าง BH จริง (dataset ตอนนั้นมีแค่ 3 ตัวอย่าง)
ถูก ML ทายผิดเป็น SV ด้วย confidence 0.63 (ผ่าน threshold ปกติสบาย ๆ) ทั้งที่
rule-based (geometry ล้วน ไม่พึ่งข้อมูลเทรน) ตอบถูก — โมเดลไม่มีทางรู้ตัวว่า
"ไม่มั่นใจ" เพราะไม่เคยเห็น class นั้นพอจะเรียนรู้ decision boundary เลย

### ⚠️ Data leakage — ทำไมต้องดู Leave-One-Clip-Out CV ไม่ใช่ random split

Random row split เสี่ยง data leakage เพราะแถวจากคลิปเดียวกัน (ท่าเดียวกัน
เฟรมใกล้กัน) หลุดไปอยู่ทั้ง train และ test พร้อมกันได้ — ตัวเลขที่ได้จะดูดีเกิน
จริง ทั้งสองสคริปต์เทรนจึงรัน **grouped (Leave-One-Clip-Out) cross-validation**
คู่กับ random split เสมอ (group ตามคอลัมน์ `clip_id`) — **เชื่อตัวเลข LOCO-CV
เท่านั้น** เวลาประเมิน accuracy จริง (ตัวเลขเปรียบเทียบดูได้ใน `docs/BENCHMARK.md`)
