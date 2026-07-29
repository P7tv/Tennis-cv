# Benchmark Report — Loeuf CV Pipeline

⚠️ **เอกสารนี้เขียนขึ้นเอง ไม่ได้มาจากการรัน `scripts/eval.py`** (ตัวรัน harness
จริง คือ `scripts/run_benchmark.py`) เพราะ harness ต้องการไฟล์ `labels/stroke_labels.csv`
ที่ยังไม่มี — ของที่มีจริงคือ label ต่อคลิปแบบ JSON (`dataset/labels/*.json`,
13 ไฟล์) คนละ format ต้องเขียน converter ก่อนถึงจะรันได้ ดูรายละเอียดใน
"ยังไม่ได้วัด" ด้านล่าง

## เป้าหมาย

Keyframe accuracy เทียบ ground-truth label (tolerance ±1 เฟรม) **เป้า TOR ≥ 0.80**
— **ยังไม่เคยวัดตัวเลขนี้จริง** (ดูเหตุผลด้านบน) นี่คือช่องว่างหลักของ
benchmark ชุดนี้

## ตัวเลขจริงที่วัดแล้ว (จาก git history + training data)

### Classifier accuracy — Leave-One-Clip-Out CV

Random row split เสี่ยง data leakage (แถวจากคลิปเดียวกันหลุดไปอยู่ทั้ง train/test)
— ตัวเลขที่เชื่อถือได้คือ grouped CV เท่านั้น (ดู `docs/FINETUNE_GUIDE.md`)

| โมเดล | Random split | LOCO-CV (เชื่อถือได้) |
|---|---|---|
| Stroke classifier (accuracy) | 92.59% | **80.30%** |
| Hit classifier (precision/recall) | 45% / 64% | **24% / 44%** |

ที่มา: commit `89840a6` ("window-shape hit-detection features + grouped
Leave-One-Clip-Out CV")

### Stroke classifier — per-class recall หลังเพิ่ม swing-window features

| Stroke type | ก่อน | หลัง |
|---|---|---|
| BH recall | 0.50 | **0.67** |
| VL precision/recall | 0.50 / 0.75 | **1.00 / 1.00** |

ที่มา: commit `806428b`

### Racket detection — ก่อน/หลัง fine-tune บนเฟรมจริงลูกค้า

| | ก่อน | หลัง |
|---|---|---|
| Detection rate (conf ≥ 0.15, นอกชุด label เอง 12 ภาพ, n=92) | 0/92 (0%) | **42/92 (45.7%)** |

ที่มา: `train_model/finetune_racket_pose.py:20-23` — racket detector เดิม
(เทรนจาก Roboflow `racket-4` เท่านั้น) แทบไม่ detect อะไรบนคลิปจริงเลย
(0.7% ของ hit candidates) เพราะมุมกล้อง/ระยะห่างต่างจากข้อมูลเทรนเดิมมาก

### Ball detection filter — แก้บั๊ก temporal continuity

| | ก่อน | หลัง |
|---|---|---|
| เฟรมที่เจอลูกบนคลิป validation (จาก 3920 เฟรม) | 391 | **469** (+~20%) |

ที่มา: commit `0b4d331` — cluster matcher เดิมเช็คแค่ระยะทางในอวกาศ ไม่จำกัด
ระยะห่างเวลา ทำให้ลูกที่ผ่านจุดเดิมซ้ำหลายรอบในคลิปยาวถูกรวมเป็น cluster เดียว
แล้วถูกตัดทิ้งเป็น false positive

### SAM 2 vs classical CV (background-subtraction tracking)

Head-to-head บนคลิปที่ classical CV หลุด track ซ้ำ ๆ (`805369423.086175.mp4`,
หน้าต่าง 150 เฟรม 280-430 ครอบคลุมช่วงหลุดจริง 325-361):

| | classical CV | SAM 2 |
|---|---|---|
| Coverage (frames 280-430) | 72.0% | **100%** |
| Mean core visibility | 0.592 | **0.818** |
| ช่วงที่ classical หลุดสนิท (325-361) | 0% (mean 0.00) | **100% (mean 0.87)** |

คลิกจุดเดียวทดสอบแยกอีกคลิป (`805110616.390966.mp4`, 149 เฟรม):
Coverage **139/149 (93%)**, mean visibility **0.77**, ความเร็ว ~1.5 วิ/เฟรม
(checkpoint tiny, CPU/MPS — ไม่มี CUDA) ที่มา: CHANGELOG `0.3.8`, `webui/README.md`

⚠️ SAM 2 ยังมีข้อจำกัด: video predictor โหลดทุกเฟรมเข้า memory พร้อมกัน (ไม่
streaming) คลิปยาวของลูกค้าจริง (70-120s) ต้องใช้ chunked processing ถึงจะรันได้
(ดู `webui/sam2_track.py`, `chunk_size` param)

### Coverage (classical CV, จาก `docs/PROBLEM_REGISTER.md`)

| ปัญหา | ก่อน | หลัง |
|---|---|---|
| Whole-frame detection ล้มเหลว (คนตัวเล็กในเฟรม) | 0% | 67% |
| คนเดียวถูกตัด track เป็น 7 ID | 34% | 74% |
| รวมทั้งหมวด detection/tracking บนคลิปทดสอบ | — | **88%** |

### Training data — ขนาด dataset ปัจจุบัน

| ไฟล์ | จำนวนแถว | รายละเอียด |
|---|---|---|
| `dataset/training/stroke_features.csv` | 132 | SV 54 · BH 26 · FH 21 · VL 20 · SL 11 (ไม่มี RS) |
| `dataset/training/hit_candidates.csv` | 585 | is_hit=1: 61 · is_hit=0: 524 · 12 คลิป (LOCO-CV fold) |

(ตรวจสอบตรงจากไฟล์จริงในโฟลเดอร์นี้ ณ วันที่เขียนเอกสารนี้)

### Test suite

`pytest tests/ -q` → **68 passed** (65 เดิม + 3 ที่เพิ่มเข้ามาสำหรับตรวจ
schema null-rule compliance — ดู `CHANGELOG.md`)

## ยังไม่ได้วัด (ไม่ใช่ "ผลไม่ดี" — คือ "ยังไม่มีตัวเลข")

- **Keyframe accuracy vs TOR ≥ 0.80** — เป้าหลักของ benchmark นี้ ยังไม่วัด
  ต้องเขียน converter จาก `dataset/labels/*.json` → `labels/stroke_labels.csv`
  ก่อน แล้วรัน `scripts/run_batch.py` + `scripts/eval.py` จริง
- **Inter-annotator agreement** — มี annotator เดียว (`atikan`) ในชุด label
  ปัจจุบัน ยังวัด agreement ระหว่างคนไม่ได้
- **Action spotting (Phase 2)** — ยังไม่มีข้อมูล session เต็มที่ label ครบ

## ห้ามอ้างเป็นผลลัพธ์

`runs/detect/**/results.csv` ในโปรเจกต์หลัก (mAP50 ~0.25 ในบาง run) — เป็น
run ที่รันแค่ 1-2 epoch แล้วหยุดกลางคัน (aborted) ไม่ใช่ผลเทรนจบจริง ตัวเลข
mAP ต่ำเพราะ epoch ต้น ๆ ไม่ใช่ตัวแทนของโมเดลที่ train เสร็จ — checkpoint
ที่ส่งมาด้วย (`checkpoints/model_v0.1.pt`) มาจาก run ที่ train ครบตาม
`configs/train.yaml` เท่านั้น
