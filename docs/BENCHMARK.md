# Benchmark Report — Loeuf CV Pipeline

⚠️ **เอกสารนี้เขียนขึ้นเอง** รวบรวมตัวเลขที่วัดไว้แล้วจากหลายที่ (git history +
training data) — ไม่ได้ generate จาก harness

## 👉 Keyframe accuracy (เกณฑ์รับงาน TOR ≥ 0.80)

**วัดจริงแล้ว → ดู [`BENCHMARK_KEYFRAME.md`](BENCHMARK_KEYFRAME.md)**
(machine-readable: `benchmark_keyframe_results.json`)

| ตัวเลข | ค่า | vs เป้า 0.80 |
|---|---|---|
| **Mode A end-to-end** (จริงตามที่ลูกค้าได้) | **0.148** | ❌ FAIL |
| Mode A matched-only (เฉพาะ stroke ที่ detect เจอ) | 0.244 | ❌ |
| **Mode B oracle-impact** (เพดานของ keyframe logic) | **0.761** | ❌ (เกือบถึง) |

Detection: recall **0.606** · precision **0.101** (TP 86 / FN 56 / FP 762)

**คอขวดอยู่ที่ hit detection ไม่ใช่ keyframe logic** — Mode B (ป้อน GT impact) ได้
0.761 แต่ end-to-end เหลือ 0.148 เพราะ `0.606 (หา stroke เจอ) × 0.29 (เจอแล้วตรง
±1 เฟรม) = 0.17` ต้องแก้ทั้งสองชั้น

วัดด้วย `scripts/run_keyframe_benchmark.py` บน ground truth จริงของลูกค้า
(13 คลิป / 142 stroke) เทียบ pipeline production (`build_loeuf_schema`) —
รายงานนั้นแยก 2 โหมด: **end-to-end** (จริงตามที่ลูกค้าได้ รวม error ของ hit
detection) กับ **oracle-impact** (ป้อน GT impact เข้าไป = เพดานของ keyframe
logic ล้วน ๆ) เพื่อให้รู้ว่าควรไปแก้ hit detection หรือ keyframe logic ก่อน

หมายเหตุ: `scripts/run_benchmark.py` (harness เดิม) ยังใช้กับข้อมูลชุดนี้ไม่ได้
เพราะออกแบบมาสำหรับคลิป single-stroke + `labels/stroke_labels.csv` ที่ไม่เคยมี
— ของจริงเป็น session ยาวหลาย stroke ต่อคลิป label เป็น JSON

## ตัวเลขจริงที่วัดแล้ว (จาก git history + training data)

### Classifier accuracy — ต้องใช้ Leave-One-**Person**-Out

Random row split เสี่ยง data leakage (แถวจากคลิปเดียวกันหลุดไปอยู่ทั้ง train/test)
— แต่ **leave-one-clip-out ยังไม่พอ** เพราะผู้เล่นคนเดียวกันอยู่หลายคลิป
(earth 4 คลิป · poom 3 คลิป · moo-grey 2 · navy 2) → group ด้วยคลิปยังมี leakage
ระดับบุคคล ตัวเลขที่เชื่อถือได้คือ **LOPO**

| โมเดล | Random split | LOCO-CV | **LOPO-CV (เชื่อถือได้)** |
|---|---|---|---|
| Stroke classifier (accuracy) | 92.59% | 80.30% | **72.7%** |
| Hit classifier (precision/recall) | 45% / 64% | 24% / 44% | — (ยังไม่วัด) |

ที่มา LOCO/random: commit `89840a6` · ที่มา LOPO: วัดใหม่ 2026-07-30 บน
`dataset/training/stroke_features.csv` (n=132) โดย map clip → ผู้เล่นจาก
`StrokeLabel.clip_id`

⚠️ **ข้อจำกัดของ dataset ที่กระทบทุกตัวเลขข้างบน** — ประเภท stroke เกือบจะ
collinear กับตัวบุคคล:

| ท่า | n | มาจากกี่คน |
|---|---|---|
| SL | 11 | **1 คน** (earth) |
| FH | 21 | 2 คน (moo-grey 20, poom 1) |
| VL | 20 | 2 คน |
| BH | 26 | 3 คน |
| SV | 64 | 6 คน ✅ |

SL มาจากคนเดียว → LOPO ให้ 0.00 โดยโครงสร้าง (ทุกโมเดลที่ทดสอบ) และยังพิสูจน์
ไม่ได้ว่าโมเดลเรียน "ท่า" หรือเรียน "ตัวบุคคล" — **การเก็บ label เพิ่มโดยกระจาย
ท่าข้ามคน สำคัญกว่าการเปลี่ยนโมเดล**

เทียบโมเดลอื่นบน feature ชุดเดิม (LOPO): RF ลึกกว่า 0.788 · HistGradientBoosting
0.780 · ExtraTrees 0.735 · **Stacking 0.644** (แย่ลง — n=132 ไม่พอเรียน
meta-model) → เพดานคือข้อมูล ไม่ใช่ model class

หมายเหตุเชิงระเบียบวิธี: 0.788 เป็นค่าที่ได้จากการ *เลือกโมเดลดีสุดจาก 11 ตัว
โดยดูคะแนน LOPO เดียวกัน* จึงมี selection bias — กำไรจริงน่าจะน้อยกว่า +6 จุด
ถ้าจะอ้างต้องทำ nested CV

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

`pytest tests/ -q` → **86 passed** (65 เดิม + 3 schema null-rule + 12 keyframe
benchmark + 6 backswing offset)

<!-- ค่าเดิมตอนเขียนเอกสารรอบแรก: 68 passed (65 + 3 schema null-rule
compliance — ดู `CHANGELOG.md`) -->

## ยังไม่ได้วัด (ไม่ใช่ "ผลไม่ดี" — คือ "ยังไม่มีตัวเลข")

- ~~**Keyframe accuracy vs TOR ≥ 0.80**~~ — **วัดแล้ว** ดู
  [`BENCHMARK_KEYFRAME.md`](BENCHMARK_KEYFRAME.md)
- **Inter-annotator agreement** — มี annotator เดียว (`atikan`) ในชุด label
  ปัจจุบัน ยังวัด agreement ระหว่างคนไม่ได้ (ต้องให้คนที่ 2 label ทับบางคลิป)
- **BL2/BL3 (ball speed, trajectory clearance)** — ยังเป็น `null` เสมอ ต้องมี
  court calibration ก่อน (ของที่มีแล้ว: `loeuf_cv/court_calibration.py`,
  `scripts/solve_camera_pose.py`)
- **Serve fields BL7/BL8/BL10** — key ครบตาม schema แล้วแต่ค่าเป็น `null`
  ยังไม่ implement logic จริง
- **Action spotting (Phase 2)** — ยังไม่มีข้อมูล session เต็มที่ label ครบ

## ห้ามอ้างเป็นผลลัพธ์

`runs/detect/**/results.csv` ในโปรเจกต์หลัก (mAP50 ~0.25 ในบาง run) — เป็น
run ที่รันแค่ 1-2 epoch แล้วหยุดกลางคัน (aborted) ไม่ใช่ผลเทรนจบจริง ตัวเลข
mAP ต่ำเพราะ epoch ต้น ๆ ไม่ใช่ตัวแทนของโมเดลที่ train เสร็จ — checkpoint
ที่ส่งมาด้วย (`checkpoints/model_v0.1.pt`) มาจาก run ที่ train ครบตาม
`configs/train.yaml` เท่านั้น
