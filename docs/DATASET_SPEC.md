# Dataset Specification

## โครงจริงที่ใช้งานอยู่ตอนนี้

โครงด้านล่าง ("โครงที่คาดหวังจากลูกค้า") เป็นแผนตอนเริ่มโปรเจกต์ — **ของจริง
ที่เกิดขึ้นต่างออกไป**: แทนที่จะเป็นคลิปแยกโฟลเดอร์ตาม stroke type (50 คลิป/
type) ข้อมูลจริงที่ได้มาเป็น **session เต็ม** (คลิปยาวมีหลาย stroke ผสมกัน)
เก็บเป็น batch `set2`/`set3`/`set4`:

```
dataset/
├── sessions/{set2,set3,set4}/*.mp4|.mov
│   └── *_stroke_labels_<annotator>_<date>.json   # 1 ไฟล์ต่อคลิป (ไม่ใช่ CSV รวม)
├── sessions_deferred/set4/                        # คลิปที่ label ทีหลัง
└── training/
    ├── hit_candidates.csv       # จาก train_model/build_training_data.py
    └── stroke_features.csv
```

รวม 13 คลิป + 13 label JSON (ดูสรุปใน `dataset/metadata.csv` ในแพ็กเกจ
handover — ไม่มีวิดีโอในแพ็กเกจนี้ ส่งแยก) — schema ของ label JSON ต่างจาก
`stroke_labels.csv` ที่คาดไว้เดิม: เก็บเป็น `{video_id, players[], strokes[]}`
ต่อไฟล์ แต่ละ stroke มี keyframe ที่ label ไว้ครบ (`backswing_peak_frame`,
`trophy_position_frame`, `impact_frame`, `follow_through_peak_frame` ฯลฯ)

Ball+racket detector ก็ไม่ได้เทรนตาม workflow เดิมด้านล่าง (extract_impact_frames
→ label → VM T4) — ใช้ Roboflow dataset (`Tennis-Ball-detection-1/`, `racket-4/`)
+ fine-tune บนเฟรมจริงลูกค้าแทน ดูรายละเอียดเต็มใน `docs/FINETUNE_GUIDE.md`
และผลลัพธ์ใน `docs/BENCHMARK.md`

---

## โครงที่คาดหวังจากลูกค้าตอนเริ่มโปรเจกต์ (แผนเดิม — ไม่ตรงกับของจริงข้างบน)

```
dataset/
├── FH/*.mp4   (50 คลิป)     ├── VL/*.mp4   (50 คลิป)
├── BH/*.mp4   (50 คลิป)     ├── SL/*.mp4   (50 คลิป)
├── SV/*.mp4   (50 คลิป)     └── RS/*.mp4   (50 คลิป)
sessions/          # Phase 2: วิดีโอ session เต็ม 5–10 อัน (ขอเพิ่ม)
labels/
├── stroke_labels.csv    # จาก labeling_tool.html — ดู LABEL_SPEC.md
└── session_labels.csv
```

- ชื่อโฟลเดอร์ = stroke type label (pipeline อ่านอัตโนมัติ)
- fps 30 หรือ 60 (ระบุใน label CSV ต่อคลิป) — 60fps จำเป็นสำหรับ KN fields
- มุมกล้อง: หลัง baseline (คลิปมุมเบี่ยง >30° จะถูก flag `side_view`)

## Dataset ที่ pipeline สร้างเอง

```
outputs/                  # JSON ต่อคลิป (run_batch.py) + batch_summary.json
dataset/ball_frames/      # เฟรม ±10 รอบ impact (extract_impact_frames.py)
                          # → สำหรับ label bounding box ลูก → train บน VM
```

## Workflow เมื่อได้ data

```bash
# 1. ประมวลผลทั้งชุด (CPU, ~1 ชม.)
python scripts/run_batch.py dataset/ -o outputs/ --workers 6 --height-cm <H>

# 2. benchmark กับเฉลยลูกค้า → docs/BENCHMARK.md
python scripts/run_benchmark.py --labels labels/stroke_labels.csv --outputs outputs/

# 3. ถ้า impact accuracy < 0.80 → เตรียม train ball detector
python scripts/extract_impact_frames.py dataset/ outputs/ -o dataset/ball_frames
# → label ลูก → train YOLOv8n/TrackNet บน VM (T4) → ผล detector เข้า loeuf_cv/ball.py
```
