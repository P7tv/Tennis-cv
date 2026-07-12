# Dataset Specification

## โครงที่คาดหวังจากลูกค้า (Client Dependency #1)

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
