# Model Card — Loeuf CV Measurement Engine

## สถาปัตยกรรม

**Rule-based measurement engine บน pretrained pose model** — ไม่มี
custom model training ใน Phase 1 (คำตอบ Feasibility Q1/Q2: ทุก core
field ทำได้ด้วย MediaPipe + rules)

```
video → BlazePose (pretrained) → resample 60fps → smoothing
     → camera calibration (per-clip) → keyframes B1–B5
     → metrics C/KN/SS + derived D → confidence/visibility → JSON 17 layers
Phase 2: + action spotting (motion energy) + rule classifier + AGG/TRE/PAT/SUM
```

| Component | Type | Version |
|---|---|---|
| Pose | MediaPipe BlazePose (pretrained, ไม่ fine-tune) | ดู `pose_model_version` ใน output |
| Keyframes / metrics / spotting / classifier | Deterministic rules | `cv_model_version` |
| Ball detector | ยังไม่มี — interface พร้อม (`ball.py`) | จะ train เมื่อ benchmark ชี้ว่าจำเป็น |

## ผลกระทบต่อ handover structure

โครง `checkpoints/ + train.py + FINETUNE_GUIDE` ใน schema doc ใช้ได้กับ
ball detector (เมื่อ train) เท่านั้น — ตัว measurement engine ไม่มี
checkpoint เพราะเป็น rules: "จูน" = แก้ค่าใน `loeuf_cv/config.py`
(ทุก threshold รวมที่เดียว) ไม่ใช่ retrain

ข้อดีที่ได้แลกมา: deterministic (คลิปเดิม → ค่าเดิมทุกครั้ง),
อธิบายได้ทุกตัวเลข, ไม่มี training data bias, รันบน CPU

## Known limitations

- มุมหลังกล้องเดียว: `depth_estimated` ทั้ง block = approximation
  (cap `low_confidence` เสมอ)
- ความเร็ว (KN) จากคลิป 30fps: resample แล้วแต่ fidelity จำกัดที่ต้นทาง
- ไม่มี court calibration: BL block, โซน "net" ยังไม่ทำงาน
- Classifier ไม่จำแนก RS (เชิง kinematics คือ FH/BH — ต้องใช้บริบท)
