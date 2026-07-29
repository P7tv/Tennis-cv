# Model Card — Loeuf CV Measurement Engine

## สถาปัตยกรรม

**Rule-based measurement engine บน pretrained pose model + custom
fine-tuned YOLO-pose detector สำหรับ ball/racket** — measurement engine
(keyframes/metrics/classifier) ยังเป็น deterministic rules เหมือนเดิม
แต่ตอนนี้มี custom model จริงแล้วสำหรับ ball+racket detection (ไม่ใช่แค่
interface ว่างเหมือนก่อนหน้านี้)

```
video → YOLO-pose (ball+racket, fine-tuned) + BlazePose (pretrained)
     → resample 60fps → smoothing → camera calibration (per-clip)
     → keyframes B1–B6 → metrics C/KN/SS + derived D
     → confidence/visibility → JSON 17 layers
Phase 2: + action spotting (motion energy) + rule classifier + AGG/TRE/PAT/SUM
```

| Component | Type | Version |
|---|---|---|
| Pose (body) | MediaPipe BlazePose (pretrained, ไม่ fine-tune) | ดู `pose_model_version` ใน output |
| Ball+racket detector | YOLO11-pose, fine-tuned บนเฟรมลูกค้าจริง | `checkpoints/model_v0.1.pt` — ดู `docs/FINETUNE_GUIDE.md` |
| Keyframes / metrics / spotting | Deterministic rules | `cv_model_version` |
| Stroke/hit classifier | RandomForest (เทรนจาก session labels) + rule-based fallback | `hit_classifier.pkl`, `stroke_classifier.pkl` |

## ผลกระทบต่อ handover structure

โครง `checkpoints/ + train.py + FINETUNE_GUIDE` ใน schema doc ใช้กับ
ball+racket detector (`checkpoints/model_v0.1.pt`, `scripts/train.py`,
`docs/FINETUNE_GUIDE.md`) และ classifier ทั้งสองตัว (`hit_classifier.pkl`,
`stroke_classifier.pkl`) — ส่วน measurement engine (C/KN/SS/D) ยังไม่มี
checkpoint เพราะเป็น rules: "จูน" = แก้ค่าใน `loeuf_cv/config.py`
(ทุก threshold รวมที่เดียว) ไม่ใช่ retrain

ข้อดีที่ได้แลกมา: deterministic (คลิปเดิม → ค่าเดิมทุกครั้ง),
อธิบายได้ทุกตัวเลข, ไม่มี training data bias, รันบน CPU

## Known limitations

- มุมหลังกล้องเดียว: `depth_estimated` ทั้ง block = approximation
  (cap `low_confidence` เสมอ)
- ความเร็ว (KN) จากคลิป 30fps: resample แล้วแต่ fidelity จำกัดที่ต้นทาง
- ไม่มี court calibration: BL2 `ball_speed_kmh`/BL3 `trajectory_clearance_cm`
  ยังเป็น `null` เสมอ (key มีครบตาม schema แต่ยังคำนวณค่าจริงไม่ได้)
- Serve-only fields (`server_position`/`target_box_correct`/`is_fault`/
  `trophy_position` keyframe) ส่ง key ครบตาม schema แล้ว แต่ยังไม่ implement
  logic จริง (ค่าเป็น `null`/`false` เสมอ)
- Classifier ไม่จำแนก RS (เชิง kinematics คือ FH/BH — ต้องใช้บริบท)
