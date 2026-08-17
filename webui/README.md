# Loeuf CV — Inference Test UI

เครื่องมือทดสอบภายในสำหรับ inspect ผล multi-person tracking + pose extraction
บนคลิปจริง โดยไม่ต้องรันสคริปต์ทีละคำสั่งด้วยมือ **ไม่ใช่ deliverable ที่ส่งลูกค้า**
— production pipeline ยังรันแบบ headless/batch ผ่าน `scripts/` ตามเดิม

## รัน

```bash
pip install -r webui/requirements.txt
streamlit run webui/app.py
```

เปิด `http://localhost:8501`

## ทำอะไรได้ (Phase 1 — MVP)

1. **Upload คลิป** + ตั้งค่า `max_players` / `model_complexity` / ส่วนสูง / มือถนัด
2. **Track Review** — ตาราง coverage/visibility ต่อ track + **overlay video** (skeleton สี
   ต่างกันต่อ track, จุดจาง = confidence ต่ำ) เพื่อตรวจด้วยตาว่า track ถูกคนจริงไหม
   ก่อนเชื่อตัวเลข — วิธีนี้เองที่จับบั๊ก wrong-identity ได้จริงระหว่าง debug (ดู
   `CHANGELOG.md` 0.3.0-0.3.2)
3. **กำหนดผู้เล่นเป้าหมาย** — เลือกได้ว่า track ไหนคือ Player 1/2 หรือ "ไม่ใช่ผู้เล่น"
   (คนป้อนบอล/คนยืนรอ) แยกออกจากการ detect อัตโนมัติที่บอกได้แค่ "มีคนขยับตรงไหน"

## ยังไม่ทำ (แผน Phase ถัดไป — ดู `docs/PROBLEM_REGISTER.md` หมวด B)

- **Phase 2**: coverage diagnostics แบบ timeline (บอกว่าเฟรมที่หายเพราะ blob ไม่เจอ
  vs pose ล้มเหลว), visibility ต่อ landmark
- **Phase 3**: ฝัง `labels/court_click_tool.html` เป็นหน้า calibration + full
  metrics/JSON viewer (17-layer schema output พร้อม highlight `low_confidence`/
  `not_detectable` fields)
- **Phase 4**: ต่อ SAM 2 click-to-track (ด้านล่าง) เข้า UI จริง — ตอนนี้มีแค่
  backend script ทดสอบแล้วผ่าน ยังไม่มีหน้าให้คลิกในเบราว์เซอร์

## Reuse

Engine ทั้งหมดเรียกจาก `loeuf_cv/` ตรง ๆ ไม่มี logic ใหม่นอกจาก UI glue —
`webui/overlay.py` คือฟังก์ชัน render skeleton ที่ดึงมาจาก diagnostic script
ที่ใช้ debug ตลอด session การแก้ multi-person tracking (2026-07-10)

---
