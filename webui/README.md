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

## Track 2 — SAM 2 click-to-track (`webui/sam2_track.py`) — prototype ทดสอบแล้ว

แก้ปัญหาหมวด B โดยตรง ("แยกผู้เล่นจากคนป้อนบอลไม่ได้" + "คนเดียวถูกตัด track
เป็นหลายท่อน") ด้วยการให้คนคลิกจุดบนผู้เล่นเป้าหมาย 1 จุด แล้วให้ SAM 2
propagate mask ผ่านทั้งคลิป มาแทนที่ `detect_blobs()` + `match_tracks()` +
`select_candidate_tracks()` ทั้งชุดใน `multi_person.py`

⚠️ **package ที่ติดตั้งจริงไม่ใช่ของ Meta โดยตรง** — `pip show sam2` (2026-07-11)
รายงาน `Home-page: https://github.com/JinsuaFeito-dev/segment-anything-2`
(fork/mirror โดยบุคคลที่สาม "Jorge Insua" ไม่ใช่ facebookresearch) — เอา
LICENSE Apache 2.0 มาด้วยเลย redistribution น่าจะยัง compliant แต่ build ที่ได้
**ขาด compiled `_C` extension** (`cannot import name '_C' from 'sam2'` ทุกครั้ง
ที่รัน) ก่อนใช้กับงานที่ส่งลูกค้าจริงควรเปลี่ยนไปติดตั้งจาก
`github.com/facebookresearch/sam2` ตรง ๆ แทน `pip install sam2`

⚠️ **ข้อจำกัดสำคัญที่เพิ่งพบ (2026-07-11)**: SAM 2 video predictor โหลดทุกเฟรม
เข้า device memory พร้อมกันทั้งคลิป (ไม่ streaming) — บนเครื่องนี้ (Mac, MPS,
ไม่มี CUDA) คลิปเกิน ~300 เฟรม (~10s) เริ่มมีปัญหา: 780 เฟรม crash ตรง ๆ
(`RuntimeError: Invalid buffer size: 9.14 GiB`) ส่วน 650 เฟรม **ไม่ crash แต่
mask พังแบบเงียบ ๆ** (bbox กลายเป็นเต็มเฟรมทุกเฟรม coverage ดูเหมือนดี 100%
ทั้งที่ track ไม่ได้จริง — จับได้เพราะเช็ค box size ก่อนเชื่อตัวเลข ดู
CHANGELOG 0.3.8) คลิปจริงของลูกค้ายาว 70-120s (~2000-3500 เฟรม) จึง **ใช้
วิธีนี้ตรง ๆ กับคลิปเต็มไม่ได้บนเครื่องนี้** ต้องมี chunked processing
(propagate เป็นช่วง ๆ แล้ว stitch) หรือรันบนเครื่อง GPU ที่มี memory เหลือพอ

### ผลทดสอบจริง (ไม่ใช่แค่เขียนโค้ด — รันจริงบน `805110616.390966.mp4` ตัด 5 วิ)

คลิกจุดเดียว (1 จุด, เฟรม 10, ตำแหน่งสะโพก near player) → SAM 2 propagate:

| | ผล |
|---|---|
| Coverage | **139/149 เฟรม (93%)** |
| Mean visibility (หลังป้อนเข้า MediaPipe) | **0.77** |
| ความถูกต้อง | ตรวจด้วยตา — bbox แม่นยำติดตัวผู้เล่นเป้าหมายตลอด ไม่หลุดไปเกาะ #52 หรือคนป้อนบอลเลย แม้มี 3-4 คนในเฟรมพร้อมกัน |
| ความเร็ว (checkpoint `tiny`, CPU/MPS บน Mac ไม่มี CUDA) | ~1.5 วิ/เฟรม (~215 วิ สำหรับคลิป 5 วิ/149 เฟรม) |

**เทียบกับ classical CV เดิม** (background subtraction + centroid tracking +
clustering ที่แก้บั๊กกันทั้งวัน 0.3.0-0.3.2): SAM 2 ได้ coverage ใกล้เคียงหรือดีกว่า
**ด้วยโค้ด tracking แค่ ~150 บรรทัด ไม่ต้องมี clustering/stitching logic เลย**
เพราะ SAM 2 รักษา identity ของวัตถุที่คลิกไว้ให้แต่ต้น ไม่ต้องเดาจาก
พฤติกรรม blob ทีหลัง

### ตั้งค่า (venv แยกจาก environment หลัก)

SAM 2 ดึง torch เวอร์ชันเฉพาะมาด้วย ซึ่งอาจชนกับ environment ของ MediaPipe —
แยก venv ให้ชัดเจน:

```bash
python3 -m venv webui/.venv-sam2
source webui/.venv-sam2/bin/activate   # Windows: webui\.venv-sam2\Scripts\activate
pip install -r webui/requirements-sam2.txt

mkdir -p webui/checkpoints
curl -L -o webui/checkpoints/sam2.1_hiera_tiny.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
```

ใช้จริง (ยังไม่มีหน้า UI ให้คลิก — เรียกจาก Python โดยตรงตอนนี้):

```python
from webui.sam2_track import track_player_with_sam2, extract_pose_from_sam2_track, ClickPrompt
from loeuf_cv.config import PipelineConfig

prompts = [ClickPrompt(frame_idx=0, x=314, y=255)]  # คลิกจุดบนตัวผู้เล่นที่เฟรมแรก
bboxes = track_player_with_sam2("clip.mp4", prompts, "webui/checkpoints/sam2.1_hiera_tiny.pt")
ts = extract_pose_from_sam2_track("clip.mp4", bboxes, PipelineConfig())
```

⚠️ `webui/.venv-sam2/` (~790MB) และ `webui/checkpoints/` (~150MB) เป็นไฟล์
local ที่สร้าง/ดาวน์โหลดใหม่ได้เสมอจาก 2 คำสั่งข้างบน — **ไม่ต้อง commit/copy
ทั้งโฟลเดอร์** ตอนย้ายไปเครื่องอื่น (เช่น Windows ที่มี GPU) แค่รันคำสั่ง
setup ใหม่ที่นั่น

### ย้ายไปรันบนเครื่องมี GPU (Windows/CUDA)

โค้ดใน `sam2_track.py` เลือก device อัตโนมัติ (`cuda` > `mps` > `cpu` — ดู
`_pick_device()`) ไม่ต้องแก้อะไรเลย แค่ setup venv ใหม่บนเครื่องนั้น (ขั้นตอน
เดียวกับข้างบน) แล้ว torch จะเจอ CUDA เอง — คาดว่าเร็วขึ้นมาก (checkpoint
`tiny` บน GPU จริงมักได้หลัก 10-40+ fps เทียบกับ ~0.7 fps ที่วัดได้บน CPU/MPS
ของเครื่องนี้)

### เทียบกับ classical CV บนคลิปที่มีบั๊กจริง (0.3.8, 2026-07-11)

ทดสอบ head-to-head บน `805369423.086175.mp4` (คลิปที่ classical CV หลุด
track ซ้ำ ๆ ตอนผู้เล่นกลับไปยืนท่าเดิม — ดู CHANGELOG 0.3.6) หน้าต่าง 150
เฟรม (280-430 เดิม) ที่ครอบคลุมช่วงหลุดจริง (325-361):

| | classical CV | SAM 2 |
|---|---|---|
| Coverage | 72.0% | **100%** |
| Mean core visibility | 0.592 | **0.818** |
| ช่วงที่ classical หลุดสนิท (325-361) | 0% (mean 0.00) | **100% (mean 0.87)** |

ตรวจ box size (33k-231k px² จาก full frame 515k px²) + ตรวจด้วยตา 5 จุด
รวมจุดกลางช่วงหลุด ก่อนเชื่อตัวเลข — bbox แม่นยำติดผู้เล่นถูกคนตลอด แก้
ปัญหาที่ classical CV แก้ไม่ได้ (0.3.6/0.3.7) ได้จริง แต่ยังจำกัดแค่หน้าต่าง
150 เฟรมเพราะข้อจำกัด memory ข้างบน

### ยังไม่ทำ

- หน้า UI ให้คลิกจริงในเบราว์เซอร์ (ตอนนี้ต้องระบุพิกัด `(x,y)` เป็นโค้ด)
- **Chunked processing สำหรับคลิปยาวเต็ม** (72-120s) — propagate เป็นช่วง
  ๆ แล้ว stitch แทนการโยนคลิปเต็มเข้า video predictor ทีเดียว (ดูข้อจำกัด
  memory ข้างบน — นี่คือ blocker หลักก่อนใช้กับคลิปจริงของลูกค้าได้)
- ทดสอบ multi-object (คลิกเลือกได้มากกว่า 1 คนพร้อมกัน — SAM 2 รองรับผ่าน
  `obj_id` ที่ต่างกัน แต่ยังไม่ได้ลองจริง)
- เปลี่ยนไปติดตั้งจาก `facebookresearch/sam2` ตรง ๆ แทน PyPI fork (ดูคำเตือนข้างบน)
- เทียบ coverage/ID-switch กับ classical CV แบบตัวเลขคู่กันชัด ๆ (2.4 ใน action plan)
