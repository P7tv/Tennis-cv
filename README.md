# Loeuf CV Pipeline — Measurement Engine

ระบบ CV สกัดข้อมูลชีวกลศาสตร์จากวิดีโอเทนนิส (มุมหลัง baseline)
Output ตาม **cv_schema_table_loeuf** — 17-layer JSON schema
(Phase 1: 13 blocks · Phase 2: +AGG/TRE/PAT/SUM)

## โครงสร้าง

```
loeuf_cv/
├── config.py          # thresholds + TBC constants (Phase 0 recalibrate)
├── pose_extractor.py  # video → BlazePose → timeseries + frame stats
├── resample.py        # M4/M5: fps < 60 → resample เป็น 60fps
├── calibration.py     # per-clip camera normalization (roll/scale/zero-ref/deviation)
├── smoothing.py       # gap-fill occlusion (NaN runs) + Savitzky-Golay
├── occlusion_fill.py  # kinematic gap-fill สำหรับข้อมือ/ข้อเท้าที่ visibility ต่ำต่อเนื่อง (ดู README ล่าง)
├── kinematics.py      # CoM (สูตร segment weighting), rotation deg, มุมข้อต่อ
├── video_quality.py   # 022-VQ pre-checks
├── keyframes.py       # 023-B: B1–B5 + sub-frame impact
├── metrics.py         # 031-035: C1–C41, KN, SS, BL, D + flat VF
├── confidence.py      # 024-A stroke_root (A9 priority logic) + 042-CF
├── visualization.py   # 043-VZ (wrist_path REQUIRED)
├── schema.py          # ประกอบ 17-layer output
├── action_spotting.py # Phase 2: motion-energy stroke spotting
├── multi_person.py    # แยกผู้เล่นหลายคน: bg-subtraction blob track + crop-zoom
├── court_calibration.py  # auto-height จากเส้นคอร์ท (experimental — ดู README ล่าง)
├── court_model.py     # model-based court registration (pinhole camera + full court model — research/experimental)
├── vanishing_point.py # focal length จาก vanishing point (ทดสอบแล้ว — ใช้ไม่ได้กับ setup กล้องมาตรฐาน ดู README)
├── classifier.py      # Phase 2: hierarchical rule classifier (ไม่คืน RS)
├── aggregation.py     # Phase 2: 051-054 AGG/TRE/PAT/SUM
├── ball.py            # ball detector interface + impact fusion (รอ VM training)
├── benchmark.py       # eval vs label CSV → accuracy/recall/agreement
├── session_pipeline.py
├── stroke_pipeline.py
├── gas_client.py      # POST → GAS (retry + backoff)
└── synthetic.py       # synthetic landmarks + stroke variants สำหรับ test/demo
scripts/   run_stroke.py · run_session.py · run_batch.py ·
           run_benchmark.py · extract_impact_frames.py · run_multi_person.py ·
           solve_camera_pose.py (จุดที่คลิก → CameraPose + overlay)
labels/    labeling_tool.html (browser tool สำหรับลูกค้า) + CSV templates
           court_click_tool.html (คลิกจุดคอร์ท → JSON ให้ solve_camera_pose.py)
docs/      D0.2_measurement_definition.md · MODEL_CARD.md · DATASET_SPEC.md
           LABEL_SPEC.md · CLIENT_LABEL_GUIDE.md · PROBLEM_REGISTER.md
notebooks/ loeuf_cv_pipeline.ipynb   # เดโมครบ executed แล้ว
tests/     test_pipeline_logic.py    # 57 tests
webui/     app.py (Streamlit) — เครื่องมือทดสอบภายใน ไม่ใช่ deliverable
           (ดู README ล่าง)
```

## Inference Test UI (`webui/`) — เครื่องมือทดสอบภายใน

```bash
pip install -r webui/requirements.txt
streamlit run webui/app.py
```

Upload คลิป → รัน multi-person tracking → ดู **overlay skeleton** ต่อ track
+ กำหนดว่า track ไหนคือผู้เล่นเป้าหมาย (แยกจากคนป้อนบอล/คนยืนรอ) — แทนที่การรัน
สคริปต์ทีละคำสั่งด้วยมือ รายละเอียด/roadmap ดู `webui/README.md`

## ใช้งาน

```bash
pip install -r requirements.txt

# Phase 1: คลิป single stroke
python scripts/run_stroke.py clip.mp4 --stroke FH --height-cm 172 -o out.json

# Phase 2: session เต็ม (strokes + aggregation)
python scripts/run_session.py match.mp4 --height-cm 172 -o session.json

# tests (ไม่ต้องมีวิดีโอ)
python -m pytest tests/ -v
```

## Decisions ที่ implement ไว้ (จุดที่ schema doc กำกวม/ขัดกันเอง)

| ประเด็น | ที่เลือก | สถานะ |
|---|---|---|
| px→cm calibration (C1) | ส่วนสูงผู้เล่นเป็น input (`--height-cm`); ไม่ระบุ → 170cm + cm fields ติด `low_confidence` | ❓ ถามลูกค้า: ใครส่ง player height |
| Per-field confidence | ยังไม่ส่ง (นิยามบอก per-field แต่ sample JSON ไม่มี block) | ❓ รอ confirm |
| Precision | ปัด 2 ตำแหน่ง (ตามตาราง type) | ❓ ขัด note "full precision" |
| C19/C20 recovery | `ready(B5) − follow_through(B4)` — label ใน doc สลับ | ❓ confirm intent |
| C31 wrist angle | ใช้ hand index keypoint แทน racket | ❓ ยอมรับ fallback ไหม |
| KN7 racket speed | fallback wrist-based (ตาม doc "ใช้ ≈ wrist_path") + `low_confidence` | ✅ ตาม doc |
| KN3 lag sign | `t(peak shoulder) − t(peak hip)` ให้ (+) = hip นำ ตาม intent | ❓ สูตรใน doc กลับด้าน |
| Rotation deg | world x-z เทียบ baseline; sign + BH flip validate Phase 0 | ⚠️ TBC |
| C37/C38 weight load | CoM position ระหว่างเท้า (approximation) + `low_confidence` | ⚠️ doc mark TBC เอง |
| C39/C40, KN5/KN6, BL | ต้องการ side view / racket / ball detector → `null` + `not_detectable` | ✅ ตาม camera_dep |

Threshold ทุกตัวที่ doc mark **"TBC after CV test"** รวมอยู่ใน
`PipelineConfig` — จูนที่เดียวจบ

## Multi-person clips (full-court / session footage)

ถ้าคลิปมีมากกว่า 1 คนในเฟรม (เช่น session footage เห็นทั้งสองฝั่งคอร์ท)
ใช้ `multi_person.py` แทน `extract_pose()` ตรง ๆ:

```bash
python scripts/run_multi_person.py session.mp4 --max-players 2
python scripts/run_multi_person.py session.mp4 --role near --stroke FH -o out.json
```

**วิธีทำงาน**: background-subtraction หา blob คนที่ขยับ (ไม่ต้องมี
person-detector model แยก เพราะกล้องเทนนิสอยู่นิ่ง) → track ID ต่อเนื่อง
ด้วย nearest-centroid matching → crop+upscale รอบแต่ละคนก่อนส่งเข้า
MediaPipe Pose เดิม → ได้ `PoseTimeseries` แยกต่อคน (`.role` = near/far/other
ตามขนาด apparent) ใช้กับ `StrokePipeline`/`SessionPipeline` เดิมได้ตรง ๆ

**Validated บนคลิปจริง (2026-07-09)**: คลิปที่ whole-frame detection ล้มเหลว
เกือบทั้งหมด (0.1% ของเฟรม, ที่เหลือเป็น false positive) หลังผ่าน
multi-person module ได้ track ที่ใช้งานได้จริง (mean visibility 0.78,
tracked 28% ของคลิป) — ดีขึ้นมากแต่ยังมีเพดานจากความละเอียดต้นทาง
(960x540, คนไกลสุด ~15% ของความสูงเฟรม) แนะนำ 1920x1080 ถ้าเป็นไปได้

**บั๊ก wrong-identity ที่เจอและแก้แล้ว (2026-07-10)** — ผู้ใช้จับได้จาก
การดูวิดีโอ overlay skeleton จริง: role "far" เคย track คนยืนนิ่งข้าง
รั้ว (ผู้ชม) แทนคู่แข่งตัวจริง สาเหตุคือ track selection เดิมเลือกจาก
"track ไหนยาวสุด" อย่างเดียว ซึ่งคนยืนนิ่ง track ต่อเนื่องได้ง่ายกว่า
ผู้เล่นจริงที่ track หลุดบ่อย (ถูกบัง/ไกล) จนกระจายเป็นหลาย track สั้น ๆ
แก้แล้วโดยเพิ่ม `select_candidate_tracks()` (`multi_person.py`) ที่กรอง
3 ชั้น: (1) ต้องมีการขยับจริง (กันคนยืนนิ่ง) (2) ไม่ใช่คนเดียวกับ track
ที่เลือกไปแล้ว (ตำแหน่งใกล้กัน**และ**ไม่เคย active พร้อมกันเลย — กัน
กรณี track เดิมหลุดแล้วได้ ID ใหม่) แต่ (3) ยังคงคนสองคนที่ยืนใกล้กันไว้
ถ้า track ได้พร้อมกันจริง (temporal overlap สูง) — สองเงื่อนไขนี้ต้อง
เช็คคู่กันเสมอ ใช้ระยะห่างอย่างเดียวไม่พอ (validate แล้วว่าจะเผลอตัดคน
ที่ยืนใกล้กันแต่เป็นคนละคนทิ้งไป) ดู CHANGELOG 0.3.0 สำหรับรายละเอียด
การ debug ทั้งหมด **แนะนำ**: render overlay skeleton ดูก่อนเชื่อผล
multi-person เสมอ (บั๊กนี้ไม่มีทางเจอจากตัวเลขสถิติอย่างเดียว)

**Track stitching — กู้ coverage ที่เสียไปคืน (2026-07-10 บ่าย)** —
คำถามต่อยอด "ทำไม skeleton หลุดบางช่วง" นำไปสู่การเจอว่าคนคนเดียวมักถูก
ตัดเป็นหลาย track ID (blob หลุดสั้น ๆ ซ้ำ ๆ ระหว่างแรลลี่เกิน
`MAX_TRACK_GAP_FRAMES`) แล้ว `select_candidate_tracks()` เดิม (0.3.0)
เลือกเก็บแค่ท่อนที่ยาวสุด ทิ้งท่อนอื่นที่เป็นคนเดียวกันไปเฉย ๆ ทั้งที่
logic ระบุถูกอยู่แล้วว่าเป็นคนเดียวกัน — แก้โดยเปลี่ยนจาก "เลือก 1 track
ต่อคน" เป็น "จัดกลุ่ม fragment ของคนเดียวกันด้วย union-find แล้ว stitch
เข้าเป็น PoseTimeseries เดียวต่อเนื่อง" (เกณฑ์จัดกลุ่มเหมือนเดิมที่
validate ไว้แล้ว แค่เปลี่ยนจาก reject เป็น merge)

**Validated บนคลิปจริงทั้งคู่**: คลิป2 near player coverage 34%→**65%**
(268→516 เฟรม, ช่วงเฟรมกว้างจาก [0,362] เป็น [0,768] เกือบเต็มคลิป)
คลิป1 near player 21%→36% — ตรงกับที่ประเมินจากการนับ fragment ด้วยมือ
เกือบเป๊ะ ยืนยันด้วยตาจาก overlay video ด้วย: เฟรม 400 ที่เคยไม่มีใคร
ถูก track เลย (ทั้งที่เห็นคนชัดในภาพ) ตอนนี้ track ได้ถูกต้องภายใต้
track_id เดิม ดู CHANGELOG 0.3.1

**แก้ 2 บั๊กเพิ่ม (0.3.2, วันเดียวกันตอนบ่าย)** — พอทดสอบคลิปที่ 2 (คู่ที่
ยืนใกล้กันจริง) เจอ union-find (single-linkage) merge คนสองคนที่เป็นคนละ
คนจริงเข้าด้วยกันผ่าน fragment ตัวกลาง → เปลี่ยนเป็น complete-linkage
(ต้อง compatible กับสมาชิกทุกคนใน cluster) แต่กลับพังอีกแบบ: fragment
ที่ควรรวมเข้า near player ถูกจัดเป็น "far" ผิด ๆ เพราะ fail กับสมาชิกเก่า
แค่ตัวเดียว (ยืนยันด้วยการดึงเฟรมจริงมาดู) → เปลี่ยนเป็น **average-linkage**
(เทียบกับจุดรวมของ cluster แทนที่จะเทียบทีละคน) แก้ได้ทั้งคู่ — ขุดลึก
ลงไปอีกพบสาเหตุจริง: background subtraction เห็นคนคนเดียวเป็น 2 blob แยก
(หัว/ไหล่ vs ลำตัว/ขา) ในเฟรมเดียวกัน แก้ที่ต้นตอด้วย `_merge_split_person_boxes()`
ใน `detect_blobs()` เลย

**ผลสุดท้ายที่ validate ครบทั้งสองคลิป**: คลิป1 (คู่ใกล้กัน) near 14%
+ far 39% = รวม **53%** (จาก ~21%/~1% เดิม) คลิป2 near 74% + far 14% =
รวม **88%** (จาก ~34%/~0% เดิม) — เช็คว่า near/far ของคลิป1 ยังแยกถูกต้อง
จริง (ไม่ใช่คนเดียวถูกตัดเป็น 2) ด้วย blob-level temporal overlap = 50.7%
(สูงกว่า threshold มาก = active พร้อมกันจริง = คนละคนแน่นอน) ดู
CHANGELOG 0.3.2 สำหรับรายละเอียดการ debug ทั้งหมด

## ข้อมือ/มือโดนบัง (occlusion) — `occlusion_fill.py`

วัดจริงบน 2 คลิปตัวอย่าง: landmark หลัก (ไหล่/สะโพก) แม่นยำ 95-100%
แต่ข้อมือ/นิ้ว visibility อยู่ที่ 0.12-0.88 (ขึ้นกับข้าง/track) บางช่วง
หายต่อเนื่องยาวถึง 65 เฟรม — ส่วนใหญ่เป็นการโดนลำตัวบังจริงระหว่างหมุนตัว
ตีลูก (ยืนยันด้วยตาจากภาพ crop จริง) ลอง crop+zoom เฉพาะแขน/มือ (แย่ลง
เพราะโมเดลคาดหวังเห็นคนทั้งตัว) และ MediaPipe Hands (detect ได้แค่ 9%)
แล้วไม่ได้ผลทั้งคู่ — ดู CHANGELOG 0.2.9 สำหรับรายละเอียด

**สิ่งที่ทำได้**: kinematic gap-fill แบบเดียวกับที่ระบบ motion-capture
มืออาชีพใช้เวลา marker หาย — ใช้ข้อศอก/เข่า (เชื่อถือได้กว่ามาก) เป็นจุด
ยึด + ความยาวท่อนแขน/ขา (calibrate จากทั้งคลิป) + สมมติทิศทางหมุนต่อเนื่อง
ราบเรียบ ผสม (blend 50/50) กับ linear interpolation ธรรมดา — validate
ด้วย held-out test จริง (ซ่อนข้อมูลที่รู้คำตอบแล้ววัด error): blend ให้
ผลสม่ำเสมอกว่าใช้วิธีเดียวโดด ๆ จริง แต่ **error สัมบูรณ์ยังสูง (~35-75%
ของความยาวท่อนแขนเอง)** ไม่ใช่ค่าที่แม่นเทียบเท่าการ detect จริง

จึงจำกัดให้ใช้เฉพาะ gap สั้น (`MAX_KINEMATIC_GAP_FRAMES=20` เฟรม) เท่านั้น
gap ยาวกว่านั้นปล่อยตามเดิม (`low_confidence`/`not_detectable`) และ
**ตั้งใจไม่แก้ visibility score เดิม** — ให้ CF1/CF3/VF ยัง flag ว่าเป็น
ค่าประมาณ ไม่ใช่ค่าที่ detect ได้จริง (เติมตำแหน่งให้ใช้งานต่อเนื่องได้
เช่น wrist_path/KN7 แต่ไม่หลอกระบบความเชื่อมั่น) wire เข้า
`StrokePipeline.process_timeseries()` แล้ว (รันก่อน `smooth_timeseries()`)

## Court calibration (auto height) — ⚠️ experimental, opt-in เท่านั้น

```bash
python scripts/run_stroke.py clip.mp4 --stroke FH --auto-height  # แทน --height-cm
```

**แนวคิด**: หาเส้นคอร์ท (baseline/service line/sidelines) → ground-plane
homography (ขนาดคอร์ทมาตรฐาน ITF) → ใช้ความสูงเน็ต (91.4cm ที่กลาง) เป็น
vertical reference → weak-perspective คำนวณส่วนสูงผู้เล่นจริงโดยไม่ต้อง
กรอกเอง (ดู `court_calibration.py`)

**สถานะจริงหลัง validate บนคลิปจริง (2026-07-09)**: สูตร geometry ถูกต้อง
และเทสผ่านหมด แต่ **การหาเส้นคอร์ทอัตโนมัติยังไม่เสถียรพอ** — focal
length ที่ประมาณได้แกว่งเกือบ 2 เท่าข้ามเฟรมในคลิปเดียวกัน (กล้องไม่ขยับ)
เพราะแสงกลางคืน/รั้ว/โครงสร้างพื้นหลังรบกวนการ classify เส้น
→ **default เป็น `False`, ไม่แนะนำให้พึ่งเป็นหลักจนกว่าจะ tune เพิ่ม**
ใช้ `--height-cm` (manual) หรือรอ GAS ส่งมา (ตามที่ schema doc ระบุ) แทน

### Model-based court registration (`court_model.py`) — research/experimental

แนวทางที่ดีกว่า 4-corner approach ข้างบน: สร้างโมเดลกล้อง pinhole เต็มรูปแบบ
(`CameraPose`: height/distance/tilt/pan/focal_length) + โมเดลเส้นคอร์ทครบ
ทุกเส้น แล้ว "ทาย" ท่ากล้องที่ทำให้เส้นที่ทำนายตรงกับ edge ที่เห็นจริง
มากที่สุด (`search_camera_pose`) — **ตอบคำถาม "ทำนายส่วนคอร์ทที่ไม่เห็น
ในเฟรมได้ไหม" ได้ว่า "ได้"**: เมื่อมีท่ากล้องที่ score ดีพอ
`project_court_lines` จะคำนวณพิกัดของทุกเส้นได้ แม้เส้นนั้นจะไม่เคย
ปรากฏในเฟรมเลยก็ตาม (เทสยืนยันแล้วว่า far baseline ที่ไกล 23.77m ยัง
คำนวณพิกัดออกมาได้)

**สถานะจริงหลัง validate**: เทสด้วยข้อมูลสังเคราะห์ผ่าน (search หาท่า
กล้องที่ตรงกับความจริงได้ เมื่อให้ search budget พอ) แต่บนคลิปจริง
ยังได้แค่ fit บางส่วน (score ~0.6-0.65, เส้น baseline พอใกล้เคียงแต่
เส้นข้างสนามยังไม่ตรง) และมีปัญหา **focal length ↔ distance ambiguity**
แบบคลาสสิก (ท่ากล้องต่างกันหลายแบบให้ภาพ 2D เหมือนกันได้ ทำให้แม้ score
สูงก็ไม่รับประกันว่าพารามิเตอร์ที่ได้ถูกต้องจริงสำหรับใช้คำนวณส่วนสูง)
→ ยังไม่ต่อเข้า pipeline หลัก เก็บเป็นโมดูลวิจัย/ทดลองเท่านั้น

### Manual point-correspondence solve — สำหรับกล้องซูมจนคอร์ทไม่เข้าเฟรมครบ

ถ้าคอร์ทไม่เข้าเฟรมครบ (ซูมมาก) หรือ auto-detect เส้นไม่น่าเชื่อถือ
(กลางคืน/สิ่งกีดขวาง) ใช้ `PointCorrespondence` + `search_camera_pose_from_points`
แทน edge-based search — ให้คนคลิกจุดที่เห็นได้จริงในเฟรม (>= 4 จุด กี่จุด
ก็ได้ ไม่ต้องครบมุม) จับคู่กับพิกัดจริงบนคอร์ท แล้ว solve pose ตรง ๆ
(random+refine search + L-BFGS-B polish รอบสุดท้าย เพราะจุดที่คลิกให้
objective ที่หา gradient ได้จริง ต่างจาก edge score ที่เป็น step function)

**ผลทดสอบ (synthetic, ยังไม่มี real click data)**: จุดบนพื้น 6 จุดที่กระจาย
ความลึกต่างกัน กู้คืนท่ากล้องได้แม่นยำเป๊ะเมื่อไม่มี noise — ข้อสันนิษฐาน
เดิมที่ว่า "ต้องมีจุดเน็ต (ไม่อยู่ระนาบพื้น) ถึงจะตัด focal-length↔distance
ambiguity ได้" ไม่จำเป็นเสมอไปถ้าจุดกระจายความลึกพอ แต่เมื่อจำลอง click
noise ~2px (คนคลิกไม่แม่นเป๊ะ) จุดเน็ตช่วยลด noise sensitivity ได้จริง
(focal_length std ลดจาก 9.2→5.0px, distance std ลดจาก 0.061→0.036m —
ดีขึ้น ~40-45% ไม่ใช่แก้ ambiguity หมดจด) แนะนำให้คลิกจุดเสาเน็ตด้วยถ้า
มองเห็นในเฟรม

**อัปเดต**: สร้าง UI แล้ว — `labels/court_click_tool.html` (เปิดเป็นไฟล์
ตรง ๆ ไม่ต้องมี server, เหมือน `labeling_tool.html`) โหลดรูป/วิดีโอ →
เลือกจุดจาก dropdown (18 จุดมาตรฐาน ตรงกับ `NAMED_POINTS` ใน
`court_model.py`) → คลิกตำแหน่งจริง → export JSON แล้วรันกับ
`scripts/solve_camera_pose.py` ได้ CameraPose + รูป overlay ตรวจสอบ fit
ด้วยตา (เส้นเขียว=โมเดลทำนาย, วงส้ม=จุดที่คลิก) — ทดสอบ end-to-end กับ
synthetic data แล้ว: กู้คืนท่ากล้องได้เป๊ะ (verified ผ่าน preview browser
จริง ไม่ใช่แค่ unit test) กล้องไม่ขยับต่อสนามหนึ่งตัว → solve ครั้งเดียว
แล้ว reuse `camera_pose.json` กับทุกคลิปจากกล้องตัวนั้นได้เลย

⚠️ **จุดที่คลิกต้องกระจายความลึก (Z) พอ** — คลิกแค่จุดที่ระดับความลึก
เดียวกันหมด (เช่น 4 มุมเสาเน็ตอย่างเดียว) เป็น degenerate case ที่
validate แล้วว่าใช้งานไม่ได้จริง (`depth_spread_m()` เช็คให้อัตโนมัติ
ใน `solve_camera_pose.py`) ต้องมีจุดอย่างน้อย 1-2 จุดจากระดับความลึก
อื่นเสมอ (เช่น baseline หรือ service line) ดู CHANGELOG 0.2.7

### `vanishing_point.py` — ลองแล้ว ไม่ใช่ทางแก้ auto ที่ใช้ได้กับ setup นี้

ทำวิจัยหาวิธี auto เต็มรูปแบบ (ไม่ต้องคลิกเลย) ตามคำขอ implement
วิธีคลาสสิก (single-view metrology): หา vanishing point จากเส้นขนาน 2
ชุดที่ตั้งฉากกันจริง (เส้นข้างสนาม ⊥ เส้น baseline/service line) แล้ว
คำนวณ focal length ด้วยสูตรปิด — ไม่ต้องมี label/โมเดลฝึกเพิ่มเลย

**ผลทดสอบตรงไปตรงมา**: สูตรถูกต้อง (แม่นเป๊ะกับ geometry สังเคราะห์)
แต่ **ใช้ไม่ได้กับ setup กล้องมาตรฐานที่แนะนำมาตลอดในโปรเจกต์นี้** —
กล้องอยู่กึ่งกลางหลัง baseline พอดี (pan≈0°) ทำให้เส้น baseline/service
line เกือบขนานกันสนิทในภาพ vanishing point เลยอยู่เกือบ infinity →
focal length ที่คำนวณได้ไวต่อ noise มาก (validate แล้ว: pan 0-2° กับ
noise แค่ 1px เพี้ยนได้ 30-65%, ต้อง pan >= 10° ถึงจะเสถียร ~3-5%)
มี reliability gate ที่คืน `None` แทนค่าที่ไม่น่าเชื่อถือ (ตั้งใจคืน
`None` บ่อยกับ footage มาตรฐานของโปรเจกต์นี้ ไม่ใช่บั๊ก)

**สรุป**: เก็บไว้เป็นเครื่องมือเสริม (มีประโยชน์ถ้ากล้องลูกค้าเผอิญ
วางเอียงมาก) แต่ **ไม่ใช่ทางแก้หลัก** — `court_click_tool.html` +
manual point-click ยังเป็นทางเดียวที่ validate แล้วว่าเชื่อถือได้จริง
ไม่ว่ากล้องจะวางมุมไหน

**ทางเลือกที่ยังไม่ทำ (ต้องตัดสินใจเพิ่ม)**: มีโมเดล deep learning
สำเร็จรูปสำหรับตรวจจับจุดคอร์ทเทนนิสโดยเฉพาะ
([yastrebksv/TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector),
14 keypoints, precision 0.96 บน dataset ของเขา) แต่ติด 2 อย่างก่อนใช้ได้จริง:
(1) repo ไม่มี LICENSE — ใช้ในงาน deliverable ที่มีค่าจ้างโดยไม่ขอ
อนุญาตผู้เขียนมีความเสี่ยงทางกฎหมาย (2) domain gap — dataset เขาน่าจะ
เป็นมุมกล้อง broadcast (สูง/ไกล) ต่างจากคลิปลูกค้า (ใกล้ ระดับคน หลัง
baseline) ต้องทดสอบจริงก่อนเชื่อ

## Scope notes

- Phase 2 aggregation (AGG/TRE/PAT/SUM) ฝั่ง CV คำนวณตาม schema ใหม่ —
  **ต่างจาก TOR v2 ที่ระบุว่า GAS aggregate** → มี pricing conversation ค้าง
- Stroke classifier สำหรับ Phase 2 ยังเป็น hook (`stroke_type_provider`)
- Handover structure ใน doc (train.py/checkpoints) ไม่ตรงกับ rule-based
  pipeline — ต้อง align deliverable กับลูกค้า
