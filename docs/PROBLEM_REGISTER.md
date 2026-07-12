# Loeuf CV Pipeline — รายการปัญหาที่พบ (Problem Register)

> สรุปปัญหาทั้งหมดที่เจอตลอดการพัฒนา แยกตามหมวดและสถานะ — สิ่งที่แก้แล้ว,
> สิ่งที่ติดข้อจำกัดจริง, และสิ่งที่ต้องให้ลูกค้าตัดสินก่อนเดินต่อ (ประกอบ Phase 0 Feasibility)
>
> **อัปเดต:** 2026-07-10 · **ทดสอบจริงบน:** 2 คลิปตัวอย่าง (960×540, กลางคืน) · **37 รายการ**

## สรุปภาพรวม

| สถานะ | จำนวน | ความหมาย |
|---|---|---|
| ✅ แก้แล้ว & validate | 9 | ทำงานได้จริงบนคลิปจริง |
| ⚠️ แก้บางส่วน | 6 | ใช้ได้ในเงื่อนไขจำกัด |
| ❌ ข้อจำกัดพื้นฐาน | 6 | แก้ด้วยซอฟต์แวร์ไม่ได้ |
| 💬 รอลูกค้าตัดสิน | 16 | scope / schema / สัญญา |

---

## A. การตรวจจับคน & tracking — *6 รายการ · แก้จบใน session นี้*

| สถานะ | ปัญหา | root cause | วิธีแก้ / ผลจริง |
|---|---|---|---|
| ✅ | Whole-frame detection ล้มเหลวเกือบ 100% | คนตัวเล็กในเฟรม (~15% ความสูง) ต่ำกว่าเพดาน MediaPipe | crop + zoom รอบตัว → detect 0% → 67% |
| ✅ | "far" เกาะคนยืนนิ่งข้างสนาม (ผู้ชม) แทนผู้เล่นจริง | เลือก track จาก "ยาวสุด" อย่างเดียว — คนนิ่ง track ต่อเนื่องง่ายกว่า | เพิ่ม `MIN_MOVEMENT_STD` กรองคนแทบไม่ขยับทิ้ง |
| ✅ | คนเดียวถูกตัดเป็น 7 track ID → coverage 34% | blob หลุด-กลับซ้ำ ๆ เกิน `MAX_TRACK_GAP_FRAMES` | **track stitching** รวม fragment คนเดียวกัน → 74% |
| ✅ | Stitching รวมคนละคนเข้าด้วยกัน (transitive merge) | single-linkage เชื่อมผ่าน fragment ตัวกลาง | เปลี่ยนเป็น **average-linkage** (เทียบกับจุดรวม cluster) |
| ✅ | Fragment ของคนเดิมถูกจัดเป็น "far" ผิด | complete-linkage เข้มไป — fail กับสมาชิกเก่าแค่ตัวเดียว | average-linkage ทนต่อ outlier สมาชิกเดี่ยว |
| ✅ | คนเดียวถูกเห็นเป็น 2 blob (หัว/ไหล่ vs ลำตัว) | background subtraction เห็นช่องว่างกลางตัว | `_merge_split_person_boxes` รวมก่อน tracking |

---

## B. หลายผู้เล่น / doubles / player identity — *5 รายการ · ใหม่ · สำคัญ*

| สถานะ | ปัญหา | root cause | ต้องตัดสิน / ทางออก |
|---|---|---|---|
| 💬 | near/far สมมติว่าเล่นเดี่ยว — doubles = 4 คน | "the near player" มี 2 คนในคู่ → label พัง | deliverable รองรับ doubles ไหม? (TOR/ลูกค้า) |
| 💬 | จำนวนคนไม่คงที่ — `max_players=2` rigid เกิน | ฉากจริง (clip2) มี 3-4 คน: ผู้เล่น + คนป้อนบอล + คนรอ | วิเคราะห์ทุกคน หรือเฉพาะ target player? |
| ❌ | แยก "ผู้เล่น" จาก "คนป้อนบอล/คนรอ" ไม่ได้ | คนป้อนบอลขยับตลอด → movement filter จับไม่ได้; CV เห็นแค่ "คนขยับ" เหมือนกันหมด | ให้คน **คลิกเลือกผู้เล่นเป้าหมาย** — CV ไม่ต้องเดา |
| ❌ | Identity ข้ามคลิป / ข้าม changeover ไม่ได้ | classical CV รักษา identity ได้แค่ในช่วง track ต่อเนื่อง | ต้องเพิ่ม re-ID (appearance/เสื้อ) = scope ใหญ่ + อาจต้อง GPU |
| 💬 | near/far = ตำแหน่ง ไม่ใช่ identity | งานวิเคราะห์ควรใช้ `player_id` เป็นหลัก | schema กำหนด player identity ไว้ field ไหน? |

---

## C. ข้อมือ / มือโดนบัง (occlusion) — *6 รายการ · partial + fundamental*

| สถานะ | ปัญหา | root cause | ผลจริง / ทำไมแก้ไม่ได้ |
|---|---|---|---|
| ❌ | ข้อมือฝั่งไกล visibility ต่ำ (0.12–0.88), หายยาว 65 เฟรม | แขนหลบหลังลำตัวในกล้องมุมเดียว = genuine occlusion | ไม่มีโมเดลไหน "มองทะลุ" พิกเซลที่ไม่มีข้อมูล |
| ❌ | crop + zoom เฉพาะแขน/มือ → **แย่ลง** | ครึ่งตัวขัดสมมติฐาน whole-body model | visibility 0.19 → 0.015 |
| ❌ | MediaPipe Hands (โมเดลเฉพาะมือ) | กลางคืน + มือกำไม้ + มุมหลัง | detect ได้แค่ **9%** (12/133 เฟรม) |
| ⚠️ | Kinematic gap-fill (ข้อศอก + ความยาวแขน) | error 35–75% ของความยาวแขน; ข้อศอกมักหายพร้อมข้อมือ | ใช้ได้แค่ gap สั้น ≤20 เฟรม; coverage จริง 0–2% |
| ⚠️ | model_complexity = 2 (MediaPipe หนักขึ้น) | ช่วยคลิปนึง แย่คลิปนึง | ไม่คงเส้นคงวา + ช้าลง 74% |
| ⚠️ | Preprocess (denoise + strong CLAHE) | ช่วยเคส "มืด/เบลอแต่ยังเห็น" ไม่ช่วยเคส occlusion จริง | ช่วย 3/4 เคส (~เท่าตัว) · ยังไม่ wire เข้า pipeline |

---

## D. Auto court calibration / หาส่วนสูงอัตโนมัติ — *5 รายการ · ยากสุด*

| สถานะ | ปัญหา | root cause | สถานะ / ผลจริง |
|---|---|---|---|
| ⚠️ | หาเส้นคอร์ทอัตโนมัติไม่เสถียร (`court_calibration`) | Hough-line โดนรั้ว/เสาไฟ/แสงกลางคืนรบกวน — focal length แกว่ง 2× | opt-in, ปิด default |
| ⚠️ | Model-based registration (`court_model`) | focal-length ↔ distance ambiguity | fit บนคลิปจริงแค่ 0.6–0.65 · research-stage |
| ⚠️ | Vanishing-point focal length (`vanishing_point`) | degenerate ตอน pan≈0 — คือ setup กล้องมาตรฐานของงานนี้พอดี | สูตรถูกแต่คืน None เกือบตลอดบน footage มาตรฐาน |
| ✅ | Manual point-click calibration | คนคลิกจุดคอร์ทที่เห็น → solve camera pose | `court_click_tool.html` — solve ครั้งเดียว reuse ทั้งกล้อง |
| ✅ | คลิกจุดระดับความลึกเดียวกัน → คำตอบเพี้ยน | degenerate จริง แม้ไม่มี noise | บังคับ depth-spread ก่อน solve |

---

## E. CNN / โมเดลสำเร็จรูป — *3 รายการ · ทดสอบแล้ว*

| สถานะ | ปัญหา | root cause | ผลจริง / ต้องตัดสิน |
|---|---|---|---|
| ❌ | TennisCourtDetector (pretrained CNN) | domain gap — เทรนจาก broadcast (สูง/ไกล) ต่างจากมุมหลังระดับคน | detect ได้แค่ 3/14 จุด · ไม่ตรงเส้นเลย |
| 💬 | repo ไม่มี LICENSE | ใช้ในงาน deliverable ที่มีค่าจ้าง = เสี่ยงทางกฎหมาย | ติดต่อผู้เขียนขออนุญาต หรือข้ามไป |
| ❌ | Fine-tune BlazePose เอง | Google ไม่เปิด training pipeline + ไม่มี label data | ต้องมีข้อมูล label จากมุมกล้องแบบนี้ก่อน |

---

## F. Environment / dependency — *2 รายการ*

| สถานะ | ปัญหา | root cause | วิธีแก้ / ต้องตัดสิน |
|---|---|---|---|
| ✅ | mediapipe 0.10.35+ ตัด legacy `mp.solutions` API | pipeline พึ่ง API เดิม | pin `mediapipe==0.10.14` |
| 💬 | ต้องมี GPU (train / RTMPose) | ธีมที่ค้างตลอดโปรเจกต์ | ผูกกับการตัดสินใจเรื่องเปลี่ยน model |

---

## G. Schema กำกวม / ขัดกันเอง — *7 รายการ · ถามลูกค้า*

| สถานะ | ปัญหา | รายละเอียด |
|---|---|---|
| 💬 | ใครส่งส่วนสูงผู้เล่น (สำหรับ px→cm) | เรากรอก, GAS ส่ง, หรือ auto-detect — กระทบความแม่นทุก field หน่วย cm |
| 💬 | per-field confidence block | นิยามบอกต้องมี แต่ sample JSON ไม่มี |
| 💬 | precision: ปัด 2 ตำแหน่ง vs "full precision" | ตาราง type กับ note ขัดกัน |
| 💬 | C19/C20 recovery formula | label ในเอกสารดูสลับกัน (ready vs follow-through) |
| 💬 | C31 wrist angle ใช้ hand index แทน racket | ไม่มี racket detector → fallback — ยอมรับไหม |
| 💬 | KN3 lag sign convention | สูตรในเอกสารดูกลับด้านจาก intent |
| 💬 | rotation degree sign convention | ยังไม่ validate กับ Phase 0 |

---

## H. Scope / สัญญา — *3 รายการ · pricing/scope*

| สถานะ | ปัญหา | รายละเอียด |
|---|---|---|
| 💬 | Phase 2 aggregation ทำฝั่ง CV แต่ TOR บอก GAS | AGG/TRE/PAT/SUM — pricing conversation ค้าง |
| 💬 | Handover structure พูดถึง train.py/checkpoints | แต่ deliverable จริงเป็น rule-based pipeline — ต้อง align ความคาดหวัง |
| 💬 | เปลี่ยนเป็น RTMPose / top-down architecture | TOR ระบุ MediaPipe/BlazePose ชัดเจน = scope change — ต้องยก Phase 0 Feasibility |

---

## ข้อจำกัดพื้นฐาน — แก้ด้วยซอฟต์แวร์ไม่ได้

รากของหลายปัญหา ❌ ข้างบน มาจากข้อจำกัดเชิงกายภาพของกล้องมุมเดียว ไม่ใช่เรื่องเลือกโมเดล/เขียนโค้ด:

1. **Single-view depth ambiguity** — lift 2D→3D จากกล้องเดียวมีความกำกวมเชิงลึกสูง
2. **Back-view occlusion** — ข้อมือ/สะโพกฝั่งไกลบังตลอดในมุมหลัง
3. **เพดานความละเอียดต้นทาง** — 960×540, คนไกล ~15% ของเฟรม (แนะนำ 1080p+)
4. **กลางคืน / แสงย้อน** — preprocess ช่วยได้บ้างแต่มีเพดาน

---

## ภาพรวม & ก้าวต่อไป

- **หมวด A** (detection/tracking) แก้จบแล้ว — coverage รวม 88% บนคลิปทดสอบ
- **หมวด C/D** แก้ได้บางส่วนและ document ตามจริงว่ายังไม่ใช่ production-grade
- **สิ่งที่โผล่มาใหม่และสำคัญสุด — หมวด B** (multi-player / identity / แยกผู้เล่นจากคนป้อนบอล): เพราะฉากจริงไม่ใช่ 1-ต่อ-1 สะอาด ๆ ทางออกที่เป็นจริงคือ "ให้คนคลิกเลือกผู้เล่นเป้าหมาย" + ยก scope doubles ให้ลูกค้ายืนยัน
- **16 รายการหมวด B/E/F/G/H** ต้องอาศัยการตัดสินใจ/ข้อมูลจากลูกค้ามากกว่าการเขียนโค้ดเพิ่ม — เหมาะยกรวมใน Phase 0 Feasibility
