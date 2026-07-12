# Label Specification — Loeuf CV Pipeline

Ground truth สำหรับ benchmark keyframe accuracy (Phase 1) และ
action spotting (Phase 2)

## ไฟล์ 1: `labels/stroke_labels.csv` — คลิป single stroke (300 คลิป)

หนึ่งแถว = หนึ่งคลิป

| คอลัมน์ | type | บังคับ | ความหมาย |
|---|---|---|---|
| `clip_path` | string | ✅ | path relative จาก dataset root (เช่น `FH/fh_001.mp4`) |
| `stroke_type` | enum | ✅ | `FH/BH/SV/VL/SL/RS` — ใส่ explicit แม้ซ้ำกับโฟลเดอร์ (กันไฟล์ย้ายผิดที่) |
| `dominant_side` | enum | ✅ | `right` / `left` |
| `fps` | int | ✅ | fps ของไฟล์ต้นทาง — **สำคัญมาก** ใช้แปลง frame → ms ตอน eval |
| `label_tier` | enum | ✅ | `core` = label แค่ backswing_peak + impact (ทั้ง 300) / `full` = ครบ 5 keyframes (~10–15 คลิป/ท่า) |
| `unit_turn_frame` | int | tier=full | frame index (0-based) — เว้นว่าง = ไม่ได้ label |
| `backswing_peak_frame` | int | ✅ | frame index |
| `impact_frame` | int | ✅ | frame index — **ตัวที่ acceptance criteria ผูกอยู่** |
| `follow_through_peak_frame` | int | tier=full | frame index |
| `ready_restored` | bool | tier=full | `true`/`false` — **แยกจากช่องว่าง**: `false` = player ไม่กลับ ready จริง (เป็นข้อมูล ไม่ใช่ label ขาด) |
| `ready_frame` | int | เมื่อ restored=true | frame index |
| `impact_ambiguous` | bool | ✅ | `true` = จังหวะ impact อยู่ระหว่างเฟรม (เลือกเฟรมหลังตาม guideline) — ใช้วิเคราะห์ inter-annotator agreement |
| `usable` | bool | ✅ | `false` = คลิปใช้ benchmark ไม่ได้ (ตัดออกจากตัวหาร accuracy) |
| `issues` | list | — | คั่นด้วย `;` ใช้ enum เดียวกับ VQ2: `motion_blur/low_light/partial_occlusion/player_too_far/side_view/partial_swing` |
| `annotator_id` | string | ✅ | ใครเป็นคน label — ใช้วัด agreement |
| `notes` | string | — | อิสระ |

### Annotation guideline (ต้องอ่านก่อน label)

1. **impact** = เฟรมแรกที่เห็นลูกสัมผัสแร็กเก็ต/เพิ่งออกจากแร็กเก็ต ถ้าจังหวะจริงอยู่ระหว่าง 2 เฟรม → **เลือกเฟรมหลัง** แล้วติ๊ก `impact_ambiguous=true`
2. **backswing_peak** = เฟรมที่ wrist/racket อยู่ไกลจาก net ที่สุดก่อนเริ่ม forward swing (จุดกลับทิศ)
3. **unit_turn** = เฟรมแรกที่เห็นไหล่+สะโพกเริ่มหมุนออกจาก net
4. **follow_through_peak** = เฟรมที่แขน/wrist อยู่จุดสูงสุดของ arc หลัง impact
5. **ready_restored** = player กลับ neutral stance + แร็กเก็ตหน้าตัว — ไม่กลับจริง (เดินไปเก็บลูก/จบ point) ให้ `false` **ห้ามเว้นว่าง**
6. คลิปเสีย (ตัดกลาง swing / มืด / เบลอหนัก) → `usable=false` + ระบุ `issues` — อย่าฝืน label
7. **20–30 คลิปแรกให้ 2 คน label ซ้ำกัน** เพื่อวัด inter-annotator agreement ก่อนลุยทั้งชุด

## ไฟล์ 2: `labels/session_labels.csv` — วิดีโอ session เต็ม (Phase 2)

หนึ่งแถว = หนึ่งจังหวะตีใน session (label ระดับ segment — ไม่ต้อง label keyframe)

| คอลัมน์ | type | บังคับ | ความหมาย |
|---|---|---|---|
| `session_path` | string | ✅ | path วิดีโอ session |
| `segment_index` | int | ✅ | ลำดับจังหวะตี เริ่มที่ 1 |
| `stroke_type` | enum | ✅ | `FH/BH/SV/VL/SL/RS` |
| `start_time_s` | float | ✅ | วินาทีเริ่ม stroke (ก่อน unit_turn เล็กน้อย) |
| `end_time_s` | float | ✅ | วินาทีจบ (หลัง follow-through) |
| `impact_time_s` | float | — | optional — ใส่บางแถวไว้ spot-check keyframe ใน context จริง |
| `annotator_id` | string | ✅ | |
| `notes` | string | — | |

ใช้วัด: spotting **recall** (เจอครบไหม — window ทับ segment = เจอ),
**precision** (window ที่ไม่ทับ segment ไหนเลย = false positive)

## Conventions

- frame index เป็น **0-based** อิงไฟล์ต้นทาง (ก่อน resample) — eval script
  แปลงเป็น ms ด้วย `frame / fps × 1000` แล้วเทียบใน time domain
  (tolerance ±1 เฟรมต้นทาง = ±33.3ms ที่ 30fps / ±16.7ms ที่ 60fps)
- ช่องว่าง = ไม่ได้ label (ต่างจาก `false` ซึ่งเป็นข้อมูลจริง)
- encoding UTF-8, comma-separated

Template: [stroke_labels_template.csv](../labels/stroke_labels_template.csv) ·
[session_labels_template.csv](../labels/session_labels_template.csv)
