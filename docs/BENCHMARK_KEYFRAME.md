# Benchmark — Keyframe Accuracy vs Ground Truth

## Verdict

**Accuracy: 0.352** (n=344, เป้า TOR ≥ 0.8) — **FAIL ⚠️**

ตัวเลขหลักใช้เกณฑ์ **"อยู่ในช่วงการเคลื่อนไหวเดียวกัน"** ซึ่งเป็นเกณฑ์ที่ลูกค้าเขียนไว้เองใน `cv_schema_table_loeuf`:

> *"Guideline นี้มีไว้เพื่อช่วยเลือก **ช่วงของ keyframe** เท่านั้น **ไม่จำเป็นต้องจับเฟรมได้ตรงเป๊ะ** หากอยู่ในช่วงการเคลื่อนไหวเดียวกันถือว่าใช้ได้"*

ในเอกสารเดียวกันลูกค้ายังกำกับ `impact` และ `follow_through_peak` ว่า **(Low Confidence)** ด้วยตัวเอง — สอดคล้องกับที่ GT 97.6% ถูก annotator flag เป็น `low_confidence`

**นิยามที่ใช้วัด**: เฟรมที่ทายต้องยังใกล้ keyframe นั้นมากกว่า keyframe อื่นของ stroke เดียวกัน (ขอบ = จุดกึ่งกลางระหว่าง keyframe ที่ติดกันใน GT) → หน้าต่างกว้างแคบเองตามจังหวะจริงของแต่ละท่า ไม่ต้องตั้งค่าคงที่ใด ๆ ดู `phase_windows()`

| ตัวเลข | เกณฑ์ | Accuracy | n |
|---|---|---|---|
| **Mode A end-to-end** | ช่วงเดียวกัน (ตามลูกค้า) | **0.352** | 344 |
| Mode B oracle-impact | ช่วงเดียวกัน (ตามลูกค้า) | 0.840 | 344 |
| Mode A end-to-end | ±1 เฟรม (เข้มกว่าที่ลูกค้าขอ) | 0.195 | 344 |
| Mode A matched-only | ±1 เฟรม เฉพาะ stroke ที่ detect เจอ (สูงหลอก) | 0.322 | 208 |
| Mode B oracle-impact | ±1 เฟรม = เพดานของ keyframe logic | 0.759 | 344 |

Mode A end-to-end = จริงตามที่ลูกค้าได้ (stroke ที่ hit detection หาไม่เจอ นับเป็นผิด ไม่ตัดออกจากตัวหาร) · เกณฑ์รวม = impact + backswing_peak ตาม spec

เทียบเป้า **TOR ≥ 0.8**: เกณฑ์ ±1 เฟรมให้ 0.195 = **FAIL ⚠️**

## Provenance

- Ground truth: 16 ไฟล์ / 172 stroke / annotator: atikan
- fps ต้นทาง: 29.97 · ความยาวรวม ~31.3 นาที
- Checkpoint: `runs/pose/tennis_ball_racket_pose_finetune_best.pt`
- git: `f42f837` · วันที่รัน: 2026-08-05 06:24
- Accuracy tolerance: ±1 เฟรม · Match tolerance: ±10 เฟรม
- **การประเมิน hit classifier: LOPO (โมเดลไม่เคยเห็นคนในคลิปที่วัด)**

  > ⚠️ ตัวเลขต่างกันมากระหว่างสองโหมด — โหมด in-sample ให้ acceptance 0.506 แต่โหมด LOPO ให้ 0.308 บนข้อมูลชุดเดียวกัน ส่วนต่างคือ "การจำคลิป" ที่ปนอยู่ (`hit_classifier.pkl` เทรนจากคลิปเดียวกับที่ใช้วัด) **ตัวเลขที่ควรบอกลูกค้าคือโหมด LOPO** เพราะเป็นตัวที่บอกได้ว่าใช้กับผู้เล่นคนใหม่แล้วจะเป็นยังไง · Mode B ไม่กระทบ (ป้อน impact จากเฉลย ไม่ได้ใช้ classifier)

## Detection (Mode A)

- TP **104** / FN **68** / FP **179**
- Recall **0.605** · Precision **0.367** · FP ต่อนาที 5.72
- Δ (pred − gt) เฉลี่ย -0.0 เฟรม · median 0.0 · p90 |Δ| 8.0
- GT ที่พลาด (68 ตัว): ระยะถึง prediction ที่ใกล้สุด median **32.0 เฟรม** · 23 ตัวอยู่ในระยะ 20 เฟรม

  > ตีความ: ถ้าระยะใกล้ ๆ = pipeline "เห็น" stroke แต่ระบุเฟรมเพี้ยน (ไปแก้ความแม่นของ impact detection) · ถ้าไกลมาก = พลาด stroke ไปเลย (ไปแก้ recall/candidate generation) — คนละปัญหาคนละทางแก้

| Clip | GT | Pred | TP | FN | FP |
|---|---|---|---|---|---|
| set2/IMG_0284 | 2 | 3 | 0 | 2 | 3 |
| set2/IMG_0300B1 | 23 | 35 | 19 | 4 | 16 |
| set2/IMG_0283B-VLbh | 10 | 14 | 3 | 7 | 11 |
| set2/IMG_0283A-VLfh | 10 | 16 | 8 | 2 | 8 |
| set4/IMG_0300A4(VL) | 8 | 7 | 3 | 5 | 4 |
| set4/IMG_0300B52(BH) | 8 | 10 | 8 | 0 | 2 |
| set4/IMG_0301A2(SV1) | 9 | 9 | 9 | 0 | 0 |
| set1/IMG_0266-SL | 10 | 14 | 6 | 4 | 8 |
| set3/IMG_0282(SL) | 11 | 19 | 6 | 5 | 13 |
| set2/IMG_0285A | 10 | 19 | 6 | 4 | 13 |
| set3/IMG_0301A4(SV1) | 11 | 11 | 4 | 7 | 7 |
| set3/IMG_0281(BH) | 16 | 14 | 11 | 5 | 3 |
| set3/IMG_0300B53-VL | 12 | 15 | 4 | 8 | 11 |
| set4/IMG_0300B7-SV1 | 11 | 21 | 8 | 3 | 13 |
| set4/IMG_0300B6-SV1 | 11 | 17 | 9 | 2 | 8 |
| sessions_deferred/set4/IMG_0294(SV1) | 10 | 59 | 0 | 10 | 59 |

## Keyframe accuracy — เกณฑ์ "ช่วงเดียวกัน" (ตามลูกค้า)

คอลัมน์ `หน้าต่าง` = ครึ่งความกว้างเฉลี่ยของช่วงที่ยอมรับ (เฟรม) — คำนวณจาก GT ของ stroke นั้นเอง ไม่ได้ตั้งค่าไว้ล่วงหน้า

| Stroke | Keyframe | n GT | Accuracy (Mode A) | หน้าต่าง ± (เฟรม) | MAE (เฟรม) |
|---|---|---|---|---|---|
| BH | unit_turn | 26 | 0.577 | 7.2 | 4.16 |
| BH | backswing_peak | 26 | 0.500 | 4.5 | 4.26 |
| BH | trophy_position | 0 | — (no GT) | — | — |
| BH | impact | 26 | 0.654 | 4.3 | 2.05 |
| BH | follow_through_peak | 25 | 0.600 | 10.5 | 3.39 |
| BH | recovery_position | 19 | 0.421 | 17.5 | 14.60 |
| FH | unit_turn | 21 | 0.762 | 11.0 | 8.63 |
| FH | backswing_peak | 21 | 0.667 | 6.3 | 3.16 |
| FH | trophy_position | 0 | — (no GT) | — | — |
| FH | impact | 21 | 0.810 | 4.6 | 2.11 |
| FH | follow_through_peak | 21 | 0.905 | 9.9 | 2.95 |
| FH | recovery_position | 16 | 0.688 | 13.7 | 11.60 |
| SL | unit_turn | 21 | 0.095 | 4.4 | 7.50 |
| SL | backswing_peak | 21 | 0.190 | 3.6 | 6.67 |
| SL | trophy_position | 0 | — (no GT) | — | — |
| SL | impact | 21 | 0.286 | 4.7 | 5.08 |
| SL | follow_through_peak | 21 | 0.381 | 7.0 | 7.00 |
| SL | recovery_position | 4 | 0.500 | 10.4 | 4.00 |
| SV | unit_turn | 0 | — (no GT) | — | — |
| SV | backswing_peak | 64 | 0.047 | 5.4 | 12.31 |
| SV | trophy_position | 64 | 0.219 | 6.0 | 5.43 |
| SV | impact | 64 | 0.453 | 5.2 | 3.69 |
| SV | follow_through_peak | 64 | 0.422 | 3.9 | 4.86 |
| SV | recovery_position | 0 | — (no GT) | — | — |
| VL | unit_turn | 1 | 0.000 | 0.5 | — |
| VL | backswing_peak | 40 | 0.175 | 5.7 | 4.72 |
| VL | trophy_position | 0 | — (no GT) | — | — |
| VL | impact | 40 | 0.275 | 3.2 | 3.00 |
| VL | follow_through_peak | 40 | 0.250 | 11.6 | 3.67 |
| VL | recovery_position | 4 | 0.000 | 13.0 | 16.00 |

### Mode B (oracle impact) — เกณฑ์เดียวกัน

| Stroke | Keyframe | n GT | Accuracy (Mode B) | หน้าต่าง ± (เฟรม) | MAE (เฟรม) |
|---|---|---|---|---|---|
| BH | unit_turn | 26 | 0.923 | 7.2 | 3.15 |
| BH | backswing_peak | 26 | 0.923 | 4.5 | 2.23 |
| BH | trophy_position | 0 | — (no GT) | — | — |
| BH | impact | 26 | 1.000 | 4.3 | 0.00 |
| BH | follow_through_peak | 25 | 0.960 | 10.5 | 1.96 |
| BH | recovery_position | 19 | 0.737 | 17.5 | 13.58 |
| FH | unit_turn | 21 | 0.905 | 11.0 | 7.71 |
| FH | backswing_peak | 21 | 0.952 | 6.3 | 0.76 |
| FH | trophy_position | 0 | — (no GT) | — | — |
| FH | impact | 21 | 1.000 | 4.6 | 0.00 |
| FH | follow_through_peak | 21 | 1.000 | 9.9 | 2.52 |
| FH | recovery_position | 16 | 0.875 | 13.7 | 9.94 |
| SL | unit_turn | 21 | 0.429 | 4.4 | 4.81 |
| SL | backswing_peak | 21 | 0.810 | 3.6 | 1.86 |
| SL | trophy_position | 0 | — (no GT) | — | — |
| SL | impact | 21 | 1.000 | 4.7 | 0.00 |
| SL | follow_through_peak | 21 | 1.000 | 7.0 | 2.71 |
| SL | recovery_position | 4 | 1.000 | 10.4 | 4.00 |
| SV | unit_turn | 0 | — (no GT) | — | — |
| SV | backswing_peak | 64 | 0.438 | 5.4 | 10.42 |
| SV | trophy_position | 64 | 0.766 | 6.0 | 2.41 |
| SV | impact | 64 | 1.000 | 5.2 | 0.00 |
| SV | follow_through_peak | 64 | 0.656 | 3.9 | 4.39 |
| SV | recovery_position | 0 | — (no GT) | — | — |
| VL | unit_turn | 1 | 0.000 | 0.5 | 17.00 |
| VL | backswing_peak | 40 | 0.700 | 5.7 | 2.48 |
| VL | trophy_position | 0 | — (no GT) | — | — |
| VL | impact | 40 | 1.000 | 3.2 | 0.00 |
| VL | follow_through_peak | 40 | 0.875 | 11.6 | 16.80 |
| VL | recovery_position | 4 | 0.250 | 13.0 | 15.25 |

## Keyframe accuracy — เกณฑ์ ±1 เฟรม (อ้างอิงแบบเข้ม)

### Mode A (end-to-end)

| Stroke | Keyframe | n GT | Accuracy | MAE (เฟรม) | median (เฟรม) |
|---|---|---|---|---|---|
| BH | unit_turn | 26 | 0.231 | 4.16 | 3.0 |
| BH | backswing_peak | 26 | 0.423 | 4.26 | 1.0 |
| BH | trophy_position | 0 | — (no GT) | — | — |
| BH | impact | 26 | 0.500 | 2.05 | 1.0 |
| BH | follow_through_peak | 25 | 0.240 | 3.39 | 3.5 |
| BH | recovery_position | 19 | 0.053 | 14.60 | 11.0 |
| FH | unit_turn | 21 | 0.048 | 8.63 | 8.0 |
| FH | backswing_peak | 21 | 0.524 | 3.16 | 1.0 |
| FH | trophy_position | 0 | — (no GT) | — | — |
| FH | impact | 21 | 0.476 | 2.11 | 1.0 |
| FH | follow_through_peak | 21 | 0.333 | 2.95 | 2.0 |
| FH | recovery_position | 16 | 0.125 | 11.60 | 6.0 |
| SL | unit_turn | 21 | 0.000 | 7.50 | 6.0 |
| SL | backswing_peak | 21 | 0.048 | 6.67 | 5.5 |
| SL | trophy_position | 0 | — (no GT) | — | — |
| SL | impact | 21 | 0.095 | 5.08 | 4.0 |
| SL | follow_through_peak | 21 | 0.048 | 7.00 | 5.5 |
| SL | recovery_position | 4 | 0.000 | 4.00 | 4.0 |
| SV | unit_turn | 0 | — (no GT) | — | — |
| SV | backswing_peak | 64 | 0.016 | 12.31 | 10.0 |
| SV | trophy_position | 64 | 0.047 | 5.43 | 5.0 |
| SV | impact | 64 | 0.094 | 3.69 | 3.0 |
| SV | follow_through_peak | 64 | 0.125 | 4.86 | 3.5 |
| SV | recovery_position | 0 | — (no GT) | — | — |
| VL | unit_turn | 1 | 0.000 | — | — |
| VL | backswing_peak | 40 | 0.125 | 4.72 | 3.5 |
| VL | trophy_position | 0 | — (no GT) | — | — |
| VL | impact | 40 | 0.175 | 3.00 | 2.0 |
| VL | follow_through_peak | 40 | 0.225 | 3.67 | 2.5 |
| VL | recovery_position | 4 | 0.000 | 16.00 | 16.0 |

### Mode B (oracle impact)

| Stroke | Keyframe | n GT | Accuracy | MAE (เฟรม) | median (เฟรม) |
|---|---|---|---|---|---|
| BH | unit_turn | 26 | 0.308 | 3.15 | 3.0 |
| BH | backswing_peak | 26 | 0.808 | 2.23 | 1.0 |
| BH | trophy_position | 0 | — (no GT) | — | — |
| BH | impact | 26 | 1.000 | 0.00 | 0.0 |
| BH | follow_through_peak | 25 | 0.440 | 1.96 | 2.0 |
| BH | recovery_position | 19 | 0.105 | 13.58 | 11.0 |
| FH | unit_turn | 21 | 0.095 | 7.71 | 7.0 |
| FH | backswing_peak | 21 | 0.905 | 0.76 | 0.0 |
| FH | trophy_position | 0 | — (no GT) | — | — |
| FH | impact | 21 | 1.000 | 0.00 | 0.0 |
| FH | follow_through_peak | 21 | 0.429 | 2.52 | 2.0 |
| FH | recovery_position | 16 | 0.250 | 9.94 | 3.5 |
| SL | unit_turn | 21 | 0.286 | 4.81 | 5.0 |
| SL | backswing_peak | 21 | 0.571 | 1.86 | 1.0 |
| SL | trophy_position | 0 | — (no GT) | — | — |
| SL | impact | 21 | 1.000 | 0.00 | 0.0 |
| SL | follow_through_peak | 21 | 0.333 | 2.71 | 3.0 |
| SL | recovery_position | 4 | 0.000 | 4.00 | 3.5 |
| SV | unit_turn | 0 | — (no GT) | — | — |
| SV | backswing_peak | 64 | 0.219 | 10.42 | 7.0 |
| SV | trophy_position | 64 | 0.438 | 2.41 | 2.0 |
| SV | impact | 64 | 1.000 | 0.00 | 0.0 |
| SV | follow_through_peak | 64 | 0.250 | 4.39 | 4.0 |
| SV | recovery_position | 0 | — (no GT) | — | — |
| VL | unit_turn | 1 | 0.000 | 17.00 | 17.0 |
| VL | backswing_peak | 40 | 0.575 | 2.48 | 1.0 |
| VL | trophy_position | 0 | — (no GT) | — | — |
| VL | impact | 40 | 1.000 | 0.00 | 0.0 |
| VL | follow_through_peak | 40 | 0.550 | 16.80 | 1.0 |
| VL | recovery_position | 4 | 0.000 | 15.25 | 15.0 |

## GT coverage — keyframe ไหนมีเฉลยบ้าง

GT ไม่ได้ label ครบทุก keyframe ทุก stroke **โดยตั้งใจ** — ช่อง `— (no GT)` ด้านบนคือ *ไม่ได้วัด* ไม่ใช่สอบตก:

- `unit_turn` ไม่มีใน SV เลย (serve ใช้ trophy_position แทน ตาม spec)
- `trophy_position` มีเฉพาะ SV
- `recovery_position` มีเมื่อ `recovery_restored = true` เท่านั้น

## ⚠️ คลิปที่ประมวลผลล้ม

GT ของคลิปเหล่านี้ถูกนับเป็น FN ทั้งก้อน (ไม่ได้ตัดออกจากตัวหาร) — ตัวเลข accuracy จึงถูกกดลงด้วยความล้มเหลวเชิงเทคนิค ไม่ใช่ความแม่น ของโมเดลเพียว ๆ:

- `set2/IMG_0283B-VLbh` [predict_a]: index 65 is out of bounds for axis 0 with size 59
- `set3/IMG_0301A4(SV1)` [predict_a]: index 45 is out of bounds for axis 0 with size 40
- `set3/IMG_0300B53-VL` [predict_a]: index 65 is out of bounds for axis 0 with size 54

## ข้อจำกัดเชิงโค้ดที่ทราบล่วงหน้า

- 🔴 **`backswing_peak` (B2) เป็นค่าประมาณจากจังหวะ impact ไม่ใช่ผลตรวจจับการเคลื่อนไหว** — schema_builder/keyframes.py ใช้ offset คงที่ (groundstroke `impact-3`, slice `impact-5`) ส่วนเสิร์ฟหาเฟรมที่ข้อมือสูงสุดในช่วง `impact-35..-9` เหตุผล: GT ของลูกค้าวาง B2 ห่าง impact แค่ median 3 เฟรม (0 จาก 78 groundstroke ที่ห่าง ≥20 เฟรม) ไม่ตรงกับนิยาม 'จุดถอยหลังสุด' เชิงพื้นที่ — ทดลองนิยามเชิงจลนศาสตร์แล้วแพ้ค่าคงที่ทุกตัว (velocity reversal 0.144 · argmax|x−x_imp| 1.5s ของเดิม 0.053 vs ค่าคงที่ ~0.49) → **ต้องยืนยันนิยาม B2 กับลูกค้า** ถ้าต้องการจุดถอยหลังสุดจริงต้อง re-label
- ✅ กิ่ง slice ของ B2 (`impact-5`) **พิสูจน์ข้ามคนได้แล้ว** (2026-07-30) — ชุด label เพิ่ม IMG_0266-SL ทำให้ SL มาจาก 2 คน (earth 11 + prem 10) เดิมมาจากคนเดียวจึง Leave-One-Person-Out ให้ 0.00 โดยโครงสร้าง ไม่ใช่เพราะ โมเดลแย่ — ตอนนี้วัดได้จริง Mode B backswing SL = 0.571
- `unit_turn` (B1) **ไม่ได้ detect** — ตั้งเป็นจุดกึ่งกลางคงที่ `backswing_peak - 0.5s` (คอมเมนต์ `# Approximation` ในโค้ด) ระยะ 15 เฟรมนี้ ตรงกับ GT median ของ `backswing_peak - unit_turn` พอดี (n=59) → ความแม่นของ B1 ผูกกับความแม่นของ B2 ทั้งหมด
- `recovery_position` (B5) ใช้เกณฑ์ `vel_mag < 0.5` บนพิกัด normalized 0-1 (schema_builder/keyframes.py) = ครึ่งความกว้างภาพต่อเฟรม ซึ่งเข้าเงื่อนไข แทบทุกครั้งที่เฟรมแรกหลัง follow-through → accuracy 0.000 ทุกท่า (median พลาด 17-35 เฟรม) ด้วยเหตุผลนี้ ไม่ใช่เพราะ tracking — **ยังไม่ได้แก้**
- `trophy_position` (B6) ออกเฉพาะเมื่อ classifier ตอบ SV (schema_builder/builder.py) → accuracy ผูกกับ stroke classifier ด้วย ไม่ใช่ความแม่นของ keyframe เพียว ๆ — ตัวเลข classifier ที่เชื่อถือได้คือ **Leave-One-Person-Out 0.727** ไม่ใช่ leave-one-clip-out 0.803 (คนเดียวกัน อยู่หลายคลิป: earth 4 คลิป, poom 3 คลิป → group ด้วยคลิปยังมี leakage ระดับบุคคล)
- BL2/BL3 (ball speed, trajectory clearance) ยังเป็น null เสมอ เพราะไม่มี court calibration — ไม่กระทบตัวเลข keyframe แต่เป็นข้อจำกัดของ output รวม

## Sensitivity — ความผิดปกติใน GT เอง

พบ label ที่ลำดับเวลาผิด 5 จุด (backswing อยู่หลัง/เท่ากับ impact, follow-through อยู่ก่อน impact, ไม่มี follow-through) — ตัวเลขเมื่อ null เฉพาะค่าที่ผิด (ไม่ตัด stroke ทั้งตัว): **0.196** (n=341)

## Failure analysis

failures ทั้งหมด 580 · GT ที่ไม่มี prediction 0

รายละเอียดเต็ม (per-stroke delta, error ต่อ keyframe, anomaly flags) ใน `docs/benchmark_keyframe_results.json`

