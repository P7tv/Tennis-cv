# Benchmark — Keyframe Accuracy vs Ground Truth

## Verdict

**Accuracy: 0.410** (n=344, เป้า TOR ≥ 0.8) — **FAIL ⚠️**

ตัวเลขหลักใช้เกณฑ์ **"อยู่ในช่วงการเคลื่อนไหวเดียวกัน"** ซึ่งเป็นเกณฑ์ที่ลูกค้าเขียนไว้เองใน `cv_schema_table_loeuf`:

> *"Guideline นี้มีไว้เพื่อช่วยเลือก **ช่วงของ keyframe** เท่านั้น **ไม่จำเป็นต้องจับเฟรมได้ตรงเป๊ะ** หากอยู่ในช่วงการเคลื่อนไหวเดียวกันถือว่าใช้ได้"*

ในเอกสารเดียวกันลูกค้ายังกำกับ `impact` และ `follow_through_peak` ว่า **(Low Confidence)** ด้วยตัวเอง — สอดคล้องกับที่ GT 97.6% ถูก annotator flag เป็น `low_confidence`

**นิยามที่ใช้วัด**: เฟรมที่ทายต้องยังใกล้ keyframe นั้นมากกว่า keyframe อื่นของ stroke เดียวกัน (ขอบ = จุดกึ่งกลางระหว่าง keyframe ที่ติดกันใน GT) → หน้าต่างกว้างแคบเองตามจังหวะจริงของแต่ละท่า ไม่ต้องตั้งค่าคงที่ใด ๆ ดู `phase_windows()`

| ตัวเลข | เกณฑ์ | Accuracy | n |
|---|---|---|---|
| **Mode A end-to-end** | ช่วงเดียวกัน (ตามลูกค้า) | **0.410** | 344 |
| Mode B oracle-impact | ช่วงเดียวกัน (ตามลูกค้า) | 0.840 | 344 |
| Mode A end-to-end | ±1 เฟรม (เข้มกว่าที่ลูกค้าขอ) | 0.218 | 344 |
| Mode A matched-only | ±1 เฟรม เฉพาะ stroke ที่ detect เจอ (สูงหลอก) | 0.284 | 264 |
| Mode B oracle-impact | ±1 เฟรม = เพดานของ keyframe logic | 0.759 | 344 |

Mode A end-to-end = จริงตามที่ลูกค้าได้ (stroke ที่ hit detection หาไม่เจอ นับเป็นผิด ไม่ตัดออกจากตัวหาร) · เกณฑ์รวม = impact + backswing_peak ตาม spec

เทียบเป้า **TOR ≥ 0.8**: เกณฑ์ ±1 เฟรมให้ 0.218 = **FAIL ⚠️**

## Provenance

- Ground truth: 16 ไฟล์ / 172 stroke / annotator: atikan
- fps ต้นทาง: 29.97 · ความยาวรวม ~31.3 นาที
- Checkpoint: `runs/pose/tennis_ball_racket_pose_finetune_best.pt`
- git: `72bac1b` · วันที่รัน: 2026-08-05 23:51
- Accuracy tolerance: ±1 เฟรม · Match tolerance: ±10 เฟรม
- **การประเมิน hit classifier: LOPO (โมเดลไม่เคยเห็นคนในคลิปที่วัด)**

  > ⚠️ ตัวเลขต่างกันมากระหว่างสองโหมด — โหมด in-sample ให้ acceptance 0.506 แต่โหมด LOPO ให้ 0.308 บนข้อมูลชุดเดียวกัน ส่วนต่างคือ "การจำคลิป" ที่ปนอยู่ (`hit_classifier.pkl` เทรนจากคลิปเดียวกับที่ใช้วัด) **ตัวเลขที่ควรบอกลูกค้าคือโหมด LOPO** เพราะเป็นตัวที่บอกได้ว่าใช้กับผู้เล่นคนใหม่แล้วจะเป็นยังไง · Mode B ไม่กระทบ (ป้อน impact จากเฉลย ไม่ได้ใช้ classifier)

## Detection (Mode A)

- TP **132** / FN **40** / FP **109**
- Recall **0.767** · Precision **0.548** · FP ต่อนาที 3.49
- Δ (pred − gt) เฉลี่ย 0.2 เฟรม · median 0.5 · p90 |Δ| 7.0
- GT ที่พลาด (40 ตัว): ระยะถึง prediction ที่ใกล้สุด median **42.0 เฟรม** · 17 ตัวอยู่ในระยะ 20 เฟรม

  > ตีความ: ถ้าระยะใกล้ ๆ = pipeline "เห็น" stroke แต่ระบุเฟรมเพี้ยน (ไปแก้ความแม่นของ impact detection) · ถ้าไกลมาก = พลาด stroke ไปเลย (ไปแก้ recall/candidate generation) — คนละปัญหาคนละทางแก้

| Clip | GT | Pred | TP | FN | FP |
|---|---|---|---|---|---|
| set2/IMG_0284 | 2 | 1 | 1 | 1 | 0 |
| set2/IMG_0300B1 | 23 | 26 | 18 | 5 | 8 |
| set2/IMG_0283B-VLbh | 10 | 16 | 7 | 3 | 9 |
| set2/IMG_0283A-VLfh | 10 | 15 | 8 | 2 | 7 |
| set4/IMG_0300A4(VL) | 8 | 3 | 2 | 6 | 1 |
| set4/IMG_0300B52(BH) | 8 | 9 | 8 | 0 | 1 |
| set4/IMG_0301A2(SV1) | 9 | 11 | 7 | 2 | 4 |
| set1/IMG_0266-SL | 10 | 10 | 9 | 1 | 1 |
| set3/IMG_0282(SL) | 11 | 12 | 5 | 6 | 7 |
| set2/IMG_0285A | 10 | 13 | 9 | 1 | 4 |
| set3/IMG_0301A4(SV1) | 11 | 15 | 11 | 0 | 4 |
| set3/IMG_0281(BH) | 16 | 18 | 16 | 0 | 2 |
| set3/IMG_0300B53-VL | 12 | 7 | 3 | 9 | 4 |
| set4/IMG_0300B7-SV1 | 11 | 15 | 11 | 0 | 4 |
| set4/IMG_0300B6-SV1 | 11 | 15 | 11 | 0 | 4 |
| sessions_deferred/set4/IMG_0294(SV1) | 10 | 55 | 6 | 4 | 49 |

## Keyframe accuracy — เกณฑ์ "ช่วงเดียวกัน" (ตามลูกค้า)

คอลัมน์ `หน้าต่าง` = ครึ่งความกว้างเฉลี่ยของช่วงที่ยอมรับ (เฟรม) — คำนวณจาก GT ของ stroke นั้นเอง ไม่ได้ตั้งค่าไว้ล่วงหน้า

| Stroke | Keyframe | n GT | Accuracy (Mode A) | หน้าต่าง ± (เฟรม) | MAE (เฟรม) |
|---|---|---|---|---|---|
| BH | unit_turn | 26 | 0.731 | 7.2 | 4.44 |
| BH | backswing_peak | 26 | 0.615 | 4.5 | 4.32 |
| BH | trophy_position | 0 | — (no GT) | — | — |
| BH | impact | 26 | 0.731 | 4.3 | 2.60 |
| BH | follow_through_peak | 25 | 0.840 | 10.5 | 3.46 |
| BH | recovery_position | 19 | 0.526 | 17.5 | 14.33 |
| FH | unit_turn | 21 | 0.667 | 11.0 | 7.88 |
| FH | backswing_peak | 21 | 0.571 | 6.3 | 3.59 |
| FH | trophy_position | 0 | — (no GT) | — | — |
| FH | impact | 21 | 0.571 | 4.6 | 2.94 |
| FH | follow_through_peak | 21 | 0.762 | 9.9 | 3.76 |
| FH | recovery_position | 16 | 0.438 | 13.7 | 15.00 |
| SL | unit_turn | 21 | 0.286 | 4.4 | 5.79 |
| SL | backswing_peak | 21 | 0.238 | 3.6 | 4.36 |
| SL | trophy_position | 0 | — (no GT) | — | — |
| SL | impact | 21 | 0.476 | 4.7 | 3.57 |
| SL | follow_through_peak | 21 | 0.476 | 7.0 | 5.43 |
| SL | recovery_position | 4 | 0.000 | 10.4 | — |
| SV | unit_turn | 0 | — (no GT) | — | — |
| SV | backswing_peak | 64 | 0.234 | 5.4 | 12.38 |
| SV | trophy_position | 64 | 0.484 | 6.0 | 3.67 |
| SV | impact | 64 | 0.641 | 5.2 | 3.29 |
| SV | follow_through_peak | 64 | 0.531 | 3.9 | 5.29 |
| SV | recovery_position | 0 | — (no GT) | — | — |
| VL | unit_turn | 1 | 0.000 | 0.5 | — |
| VL | backswing_peak | 40 | 0.075 | 5.7 | 6.40 |
| VL | trophy_position | 0 | — (no GT) | — | — |
| VL | impact | 40 | 0.200 | 3.2 | 3.70 |
| VL | follow_through_peak | 40 | 0.275 | 11.6 | 3.80 |
| VL | recovery_position | 4 | 0.000 | 13.0 | 25.00 |

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
| BH | unit_turn | 26 | 0.269 | 4.44 | 3.0 |
| BH | backswing_peak | 26 | 0.462 | 4.32 | 2.0 |
| BH | trophy_position | 0 | — (no GT) | — | — |
| BH | impact | 26 | 0.538 | 2.60 | 1.0 |
| BH | follow_through_peak | 25 | 0.320 | 3.46 | 3.0 |
| BH | recovery_position | 19 | 0.105 | 14.33 | 12.0 |
| FH | unit_turn | 21 | 0.048 | 7.88 | 6.0 |
| FH | backswing_peak | 21 | 0.381 | 3.59 | 2.0 |
| FH | trophy_position | 0 | — (no GT) | — | — |
| FH | impact | 21 | 0.381 | 2.94 | 2.0 |
| FH | follow_through_peak | 21 | 0.286 | 3.76 | 3.0 |
| FH | recovery_position | 16 | 0.000 | 15.00 | 12.5 |
| SL | unit_turn | 21 | 0.000 | 5.79 | 4.5 |
| SL | backswing_peak | 21 | 0.095 | 4.36 | 4.0 |
| SL | trophy_position | 0 | — (no GT) | — | — |
| SL | impact | 21 | 0.048 | 3.57 | 3.5 |
| SL | follow_through_peak | 21 | 0.095 | 5.43 | 4.0 |
| SL | recovery_position | 4 | 0.000 | — | — |
| SV | unit_turn | 0 | — (no GT) | — | — |
| SV | backswing_peak | 64 | 0.078 | 12.38 | 10.0 |
| SV | trophy_position | 64 | 0.156 | 3.67 | 3.0 |
| SV | impact | 64 | 0.297 | 3.29 | 3.0 |
| SV | follow_through_peak | 64 | 0.203 | 5.29 | 4.0 |
| SV | recovery_position | 0 | — (no GT) | — | — |
| VL | unit_turn | 1 | 0.000 | — | — |
| VL | backswing_peak | 40 | 0.025 | 6.40 | 5.0 |
| VL | trophy_position | 0 | — (no GT) | — | — |
| VL | impact | 40 | 0.125 | 3.70 | 4.0 |
| VL | follow_through_peak | 40 | 0.175 | 3.80 | 4.0 |
| VL | recovery_position | 4 | 0.000 | 25.00 | 25.0 |

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

## ข้อจำกัดเชิงโค้ดที่ทราบล่วงหน้า

- 🔴 **`backswing_peak` (B2) เป็นค่าประมาณจากจังหวะ impact ไม่ใช่ผลตรวจจับการเคลื่อนไหว** — schema_builder/keyframes.py ใช้ offset คงที่ (groundstroke `impact-3`, slice `impact-5`) ส่วนเสิร์ฟหาเฟรมที่ข้อมือสูงสุดในช่วง `impact-35..-9` เหตุผล: GT ของลูกค้าวาง B2 ห่าง impact แค่ median 3 เฟรม (0 จาก 78 groundstroke ที่ห่าง ≥20 เฟรม) ไม่ตรงกับนิยาม 'จุดถอยหลังสุด' เชิงพื้นที่ — ทดลองนิยามเชิงจลนศาสตร์แล้วแพ้ค่าคงที่ทุกตัว (velocity reversal 0.144 · argmax|x−x_imp| 1.5s ของเดิม 0.053 vs ค่าคงที่ ~0.49) → **ต้องยืนยันนิยาม B2 กับลูกค้า** ถ้าต้องการจุดถอยหลังสุดจริงต้อง re-label
- ✅ กิ่ง slice ของ B2 (`impact-5`) **พิสูจน์ข้ามคนได้แล้ว** (2026-07-30) — ชุด label เพิ่ม IMG_0266-SL ทำให้ SL มาจาก 2 คน (earth 11 + prem 10) เดิมมาจากคนเดียวจึง Leave-One-Person-Out ให้ 0.00 โดยโครงสร้าง ไม่ใช่เพราะ โมเดลแย่ — ตอนนี้วัดได้จริง Mode B backswing SL = 0.571
- `unit_turn` (B1) **ไม่ได้ detect** — ตั้งเป็นจุดกึ่งกลางคงที่ `backswing_peak - 0.5s` (คอมเมนต์ `# Approximation` ในโค้ด) ระยะ 15 เฟรมนี้ ตรงกับ GT median ของ `backswing_peak - unit_turn` พอดี (n=59) → ความแม่นของ B1 ผูกกับความแม่นของ B2 ทั้งหมด
- `recovery_position` (B5) ใช้เกณฑ์ `vel_mag < 0.5` บนพิกัด normalized 0-1 (schema_builder/keyframes.py) = ครึ่งความกว้างภาพต่อเฟรม ซึ่งเข้าเงื่อนไข แทบทุกครั้งที่เฟรมแรกหลัง follow-through → accuracy 0.000 ทุกท่า (median พลาด 17-35 เฟรม) ด้วยเหตุผลนี้ ไม่ใช่เพราะ tracking — **ยังไม่ได้แก้**
- `trophy_position` (B6) ออกเฉพาะเมื่อ classifier ตอบ SV (schema_builder/builder.py) → accuracy ผูกกับ stroke classifier ด้วย ไม่ใช่ความแม่นของ keyframe เพียว ๆ — ตัวเลข classifier ที่เชื่อถือได้คือ **Leave-One-Person-Out 0.727** ไม่ใช่ leave-one-clip-out 0.803 (คนเดียวกัน อยู่หลายคลิป: earth 4 คลิป, poom 3 คลิป → group ด้วยคลิปยังมี leakage ระดับบุคคล)
- BL2/BL3 (ball speed, trajectory clearance) ยังเป็น null เสมอ เพราะไม่มี court calibration — ไม่กระทบตัวเลข keyframe แต่เป็นข้อจำกัดของ output รวม

## Sensitivity — ความผิดปกติใน GT เอง

พบ label ที่ลำดับเวลาผิด 5 จุด (backswing อยู่หลัง/เท่ากับ impact, follow-through อยู่ก่อน impact, ไม่มี follow-through) — ตัวเลขเมื่อ null เฉพาะค่าที่ผิด (ไม่ตัด stroke ทั้งตัว): **0.220** (n=341)

## Failure analysis

failures ทั้งหมด 560 · GT ที่ไม่มี prediction 0

รายละเอียดเต็ม (per-stroke delta, error ต่อ keyframe, anomaly flags) ใน `docs/benchmark_keyframe_results.json`

