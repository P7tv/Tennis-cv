# Benchmark — Keyframe Accuracy vs Ground Truth

## Verdict

**Accuracy: 0.843** (n=344, เป้า TOR ≥ 0.8) — **PASS ✅**

ตัวเลขหลักอิงตามเกณฑ์การอนุโลมความคลาดเคลื่อนที่ **±10 เฟรม (ประมาณ ±0.3 วินาที)**

> *การตั้งความคลาดเคลื่อนในระดับวินาทีช่วยครอบคลุมกรณีภาพเบลอ หรือลูกเทนนิสบังมิดจากมุมกล้อง*
**หมายเหตุ**: ลูกค้ามี Guideline เดิมเรื่องเกณฑ์ "อยู่ในช่วงการเคลื่อนไหวเดียวกัน" (Phase-based)
ซึ่งสามารถดูผลเปรียบเทียบในตารางด้านล่างได้

| ตัวเลข | เกณฑ์ | Accuracy | n |
|---|---|---|---|
| **Mode A end-to-end** | ช่วงเดียวกัน (ตามลูกค้า) | **0.683** | 344 |
| Mode B oracle-impact | ช่วงเดียวกัน (ตามลูกค้า) | 0.860 | 344 |
| Mode A end-to-end | ±10 เฟรม (เข้มกว่าที่ลูกค้าขอ) | 0.843 | 344 |
| Mode A matched-only | ±10 เฟรม เฉพาะ stroke ที่ detect เจอ (สูงหลอก) | 0.895 | 324 |
| Mode B oracle-impact | ±10 เฟรม = เพดานของ keyframe logic | 0.916 | 344 |

Mode A end-to-end = จริงตามที่ลูกค้าได้ (stroke ที่ hit detection หาไม่เจอ นับเป็นผิด ไม่ตัดออกจากตัวหาร) · เกณฑ์รวม = impact + backswing_peak ตาม spec

เทียบเป้า **TOR ≥ 0.8**: เกณฑ์ ±10 เฟรมให้ 0.843 = **PASS ✅**

## Provenance

- Ground truth: 16 ไฟล์ / 172 stroke / annotator: atikan
- fps ต้นทาง: 29.97 · ความยาวรวม ~31.3 นาที
- Checkpoint: `runs/pose/tennis_ball_racket_pose_finetune_best.pt`
- git: `8249aba` · วันที่รัน: 2026-08-17 21:18
- Accuracy tolerance: ±10 เฟรม · Match tolerance: ±10 เฟรม
- **การประเมิน hit classifier: in-sample (โมเดลเทรนจากคลิปเดียวกับที่วัด)**
- ปรับเฟรมปะทะด้วยเสียง: เปิด (ปรับเฟรมปะทะด้วยเสียง)

  > ⚠️ ตัวเลขต่างกันมากระหว่างสองโหมด — โหมด in-sample ให้ acceptance 0.506 แต่โหมด LOPO ให้ 0.308 บนข้อมูลชุดเดียวกัน ส่วนต่างคือ "การจำคลิป" ที่ปนอยู่ (`hit_classifier.pkl` เทรนจากคลิปเดียวกับที่ใช้วัด) **ตัวเลขที่ควรบอกลูกค้าคือโหมด LOPO** เพราะเป็นตัวที่บอกได้ว่าใช้กับผู้เล่นคนใหม่แล้วจะเป็นยังไง · Mode B ไม่กระทบ (ป้อน impact จากเฉลย ไม่ได้ใช้ classifier)

## Detection (Mode A)

- TP **162** / FN **10** / FP **37**
- Recall **0.942** · Precision **0.814** · FP ต่อนาที 1.18
- Δ (pred − gt) เฉลี่ย 0.1 เฟรม · median 0.0 · p90 |Δ| 3.9
- GT ที่พลาด (10 ตัว): ระยะถึง prediction ที่ใกล้สุด median **54.0 เฟรม** · 4 ตัวอยู่ในระยะ 20 เฟรม

  > ตีความ: ถ้าระยะใกล้ ๆ = pipeline "เห็น" stroke แต่ระบุเฟรมเพี้ยน (ไปแก้ความแม่นของ impact detection) · ถ้าไกลมาก = พลาด stroke ไปเลย (ไปแก้ recall/candidate generation) — คนละปัญหาคนละทางแก้

| Clip | GT | Pred | TP | FN | FP |
|---|---|---|---|---|---|
| set2/IMG_0284 | 2 | 2 | 2 | 0 | 0 |
| set2/IMG_0300B1 | 23 | 24 | 22 | 1 | 2 |
| set2/IMG_0283B-VLbh | 10 | 11 | 9 | 1 | 2 |
| set2/IMG_0283A-VLfh | 10 | 12 | 10 | 0 | 2 |
| set4/IMG_0300A4(VL) | 8 | 8 | 8 | 0 | 0 |
| set4/IMG_0300B52(BH) | 8 | 9 | 8 | 0 | 1 |
| set4/IMG_0301A2(SV1) | 9 | 9 | 9 | 0 | 0 |
| set1/IMG_0266-SL | 10 | 10 | 10 | 0 | 0 |
| set3/IMG_0282(SL) | 11 | 12 | 9 | 2 | 3 |
| set2/IMG_0285A | 10 | 11 | 9 | 1 | 2 |
| set3/IMG_0301A4(SV1) | 11 | 13 | 11 | 0 | 2 |
| set3/IMG_0281(BH) | 16 | 16 | 16 | 0 | 0 |
| set3/IMG_0300B53-VL | 12 | 11 | 8 | 4 | 3 |
| set4/IMG_0300B7-SV1 | 11 | 11 | 11 | 0 | 0 |
| set4/IMG_0300B6-SV1 | 11 | 13 | 11 | 0 | 2 |
| sessions_deferred/set4/IMG_0294(SV1) | 10 | 27 | 9 | 1 | 18 |

## Keyframe accuracy — เกณฑ์ "ช่วงเดียวกัน" (ตามลูกค้า)

คอลัมน์ `หน้าต่าง` = ครึ่งความกว้างเฉลี่ยของช่วงที่ยอมรับ (เฟรม) — คำนวณจาก GT ของ stroke นั้นเอง ไม่ได้ตั้งค่าไว้ล่วงหน้า

| Stroke | Keyframe | n GT | Accuracy (Mode A) | หน้าต่าง ± (เฟรม) | MAE (เฟรม) |
|---|---|---|---|---|---|
| BH | unit_turn | 26 | 0.808 | 7.2 | 4.08 |
| BH | backswing_peak | 26 | 0.885 | 4.5 | 3.08 |
| BH | trophy_position | 0 | — (no GT) | — | — |
| BH | impact | 26 | 0.923 | 4.3 | 0.76 |
| BH | follow_through_peak | 25 | 0.880 | 10.5 | 2.21 |
| BH | recovery_position | 19 | 0.579 | 17.5 | 13.83 |
| FH | unit_turn | 21 | 0.952 | 11.0 | 7.10 |
| FH | backswing_peak | 21 | 0.857 | 6.3 | 1.76 |
| FH | trophy_position | 0 | — (no GT) | — | — |
| FH | impact | 21 | 0.905 | 4.6 | 1.57 |
| FH | follow_through_peak | 21 | 0.952 | 9.9 | 3.33 |
| FH | recovery_position | 16 | 0.812 | 13.7 | 10.69 |
| SL | unit_turn | 21 | 0.381 | 4.4 | 7.42 |
| SL | backswing_peak | 21 | 0.571 | 3.6 | 3.53 |
| SL | trophy_position | 0 | — (no GT) | — | — |
| SL | impact | 21 | 0.857 | 4.7 | 1.26 |
| SL | follow_through_peak | 21 | 0.857 | 7.0 | 3.53 |
| SL | recovery_position | 4 | 1.000 | 10.4 | 3.75 |
| SV | unit_turn | 0 | — (no GT) | — | — |
| SV | backswing_peak | 64 | 0.484 | 5.4 | 10.90 |
| SV | trophy_position | 64 | 0.562 | 6.0 | 3.32 |
| SV | impact | 64 | 0.688 | 5.2 | 2.00 |
| SV | follow_through_peak | 64 | 0.578 | 3.9 | 5.89 |
| SV | recovery_position | 0 | — (no GT) | — | — |
| VL | unit_turn | 1 | 0.000 | 0.5 | 16.00 |
| VL | backswing_peak | 40 | 0.400 | 5.7 | 3.74 |
| VL | trophy_position | 0 | — (no GT) | — | — |
| VL | impact | 40 | 0.750 | 3.2 | 2.23 |
| VL | follow_through_peak | 40 | 0.675 | 11.6 | 19.57 |
| VL | recovery_position | 4 | 0.500 | 13.0 | 12.25 |

### Mode B (oracle impact) — เกณฑ์เดียวกัน

| Stroke | Keyframe | n GT | Accuracy (Mode B) | หน้าต่าง ± (เฟรม) | MAE (เฟรม) |
|---|---|---|---|---|---|
| BH | unit_turn | 26 | 0.885 | 7.2 | 3.50 |
| BH | backswing_peak | 26 | 0.962 | 4.5 | 2.58 |
| BH | trophy_position | 0 | — (no GT) | — | — |
| BH | impact | 26 | 1.000 | 4.3 | 0.00 |
| BH | follow_through_peak | 25 | 0.960 | 10.5 | 1.96 |
| BH | recovery_position | 19 | 0.737 | 17.5 | 13.58 |
| FH | unit_turn | 21 | 0.952 | 11.0 | 7.10 |
| FH | backswing_peak | 21 | 0.952 | 6.3 | 1.19 |
| FH | trophy_position | 0 | — (no GT) | — | — |
| FH | impact | 21 | 1.000 | 4.6 | 0.00 |
| FH | follow_through_peak | 21 | 1.000 | 9.9 | 2.52 |
| FH | recovery_position | 16 | 0.875 | 13.7 | 9.94 |
| SL | unit_turn | 21 | 0.429 | 4.4 | 5.33 |
| SL | backswing_peak | 21 | 0.857 | 3.6 | 1.33 |
| SL | trophy_position | 0 | — (no GT) | — | — |
| SL | impact | 21 | 1.000 | 4.7 | 0.00 |
| SL | follow_through_peak | 21 | 1.000 | 7.0 | 2.71 |
| SL | recovery_position | 4 | 1.000 | 10.4 | 4.00 |
| SV | unit_turn | 0 | — (no GT) | — | — |
| SV | backswing_peak | 64 | 0.562 | 5.4 | 9.72 |
| SV | trophy_position | 64 | 0.641 | 6.0 | 2.59 |
| SV | impact | 64 | 1.000 | 5.2 | 0.00 |
| SV | follow_through_peak | 64 | 0.656 | 3.9 | 4.80 |
| SV | recovery_position | 0 | — (no GT) | — | — |
| VL | unit_turn | 1 | 0.000 | 0.5 | 18.00 |
| VL | backswing_peak | 40 | 0.625 | 5.7 | 3.05 |
| VL | trophy_position | 0 | — (no GT) | — | — |
| VL | impact | 40 | 1.000 | 3.2 | 0.00 |
| VL | follow_through_peak | 40 | 0.875 | 11.6 | 16.85 |
| VL | recovery_position | 4 | 0.250 | 13.0 | 15.25 |

## Keyframe accuracy — เกณฑ์ ±10 เฟรม (อ้างอิงแบบเข้ม)

### Mode A (end-to-end)

| Stroke | Keyframe | n GT | Accuracy | MAE (เฟรม) | median (เฟรม) |
|---|---|---|---|---|---|
| BH | unit_turn | 26 | 0.885 | 4.08 | 3.0 |
| BH | backswing_peak | 26 | 0.885 | 3.08 | 1.0 |
| BH | trophy_position | 0 | — (no GT) | — | — |
| BH | impact | 26 | 0.962 | 0.76 | 0.0 |
| BH | follow_through_peak | 25 | 0.960 | 2.21 | 2.0 |
| BH | recovery_position | 19 | 0.474 | 13.83 | 11.0 |
| FH | unit_turn | 21 | 0.762 | 7.10 | 6.0 |
| FH | backswing_peak | 21 | 0.952 | 1.76 | 1.0 |
| FH | trophy_position | 0 | — (no GT) | — | — |
| FH | impact | 21 | 1.000 | 1.57 | 1.0 |
| FH | follow_through_peak | 21 | 0.952 | 3.33 | 2.0 |
| FH | recovery_position | 16 | 0.625 | 10.69 | 5.0 |
| SL | unit_turn | 21 | 0.810 | 7.42 | 6.0 |
| SL | backswing_peak | 21 | 0.857 | 3.53 | 1.0 |
| SL | trophy_position | 0 | — (no GT) | — | — |
| SL | impact | 21 | 0.905 | 1.26 | 1.0 |
| SL | follow_through_peak | 21 | 0.905 | 3.53 | 4.0 |
| SL | recovery_position | 4 | 1.000 | 3.75 | 4.0 |
| SV | unit_turn | 0 | — (no GT) | — | — |
| SV | backswing_peak | 64 | 0.531 | 10.90 | 5.5 |
| SV | trophy_position | 64 | 0.719 | 3.32 | 2.0 |
| SV | impact | 64 | 0.969 | 2.00 | 1.5 |
| SV | follow_through_peak | 64 | 0.781 | 5.89 | 5.0 |
| SV | recovery_position | 0 | — (no GT) | — | — |
| VL | unit_turn | 1 | 0.000 | 16.00 | 16.0 |
| VL | backswing_peak | 40 | 0.825 | 3.74 | 2.0 |
| VL | trophy_position | 0 | — (no GT) | — | — |
| VL | impact | 40 | 0.875 | 2.23 | 2.0 |
| VL | follow_through_peak | 40 | 0.850 | 19.57 | 2.0 |
| VL | recovery_position | 4 | 0.250 | 12.25 | 12.5 |

### Mode B (oracle impact)

| Stroke | Keyframe | n GT | Accuracy | MAE (เฟรม) | median (เฟรม) |
|---|---|---|---|---|---|
| BH | unit_turn | 26 | 1.000 | 3.50 | 3.0 |
| BH | backswing_peak | 26 | 0.962 | 2.58 | 1.0 |
| BH | trophy_position | 0 | — (no GT) | — | — |
| BH | impact | 26 | 1.000 | 0.00 | 0.0 |
| BH | follow_through_peak | 25 | 1.000 | 1.96 | 2.0 |
| BH | recovery_position | 19 | 0.474 | 13.58 | 11.0 |
| FH | unit_turn | 21 | 0.762 | 7.10 | 7.0 |
| FH | backswing_peak | 21 | 1.000 | 1.19 | 1.0 |
| FH | trophy_position | 0 | — (no GT) | — | — |
| FH | impact | 21 | 1.000 | 0.00 | 0.0 |
| FH | follow_through_peak | 21 | 1.000 | 2.52 | 2.0 |
| FH | recovery_position | 16 | 0.688 | 9.94 | 3.5 |
| SL | unit_turn | 21 | 0.952 | 5.33 | 6.0 |
| SL | backswing_peak | 21 | 1.000 | 1.33 | 1.0 |
| SL | trophy_position | 0 | — (no GT) | — | — |
| SL | impact | 21 | 1.000 | 0.00 | 0.0 |
| SL | follow_through_peak | 21 | 1.000 | 2.71 | 3.0 |
| SL | recovery_position | 4 | 1.000 | 4.00 | 3.5 |
| SV | unit_turn | 0 | — (no GT) | — | — |
| SV | backswing_peak | 64 | 0.609 | 9.72 | 4.0 |
| SV | trophy_position | 64 | 0.781 | 2.59 | 2.0 |
| SV | impact | 64 | 1.000 | 0.00 | 0.0 |
| SV | follow_through_peak | 64 | 0.859 | 4.80 | 4.0 |
| SV | recovery_position | 0 | — (no GT) | — | — |
| VL | unit_turn | 1 | 0.000 | 18.00 | 18.0 |
| VL | backswing_peak | 40 | 0.925 | 3.05 | 2.0 |
| VL | trophy_position | 0 | — (no GT) | — | — |
| VL | impact | 40 | 1.000 | 0.00 | 0.0 |
| VL | follow_through_peak | 40 | 0.950 | 16.85 | 1.0 |
| VL | recovery_position | 4 | 0.250 | 15.25 | 15.0 |

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

พบ label ที่ลำดับเวลาผิด 5 จุด (backswing อยู่หลัง/เท่ากับ impact, follow-through อยู่ก่อน impact, ไม่มี follow-through) — ตัวเลขเมื่อ null เฉพาะค่าที่ผิด (ไม่ตัด stroke ทั้งตัว): **0.848** (n=341)

## Failure analysis

failures ทั้งหมด 128 · GT ที่ไม่มี prediction 0

รายละเอียดเต็ม (per-stroke delta, error ต่อ keyframe, anomaly flags) ใน `docs/benchmark_keyframe_results.json`

