# Benchmark — Keyframe Accuracy vs Ground Truth

## Verdict

**Accuracy: 0.192** (n=344, เป้า TOR ≥ 0.8) — **FAIL ⚠️**

ตัวเลขหลัก = **Mode A end-to-end** บน GT ทุก stroke (stroke ที่ hit detection หาไม่เจอ นับเป็นผิด ไม่ตัดออกจากตัวหาร) เกณฑ์ = impact + backswing_peak ตาม spec

| ตัวเลข | ความหมาย | Accuracy | n |
|---|---|---|---|
| **Mode A end-to-end** | จริงตามที่ลูกค้าได้ | **0.192** | 344 |
| Mode A matched-only | เฉพาะ stroke ที่ detect เจอ (สูงหลอก) | 0.268 | 246 |
| Mode B oracle-impact | ป้อน GT impact เข้าไป = เพดานของ keyframe logic | 0.759 | 344 |

## Provenance

- Ground truth: 16 ไฟล์ / 172 stroke / annotator: atikan
- fps ต้นทาง: 29.97 · ความยาวรวม ~31.3 นาที
- Checkpoint: `runs/pose/tennis_ball_racket_pose_finetune_best.pt`
- git: `e83cbe9` · วันที่รัน: 2026-08-01 07:58
- Accuracy tolerance: ±1 เฟรม · Match tolerance: ±10 เฟรม

## Detection (Mode A)

- TP **123** / FN **49** / FP **283**
- Recall **0.715** · Precision **0.303** · FP ต่อนาที 9.05
- Δ (pred − gt) เฉลี่ย 0.0 เฟรม · median 0.0 · p90 |Δ| 8.0
- GT ที่พลาด (49 ตัว): ระยะถึง prediction ที่ใกล้สุด median **41.0 เฟรม** · 14 ตัวอยู่ในระยะ 20 เฟรม

  > ตีความ: ถ้าระยะใกล้ ๆ = pipeline "เห็น" stroke แต่ระบุเฟรมเพี้ยน (ไปแก้ความแม่นของ impact detection) · ถ้าไกลมาก = พลาด stroke ไปเลย (ไปแก้ recall/candidate generation) — คนละปัญหาคนละทางแก้

| Clip | GT | Pred | TP | FN | FP |
|---|---|---|---|---|---|
| set2/IMG_0284 | 2 | 3 | 0 | 2 | 3 |
| set2/IMG_0300B1 | 23 | 25 | 19 | 4 | 6 |
| set2/IMG_0283B-VLbh | 10 | 18 | 7 | 3 | 11 |
| set2/IMG_0283A-VLfh | 10 | 7 | 2 | 8 | 5 |
| set4/IMG_0300A4(VL) | 8 | 5 | 4 | 4 | 1 |
| set4/IMG_0300B52(BH) | 8 | 9 | 8 | 0 | 1 |
| set4/IMG_0301A2(SV1) | 9 | 12 | 9 | 0 | 3 |
| set1/IMG_0266-SL | 10 | 13 | 7 | 3 | 6 |
| set3/IMG_0282(SL) | 11 | 26 | 8 | 3 | 18 |
| set2/IMG_0285A | 10 | 26 | 8 | 2 | 18 |
| set3/IMG_0301A4(SV1) | 11 | 33 | 9 | 2 | 24 |
| set3/IMG_0281(BH) | 16 | 22 | 14 | 2 | 8 |
| set3/IMG_0300B53-VL | 12 | 13 | 3 | 9 | 10 |
| set4/IMG_0300B7-SV1 | 11 | 29 | 11 | 0 | 18 |
| set4/IMG_0300B6-SV1 | 11 | 20 | 9 | 2 | 11 |
| sessions_deferred/set4/IMG_0294(SV1) | 10 | 145 | 5 | 5 | 140 |

## Keyframe accuracy — Mode A (end-to-end)

| Stroke | Keyframe | n GT | Accuracy | MAE (เฟรม) | median (เฟรม) |
|---|---|---|---|---|---|
| BH | unit_turn | 26 | 0.192 | 4.39 | 3.0 |
| BH | backswing_peak | 26 | 0.423 | 4.39 | 2.0 |
| BH | trophy_position | 0 | — (no GT) | — | — |
| BH | impact | 26 | 0.462 | 2.70 | 1.0 |
| BH | follow_through_peak | 25 | 0.120 | 6.18 | 5.0 |
| BH | recovery_position | 19 | 0.000 | 40.00 | 34.0 |
| FH | unit_turn | 21 | 0.048 | 8.94 | 8.5 |
| FH | backswing_peak | 21 | 0.476 | 2.72 | 1.0 |
| FH | trophy_position | 0 | — (no GT) | — | — |
| FH | impact | 21 | 0.429 | 2.39 | 1.5 |
| FH | follow_through_peak | 21 | 0.000 | 8.50 | 9.0 |
| FH | recovery_position | 16 | 0.000 | 37.15 | 37.0 |
| SL | unit_turn | 21 | 0.048 | 5.40 | 5.0 |
| SL | backswing_peak | 21 | 0.143 | 6.67 | 7.0 |
| SL | trophy_position | 0 | — (no GT) | — | — |
| SL | impact | 21 | 0.143 | 5.20 | 4.0 |
| SL | follow_through_peak | 21 | 0.000 | 11.80 | 12.0 |
| SL | recovery_position | 4 | 0.000 | 20.00 | 19.0 |
| SV | unit_turn | 0 | — (no GT) | — | — |
| SV | backswing_peak | 64 | 0.016 | 12.53 | 10.0 |
| SV | trophy_position | 64 | 0.031 | 6.91 | 6.0 |
| SV | impact | 64 | 0.188 | 3.31 | 3.0 |
| SV | follow_through_peak | 64 | 0.094 | 8.22 | 8.0 |
| SV | recovery_position | 0 | — (no GT) | — | — |
| VL | unit_turn | 1 | 0.000 | — | — |
| VL | backswing_peak | 40 | 0.050 | 6.00 | 4.5 |
| VL | trophy_position | 0 | — (no GT) | — | — |
| VL | impact | 40 | 0.075 | 4.75 | 5.0 |
| VL | follow_through_peak | 40 | 0.025 | 8.31 | 8.0 |
| VL | recovery_position | 4 | 0.000 | 32.00 | 32.0 |

## Keyframe accuracy — Mode B (oracle impact)

| Stroke | Keyframe | n GT | Accuracy | MAE (เฟรม) | median (เฟรม) |
|---|---|---|---|---|---|
| BH | unit_turn | 26 | 0.308 | 3.15 | 3.0 |
| BH | backswing_peak | 26 | 0.808 | 2.23 | 1.0 |
| BH | trophy_position | 0 | — (no GT) | — | — |
| BH | impact | 26 | 1.000 | 0.00 | 0.0 |
| BH | follow_through_peak | 25 | 0.200 | 5.04 | 5.0 |
| BH | recovery_position | 19 | 0.000 | 39.47 | 35.0 |
| FH | unit_turn | 21 | 0.095 | 7.71 | 7.0 |
| FH | backswing_peak | 21 | 0.905 | 0.76 | 0.0 |
| FH | trophy_position | 0 | — (no GT) | — | — |
| FH | impact | 21 | 1.000 | 0.00 | 0.0 |
| FH | follow_through_peak | 21 | 0.000 | 8.33 | 9.0 |
| FH | recovery_position | 16 | 0.000 | 34.75 | 28.0 |
| SL | unit_turn | 21 | 0.286 | 4.81 | 5.0 |
| SL | backswing_peak | 21 | 0.571 | 1.86 | 1.0 |
| SL | trophy_position | 0 | — (no GT) | — | — |
| SL | impact | 21 | 1.000 | 0.00 | 0.0 |
| SL | follow_through_peak | 21 | 0.095 | 8.90 | 10.0 |
| SL | recovery_position | 4 | 0.000 | 16.25 | 17.5 |
| SV | unit_turn | 0 | — (no GT) | — | — |
| SV | backswing_peak | 64 | 0.219 | 10.42 | 7.0 |
| SV | trophy_position | 64 | 0.094 | 6.54 | 5.0 |
| SV | impact | 64 | 1.000 | 0.00 | 0.0 |
| SV | follow_through_peak | 64 | 0.109 | 7.00 | 7.0 |
| SV | recovery_position | 0 | — (no GT) | — | — |
| VL | unit_turn | 1 | 0.000 | 17.00 | 17.0 |
| VL | backswing_peak | 40 | 0.575 | 2.48 | 1.0 |
| VL | trophy_position | 0 | — (no GT) | — | — |
| VL | impact | 40 | 1.000 | 0.00 | 0.0 |
| VL | follow_through_peak | 40 | 0.050 | 22.68 | 7.0 |
| VL | recovery_position | 4 | 0.000 | 31.50 | 32.0 |

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

พบ label ที่ลำดับเวลาผิด 5 จุด (backswing อยู่หลัง/เท่ากับ impact, follow-through อยู่ก่อน impact, ไม่มี follow-through) — ตัวเลขเมื่อ null เฉพาะค่าที่ผิด (ไม่ตัด stroke ทั้งตัว): **0.194** (n=341)

## Failure analysis

failures ทั้งหมด 606 · GT ที่ไม่มี prediction 0

รายละเอียดเต็ม (per-stroke delta, error ต่อ keyframe, anomaly flags) ใน `docs/benchmark_keyframe_results.json`

