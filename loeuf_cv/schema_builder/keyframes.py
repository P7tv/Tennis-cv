import numpy as np

# ---------------------------------------------------------------------------
# 🔴 หน่วยของค่าคงที่ในไฟล์นี้ — อ่านก่อนแก้
# ---------------------------------------------------------------------------
# ตัวเลข offset ทุกตัวด้านล่างคาลิเบรตจาก GT ของลูกค้าซึ่งถ่ายที่ 29.97fps
# ทั้งชุด ค่าที่เขียนไว้จึงเป็น "จำนวนเฟรมที่ 29.97fps" ไม่ใช่จำนวนเฟรมสัมบูรณ์
#
# ⚠️ ถ้าใช้ค่าเหล่านี้เป็นเฟรมตรง ๆ กับคลิป 60fps จะได้ระยะเวลาเหลือครึ่งเดียว
# เงียบ ๆ (เช่น follow_through ของ FH ที่ตั้งใจให้เป็น 500 ms จะกลายเป็น 250 ms)
# — ไม่ error ไม่มีอะไรเตือน แค่ accuracy ตกทั้งกระดาน  ลูกค้าแจ้งว่าชุดคลิป
# รอบหน้าจะถ่ายที่ >= 60fps ตาม spec (MT4 "fps >= 60 required for KN")
#
# ทุกจุดที่ใช้ค่าพวกนี้จึงต้องผ่าน _f() เพื่อสเกลตาม fps จริงของคลิป
# ที่ fps == CALIBRATION_FPS ผลลัพธ์เท่าเดิมเป๊ะ (สเกล = 1.0) → ตัวเลข
# benchmark ที่วัดไว้ทั้งหมดไม่เปลี่ยน
CALIBRATION_FPS = 29.97

# fps ที่ต่างจากค่าคาลิเบรตน้อยกว่านี้ ให้ถือว่า "เท่ากัน" — cv2 อ่าน fps ของ
# คลิปชุดนี้ได้ 29.9735 ไม่ใช่ 29.97 เป๊ะ ไม่ควรให้ความต่าง 0.01% ไปขยับผล
# (นิยามเดียวกับ hit_detection.FPS_EQUAL_TOLERANCE — ดูเหตุผลเต็มที่นั่น)
FPS_EQUAL_TOLERANCE = 0.02


def _f(frames_at_calib: float, fps: float | None) -> int:
    """แปลง 'จำนวนเฟรมที่ 29.97fps' → จำนวนเฟรมที่ fps จริงของคลิป

    fps ที่เป็น None/0 → ถือว่าเป็นคลิปคาลิเบรต (คืนค่าเดิม) เพื่อไม่ให้
    caller เก่าที่ยังไม่ส่ง fps พังแบบเงียบ ๆ เป็น 0 เฟรม
    """
    if not fps or fps <= 0:
        return int(round(frames_at_calib))
    r = float(fps) / CALIBRATION_FPS
    if abs(r - 1.0) < FPS_EQUAL_TOLERANCE:
        return int(round(frames_at_calib))
    return int(round(frames_at_calib * r))


# ---------------------------------------------------------------------------
# Backswing peak (B2) — ค่าคงที่ที่คาลิเบรตกับ ground truth ของลูกค้า
# ---------------------------------------------------------------------------
# ⚠️ อ่านก่อนแก้ตัวเลขพวกนี้ — มันไม่ใช่ค่าที่เดามา และไม่ใช่ "การตรวจจับ" ด้วย
#
# ของเดิมหา backswing_peak = argmax|x - x_impact| ในช่วง 1.5s ก่อน impact
# (นิยาม "ข้อมือถอยไปไกลสุดในแนวนอน") วัดกับ GT จริง 132 stroke แล้วได้
# accuracy = 0.053 และ 44-71% ของ prediction ไปกองที่ "ขอบหน้าต่างค้นหา" พอดี
# = ค่าที่ได้เป็น artifact ของขนาดหน้าต่าง ไม่ใช่จุดที่วัดได้จากการเคลื่อนไหว
#
# สาเหตุ: GT ของลูกค้าวาง backswing_peak ไว้ใกล้ impact มาก —
#   groundstroke (FH/BH/VL/SL): median 3 เฟรม (100 ms) ก่อน impact
#   และ **0 จาก 78 stroke** ที่ห่าง >= 20 เฟรม (85.9% อยู่ใน 5 เฟรม)
#   serve (SV): median 25 เฟรม (ช่วง 9-35)
# ระยะ 3 เฟรมก่อน impact ไม่ใช่ "จุดถอยหลังสุด" ในเชิงพื้นที่ใด ๆ → นิยามที่
# annotator ใช้จริงต่างจากที่โค้ดเดิมพยายามวัด
#
# ทดลองนิยามเชิงจลนศาสตร์แล้วทั้งหมด แพ้ค่าคงที่ชัดเจน (accuracy, tol ±1 เฟรม):
#   argmax|x-x_imp| 1.5s (ของเดิม) 0.053 · 0.3s 0.076 · velocity reversal 0.144
#   · min speed 0.053 · ค่าคงที่ตามด้านล่าง 0.49
#
# 🔴 สิ่งที่ต้องยืนยันกับลูกค้า: B2 ที่ออกจากที่นี่เป็น "ค่าประมาณจากจังหวะ
# impact" ไม่ใช่ผลตรวจจับการเคลื่อนไหว (สถานะเดียวกับ unit_turn ที่เขียน
# # Approximation ไว้อยู่แล้ว) ถ้านิยามที่ลูกค้าต้องการคือจุดถอยหลังสุดจริง ๆ
# ต้อง re-label แล้วเปลี่ยนมาใช้ velocity reversal
#
# ที่มาของตัวเลข + สคริปต์ทดลอง: docs/BENCHMARK_KEYFRAME.md

# groundstroke — GT median = 3 เฟรม (FH/BH/VL); LOPO 4/5 fold เลือกค่านี้ตรงกัน
GROUND_BACKSWING_OFFSET = 3

# slice — GT median = 5 เฟรม (ง้างค้างนานกว่า groundstroke เล็กน้อย)
# ✅ 2026-07-30: พิสูจน์ข้ามคนได้แล้ว — ชุด label เพิ่ม IMG_0266-SL (prem 10 stroke)
# ทำให้ SL มาจาก 2 คน (earth 11 + prem 10) เดิมมาจากคนเดียวจึง LOPO ให้ 0.00
# โดยโครงสร้าง ตอนนี้วัดได้จริง: Mode B backswing SL = 0.571 (LOPO per-type 0.667)
SLICE_BACKSWING_OFFSET = 5

# แยก slice จาก FH/BH ด้วยทิศความเร็วแนวดิ่งของข้อมือตอน impact
# (y ในพิกัดภาพเพิ่มลงล่าง → vy > 0 = ข้อมือ "สับลง")
# วัดบน GT: SL 72.7% เข้าเงื่อนไข · FH 0% · BH 0% · VL 11.1% → แยก SL จาก
# FH/BH ได้สะอาด แลกกับกิน VL ไป 2 ตัว
SLICE_VY_THRESHOLD = 0.01
SLICE_VY_LOOKBACK = 2          # เฟรมที่ใช้คิด vy (impact-2 → impact)

# ❌ เคยลองเพิ่ม "กิ่ง volley" แยกจาก groundstroke แล้ว **แย่ลง** — ไม่ต้องลองซ้ำ
# แนวคิด: volley เหวี่ยงสั้นกว่า จึงแยกด้วย amplitude ที่ข้อมือกวาดในแนวนอน
# (หารความกว้างไหล่) วัดบน GT 160 stroke: VL median 2.75 · FH 6.99 · BH 6.69
# · SL 4.35 · SV 4.59 → ดูเหมือนแยกได้ แต่ p25 ของ FH (3.74) และ BH (4.15)
# ต่ำกว่า threshold 4.0 → 25% ของ groundstroke ถูกส่งเข้ากิ่ง volley ผิด
# ผลจริง (Mode B backswing_peak): FH 0.905→0.810 · BH 0.808→0.577 ·
# VL 0.575→0.450 (ตัวที่ควรดีขึ้นกลับแย่ เพราะ offset 4 แพ้ 3 สำหรับ VL) ·
# SL 0.571→0.762  รวม acceptance 0.759→0.733
VOLLEY_AMP_WINDOW = 30         # เก็บไว้ให้ _is_volley_swing ที่ยังไม่ถูกเรียกใช้
VOLLEY_AMP_THRESHOLD = 4.0

# serve — ไม่ใช้ค่าคงที่ แต่หาเฟรมที่ข้อมืออยู่ "สูงสุด" (y น้อยสุด) ในช่วงนี้
# = จังหวะ trophy/ง้างสุดของการเสิร์ฟ ซึ่งแปรผันตามความสูงที่แต่ละคนโยนลูก
# (GT SV กระจาย 9-35 เฟรม ค่าคงที่เดียวจึงได้แค่ ~0.20-0.25)
SERVE_SEARCH_MIN = 9
SERVE_SEARCH_MAX = 35
SERVE_FALLBACK_OFFSET = 24     # ใช้เมื่อช่วงค้นหาไม่มีข้อมูล (NaN ล้วน)


def _is_serve_pose(wrist_path, head_path, impact_frame: int) -> bool:
    """ท่าเสิร์ฟ = ตอน impact ข้อมืออยู่เหนือหัว (y น้อยกว่า)

    ใช้ pose ตรง ๆ ไม่พึ่ง stroke classifier เพราะ extract_keyframes() ถูกเรียก
    *ก่อน* classify_stroke() (schema_builder/builder.py) — และไม่ผูกความแม่นของ
    keyframe ไว้กับ classifier ที่ LOPO ได้ ~0.73

    วัดบน GT 132 stroke: recall 72.2% precision 95.1%
    """
    if head_path is None:
        return False
    if impact_frame < 0 or impact_frame >= len(head_path):
        return False
    wy = wrist_path[impact_frame, 1]
    hy = head_path[impact_frame, 1]
    if np.isnan(wy) or np.isnan(hy):
        return False
    return bool(wy < hy)


def _serve_backswing(wrist_path, impact_frame: int, fps: float | None) -> int:
    """เฟรมที่ข้อมืออยู่สูงสุด (min y) ในช่วง trophy ที่เป็นไปได้"""
    ws = max(0, impact_frame - _f(SERVE_SEARCH_MAX, fps))
    we = max(0, impact_frame - _f(SERVE_SEARCH_MIN, fps))
    if we > ws:
        seg = wrist_path[ws:we, 1]
        if not np.isnan(seg).all():
            return ws + int(np.nanargmin(seg))
    return max(0, impact_frame - _f(SERVE_FALLBACK_OFFSET, fps))


def _is_volley_swing(wrist_path, impact_frame: int,
                     shoulder_width: float | None,
                     fps: float | None = None) -> bool:
    """volley = ข้อมือกวาดในแนวนอนสั้น ๆ (บล็อกลูก ไม่เหวี่ยงเต็มวง)

    หาร amplitude ด้วยความกว้างไหล่เพื่อให้เทียบข้ามขนาดคน/ระยะกล้องได้
    ถ้าไม่รู้ความกว้างไหล่ → คืน False (ตกไปใช้ offset ของ groundstroke)
    """
    if not shoulder_width or shoulder_width <= 1e-6:
        return False
    a = max(0, impact_frame - _f(VOLLEY_AMP_WINDOW, fps))
    seg = wrist_path[a:impact_frame + 1, 0]
    if len(seg) < 3 or np.isnan(seg).all():
        return False
    amp = (np.nanmax(seg) - np.nanmin(seg)) / shoulder_width
    return bool(amp < VOLLEY_AMP_THRESHOLD)


def _wrist_vy(wrist_path, impact_frame: int, fps: float | None) -> float:
    """ความเร็วแนวดิ่งของข้อมือตอน impact (บวก = สับลง)

    lookback สเกลตาม fps ด้วย เพราะ SLICE_VY_THRESHOLD เป็น "ระยะที่ข้อมือ
    เลื่อนลงภายในช่วงเวลานั้น" — ถ้าคงจำนวนเฟรมไว้ ช่วงเวลาจะสั้นลงครึ่งหนึ่ง
    ที่ 60fps ระยะที่วัดได้ก็หดตาม แล้ว threshold เดิมจะแทบไม่ trigger เลย
    (SL จะถูกจัดเป็น groundstroke หมด) — สเกลแล้วเทียบเวลาเท่ากันทั้งสอง fps
    """
    i0 = impact_frame - max(1, _f(SLICE_VY_LOOKBACK, fps))
    if i0 < 0 or impact_frame >= len(wrist_path):
        return 0.0
    v = wrist_path[impact_frame, 1] - wrist_path[i0, 1]
    return 0.0 if np.isnan(v) else float(v)


def extract_keyframes(impact_frame: int, wrist_path: np.ndarray, fps: float,
                      max_frames: int, head_path: np.ndarray | None = None,
                      shoulder_width: float | None = None):
    """
    วิเคราะห์หา 5 Keyframes หลักจาก trajectory ของข้อมือ
    wrist_path: numpy array (N, 2) พิกัด x, y ของข้อมือข้างที่ถือไม้ (dominant_wrist)
    head_path: (N, 2) พิกัดจมูก — ใช้แยกท่าเสิร์ฟสำหรับ backswing_peak
               ถ้าไม่ส่งมา จะถือว่าไม่ใช่เสิร์ฟทุกครั้ง (B2 จะเพี้ยนกับ SV)
    shoulder_width: ความกว้างไหล่ (พิกัด normalized เดียวกับ wrist_path) ใช้แยก
               volley ออกจาก groundstroke — ไม่ส่งมาก็ได้ (volley จะใช้ offset
               ของ groundstroke แทน)
    """
    keyframes = {
        "unit_turn": None,
        "backswing_peak": None,
        "impact": impact_frame,
        "follow_through_peak": None,
        "recovery_position": None
    }

    if wrist_path is None or len(wrist_path) == 0:
        return keyframes

    N = len(wrist_path)
    if impact_frame < 0 or impact_frame >= N:
        return keyframes

    # 1. Backswing Peak (B2) — ดูคอมเมนต์บล็อกค่าคงที่ด้านบนก่อนแก้
    if _is_serve_pose(wrist_path, head_path, impact_frame):
        keyframes["backswing_peak"] = _serve_backswing(wrist_path, impact_frame, fps)
    elif _wrist_vy(wrist_path, impact_frame, fps) > SLICE_VY_THRESHOLD:
        keyframes["backswing_peak"] = max(0, impact_frame - _f(SLICE_BACKSWING_OFFSET, fps))
    else:
        keyframes["backswing_peak"] = max(0, impact_frame - _f(GROUND_BACKSWING_OFFSET, fps))

    # 2. Unit Turn (B1):
    # เกิดก่อน backswing peak ประมาณ 0.5-1 วิ
    # (ระยะ 0.5*fps = 15 เฟรม ตรงกับ GT median ของ backswing_peak - unit_turn
    #  พอดี — ตรวจกับ label จริง 59 stroke แล้ว)
    if keyframes["backswing_peak"] is not None:
        bp = keyframes["backswing_peak"]
        ut_search_start = max(0, bp - int(1.0 * fps))
        if bp > ut_search_start:
            keyframes["unit_turn"] = int(np.mean([ut_search_start, bp])) # Approximation

    # 3. Follow Through Peak (B4) + 4. Recovery Position (B5)
    # ค่าเริ่มต้นก่อนรู้ stroke_type — builder จะเรียก refine_keyframes_for_type()
    # ทับอีกทีหลังจำแนกท่าได้ (ดูคอมเมนต์ที่ฟังก์ชันนั้น)
    _apply_type_offsets(keyframes, None, impact_frame, N, fps)

    return keyframes


# ─────────────────────────────────────────────────────────
# offset ต่อ stroke type สำหรับ B4/B5/B6
# ─────────────────────────────────────────────────────────
#
# ที่มา: median ของ (keyframe − impact) จาก GT ลูกค้า 172 stroke
# (scripts/probe_broken_keyframes.py) — ใช้ median ไม่ใช่ค่าที่ fit ได้ดีสุด
# เพราะ median ทนต่อ outlier และ generalize ข้ามคนดีกว่า
#
# ทำไมถึงเลิกใช้วิธี "หาจุดสูงสุดของข้อมือ" กับ B4: วัดแล้วได้ 0.000-0.200
# ส่วน offset คงที่ให้ 26.9% แบบ Leave-One-Person-Out (BH 40% · VL 52%)
# — รูปแบบเดียวกับ backswing_peak ที่นิยามเชิงจลนศาสตร์แพ้ค่าคงที่
#
# ทำไม B5 ยังต่ำอยู่ดี (LOPO 10.3%): GT ของ recovery มี sd 15-16 เฟรม
# = ตัวนิยามเองก็ไม่ชัด ไม่ใช่ว่าโมเดลแย่ แต่ 10.3% ยังดีกว่าของเดิมที่ 0.000
# (ของเดิมใช้เกณฑ์ vel_mag < 0.5 บนพิกัด normalized 0-1 = ครึ่งความกว้างภาพ
#  ต่อเฟรม ซึ่งเข้าเงื่อนไขทันทีที่เฟรมแรกเสมอ → recovery = follow_through)
#
# ❌ เคยลองเปลี่ยน unit_turn (B1) มาใช้ offset ต่อท่าแบบเดียวกันแล้ว **แย่ลง**
#    (LOPO 19.1% เทียบกับของเดิมที่ได้ BH 0.308 / SL 0.286) — ไม่ต้องลองซ้ำ
FOLLOW_THROUGH_OFFSET = {"BH": 13, "FH": 15, "SL": 12, "SV": 9, "VL": 8}
FOLLOW_THROUGH_OFFSET_DEFAULT = 10
RECOVERY_OFFSET = {"BH": 42, "FH": 34, "SL": 31, "VL": 38}
RECOVERY_OFFSET_DEFAULT = 38
TROPHY_OFFSET = -14          # SV เท่านั้น (trophy − impact)


def _apply_type_offsets(keyframes: dict, stroke_type, impact_frame: int, n: int,
                        fps: float | None):
    ft = FOLLOW_THROUGH_OFFSET.get(stroke_type, FOLLOW_THROUGH_OFFSET_DEFAULT)
    keyframes["follow_through_peak"] = min(n - 1, impact_frame + _f(ft, fps))
    rc = RECOVERY_OFFSET.get(stroke_type, RECOVERY_OFFSET_DEFAULT)
    keyframes["recovery_position"] = min(n - 1, impact_frame + _f(rc, fps))


def refine_keyframes_for_type(keyframes: dict, stroke_type: str,
                              impact_frame: int, max_frames: int,
                              fps: float | None = None) -> dict:
    """ปรับ B4/B5 ให้ตรงกับ stroke_type หลังจำแนกท่าได้แล้ว

    extract_keyframes() ถูกเรียกก่อน classify_stroke() (classifier ต้องใช้
    metric ที่คำนวณจาก keyframe ก่อน) จึงยังไม่รู้ท่าตอนนั้น — builder เรียก
    ฟังก์ชันนี้ทับอีกรอบเมื่อรู้ท่าแล้ว

    fps: ไม่ส่ง = ถือว่าเป็นคลิป 29.97fps (ค่าคาลิเบรต) — ดู _f()
    """
    if impact_frame is None or max_frames <= 0:
        return keyframes
    _apply_type_offsets(keyframes, stroke_type, impact_frame, max_frames, fps)
    return keyframes


def trophy_position_frame(stroke_type: str, impact_frame: int,
                          max_frames: int, fps: float | None = None) -> int | None:
    """B6 — serve เท่านั้น

    เดิมใช้ค่าเดียวกับ backswing_peak ของ SV ซึ่งวัดได้แค่ 0.094 เพราะ
    backswing ของ SV เองก็ยังต่ำ (0.219) — ผูกกับ impact ตรง ๆ ได้ LOPO 37.5%
    """
    if stroke_type != "SV" or impact_frame is None:
        return None
    return max(0, min(max_frames - 1, impact_frame + _f(TROPHY_OFFSET, fps)))
