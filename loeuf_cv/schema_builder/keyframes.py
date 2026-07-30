import numpy as np

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
# ⚠️ กิ่งนี้ **ยังพิสูจน์ข้ามคนไม่ได้** เพราะ SL ในชุด label มาจากคนเดียว
# (11 stroke / 1 คน) ตอน Leave-One-Person-Out จึงวัดค่านี้ไม่ได้เลย
# 4/5 fold ที่มี SL อยู่ใน train เลือก 5 ตรงกัน — ต้องเก็บ SL จากคนอื่นมายืนยัน
SLICE_BACKSWING_OFFSET = 5

# แยก slice จาก FH/BH ด้วยทิศความเร็วแนวดิ่งของข้อมือตอน impact
# (y ในพิกัดภาพเพิ่มลงล่าง → vy > 0 = ข้อมือ "สับลง")
# วัดบน GT: SL 72.7% เข้าเงื่อนไข · FH 0% · BH 0% · VL 11.1% → แยก SL จาก
# FH/BH ได้สะอาด แลกกับกิน VL ไป 2 ตัว
SLICE_VY_THRESHOLD = 0.01
SLICE_VY_LOOKBACK = 2          # เฟรมที่ใช้คิด vy (impact-2 → impact)

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


def _serve_backswing(wrist_path, impact_frame: int) -> int:
    """เฟรมที่ข้อมืออยู่สูงสุด (min y) ในช่วง trophy ที่เป็นไปได้"""
    ws = max(0, impact_frame - SERVE_SEARCH_MAX)
    we = max(0, impact_frame - SERVE_SEARCH_MIN)
    if we > ws:
        seg = wrist_path[ws:we, 1]
        if not np.isnan(seg).all():
            return ws + int(np.nanargmin(seg))
    return max(0, impact_frame - SERVE_FALLBACK_OFFSET)


def _wrist_vy(wrist_path, impact_frame: int) -> float:
    """ความเร็วแนวดิ่งของข้อมือตอน impact (บวก = สับลง)"""
    i0 = impact_frame - SLICE_VY_LOOKBACK
    if i0 < 0 or impact_frame >= len(wrist_path):
        return 0.0
    v = wrist_path[impact_frame, 1] - wrist_path[i0, 1]
    return 0.0 if np.isnan(v) else float(v)


def extract_keyframes(impact_frame: int, wrist_path: np.ndarray, fps: float,
                      max_frames: int, head_path: np.ndarray | None = None):
    """
    วิเคราะห์หา 5 Keyframes หลักจาก trajectory ของข้อมือ
    wrist_path: numpy array (N, 2) พิกัด x, y ของข้อมือข้างที่ถือไม้ (dominant_wrist)
    head_path: (N, 2) พิกัดจมูก — ใช้แยกท่าเสิร์ฟสำหรับ backswing_peak
               ถ้าไม่ส่งมา จะถือว่าไม่ใช่เสิร์ฟทุกครั้ง (B2 จะเพี้ยนกับ SV)
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
        keyframes["backswing_peak"] = _serve_backswing(wrist_path, impact_frame)
    elif _wrist_vy(wrist_path, impact_frame) > SLICE_VY_THRESHOLD:
        keyframes["backswing_peak"] = max(0, impact_frame - SLICE_BACKSWING_OFFSET)
    else:
        keyframes["backswing_peak"] = max(0, impact_frame - GROUND_BACKSWING_OFFSET)

    # 2. Unit Turn (B1):
    # เกิดก่อน backswing peak ประมาณ 0.5-1 วิ
    # (ระยะ 0.5*fps = 15 เฟรม ตรงกับ GT median ของ backswing_peak - unit_turn
    #  พอดี — ตรวจกับ label จริง 59 stroke แล้ว)
    if keyframes["backswing_peak"] is not None:
        bp = keyframes["backswing_peak"]
        ut_search_start = max(0, bp - int(1.0 * fps))
        if bp > ut_search_start:
            keyframes["unit_turn"] = int(np.mean([ut_search_start, bp])) # Approximation

    # 3. Follow Through Peak (B4):
    # จุดสูงสุดของการเหวี่ยงข้อมือหลัง impact ในช่วง 1.0 วิ
    window_end = min(N, impact_frame + int(1.0 * fps))
    if window_end > impact_frame:
        path_after_y = wrist_path[impact_frame:window_end, 1]
        if not np.isnan(path_after_y).all():
            # Y น้อยสุด = จุดที่อยู่สูงที่สุดบนจอ (จอภาพ 0 อยู่บนสุด)
            peak_idx_local = np.nanargmin(path_after_y)
            keyframes["follow_through_peak"] = impact_frame + int(peak_idx_local)

    # 4. Recovery Position (B5):
    # จุดที่ผู้เล่นกลับมาหยุดนิ่ง หลัง follow_through
    if keyframes["follow_through_peak"] is not None:
        ftp = keyframes["follow_through_peak"]
        rec_end = min(N, ftp + int(2.5 * fps))
        if rec_end > ftp:
            vel_x = np.diff(wrist_path[ftp:rec_end, 0])
            vel_y = np.diff(wrist_path[ftp:rec_end, 1])
            vel_mag = np.sqrt(vel_x**2 + vel_y**2)
            for i, v in enumerate(vel_mag):
                if v < 0.5: # ความเร็วน้อยมาก
                    keyframes["recovery_position"] = ftp + i
                    break
            if keyframes["recovery_position"] is None:
                keyframes["recovery_position"] = rec_end - 1

    return keyframes
