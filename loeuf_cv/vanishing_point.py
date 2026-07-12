"""Vanishing-point-based focal length recovery — closed-form fix for the
focal-length↔distance ambiguity validated empirically in court_model.py
(CHANGELOG 0.2.6/0.2.7: same-depth points, or unconstrained focal_length
bounds in general, let the random search converge on wrong-but-high-scoring
poses even with zero noise).

แนวคิด (single-view metrology — Criminisi/Reid/Zisserman): สนามเทนนิสเป็น
สี่เหลี่ยมผืนผ้า → เส้นข้างสนาม (ขนานกัน 1 ชุด) กับเส้น baseline/service
line (ขนานกันอีกชุด) ตั้งฉากกันจริงในโลก 3 มิติ เมื่อฉายลงภาพ เส้นขนาน
แต่ละชุดลู่เข้าหา "vanishing point" (VP) จุดเดียว — เพราะสองชุดตั้งฉากกัน
จริง ตำแหน่ง VP ทั้งสองจุดจึงคำนวณ focal length ได้ตรง ๆ ด้วยสูตรปิด
ไม่ต้อง search เดาเหมือน court_model.py เดิม (ซึ่งเป็นที่มาของปัญหา
ambiguity ที่เจอมาตลอด)

ไม่ต้องมี label/ข้อมูลฝึกโมเดลเพิ่ม — เป็นเรขาคณิตล้วน คำนวณจาก
line segment ที่ detect ได้ในเฟรมเดียว

⚠️ ข้อจำกัดสำคัญที่ validate แล้ว (2026-07-09, ก่อนใช้งานจริงต้องอ่าน):
วิธีนี้ใช้ไม่ได้ดีกับ setup กล้องมาตรฐานที่แนะนำมาตลอดในโปรเจกต์นี้ —
กล้องอยู่ "กึ่งกลางหลัง baseline พอดี" (pan_deg≈0) ทำให้เส้นทิศทางกว้าง
(baseline/service line) เกือบขนานกันสนิทในภาพ vanishing point ของกลุ่ม
นั้นเลยอยู่ไกลจนเกือบเป็น infinity — focal length ที่คำนวณได้เลย"ไวต่อ
noise มาก" (validated: pan=0-2° กับ noise 1px บน endpoint ทำให้ focal
เพี้ยนได้ 30-65%, ส่วน pan>=10° เพี้ยนแค่ ~3-5%) มีการันตี reliability
gate (`_reliability_check`) ที่จะคืน `None` แทนค่าที่ไม่น่าเชื่อถือ —
แปลว่าฟังก์ชันนี้จะคืน `None` บ่อยมากสำหรับ footage แบบมาตรฐานของ
โปรเจกต์นี้ นี่คือพฤติกรรมที่ตั้งใจ ไม่ใช่บั๊ก ใช้เป็น "hint เสริมเมื่อ
มีโชค" ไม่ใช่ทางแก้หลัก — ทางแก้หลักที่ validate แล้วว่าเสถียรกว่าคือ
manual point-click + depth-spread requirement (ดู court_model.py,
court_click_tool.html, CHANGELOG 0.2.6/0.2.7)
"""

import cv2
import numpy as np

COURT_SURFACE_TOP_FRAC = 0.35  # ตัดโซนรั้ว/หลังคา/ท้องฟ้าทิ้ง เหมือน court_model.py
MIN_SEGMENT_LENGTH_PX = 40
ANGLE_CONSISTENCY_DEG = 5.0  # segment ถือว่า "เห็นด้วย" กับ VP candidate ถ้ามุมต่างไม่เกินนี้
MIN_VP_INLIERS = 3


def detect_line_segments(frame: np.ndarray,
                         court_surface_top_frac: float = COURT_SURFACE_TOP_FRAC,
                         min_length_px: float = MIN_SEGMENT_LENGTH_PX) -> np.ndarray:
    """หา line segment ของเส้นขาว (LSD บนพื้นที่ mask เฉพาะโซนพื้นสนาม)
    คืน array shape (N, 4) แต่ละแถว (x1,y1,x2,y2)"""
    masked = frame.copy()
    masked[:int(frame.shape[0] * court_surface_top_frac), :] = 0
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(bright)[0]
    if lines is None:
        return np.empty((0, 4))
    segs = lines.reshape(-1, 4)
    lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    return segs[lengths >= min_length_px]


def _line_intersection(seg_a, seg_b):
    """จุดตัดของเส้นตรงที่ผ่าน seg_a, seg_b (เส้นเต็ม ไม่ใช่แค่ segment)
    คืน None ถ้าขนานกัน (หรือเกือบขนาน — ไม่มีจุดตัดที่เสถียร)"""
    x1, y1, x2, y2 = seg_a
    x3, y3, x4, y4 = seg_b
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d
    return (px, py)


def _segment_angle(seg) -> float:
    x1, y1, x2, y2 = seg
    return np.arctan2(y2 - y1, x2 - x1) % np.pi  # 0..pi, ไม่สนใจทิศทาง


def _angle_diff(a: float, b: float) -> float:
    diff = abs(a - b) % np.pi
    return min(diff, np.pi - diff)


def _vp_consensus(vp, segments) -> list:
    """เส้นที่ "เห็นด้วย" กับ VP candidate — ทิศทางของ segment ต้องชี้
    เข้าหา vp ภายในมุมที่ยอมรับได้ (ANGLE_CONSISTENCY_DEG)"""
    inliers = []
    for seg in segments:
        x1, y1, x2, y2 = seg
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        to_vp_angle = np.arctan2(vp[1] - my, vp[0] - mx) % np.pi
        if np.degrees(_angle_diff(to_vp_angle, _segment_angle(seg))) < ANGLE_CONSISTENCY_DEG:
            inliers.append(seg)
    return inliers


def _estimate_one_vp(segments: list, rng: np.random.Generator, n_iters: int = 500):
    """RANSAC: สุ่มจับคู่ segment หาจุดตัดเป็น VP candidate นับเส้นอื่นที่
    เห็นด้วย เลือก candidate ที่มี inlier มากสุด"""
    best_vp, best_inliers = None, []
    n = len(segments)
    if n < 2:
        return None, []
    for _ in range(n_iters):
        i, j = rng.choice(n, size=2, replace=False)
        vp = _line_intersection(segments[i], segments[j])
        if vp is None:
            continue
        inliers = _vp_consensus(vp, segments)
        if len(inliers) > len(best_inliers):
            best_vp, best_inliers = vp, inliers
    return best_vp, best_inliers


def estimate_two_vanishing_points(segments, rng: np.random.Generator | None = None
                                  ) -> tuple:
    """หา VP สองจุดจากเส้นขนานสองชุดที่ตั้งฉากกันจริง (sideline ทิศทาง
    หนึ่ง, baseline/service line อีกทิศทางหนึ่ง) ด้วย RANSAC สองรอบ
    (รอบแรกหา VP ที่มี inlier เยอะสุด, ตัดเส้นที่ใช้ไปแล้วออก, รอบสองหา
    VP ที่เหลือ) — คืน (vp1, vp2) หรือ (None, None) ถ้าหลักฐานไม่พอ
    (กันไม่ให้ผลลัพธ์ปลอมจาก line ไม่กี่เส้นที่บังเอิญตัดกัน)"""
    rng = rng or np.random.default_rng()
    segments = list(segments)
    vp1, inliers1 = _estimate_one_vp(segments, rng)
    if vp1 is None or len(inliers1) < MIN_VP_INLIERS:
        return None, None

    inlier_ids1 = {id(s) for s in inliers1}
    remaining = [s for s in segments if id(s) not in inlier_ids1]
    vp2, inliers2 = _estimate_one_vp(remaining, rng)
    if vp2 is None or len(inliers2) < MIN_VP_INLIERS:
        return None, None
    return tuple(vp1), tuple(vp2)


def focal_length_from_vanishing_points(vp1: tuple, vp2: tuple,
                                       principal_point: tuple) -> float | None:
    """สูตรปิด (closed-form): focal length จาก vanishing point สองจุดของ
    เส้นขนานสองชุดที่ตั้งฉากกันจริงในโลก 3 มิติ

        f = sqrt( -(v1x-cx)(v2x-cx) - (v1y-cy)(v2y-cy) )

    (Criminisi/Reid/Zisserman single-view metrology) คืน None ถ้าค่าใน
    sqrt ติดลบ — แปลว่า VP ที่ได้ไม่สอดคล้องกับกล้อง pinhole จริง (เช่น
    line detection ผิดพลาด หรือเส้นสองชุดไม่ได้ตั้งฉากกันจริงตามที่สมมติ)
    """
    cx, cy = principal_point
    dot = (vp1[0] - cx) * (vp2[0] - cx) + (vp1[1] - cy) * (vp2[1] - cy)
    if -dot <= 0:
        return None
    return float(np.sqrt(-dot))


MAX_FOCAL_RELATIVE_STD = 0.15  # ยอมรับได้ถ้า bootstrap แล้ว focal ยังไม่แกว่งเกินนี้
N_RELIABILITY_TRIALS = 15


def _reliability_check(inliers1: list, vp2: tuple, principal_point: tuple,
                       rng: np.random.Generator,
                       max_relative_std: float, n_trials: int) -> tuple[bool, float]:
    """⚠️ ข้อจำกัดสำคัญที่ validate แล้ว (2026-07-09): ถ้ากล้องอยู่ตรงกลาง
    หลัง baseline พอดี (pan_deg≈0 — คือ setup มาตรฐานที่แนะนำมาตลอดใน
    โปรเจกต์นี้!) เส้นทิศทางกว้าง (baseline/service line) จะเกือบขนานกัน
    สนิทในภาพ ทำให้ vanishing point ของกลุ่มนั้นอยู่ไกลจนเกือบเป็น
    infinity — ผลคือ focal length ที่คำนวณได้ไวต่อ noise ของ line
    detection มาก (validated: pan=0-2° กับ noise แค่ 1px บน endpoint
    ทำให้ focal สูงต่ำเพี้ยนได้เกิน 30-60%, ขณะที่ pan>=10° เพี้ยนแค่
    ~5%) → ต้อง bootstrap เช็คความเสถียรก่อนเชื่อผลลัพธ์ ไม่งั้นจะได้
    ตัวเลขผิดแบบมั่นใจเกินจริง (เหมือน degenerate score bug ที่เจอใน
    court_model.py มาก่อน)"""
    if len(inliers1) < 2:
        return False, 0.0
    focals = []
    for _ in range(n_trials):
        i, j = rng.choice(len(inliers1), size=2, replace=False)
        vp1_b = _line_intersection(inliers1[i], inliers1[j])
        if vp1_b is None:
            continue
        f_b = focal_length_from_vanishing_points(vp1_b, vp2, principal_point)
        if f_b is not None:
            focals.append(f_b)
    if len(focals) < max(3, n_trials // 2):
        return False, 0.0
    rel_std = float(np.std(focals) / max(np.mean(focals), 1e-6))
    return rel_std <= max_relative_std, rel_std


def estimate_focal_length_from_frame(frame: np.ndarray, frame_shape: tuple,
                                     principal_point: tuple | None = None,
                                     rng: np.random.Generator | None = None,
                                     max_focal_relative_std: float = MAX_FOCAL_RELATIVE_STD,
                                     n_reliability_trials: int = N_RELIABILITY_TRIALS
                                     ) -> float | None:
    """เอนด์ทูเอนด์: เฟรม → line segments → 2 VP → focal length (หรือ None
    ถ้าหลักฐานไม่พอ/ไม่สอดคล้อง/ไม่เสถียรพอ) — ใช้เป็น hint แคบช่วง
    focal_length_px ให้ search_camera_pose[_from_points] ใน court_model.py
    แทนที่จะปล่อยให้ search กว้างทั่ว DEFAULT_POSE_BOUNDS (ต้นตอ ambiguity)

    ⚠️ ตั้งใจคืน None บ่อยกว่าที่คิด — โดยเฉพาะกล้อง setup มาตรฐาน (อยู่
    กลางหลัง baseline, pan≈0) เพราะ validate แล้วว่า unreliable จริงใน
    setup นั้น (ดู _reliability_check) การคืน None ตรงนี้คือพฤติกรรมที่
    ตั้งใจ ไม่ใช่บั๊ก — ดีกว่าคืนค่าที่มั่นใจผิด ๆ
    """
    rng = rng or np.random.default_rng()
    h, w = frame_shape[:2]
    cx, cy = principal_point or (w / 2.0, h / 2.0)
    segments = list(detect_line_segments(frame))

    vp1, inliers1 = _estimate_one_vp(segments, rng)
    if vp1 is None or len(inliers1) < MIN_VP_INLIERS:
        return None
    inlier_ids1 = {id(s) for s in inliers1}
    remaining = [s for s in segments if id(s) not in inlier_ids1]
    vp2, inliers2 = _estimate_one_vp(remaining, rng)
    if vp2 is None or len(inliers2) < MIN_VP_INLIERS:
        return None

    f = focal_length_from_vanishing_points(vp1, vp2, (cx, cy))
    if f is None:
        return None

    reliable, _rel_std = _reliability_check(
        inliers1, vp2, (cx, cy), rng, max_focal_relative_std, n_reliability_trials)
    return f if reliable else None
