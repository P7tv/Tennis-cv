"""
loeuf_cv/hit_detection.py

Fusion approach for hit detection:
  PRIMARY:   Wrist Acceleration Peak (จาก MediaPipe Pose) — ทำงานได้แม้ตอนกลางคืน/จับลูกไม่ได้
  SECONDARY: Ball Direction Change (Inflection) — ยืนยัน Timing และ Confidence
  FALLBACK:  Racket Proximity — ใช้เมื่อ Ball ไม่มีข้อมูล
"""
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────
# Ball Trajectory (ไม่เปลี่ยน)
# ─────────────────────────────────────────────────────────

# คัด detection ของ "ลูกที่นอนนิ่ง" (ลูกบนพื้นสนาม / ในรถเข็นซ้อม) ออกก่อน
# เข้าตัวติดตาม — ในคลิปมุมกว้างของ dataset จริง 87.7% ของ detection เป็น
# ลูกนิ่ง ทำให้ Kalman association ล็อกผิดตัวได้ เพราะเงื่อนไขเดิมคือ
# "bbox ที่ใกล้ตำแหน่งทำนายที่สุด ภายใน 200px" โดยไม่สนว่าเคลื่อนที่หรือไม่
# ดู docs/BALL_DETECTION_ISSUE.md สำหรับตัวเลขที่วัดได้
STATIC_BALL_RADIUS = 8.0   # อยู่ห่างกัน < 8px ถือว่าเป็น "จุดเดียวกัน"
STATIC_BALL_SPAN = 30      # ครอบครองจุดเดิมนานเกิน 30 เฟรม (1 วิ) = ของนิ่ง
STATIC_BALL_MIN_HITS = 6   # และต้องเจอซ้ำอย่างน้อยเท่านี้ครั้ง

# Kalman noise — เชื่อโมเดลกับค่าที่วัดได้พอ ๆ กัน (ค่ากลาง ไม่ได้ fit มา)
# ค่าเดิมคือ 1e-2 / 5.0 = เชื่อโมเดล "ความเร็วคงที่" มากกว่าค่าที่วัดได้ 500
# เท่า ซึ่งบังคับ trajectory ให้เป็นเส้นตรงจนฟิสิกส์จริงหายไป
PROCESS_NOISE = 1.0        # Q — เผื่อให้ลูกเร่ง/เปลี่ยนทิศได้ (แรงโน้มถ่วง ฯลฯ)
MEASUREMENT_NOISE = 1.0    # R — จุดกึ่งกลาง bbox ลูกแม่นราว ±1px

# ---------------------------------------------------------------------------
# 🔴 หน่วยเวลา — ค่าคงที่ "จำนวนเฟรม" และ "px ต่อเฟรม" ในไฟล์นี้
# ---------------------------------------------------------------------------
# dataset ที่ใช้จูนทุกอย่าง (รวมทั้ง hit_classifier.pkl ที่เทรนไว้) ถ่ายที่
# 29.97fps ทั้งชุด ค่าที่เป็น "เฟรม" หรือ "px ต่อเฟรม" จึงผูกกับ fps นั้น
#
# ⚠️ ลูกค้าแจ้งว่าคลิปชุดหน้าจะถ่าย >= 60fps ตาม spec — ถ้าไม่สเกล:
#   · หน้าต่างฟีเจอร์ 8 เฟรม จะครอบเวลาแค่ครึ่งเดียว
#   · wrist speed (px/เฟรม) จะเหลือครึ่งหนึ่งที่การเคลื่อนไหวเท่ากัน
#     -> ฟีเจอร์หลุดออกนอกช่วงที่ hit_classifier เคยเห็น = train/serve skew
#     -> hit detection พังทั้งกระดานโดยไม่มี error ใด ๆ
#
# วิธีรับมือ: แปลงทุกอย่างกลับไปอยู่ในหน่วย "เทียบเท่า 29.97fps" ก่อนป้อน
# โมเดล แทนที่จะเทรนโมเดลใหม่ (ยังไม่มีข้อมูล 60fps ให้เทรน)
# ที่ fps == CALIBRATION_FPS ตัวคูณเป็น 1.0 พอดี -> พฤติกรรมเดิมไม่เปลี่ยน
CALIBRATION_FPS = 29.97

# fps ที่ต่างจากค่าคาลิเบรตน้อยกว่านี้ ให้ถือว่า "เท่ากัน"
#
# ทำไมต้องมี: cv2 อ่าน fps ของคลิปชุดนี้ได้ 29.973518362654733 ไม่ใช่ 29.97
# เป๊ะ ตัวคูณ 1.000117 ทำให้ฟีเจอร์ speed ขยับ 0.01% ซึ่งพอจะพลิก candidate
# ที่อยู่ติดขอบ threshold ของ ML ได้ -> ตัวเลข benchmark เปลี่ยนโดยไม่มีเหตุผล
# เชิงเนื้อหา (วัดจริงแล้ว: recall 0.866 -> 0.860)
# ±2% ปลอดภัย: offset ที่ยาวสุดคือ 42 เฟรม -> คลาดไม่ถึง 1 เฟรม
# และยังห่างจาก 60fps (ratio 2.0) มากพอที่จะไม่กลืนกัน
FPS_EQUAL_TOLERANCE = 0.02


def _fps_ratio(fps: float | None) -> float:
    """fps จริง / fps ที่คาลิเบรต — 1.0 เมื่อไม่รู้ fps หรือใกล้ค่าคาลิเบรต"""
    if not fps or fps <= 0:
        return 1.0
    r = float(fps) / CALIBRATION_FPS
    return 1.0 if abs(r - 1.0) < FPS_EQUAL_TOLERANCE else r


def _f(frames_at_calib: float, fps: float | None) -> int:
    """'จำนวนเฟรมที่ 29.97fps' -> จำนวนเฟรมที่ fps จริง (อย่างน้อย 1)"""
    return max(1, int(round(frames_at_calib * _fps_ratio(fps))))


def filter_static_ball_bboxes(
    ball_bboxes: dict,
    total_frames: int,
    radius: float = STATIC_BALL_RADIUS,
    span_frames: int | None = None,
    min_hits: int = STATIC_BALL_MIN_HITS,
    fps: float | None = None,
) -> dict:
    """ตัด detection ที่อยู่กับที่ออก คืน dict รูปเดิม (frame -> list[bbox])

    เกณฑ์: ดูว่า "จุดนี้" ถูก detection ครอบครองยาวนานแค่ไหน — ถ้ามี
    detection อยู่ในรัศมี 8px เดิม ตั้งแต่เฟรมแรกถึงเฟรมสุดท้ายห่างกัน
    เกิน 30 เฟรม (1 วินาที) และเจอซ้ำ >= 6 ครั้ง = ไม่ใช่ลูกที่กำลังเล่น

    ทำไมใช้ 'ช่วงเวลา' ไม่ใช่ 'สัดส่วนเฟรมที่เจอ': ลูกนิ่งใน dataset จริง
    ถูก detect แบบติด ๆ ดับ ๆ (เจอราว 40-50% ของเฟรม) เกณฑ์สัดส่วนจึง
    ปล่อยหลุด — แต่ช่วงเวลาที่มันครอบครองจุดนั้นยาวเป็นพันเฟรมเสมอ

    ความปลอดภัย: ลูกที่ลอยจริงอยู่ในวง 8px เดิมได้ไม่กี่เฟรม แม้ตอนถึง
    จุดสูงสุดของการโยนเสิร์ฟที่ความเร็วแนวดิ่งเป็น 0 — แรงโน้มถ่วง
    4.4 px/frame² พาออกจากรัศมีใน ~1.9 เฟรม (span ~4) ห่างจาก 30 มาก

    ⚠️ ใช้ไม่ได้ถ้ากล้องขยับ (pan/handheld) เพราะลูกนิ่งจะเลื่อนในภาพ —
    ในกรณีนั้นฟังก์ชันนี้แค่ไม่ตัดอะไรเลย ไม่ได้ทำให้แย่ลง

    span_frames: ไม่ส่ง = คิดจาก STATIC_BALL_SPAN (1 วินาที) สเกลตาม fps —
    เกณฑ์นี้เป็น "เวลา" ไม่ใช่จำนวนเฟรม ถ้าคงไว้ 30 เฟรมที่คลิป 60fps จะ
    กลายเป็น 0.5 วิ แล้วไปตัดลูกที่เล่นจริงซึ่งค้างจุดเดิมนานหน่อยทิ้ง
    """
    if span_frames is None:
        span_frames = _f(STATIC_BALL_SPAN, fps)
    # ปักหมุด detection ลงตาราง cell ขนาด radius เพื่อไม่ต้องเทียบทุกคู่
    grid: dict[tuple[int, int], list[tuple[int, float, float]]] = {}
    for i in range(total_frames):
        for bx, by, bw, bh in ball_bboxes.get(i, []):
            cx, cy = bx + bw / 2.0, by + bh / 2.0
            grid.setdefault((int(cx // radius), int(cy // radius)),
                            []).append((i, cx, cy))

    out: dict[int, list] = {}
    r2 = radius * radius
    for i in range(total_frames):
        bboxes = ball_bboxes.get(i, [])
        if not bboxes:
            continue
        keep = []
        for bb in bboxes:
            bx, by, bw, bh = bb
            cx, cy = bx + bw / 2.0, by + bh / 2.0
            gx, gy = int(cx // radius), int(cy // radius)
            frames = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for (j, ox, oy) in grid.get((gx + dx, gy + dy), ()):
                        if (ox - cx) ** 2 + (oy - cy) ** 2 < r2:
                            frames.append(j)
            occupied = max(frames) - min(frames)
            if not (occupied > span_frames and len(frames) >= min_hits):
                keep.append(bb)
        if keep:
            out[i] = keep
    return out


def extract_ball_trajectory(ball_bboxes: dict, total_frames: int, max_gap: int = 15) -> np.ndarray:
    """
    สร้าง 2D Trajectory (x, y) ของลูกเทนนิสจาก Bounding boxes ที่ได้
    คืนค่าเป็น numpy array [total_frames, 2] ช่องไหนไม่มีจะปล่อยเป็น np.nan
    """
    traj = np.full((total_frames, 2), np.nan)

    last_pt = None
    for i in range(total_frames):
        bboxes = ball_bboxes.get(i, [])
        if not bboxes:
            continue

        best_pt = None
        if last_pt is None:
            bx, by, bw, bh = bboxes[0]
            best_pt = (bx + bw / 2.0, by + bh / 2.0)
        else:
            min_dist = float('inf')
            for bx, by, bw, bh in bboxes:
                cx, cy = bx + bw / 2.0, by + bh / 2.0
                dist = ((cx - last_pt[0])**2 + (cy - last_pt[1])**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    best_pt = (cx, cy)

            if min_dist > 200:
                bx, by, bw, bh = bboxes[0]
                best_pt = (bx + bw / 2.0, by + bh / 2.0)

        traj[i] = best_pt
        last_pt = best_pt

    # Gap-filling (Linear Interpolation)
    isnan = np.isnan(traj[:, 0])
    valid = ~isnan
    if not valid.any():
        return traj

    edges = np.diff(isnan.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if isnan[0]: starts.insert(0, 0)
    if isnan[-1]: ends.append(total_frames)

    idx = np.arange(total_frames)
    for s, e in zip(starts, ends):
        if e - s <= max_gap and s > 0 and e < total_frames:
            traj[s:e, 0] = np.interp(idx[s:e], idx[valid], traj[valid, 0])
            traj[s:e, 1] = np.interp(idx[s:e], idx[valid], traj[valid, 1])

    return traj


def extract_ball_trajectory_kalman(
    ball_bboxes: dict,
    total_frames: int,
    max_gap: int | None = None,
    drop_static: bool = True,
    return_measured: bool = False,
    fps: float | None = None,
):
    """
    Ball Trajectory ด้วย Kalman Filter (ดีกว่า Linear Interpolation):
    - คาด (predict) ตำแหน่งลูกในเฟรมที่จับไม่ได้ โดยใช้ velocity จากเฟรมก่อน
    - ทนต่อ YOLO miss ได้ถึง ~30 เฟรม (1 วินาทีที่ 30fps)
    - ใช้แทน extract_ball_trajectory ได้ทันที (API เหมือนกันทุกอย่าง)

    State: [x, y, vx, vy]  — ตำแหน่งและความเร็ว

    drop_static: คัดลูกที่นอนนิ่ง (บนพื้น/ในรถเข็น) ออกก่อน ดู
                 filter_static_ball_bboxes() — ปิดได้ถ้าต้องการพฤติกรรมเดิม
    return_measured: คืน (traj, measured) โดย measured[i] = True เฉพาะเฟรมที่
                 มี detection จริง ไม่ใช่ค่าที่ Kalman เดาต่อ — จำเป็นสำหรับ
                 ตรรกะที่ "เชื่อตำแหน่งลูกแล้วตัดสินใจ" เพราะช่วง coasting
                 ตำแหน่งลูกเป็นของสมมติ (ดู detect_hit_events ด่าน ball gate)
    """
    import cv2

    # max_gap เป็น "เวลาที่ยอมให้ลูกหายก่อนตัดสายพาน" = 1 วินาที ไม่ใช่ 30 เฟรม
    if max_gap is None:
        max_gap = _f(30, fps)
    if drop_static:
        ball_bboxes = filter_static_ball_bboxes(ball_bboxes, total_frames,
                                                fps=fps)

    # ─── Kalman Filter Setup ───
    kf = cv2.KalmanFilter(4, 2)
    # State transition: x_new = x + vx, vx_new = vx (constant velocity model)
    kf.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float32)
    # Q >> Q เดิม (1e-2) โดยตั้งใจ: โมเดล transition เป็น "ความเร็วคงที่" ซึ่ง
    # ไม่มีแรงโน้มถ่วงอยู่ในสมการ ถ้า Q เล็กเทียบกับ R ฟิลเตอร์จะเชื่อโมเดล
    # มากกว่าค่าที่วัดได้ แล้วบังคับ trajectory ให้เป็นเส้นตรง — วัดจริงแล้ว
    # ความเร่งแนวดิ่งเหลือ 0.05 px/frame² ทั้งที่ detection ดิบให้ 1.00
    # (กลบฟิสิกส์จริงทิ้ง 20 เท่า) ดู docs/BALL_DETECTION_ISSUE.md
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * PROCESS_NOISE
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * MEASUREMENT_NOISE
    kf.errorCovPost = np.eye(4, dtype=np.float32)

    traj = np.full((total_frames, 2), np.nan)
    measured = np.zeros(total_frames, dtype=bool)
    initialized = False
    frames_since_detection = 0
    last_pt = None

    for i in range(total_frames):
        bboxes = ball_bboxes.get(i, [])

        # predict หนึ่งครั้งต่อเฟรมเสมอ — เดิมเรียกซ้ำได้เมื่อเฟรมนั้นมี bbox
        # แต่ทุกตัวไกลเกิน 200px (predict ตอน associate + predict ตอนเติม gap)
        # ทำให้ state เดินหน้าไป 2 เฟรมในเฟรมเดียว
        predicted = None
        if initialized:
            predicted = np.asarray(kf.predict(), dtype=float).reshape(-1)

        # เลือก bbox ที่ใกล้ตำแหน่ง predict ล่าสุดที่สุด
        measurement = None
        if bboxes:
            if predicted is None or last_pt is None:
                bx, by, bw, bh = bboxes[0]
                measurement = (bx + bw / 2.0, by + bh / 2.0)
            else:
                px, py = predicted[0], predicted[1]
                min_dist = float('inf')
                for bx, by, bw, bh in bboxes:
                    cx, cy = bx + bw / 2.0, by + bh / 2.0
                    d = ((cx - px)**2 + (cy - py)**2)**0.5
                    if d < min_dist:
                        min_dist = d
                        measurement = (cx, cy)
                # ถ้า bbox ที่ใกล้ที่สุดยังไกลเกิน 200px กว่า predict → ไม่น่าใช่ลูกเดิม
                if min_dist > 200:
                    measurement = None

        if measurement is not None:
            frames_since_detection = 0
            m = np.array([[measurement[0]], [measurement[1]]], dtype=np.float32)
            if not initialized:
                kf.statePre = np.array([[measurement[0]], [measurement[1]], [0], [0]], dtype=np.float32)
                kf.statePost = kf.statePre.copy()
                initialized = True
            kf.correct(m)
            state_val = np.asarray(kf.statePost, dtype=float).reshape(-1)
            traj[i, 0] = state_val[0]
            traj[i, 1] = state_val[1]
            measured[i] = True
            last_pt = (traj[i, 0], traj[i, 1])
        elif initialized:
            frames_since_detection += 1
            if frames_since_detection <= max_gap:
                # ใช้ predict แทน (ลูกยังน่าจะอยู่แถวนี้)
                traj[i, 0] = predicted[0]
                traj[i, 1] = predicted[1]
            else:
                # หลุดนานเกิน max_gap → reset (อาจเป็นลูกใหม่)
                initialized = False
                last_pt = None

    return (traj, measured) if return_measured else traj


# ─────────────────────────────────────────────────────────
# PRIMARY: Wrist Acceleration Peak Detection
# ─────────────────────────────────────────────────────────

def _compute_wrist_speed(pose_ts, wrist_idx: int, width: int, height: int) -> np.ndarray:
    """คำนวณ speed ของข้อมือ (pixels/frame) ตลอดทั้งคลิป"""
    n = len(pose_ts.landmarks)
    speed = np.full(n, np.nan)
    lm = pose_ts.landmarks
    vis = pose_ts.visibility

    for i in range(1, n - 1):
        if vis[i - 1, wrist_idx] > 0.2 and vis[i + 1, wrist_idx] > 0.2:
            if not np.isnan(lm[i - 1, wrist_idx, 0]) and not np.isnan(lm[i + 1, wrist_idx, 0]):
                dx = (lm[i + 1, wrist_idx, 0] - lm[i - 1, wrist_idx, 0]) * width
                dy = (lm[i + 1, wrist_idx, 1] - lm[i - 1, wrist_idx, 1]) * height
                speed[i] = (dx**2 + dy**2)**0.5 / 2.0  # px/frame
    return speed


# หน้าต่างหา local max ของความเร็วข้อมือ และระยะที่ถือว่า peak สองตัวเป็น
# ตัวเดียวกัน (หน่วย: เฟรมที่ 29.97fps — สเกลด้วย _f() ที่จุดเรียก)
#
# เดิม merge = window*2 = 16 เฟรม ซึ่งกว้างเกินไป: จังหวะปะทะกับจังหวะพีคของ
# follow-through อยู่ห่างกันราว 10-15 เฟรม พอถูกยุบเป็นตัวเดียวโดยเลือก
# "ตัวที่เร็วกว่า" เฟรมปะทะของลูกที่ตีแรงจะแพ้เสมอ
#
# วัดผลของการลดเหลือ 8 (เพดาน recall ระดับ candidate บน GT 172 ลูก):
#   merge 16 -> 95.3% · candidate 2,250
#   merge  8 -> 99.4% · candidate 2,909  (+29% แลกกับเพดาน +4.1 จุด)
# ลดต่ำกว่า 8 ไม่ได้เพดานเพิ่มอีก มีแต่ candidate บวม
PEAK_WINDOW = 8
PEAK_MERGE_WINDOW = 8


def _find_wrist_peaks(speed: np.ndarray, min_speed_px: float = 8.0,
                      window: int = 8, merge_window: int | None = None) -> list[int]:
    """
    หา Local Maximum ของ speed ที่เกิน threshold
    → จุดเหล่านี้คือ "จังหวะที่แขนขยับเร็วสุด" ซึ่งสอดคล้องกับจังหวะตีลูก

    merge_window  ระยะที่ถือว่า peak สองตัวเป็นตัวเดียวกัน (default = window*2)

    🔴 ทำไม merge_window ต้องปรับได้ — วัดแล้วพบว่า GT ที่หลุดตั้งแต่ด่านนี้
    ไม่ใช่ลูกที่ตีเบา แต่เป็นลูกที่ **ตีแรงที่สุด** (speed median 108 px เทียบ
    กับ 49 px ของลูกที่จับได้) เพราะตอนตีแรง จังหวะ follow-through เร็วกว่า
    จังหวะปะทะ พอ de-dup เลือก "ตัวที่ speed สูงกว่า" เฟรมปะทะจึงถูกเขี่ยทิ้ง
    แล้วเหลือ candidate ที่อยู่ห่างจากเฉลยเกิน tolerance
    ดู scripts/diagnose_missed_candidates.py

    การเลือกด้วย speed ดิบตรงนี้เป็นการตัดสินใจที่ "เร็วเกินไป" — ปลายทางมี
    NMS ที่เลือกด้วย ml_prob ซึ่งรู้จักรูปทรงวงสวิงและมีข้อมูลมากกว่ามาก
    ปล่อย candidate ให้เยอะขึ้นแล้วให้ปลายทางตัดสินจึงดีกว่า แต่ทำได้ก็ต่อ
    เมื่อ precision ดีพอ (ตอนใช้ฟีเจอร์เดิม 39.8% ยังไม่พอ)
    """
    if merge_window is None:
        merge_window = window * 2
    n = len(speed)
    peaks = []
    for i in range(window, n - window):
        if np.isnan(speed[i]) or speed[i] < min_speed_px:
            continue
        local_max = np.nanmax(speed[max(0, i - window): i + window + 1])
        if speed[i] == local_max:
            peaks.append(i)

    # De-duplicate: ถ้า peak ใกล้กัน < merge_window ให้เอา peak ที่ speed สูงสุด
    if not peaks or merge_window <= 1:
        return peaks
    merged = [peaks[0]]
    for p in peaks[1:]:
        if p - merged[-1] < merge_window:
            if speed[p] > speed[merged[-1]]:
                merged[-1] = p
        else:
            merged.append(p)
    return merged


# ─────────────────────────────────────────────────────────
# SECONDARY: Ball Direction Change
# ─────────────────────────────────────────────────────────

def _find_ball_inflections(ball_traj: np.ndarray) -> set[int]:
    """หาเฟรมที่ลูกเปลี่ยนทิศทาง (cos similarity < 0 = มุม > 90°)"""
    total = ball_traj.shape[0]
    vx = np.full(total, np.nan)
    vy = np.full(total, np.nan)

    for i in range(1, total - 1):
        if not np.isnan(ball_traj[i - 1, 0]) and not np.isnan(ball_traj[i + 1, 0]):
            vx[i] = ball_traj[i + 1, 0] - ball_traj[i - 1, 0]
            vy[i] = ball_traj[i + 1, 1] - ball_traj[i - 1, 1]

    inflections = set()
    for i in range(2, total - 2):
        if np.isnan(vx[i - 1]) or np.isnan(vx[i + 1]):
            continue
        v1 = np.array([vx[i - 1], vy[i - 1]])
        v2 = np.array([vx[i + 1], vy[i + 1]])
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 1.5 and n2 > 1.5:
            cos_sim = np.dot(v1, v2) / (n1 * n2)
            if cos_sim < 0.0:
                # เพิ่ม window ±3 รอบๆ ด้วย เผื่อ peak ไม่ตรงเฟรม
                for offset in range(-3, 4):
                    inflections.add(i + offset)
    return inflections


# ─────────────────────────────────────────────────────────
# ML FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────
HIT_WINDOW = 8  # เฟรม — เท่ากับ window ที่ _find_wrist_peaks ใช้หา local max อยู่แล้ว

# ค่า probability ต่ำกว่านี้จาก hit_classifier → ทิ้ง candidate
#
# 0.6 มาจาก 2 หลักฐานที่ตรงกัน (2026-08-01):
#   - Leave-One-Person-Out: F1 สูงสุดที่ 0.6 (0.461) — ตัวเลขที่เชื่อถือได้
#   - benchmark 16 คลิป: 0.4→0.6 ทำให้ precision 0.443→0.674 (FP 196→72)
#     โดย recall ลดแค่ 0.907→0.866 และ acceptance ไม่เปลี่ยนเลย (0.267)
# เดิมเป็น 0.4 ซึ่ง permissive เกินไป
#
# ปรับผ่าน param ml_prob_threshold ของ detect_hit_events() ได้
# ⚠️ การแก้ตัวแปรนี้ตอน runtime ไม่มีผล — มันถูกผูกเป็น default argument
# ตั้งแต่ตอนนิยามฟังก์ชัน ต้องส่งเป็น argument เท่านั้น
ML_PROB_THRESHOLD = 0.6

# prob เกินนี้ → ยก confidence เป็น HIGH (มีผลต่อ tie-break ตอน de-dup)
ML_CONFIDENT_PROB = 0.8

HIT_CLASSIFIER_FILENAME = "hit_classifier.pkl"
_hit_clf_cache: tuple | None = None


def _load_hit_classifier(path: str | None = None) -> tuple:
    """โหลด hit_classifier.pkl แบบไม่ขึ้นกับ current working directory

    ⚠️ ของเดิมเปิดไฟล์ด้วย relative path `"hit_classifier.pkl"` ตรง ๆ ซึ่งหมายถึง
    "หาจาก CWD" ผลคือถ้าโปรเซสเรียกจากโฟลเดอร์อื่น โมเดลจะโหลดไม่ได้ **แบบเงียบ ๆ**
    แล้ว detection ตกไปใช้ candidate ดิบทั้งหมดโดยไม่กรอง — FP พุ่งหลายเท่า
    โดยไม่มี error ให้เห็น

    เคสที่โดนจริง 2 เคส:
      1. scripts/run_keyframe_benchmark.py ครอบ os.chdir ไป temp dir (กัน
         hit_candidates.csv เขียนลง repo) → ตัวเลข benchmark ที่วัดก่อนแก้นี้
         วัดโดยไม่มี classifier เลย
      2. dist/loeuf-cv/ ที่ส่งลูกค้า — ถ้ารัน predict.py จากโฟลเดอร์อื่น
         จะได้ FP เยอะขึ้นหลายเท่าเงียบ ๆ เหมือนกัน

    ลำดับการค้นหา: LOEUF_HIT_CLASSIFIER (env) → รากโปรเจกต์ (แม่ของ loeuf_cv/)
    → CWD (ไว้เพื่อ backward compat)
    """
    global _hit_clf_cache
    if _hit_clf_cache is not None and path is None:
        return _hit_clf_cache

    import os
    import pickle

    candidates = []
    if path:
        candidates.append(Path(path))
    env = os.environ.get("LOEUF_HIT_CLASSIFIER")
    if env:
        candidates.append(Path(env))
    candidates.append(Path(__file__).resolve().parent.parent / HIT_CLASSIFIER_FILENAME)
    candidates.append(Path.cwd() / HIT_CLASSIFIER_FILENAME)

    for p in candidates:
        try:
            if p.is_file():
                with open(p, "rb") as f:
                    model, cols = pickle.load(f)
                if path is None:
                    _hit_clf_cache = (model, cols)
                return model, cols
        except Exception as e:
            print(f"Failed to load hit classifier from {p}: {e}")

    # ไม่เจอ → เตือนให้ดังพอ ไม่ให้หายเงียบเหมือนของเดิม
    print(f"WARNING: ไม่พบ {HIT_CLASSIFIER_FILENAME} (ค้นแล้ว: "
          f"{', '.join(str(c) for c in candidates)}) — hit detection จะไม่กรอง "
          f"candidate ด้วย ML ทำให้ false positive สูงกว่าปกติหลายเท่า")
    return None, []


IMPACT_REFINER_FILENAME = "impact_refiner.pkl"
_impact_refiner_cache = None


def _load_impact_refiner(path: str | None = None) -> tuple:
    """โหลดตัวปรับเฟรมปะทะ — ไม่มีไฟล์ = ข้ามการปรับ ไม่ใช่ error

    ต่างจาก hit_classifier ตรงที่ตัวนี้เป็นของเสริม ไม่มีก็ทำงานได้ (แค่ได้
    เฟรมปะทะหยาบกว่า) จึงไม่เตือนดัง ๆ เมื่อหาไม่เจอ
    ลำดับค้นหาเหมือน _load_hit_classifier: LOEUF_IMPACT_REFINER -> รากโปรเจกต์ -> CWD
    """
    global _impact_refiner_cache
    if _impact_refiner_cache is not None and path is None:
        return _impact_refiner_cache

    import os
    import pickle

    env = os.environ.get("LOEUF_IMPACT_REFINER")
    # ตั้งเป็นค่าว่าง = "ปิดการปรับ" อย่างชัดเจน ไม่ใช่ "ไม่ได้ตั้ง"
    # จำเป็นเพราะถ้าปล่อยให้ไหลไปหาไฟล์ที่รากโปรเจกต์ ตอนรัน benchmark แบบ LOPO
    # จะเผลอหยิบตัวที่เทรนจากทุกคนมาใช้ = ปนเปื้อนเงียบ ๆ ที่ขั้นนี้แทน
    if env == "" and path is None:
        return None, [], 0

    candidates = []
    if path:
        candidates.append(Path(path))
    if env:
        candidates.append(Path(env))
    candidates.append(Path(__file__).resolve().parent.parent / IMPACT_REFINER_FILENAME)
    candidates.append(Path.cwd() / IMPACT_REFINER_FILENAME)

    for p in candidates:
        try:
            if p.is_file():
                with open(p, "rb") as f:
                    model, cols, max_shift = pickle.load(f)
                if path is None:
                    _impact_refiner_cache = (model, cols, max_shift)
                return model, cols, max_shift
        except Exception as e:
            print(f"Failed to load impact refiner from {p}: {e}")
    return None, [], 0


def _refine_impact_frames(events: list[dict], total_frames: int,
                          fps: float) -> list[dict]:
    """เลื่อนเฟรมปะทะของ event ที่เลือกแล้ว ให้เข้าใกล้จังหวะปะทะจริงขึ้น

    🔴 ทำไมต้องมีขั้นนี้: hit_classifier ถูกเทรนให้ตอบว่า "หน้าต่างนี้เหมือน
    การตีแค่ไหน" (label = |frame - เฉลย| <= 5 เฟรม) ซึ่งไม่ใช่คำถามว่า "เฟรมนี้
    คือจังหวะปะทะพอดีไหม" พอ NMS เลือกด้วย ml_prob เฟรม follow-through ที่
    ความเร็วสูงกว่าจึงชนะบ่อย

    วัดจากของจริง (LOPO, 132 stroke ที่ตรวจเจอ): 37 เคสที่เลือกผิด ตัวที่ควร
    เลือกมี prob ต่ำกว่า **100%** ของเคส และ 30 ใน 37 ผ่านเกณฑ์แล้วแต่แพ้ NMS

    ทำไมสำคัญกว่าที่คิด: backswing_peak = impact - ค่าคงที่ ความคลาดของ impact
    จึงส่งต่อไปทั้งดุ้น keyframe ทั้งสองตกพร้อมกัน — ก้อนนี้กิน 23.5% ของคะแนน
    ที่หายไป พอ ๆ กับก้อน "หาลูกไม่เจอ" (23.3%)

    ⚠️ ตัวปรับต้องเทรนบนประชากร "เฟรมที่ NMS เลือกแล้ว" เท่านั้น ไม่ใช่ candidate
    ทุกตัวในรัศมี — สองประชากรนี้เอนเอียงคนละทิศ (+1 vs -0.5 เฟรม) เทรนผิด
    ประชากรแล้วโมเดลจะเลื่อนผิดทาง ดู scripts/train_impact_refiner.py
    """
    model, cols, max_shift = _load_impact_refiner()
    if model is None or not events:
        return events

    import pandas as pd
    rows = [{c: e.get("features", {}).get(c, 0) for c in cols} for e in events]
    try:
        shifts = model.predict(pd.DataFrame(rows))
    except Exception as e:
        print(f"impact refiner predict failed, ใช้เฟรมเดิม: {e}")
        return events

    for e, s in zip(events, shifts):
        d = int(np.clip(round(float(s)), -max_shift, max_shift))
        f = int(np.clip(e["frame"] + d, 0, max(0, total_frames - 1)))
        if f != e["frame"]:
            e["frame_before_refine"] = e["frame"]
            e["frame"] = f
            e["timestamp_sec"] = round(f / fps, 2) if fps else e["timestamp_sec"]
    return sorted(events, key=lambda x: x["frame"])


def _hit_window_features(f: int, speed: np.ndarray, window: int = HIT_WINDOW) -> dict:
    """รูปทรงของ wrist speed รอบๆ frame ผู้สมัคร (candidate) แทนที่จะดูแค่ค่าเดียว ณ frame นั้น

    แรงจูงใจ: การตีจริงคือ peak แหลมสั้นๆ แล้ว "หยุด/ชะลอ" ทันที (follow-through)
    ต่างจากการวิ่ง/ปรับท่าที่ speed จะสูงต่อเนื่องก่อน-หลัง หรือแกว่งเป็นจังหวะ (periodic)
    ไม่ใช่ peak เดี่ยวๆ — ใช้เทคนิคเดียวกับ swing_window_features() ที่ช่วย stroke
    classifier ไปแล้ว (loeuf_cv/schema_builder/classifier.py)
    """
    out = {
        "speed_pre_mean": 0.0,
        "speed_post_mean": 0.0,
        "speed_decel_ratio": 1.0,
        "speed_peak_sharpness": 1.0,
        "speed_std_window": 0.0,
    }
    if speed is None:
        return out

    n = len(speed)
    pre = speed[max(0, f - window):f]
    post = speed[f + 1:min(n, f + window + 1)]
    pre = pre[~np.isnan(pre)]
    post = post[~np.isnan(post)]
    peak = speed[f] if f < n and not np.isnan(speed[f]) else 0.0

    if pre.size > 0:
        out["speed_pre_mean"] = float(np.mean(pre))
    if post.size > 0:
        out["speed_post_mean"] = float(np.mean(post))

    surround_mean = (out["speed_pre_mean"] + out["speed_post_mean"]) / 2.0
    eps = 1e-6
    out["speed_decel_ratio"] = float(out["speed_post_mean"] / (peak + eps))
    out["speed_peak_sharpness"] = float(peak / (surround_mean + eps))

    win_all = speed[max(0, f - window):min(n, f + window + 1)]
    win_all = win_all[~np.isnan(win_all)]
    if win_all.size > 0:
        out["speed_std_window"] = float(np.std(win_all))

    return out


def _extract_hit_features(f: int, ball_traj: np.ndarray, speed: np.ndarray, d_px: float, width: int, height: int, racket_bboxes: dict | None, scale_ctx=None, fps: float | None = None) -> dict:
    total_frames = len(ball_traj)
    features = {
        "wrist_speed": 0.0,
        "ball_dist": 9999.0,
        "ball_vel_before": 0.0,
        "ball_vel_after": 0.0,
        "ball_vel_change": 0.0,
        "ball_angle_change": 1.0,
        "racket_dist": 9999.0,
        **_hit_window_features(f, speed, _f(HIT_WINDOW, fps)),
    }

    # Wrist Speed
    if speed is not None and f < len(speed) and not np.isnan(speed[f]):
        features["wrist_speed"] = float(speed[f])

    # Ball Distance
    if d_px is not None:
        features["ball_dist"] = float(d_px)
        
    # Ball Velocity (before and after) — W เป็นระยะเวลา ไม่ใช่จำนวนเฟรม
    W = _f(3, fps)
    if f >= W and f < total_frames - W:
        b_before = ball_traj[f - W]
        b_curr = ball_traj[f]
        b_after = ball_traj[f + W]
        
        if not np.isnan(b_before).any() and not np.isnan(b_curr).any() and not np.isnan(b_after).any():
            v1 = b_curr - b_before
            v2 = b_after - b_curr
            
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            features["ball_vel_before"] = float(n1)
            features["ball_vel_after"] = float(n2)
            features["ball_vel_change"] = float(n2 - n1)
            
            if n1 > 0 and n2 > 0:
                cos_sim = np.dot(v1, v2) / (n1 * n2)
                features["ball_angle_change"] = float(cos_sim)

    # Racket Distance
    # racket_bboxes[f] มักเป็น [] (list ว่าง) ไม่ใช่ key หาย — yolo_track.py
    # init ทุกเฟรมไว้เป็น [] เสมอ ต้องเช็ค truthy ของ list ด้วย ไม่ใช่แค่ f in racket_bboxes
    # ไม่งั้น best_r_dist ค้างที่ float('inf') แล้วหลุดเข้า features → ML predict_proba
    # พังด้วย ValueError (inf ไม่ผ่าน sklearn input validation)
    if racket_bboxes and racket_bboxes.get(f) and not np.isnan(ball_traj[min(f, total_frames - 1)]).any():
        ball_pos = ball_traj[min(f, total_frames - 1)]
        best_r_dist = float('inf')
        for rx, ry, rw, rh in racket_bboxes[f]:
            rcx, rcy = rx + rw / 2.0, ry + rh / 2.0
            best_r_dist = min(best_r_dist, ((ball_pos[0] - rcx)**2 + (ball_pos[1] - rcy)**2)**0.5)
        features["racket_dist"] = float(best_r_dist)

    if scale_ctx is not None:
        features.update(_scaled_features(f, features, scale_ctx, fps))

    return features


# ─────────────────────────────────────────────────────────
# ฟีเจอร์ที่ normalize ด้วยขนาดตัว (ข้ามคน/ข้ามระยะกล้องได้)
# ─────────────────────────────────────────────────────────
#
# ทำไมต้องมี: wrist_speed ดิบเป็น px/frame ซึ่งขึ้นกับระยะกล้องและขนาดตัวคน
# คนที่ยืนไกลกล้องได้ค่าต่ำกว่าคนที่ยืนใกล้ทั้งที่ตีแรงเท่ากัน -> โมเดลเรียน
# "ระยะกล้อง" แทน "การตี" แล้วย้ายข้ามคนไม่ได้
#
# วัดด้วย Leave-One-Person-Out: F1 0.365 -> 0.461 · precision 29.6% -> 39.8%
# และฟีเจอร์ 4 อันดับแรกกลายเป็นตัวที่ normalize แล้วทั้งหมด
# ดู scripts/train_hit_classifier_from_cache.py

SCALED_FEATURE_COLS = [
    "wrist_speed_bw", "ball_dist_bw", "racket_dist_bw",
    "speed_pre_bw", "speed_post_bw", "speed_std_bw",
    "wrist_dir_change", "wrist_y_rel", "wrist_x_rel",
]


def body_scale_context(pose, width: int, height: int, fps: float | None = None):
    """คำนวณไม้บรรทัด (ความกว้างไหล่ px) + สัญญาณต่อเฟรมที่ไม่มีหน่วย px

    เรียกครั้งเดียวต่อ track แล้วส่งต่อให้ _extract_hit_features ทุก candidate

    stencil ของ wrist_dir_change (เดิม +-2 เฟรมตายตัว) สเกลตาม fps ด้วย —
    มุมหักเหที่วัดได้ขึ้นกับ "ช่วงเวลา" ที่คร่อม ไม่ใช่จำนวนเฟรม ถ้าคงไว้ 2
    เฟรมที่ 60fps จะคร่อมเวลาแค่ครึ่งเดียว มุมที่ได้ตื้นกว่าเดิมทุกจุด
    """
    d = _f(2, fps)
    from .config import L_SHOULDER, L_WRIST, R_SHOULDER, R_WRIST

    lm = pose.landmarks
    n = len(lm)
    sw = np.abs(lm[:, R_SHOULDER, 0] - lm[:, L_SHOULDER, 0]) * width
    bw = float(np.nanmedian(sw)) if not np.isnan(sw).all() else 0.0
    if not np.isfinite(bw) or bw < 1.0:
        bw = max(1.0, height * 0.1)   # fallback กันหารศูนย์

    sh_x = (lm[:, R_SHOULDER, 0] + lm[:, L_SHOULDER, 0]) / 2.0 * width
    sh_y = (lm[:, R_SHOULDER, 1] + lm[:, L_SHOULDER, 1]) / 2.0 * height

    dir_change = np.full(n, 1.0)
    y_rel = np.zeros(n)
    x_rel = np.zeros(n)
    for widx in (R_WRIST, L_WRIST):
        wx, wy = lm[:, widx, 0] * width, lm[:, widx, 1] * height
        for f in range(d, n - d):
            if np.isnan([wx[f - d], wx[f], wx[f + d]]).any():
                continue
            v1 = np.array([wx[f] - wx[f - d], wy[f] - wy[f - d]])
            v2 = np.array([wx[f + d] - wx[f], wy[f + d] - wy[f]])
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 <= 0 or n2 <= 0:
                continue
            c = float(np.dot(v1, v2) / (n1 * n2))
            # เก็บข้อมือข้างที่หักเหมากกว่า (ข้างที่กำลังตี)
            if c < dir_change[f]:
                dir_change[f] = c
                y_rel[f] = (wy[f] - sh_y[f]) / bw
                x_rel[f] = abs(wx[f] - sh_x[f]) / bw
    return bw, {"wrist_dir_change": dir_change,
                "wrist_y_rel": y_rel, "wrist_x_rel": x_rel}


def _scaled_features(f: int, features: dict, scale_ctx, fps: float | None = None) -> dict:
    """แปลงฟีเจอร์ดิบ -> หน่วยที่ hit_classifier เคยเห็นตอนเทรน

    หาร bw (ความกว้างไหล่) = ตัดผลของขนาดคน/ระยะกล้อง
    หาร fps_ratio = ตัดผลของเฟรมเรต — ฟีเจอร์กลุ่ม speed_* เป็น "px ต่อเฟรม"
    การเคลื่อนไหวเดียวกันที่ 60fps จะให้ค่าครึ่งเดียวของที่ 29.97fps ซึ่งเป็น
    คนละ distribution กับที่โมเดลเทรนมา (โมเดลเทรนจากคลิป 29.97fps ล้วน)
    ตัวคูณกลับให้เป็น "px ต่อเฟรมเทียบเท่า 29.97fps" -> โมเดลเดิมใช้ต่อได้
    ระยะทาง (ball_dist / racket_dist) ไม่ขึ้นกับ fps จึงไม่ต้องแตะ
    """
    bw, extra = scale_ctx
    r = _fps_ratio(fps)          # 1.0 ที่ 29.97fps -> ไม่เปลี่ยนพฤติกรรมเดิม
    out = {
        "wrist_speed_bw": features.get("wrist_speed", 0.0) / bw * r,
        "ball_dist_bw": min(features.get("ball_dist", 9999.0), 9999.0) / bw,
        "racket_dist_bw": min(features.get("racket_dist", 9999.0), 9999.0) / bw,
        "speed_pre_bw": features.get("speed_pre_mean", 0.0) / bw * r,
        "speed_post_bw": features.get("speed_post_mean", 0.0) / bw * r,
        "speed_std_bw": features.get("speed_std_window", 0.0) / bw * r,
    }
    for k, v in extra.items():
        out[k] = float(v[f]) if f < len(v) and not np.isnan(v[f]) else 0.0
    return out

# ─────────────────────────────────────────────────────────
# FUSION: Primary Wrist + Secondary Ball + Fallback Racket
# ─────────────────────────────────────────────────────────

def detect_hit_events(
    ball_traj: np.ndarray,
    tracks: list,
    fps: float,
    width: int,
    height: int,
    proximity_thresh: float = 0.30,
    racket_bboxes: dict | None = None,
    min_wrist_speed_px: float = 8.0,
    ml_prob_threshold: float = ML_PROB_THRESHOLD,
    nms_by_prob: bool = True,
    return_candidates: bool = False,
    ball_measured: np.ndarray | None = None,
) -> list[dict]:
    """
    Fusion Hit Detection:
    - PRIMARY:  Wrist Acceleration Peak (ทำงานได้แม้ไม่มีลูก)
    - SECONDARY: Ball Direction Change (ยืนยัน Timing)
    - FALLBACK: Racket Proximity (ถ้า Wrist confidence ต่ำ)

    ทุก candidate จาก Wrist จะได้ Confidence Score:
      HIGH   = Wrist Peak + Ball Inflection ตรงกัน (±5 frame)
      MEDIUM = Wrist Peak เท่านั้น (ไม่มีลูก หรือ Ball ไม่ตรง)
      LOW    = Ball Inflection + Racket Proximity (Wrist ไม่เจอ)
    """
    from .config import L_WRIST, R_WRIST
    # import ในฟังก์ชันเพราะ swing_shape/racket_shape ใช้ _f/_fps_ratio ของ
    # ไฟล์นี้ — import ระดับโมดูลจะวนกลับมาหากันเอง
    from .racket_shape import racket_shape_features
    from .swing_shape import swing_shape_3d_features, swing_shape_features

    total_frames = ball_traj.shape[0]
    px_thresh = height * proximity_thresh
    CONFIRM_WINDOW = _f(5, fps)  # ~0.17 วิ (สเกลตาม fps ดูหัวไฟล์)

    # min_wrist_speed_px เป็น "px ต่อเฟรม" — การเคลื่อนไหวเดียวกันที่ 60fps
    # ให้ค่าครึ่งเดียว ถ้าไม่หารตามจะกรอง candidate ที่ควรผ่านทิ้งเกือบหมด
    min_wrist_speed_px = min_wrist_speed_px / _fps_ratio(fps)

    # ─── Secondary: Ball Inflections ───
    ball_inflections = _find_ball_inflections(ball_traj)

    all_candidates: list[dict] = []

    # ไม้บรรทัดขนาดตัวต่อ track — คำนวณครั้งเดียว ใช้ซ้ำทุก candidate
    scale_by_track = {t.track_id: body_scale_context(t.pose, width, height, fps)
                      for t in tracks}

    # ─── Primary: Wrist Peaks per Track ───
    for t in tracks:
        n_pose = len(t.pose.landmarks)
        for wrist_idx, side in [(R_WRIST, "right"), (L_WRIST, "left")]:
            speed = _compute_wrist_speed(t.pose, wrist_idx, width, height)
            peaks = _find_wrist_peaks(speed, min_speed_px=min_wrist_speed_px,
                                      window=_f(PEAK_WINDOW, fps),
                                      merge_window=_f(PEAK_MERGE_WINDOW, fps))

            for f in peaks:
                if f >= n_pose:
                    continue

                # Ball confirmation
                ball_confirmed = any(f + offset in ball_inflections for offset in range(-CONFIRM_WINDOW, CONFIRM_WINDOW + 1))
                confidence = "HIGH" if ball_confirmed else "MEDIUM"

                # ตรวจสอบว่า ball ใกล้ wrist ไหม (optional cross-check)
                ball_nearby = False
                if not np.isnan(ball_traj[min(f, total_frames - 1), 0]):
                    ball_pos = ball_traj[min(f, total_frames - 1)]
                    lm = t.pose.landmarks[f]
                    vis = t.pose.visibility[f]
                    if vis[wrist_idx] > 0.2 and not np.isnan(lm[wrist_idx, 0]):
                        wx = lm[wrist_idx, 0] * width
                        wy = lm[wrist_idx, 1] * height
                        d = ((ball_pos[0] - wx)**2 + (ball_pos[1] - wy)**2)**0.5
                        ball_nearby = d < px_thresh

                # กรองพวกที่แกว่งแขนเฉยๆ (MEDIUM confidence แต่ลูกอยู่ไกลมาก)
                if confidence == "MEDIUM" and not ball_nearby:
                    # ถ้าระบบหาลูกไม่เจอเลยแถวนั้น อนุโลมให้ผ่าน (อาจจะโดนบัง)
                    # แต่ถ้าหาเจอแล้วอยู่ไกล ให้ปัดตก
                    #
                    # ⚠️ ต้องเชื่อ "เฉพาะตำแหน่งที่มาจาก detection จริง" — ช่วงที่
                    # Kalman เดาต่อ (coasting) ตำแหน่งลูกเป็นของสมมติ การเอามาปัด
                    # candidate ทิ้งคือการตัด stroke จริงด้วยหลักฐานที่ระบบแต่งเอง
                    # วัดแล้วด่านนี้กิน recall ไป 17% (109 -> 90 จาก GT 112)
                    # ดู scripts/diagnose_hit_recall.py
                    fi = min(f, total_frames - 1)
                    ball_is_real = (ball_measured is None
                                    or bool(ball_measured[fi]))
                    if ball_is_real and not np.isnan(ball_traj[fi, 0]):
                        continue

                d_val = float(d) if ball_nearby else None
                feats = _extract_hit_features(f, ball_traj, speed, d_val, width,
                                              height, racket_bboxes,
                                              scale_ctx=scale_by_track[t.track_id],
                                              fps=fps)
                feats.update(swing_shape_features(
                    t.pose.landmarks, f, fps=fps, width=width, height=height,
                    side=side))
                feats.update(swing_shape_3d_features(
                    t.pose.world_landmarks, f, fps=fps, side=side))
                feats.update(racket_shape_features(
                    racket_bboxes, t.pose.landmarks, f, fps=fps, width=width,
                    height=height, side=side,
                    wrist_speed_px=float(speed[f]) if not np.isnan(speed[f])
                    else None))

                all_candidates.append({
                    "frame": f,
                    "timestamp_sec": round(f / fps, 2),
                    "player_id": t.track_id,
                    "player_role": t.role,
                    "wrist_speed_px": round(float(speed[f]), 1),
                    "wrist_side": side,
                    "ball_confirmed": ball_confirmed,
                    "ball_nearby": ball_nearby,
                    "confidence": confidence,
                    "distance_px": round(d, 1) if ball_nearby else None,
                    "source": "wrist_peak",
                    "features": feats,
                })

    # ─── Fallback: Ball Inflections not covered by any Wrist Peak ───
    covered_frames = set()
    for c in all_candidates:
        for offset in range(-CONFIRM_WINDOW, CONFIRM_WINDOW + 1):
            covered_frames.add(c["frame"] + offset)

    # Group ball inflections into candidates
    uncovered = sorted(f for f in ball_inflections if f not in covered_frames and 0 <= f < total_frames)
    if uncovered:
        groups: list[list[int]] = []
        curr = [uncovered[0]]
        for f in uncovered[1:]:
            if f - curr[-1] <= 5:
                curr.append(f)
            else:
                groups.append(curr)
                curr = [f]
        groups.append(curr)

        for group in groups:
            f = int(np.median(group))
            ball_pos = ball_traj[f]
            if np.isnan(ball_pos).any():
                continue

            best_dist = float('inf')
            best_track = None
            for t in tracks:
                if f >= len(t.pose.visibility):
                    continue
                vis = t.pose.visibility[f]
                lm = t.pose.landmarks[f]
                for wrist_idx in (L_WRIST, R_WRIST):
                    if vis[wrist_idx] > 0.2 and not np.isnan(lm[wrist_idx, 0]):
                        wx = lm[wrist_idx, 0] * width
                        wy = lm[wrist_idx, 1] * height
                        d = ((ball_pos[0] - wx)**2 + (ball_pos[1] - wy)**2)**0.5
                        if d < best_dist:
                            best_dist = d
                            best_track = t

            # Racket fallback
            if best_dist >= px_thresh and racket_bboxes and f in racket_bboxes:
                for rx, ry, rw, rh in racket_bboxes[f]:
                    rcx, rcy = rx + rw / 2.0, ry + rh / 2.0
                    d = ((ball_pos[0] - rcx)**2 + (ball_pos[1] - rcy)**2)**0.5
                    if d < best_dist:
                        best_dist = d
                        for t in tracks:
                            if f >= len(t.pose.visibility):
                                continue
                            best_track = t

            if best_dist < px_thresh and best_track is not None:
                feats = _extract_hit_features(
                    f, ball_traj, None, best_dist, width, height, racket_bboxes,
                    scale_ctx=scale_by_track.get(best_track.track_id), fps=fps)
                # candidate สายนี้ไม่รู้ว่าใช้มือไหน -> ให้ swing_shape เลือก
                # ข้างที่ข้อมือเร็วกว่าเอง (side=None)
                feats.update(swing_shape_features(
                    best_track.pose.landmarks, f, fps=fps, width=width,
                    height=height))
                feats.update(swing_shape_3d_features(
                    best_track.pose.world_landmarks, f, fps=fps))
                feats.update(racket_shape_features(
                    racket_bboxes, best_track.pose.landmarks, f, fps=fps,
                    width=width, height=height))
                all_candidates.append({
                    "frame": f,
                    "timestamp_sec": round(f / fps, 2),
                    "player_id": best_track.track_id,
                    "player_role": best_track.role,
                    "wrist_speed_px": None,
                    "wrist_side": None,
                    "ball_confirmed": True,
                    "ball_nearby": True,
                    "confidence": "LOW",
                    "distance_px": round(best_dist, 1),
                    "source": "ball_inflection",
                    "features": feats,
                })

    # ─── De-duplicate: ถ้า candidates ใกล้กันเกินไป ให้รวมกัน ───
    all_candidates.sort(key=lambda x: x["frame"])
    MIN_GAP = int(fps * 0.75)  # เช่น 30fps * 0.75 = 22 เฟรม (คนเราตีลูกติดกันเร็วกว่า 0.75 วิได้ยากมาก)

    # ─── Integration with ML Classifier ───
    ml_model, feature_cols = _load_hit_classifier()

    # ─── Pass 1: คิด ml_prob ให้ "ทุก" candidate ก่อน ยังไม่กรอง ───
    # แยกออกมาเพื่อให้ return_candidates ส่งข้อมูลดิบพร้อม prob ออกไป tune ได้
    # โดยไม่ต้องรัน YOLO ใหม่ (tracking แพงกว่า detection ~95 เท่า)
    for c in all_candidates:
        c.setdefault("ml_prob", None)
        if ml_model is None:
            continue
        import pandas as pd
        X_dict = {col: c.get("features", {}).get(col, 0) for col in feature_cols}
        try:
            prob = float(ml_model.predict_proba(pd.DataFrame([X_dict]))[0][1])
            c["ml_prob"] = round(prob, 4)
            if prob > ML_CONFIDENT_PROB:
                c["confidence"] = "HIGH"
        except Exception as e:
            # อย่าให้ candidate เดียวพัง detection ทั้งคลิป — ตกไปใช้
            # rule-based confidence เดิมของ candidate นี้แทน (ไม่ drop, ไม่ boost)
            print(f"ML predict failed for frame {c['frame']}, falling back to rule-based: {e}")

    if return_candidates:
        return all_candidates

    # ─── Pass 2: กรองด้วย threshold ───
    kept = [c for c in all_candidates
            if c["ml_prob"] is None or c["ml_prob"] >= ml_prob_threshold]

    # ─── Pass 3: de-dup ───
    if nms_by_prob:
        # NMS แบบมาตรฐาน: เรียงตามคะแนนมาก→น้อย แล้วกดเพื่อนบ้านที่ใกล้กว่า MIN_GAP
        # ต่างจาก de-dup เดิมที่ไล่ตาม "ลำดับเฟรม" แล้วเก็บตัวที่มาก่อน ซึ่งทำให้
        # ได้เฟรมที่ไม่ใช่ตัวคะแนนสูงสุดในกลุ่ม → ตำแหน่ง impact เพี้ยน
        chosen: list[dict] = []
        for c in sorted(kept, key=lambda x: -(x["ml_prob"] if x["ml_prob"] is not None else 0.0)):
            if all(abs(c["frame"] - k["frame"]) >= MIN_GAP for k in chosen):
                chosen.append(c)
        # ปรับเฟรมหลัง NMS เท่านั้น — ตัวปรับเทรนบนประชากรนี้ ไม่ใช่ candidate ดิบ
        return _refine_impact_frames(sorted(chosen, key=lambda x: x["frame"]),
                                     total_frames, fps)

    final: list[dict] = []
    for c in kept:
        # 2. Minimum Gap De-duplication
        if not final or c["frame"] - final[-1]["frame"] >= MIN_GAP:
            final.append(c)
        else:
            # เปรียบ confidence
            conf_rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
            rank_curr = conf_rank.get(c["confidence"], 0)
            rank_prev = conf_rank.get(final[-1]["confidence"], 0)
            
            # ถ้า rank สูงกว่า ให้แทนที่
            if rank_curr > rank_prev:
                final[-1] = c
            # ถ้า rank เท่ากัน ให้ดูว่าใครลูกใกล้กว่า (หรือถ้ามีคนนึงใกล้ อีกคนไกล ให้เอาคนใกล้)
            elif rank_curr == rank_prev:
                if c["ball_nearby"] and not final[-1]["ball_nearby"]:
                    final[-1] = c
                elif c["source"] == "wrist_peak" and final[-1]["source"] == "wrist_peak":
                    # ถ้า peak ทั้งคู่ เอาอันที่แกว่งแขนแรงสุด
                    if (c["wrist_speed_px"] or 0) > (final[-1]["wrist_speed_px"] or 0):
                        final[-1] = c

    # ─── Export Features for ML Training ───
    import os
    import csv
    csv_file = "hit_candidates.csv"
    file_exists = os.path.isfile(csv_file)
    try:
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["is_hit", "frame", "confidence", "wrist_speed", "ball_dist", "ball_vel_before", "ball_vel_after", "ball_vel_change", "ball_angle_change", "racket_dist"])
            
            for c in final:
                feats = c.get("features", {})
                writer.writerow([
                    "", # user will fill this
                    c["frame"],
                    c["confidence"],
                    feats.get("wrist_speed", 0),
                    feats.get("ball_dist", 9999),
                    feats.get("ball_vel_before", 0),
                    feats.get("ball_vel_after", 0),
                    feats.get("ball_vel_change", 0),
                    feats.get("ball_angle_change", 1),
                    feats.get("racket_dist", 9999)
                ])
    except Exception as e:
        print(f"Warning: could not write to {csv_file}: {e}")

    return _refine_impact_frames(final, total_frames, fps)
