"""
loeuf_cv/bounce_detection.py

ตรวจจับจังหวะที่ลูกเทนนิสกระทบพื้น (Bounce) และแปลงตำแหน่งเป็นพิกัดจริงบนสนาม

หลักการ:
  - ลูกก่อน Bounce: กำลังลงมา (vy > 0 ในพิกัดภาพ เพราะ Y เพิ่มลงด้านล่าง)
  - จุด Bounce:    local minimum ของ Y (จุดต่ำสุดก่อนขึ้น)
  - หลัง Bounce:  vy เปลี่ยนเป็นลบ (ขึ้นไป)
  - แปลง pixel → court (X, Z) เมตร ผ่าน Court Homography
"""
import numpy as np


# ─────────────────────────────────────────────────────────
# Bounce Frame Detection
# ─────────────────────────────────────────────────────────

def detect_bounce_frames(
    ball_traj: np.ndarray,
    min_speed_px: float = 3.0,
    smooth_window: int = 5,
) -> list[dict]:
    """
    หาเฟรมที่ลูกเทนนิสกระทบพื้น (Bounce Events)

    Args:
        ball_traj: [total_frames, 2] trajectory (x, y) — nan คือไม่มีข้อมูล
        min_speed_px: ความเร็วขั้นต่ำ (px/frame) ก่อน/หลัง bounce — กันลูกวางนิ่ง
        smooth_window: จำนวนเฟรม smooth ก่อนหา vy

    Returns:
        list of dicts: {"frame": int, "x_px": float, "y_px": float, "vy_before": float, "vy_after": float}
    """
    total = ball_traj.shape[0]
    if total < 5:
        return []

    # Smooth trajectory ก่อน (กัน noise กระโดด)
    from numpy.lib.stride_tricks import sliding_window_view
    y_raw = ball_traj[:, 1].copy()
    x_raw = ball_traj[:, 0].copy()

    # Simple moving average (nan-safe)
    y_smooth = _nansmooth(y_raw, smooth_window)
    x_smooth = _nansmooth(x_raw, smooth_window)

    # คำนวณ vy (derivative of y)
    vy = np.full(total, np.nan)
    for i in range(1, total - 1):
        if not np.isnan(y_smooth[i - 1]) and not np.isnan(y_smooth[i + 1]):
            vy[i] = (y_smooth[i + 1] - y_smooth[i - 1]) / 2.0

    bounces = []
    for i in range(2, total - 2):
        if np.isnan(vy[i - 1]) or np.isnan(vy[i + 1]) or np.isnan(y_smooth[i]):
            continue

        # Bounce signature: vy เปลี่ยนจาก + เป็น - (ลงมาแล้วขึ้น)
        # vy[i-1] > 0  (กำลังลง) และ vy[i+1] < 0 (เริ่มขึ้น)
        if vy[i - 1] > min_speed_px and vy[i + 1] < -min_speed_px:
            # ตรวจสอบว่าเป็น local minimum ของ y จริงๆ
            y_window = y_smooth[max(0, i - 3): i + 4]
            if not np.isnan(y_window).all() and float(y_smooth[i]) >= np.nanmax(y_window) * 0.95:
                bounces.append({
                    "frame": i,
                    "x_px": float(x_smooth[i]) if not np.isnan(x_smooth[i]) else float(x_raw[i]),
                    "y_px": float(y_smooth[i]) if not np.isnan(y_smooth[i]) else float(y_raw[i]),
                    "vy_before": round(float(vy[i - 1]), 2),
                    "vy_after": round(float(vy[i + 1]), 2),
                })

    # De-duplicate: ถ้า bounce ใกล้กัน < 10 เฟรม เก็บอันที่ vy_before ใหญ่สุด
    if not bounces:
        return []
    merged = [bounces[0]]
    for b in bounces[1:]:
        if b["frame"] - merged[-1]["frame"] < 10:
            if b["vy_before"] > merged[-1]["vy_before"]:
                merged[-1] = b
        else:
            merged.append(b)

    return merged


def _nansmooth(arr: np.ndarray, window: int) -> np.ndarray:
    """Moving average แบบ nan-safe"""
    out = arr.copy().astype(float)
    half = window // 2
    for i in range(len(arr)):
        segment = arr[max(0, i - half): i + half + 1]
        valid = segment[~np.isnan(segment)]
        if len(valid) > 0:
            out[i] = float(np.mean(valid))
    return out


# ─────────────────────────────────────────────────────────
# Court Coordinate Conversion
# ─────────────────────────────────────────────────────────

def bounce_pixel_to_court(
    x_px: float,
    y_px: float,
    homography: np.ndarray,
) -> tuple[float, float] | tuple[None, None]:
    """
    แปลงพิกัด pixel (x_px, y_px) บนพื้นสนาม → พิกัดจริง (X_m, Z_m) เมตร
    ใช้ Court Homography จาก court_calibration.py

    Returns:
        (X_m, Z_m) — X = ข้าง (ขวา+), Z = ความลึกจากกล้อง
        (None, None) ถ้า homography เป็น None หรือคำนวณไม่ได้
    """
    if homography is None:
        return None, None
    try:
        pt = np.array([x_px, y_px, 1.0], dtype=np.float64)
        world = homography @ pt
        if abs(world[2]) < 1e-8:
            return None, None
        world = world / world[2]
        return float(world[0]), float(world[1])
    except Exception:
        return None, None


def is_in_court(x_m: float | None, z_m: float | None) -> bool:
    """
    True ถ้าพิกัด (x_m, z_m) อยู่ในขอบเขตสนามเทนนิสมาตรฐาน
    (กว้าง ±4.115m, ลึก 0–23.77m)
    """
    if x_m is None or z_m is None:
        return False
    HALF_W = 4.115 + 1.0   # +1m margin
    COURT_LENGTH = 23.77 + 2.0  # +2m margin
    return (-HALF_W <= x_m <= HALF_W) and (-2.0 <= z_m <= COURT_LENGTH)


# ─────────────────────────────────────────────────────────
# Enrich Hit Events with Bounce Data
# ─────────────────────────────────────────────────────────

def add_bounce_to_hits(
    hits: list[dict],
    ball_traj: np.ndarray,
    homography: np.ndarray | None,
    fps: float,
    bounce_window_frames: int = 45,
) -> list[dict]:
    """
    หา Bounce ที่ตามหลัง Hit Event แต่ละครั้ง แล้วเพิ่ม field เข้าไปใน hit:
      - bounce_frame
      - bounce_court_x_m  (None ถ้าไม่มี homography)
      - bounce_court_z_m
      - bounce_in_court   (bool)
      - bounce_delay_ms   (เวลาระหว่าง hit และ bounce)

    bounce_window_frames: หน้าต่างค้นหา bounce หลัง hit (default = 45 เฟรม = 1.5s ที่ 30fps)
    """
    bounces = detect_bounce_frames(ball_traj)

    enriched = []
    for hit in hits:
        h = dict(hit)
        h_frame = h["frame"]

        # หา bounce ที่เกิดขึ้นหลัง hit และอยู่ใน window
        next_bounce = None
        for b in bounces:
            if h_frame < b["frame"] <= h_frame + bounce_window_frames:
                next_bounce = b
                break  # เอาอันแรกที่เจอ (ใกล้ hit ที่สุด)

        if next_bounce is not None:
            x_m, z_m = bounce_pixel_to_court(
                next_bounce["x_px"], next_bounce["y_px"], homography
            )
            h["bounce_frame"] = next_bounce["frame"]
            h["bounce_court_x_m"] = round(x_m, 2) if x_m is not None else None
            h["bounce_court_z_m"] = round(z_m, 2) if z_m is not None else None
            h["bounce_in_court"] = is_in_court(x_m, z_m)
            h["bounce_delay_ms"] = round((next_bounce["frame"] - h_frame) / fps * 1000, 0)
        else:
            h["bounce_frame"] = None
            h["bounce_court_x_m"] = None
            h["bounce_court_z_m"] = None
            h["bounce_in_court"] = None
            h["bounce_delay_ms"] = None

        enriched.append(h)

    return enriched
