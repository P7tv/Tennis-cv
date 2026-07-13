"""Court geometry calibration — auto-derive player height + real-world
scale จากเส้นคอร์ท (ไม่ต้องพึ่งผู้ใช้กรอกส่วนสูงเอง)

แนวคิด (single-view metrology แบบง่าย, weak-perspective):
1. หาเส้นคอร์ท (baseline, service line, sidelines) ด้วย classical CV
   (color threshold เส้นขาว + Hough line) → 4 จุดตัด → ground-plane
   homography (พิกัดภาพ ↔ พิกัดจริงบนพื้นสนาม หน่วยเมตร ตามขนาดคอร์ท
   มาตรฐาน ITF)
2. หา net band (แถบพาดสนาม) → ความสูงพิกเซลของเน็ต ณ ตำแหน่งที่รู้ความลึก
   จริง (11.885m จาก baseline) และรู้ความสูงจริงมาตรฐาน (0.914m กลางเน็ต)
3. weak-perspective: pixel_height = f_px × real_height ÷ depth
   → คำนวณ f_px จากเน็ต (reference) → ใช้ f_px หา real_height ผู้เล่น
   จาก pixel_height (ankle→head) + depth (จาก homography ตำแหน่งเท้า)

⚠️ ข้อสมมติ: กล้องไม่เอียงขึ้น-ลง (pitch ≈ 0) — ถ้ากล้องก้ม/เงยมาก
(เช่น GoPro มุมสูง) ค่าจะมี systematic bias เพราะยังไม่ทำ vertical-
vanishing-point correction เต็มรูปแบบ (Criminisi single-view metrology
เต็มรูปแบบ) — เก็บไว้เป็น refinement ถ้า Phase 0 พบว่าจำเป็น

เป็น "ทางเลือกเสริม" ให้ body_height_cm ที่ผู้ใช้กรอก ไม่ใช่บังคับ —
ถ้า calibrate ไม่สำเร็จ (หาเส้นคอร์ทไม่พอ/net ไม่ชัด) กลับไปใช้
body-relative scale เดิมพร้อม flag ว่า auto-calibration ไม่สำเร็จ
"""

from dataclasses import dataclass

import cv2
import numpy as np

# ITF standard court dimensions (m) — origin ที่กึ่งกลาง near baseline
# (baseline ฝั่งกล้อง), X = ข้าง (ขวา+), Z = ความลึกจากกล้อง (ห่างจาก net -)
SINGLES_HALF_WIDTH_M = 8.23 / 2       # 4.115
BASELINE_TO_SERVICE_M = 5.485
BASELINE_TO_NET_M = 11.885
NET_HEIGHT_CENTER_M = 0.914

WORLD_POINTS = {
    # 4 มุมมาตรฐาน (สนามเดี่ยว)
    "near_baseline_left": (-SINGLES_HALF_WIDTH_M, 0.0),
    "near_baseline_right": (SINGLES_HALF_WIDTH_M, 0.0),
    "service_line_left": (-SINGLES_HALF_WIDTH_M, BASELINE_TO_SERVICE_M),
    "service_line_right": (SINGLES_HALF_WIDTH_M, BASELINE_TO_SERVICE_M),
    
    # จุดกึ่งกลาง (ทางเลือกเมื่อมุมซ้าย/ขวาหลุดจอ)
    "center_baseline": (0.0, 0.0),
    "center_service_line": (0.0, BASELINE_TO_SERVICE_M),
    
    # เส้นใต้เน็ต (ทางเลือกเมื่อเส้นหลังหลุดจอทั้งซ้ายขวา)
    "net_left": (-SINGLES_HALF_WIDTH_M, BASELINE_TO_NET_M),
    "net_right": (SINGLES_HALF_WIDTH_M, BASELINE_TO_NET_M),
    "center_net": (0.0, BASELINE_TO_NET_M),
}

MIN_LINE_LENGTH_FRAC = 0.08
HORIZONTAL_ANGLE_MAX_DEG = 12.0
DIAGONAL_ANGLE_MIN_DEG = 12.0   # ⚠️ validated บนคลิปจริง 2026-07-09: sideline
                                # ปรากฏที่ ~15-25° ในกล้องมุมหลัง ไม่ใช่ >30°
                                # ตามที่เดาไว้แต่แรกจาก synthetic test เท่านั้น
BRIGHT_LINE_THRESHOLD = 170
COURT_SURFACE_TOP_FRAC = 0.35  # ตัดโซนรั้ว/หลังคา/เสาไฟ/ท้องฟ้าทิ้งก่อนหาเส้น
                                # (validated: เสาไฟถูกจับเป็น false-positive sideline)
FRAME_BOUNDS_MARGIN = 0.3       # จุดตัดเลยขอบเฟรมได้ไม่เกินสัดส่วนนี้ (กัน
                                # sideline ปลอมที่เกือบขนาน baseline)


@dataclass
class CourtCalibration:
    homography: np.ndarray | None      # image px (u,v,1) -> world (X,Z,1)
    focal_length_px: float | None
    valid: bool
    reason: str = ""

    def depth_of(self, image_point: tuple) -> float | None:
        """ความลึกจริง (Z, เมตร) ของจุดบนพื้นสนาม"""
        if not self.valid or self.homography is None:
            return None
        return ground_depth_from_image_point(self.homography, image_point)[1]


# ---------- pure geometry math (testable โดยไม่ต้องมีภาพจริง) ----------

def ground_depth_from_image_point(homography: np.ndarray,
                                  image_point: tuple) -> tuple:
    """แปลงจุดภาพ (u,v) บนพื้นสนาม → พิกัดจริง (X, Z) เมตร"""
    pt = np.array([image_point[0], image_point[1], 1.0])
    world = homography @ pt
    world = world / world[2]
    return float(world[0]), float(world[1])


def image_point_from_world(homography: np.ndarray, world_point: tuple) -> tuple:
    """แปลงพิกัดจริงบนพื้นสนาม (X, Z) เมตร → จุดภาพ (u, v) — inverse
    homography ใช้คาดตำแหน่งวัตถุที่รู้ระยะจริง (เช่นเน็ต) ในภาพ แม่นกว่า
    การเดา region กว้าง ๆ (validated บนคลิปจริง 2026-07-09: ค้นหาแบบกว้าง
    ไปเจอท้องฟ้ามืดตอนกลางคืนแทนเน็ตจริง)"""
    H_inv = np.linalg.inv(homography)
    pt = np.array([world_point[0], world_point[1], 1.0])
    img = H_inv @ pt
    img = img / img[2]
    return float(img[0]), float(img[1])


def solve_ground_homography(image_points: dict) -> np.ndarray | None:
    """image_points: {name: (u,v)} ต้องมีอย่างน้อย 4 จุดจาก WORLD_POINTS
    คืน homography (image → world ground plane) หรือ None ถ้าจุดไม่พอ"""
    names = [n for n in WORLD_POINTS if n in image_points]
    if len(names) < 4:
        return None
    src = np.array([image_points[n] for n in names], dtype=np.float64)
    dst = np.array([WORLD_POINTS[n] for n in names], dtype=np.float64)
    H, _ = cv2.findHomography(src, dst, method=0)
    return H


def estimate_focal_length_px(net_pixel_height: float, net_depth_m: float,
                             net_real_height_m: float = NET_HEIGHT_CENTER_M
                             ) -> float:
    """weak-perspective: f_px = pixel_height × depth ÷ real_height"""
    return net_pixel_height * net_depth_m / net_real_height_m


def estimate_height_cm(player_pixel_height: float, player_depth_m: float,
                       focal_length_px: float) -> float:
    """weak-perspective: real_height = pixel_height × depth ÷ f_px"""
    real_height_m = player_pixel_height * player_depth_m / focal_length_px
    return real_height_m * 100.0


# ---------- perception (classical CV — best-effort, ต้องจูนกับข้อมูลจริง) ----------

def _apply_gamma(frame: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    """Gamma correction — ยกแสงภาพมืด ช่วยให้เส้นคอร์ทขาวชัดขึ้นในคลิปกลางคืน"""
    inv_gamma = 1.0 / gamma
    table = (np.arange(256) / 255.0) ** inv_gamma * 255
    return cv2.LUT(frame, table.astype(np.uint8))


def _auto_brightness_ok(gray: np.ndarray, thresh: int = 80) -> bool:
    """True ถ้าภาพสว่างพอ (median brightness > thresh)"""
    return float(np.median(gray)) > thresh


def detect_court_lines(frame: np.ndarray) -> list[tuple]:
    """หาเส้นตรงสว่าง (เส้นคอร์ทขาว/ครีมตัดกับพื้นสนามเข้มกว่า)
    คืน list ของ (x1, y1, x2, y2)

    ตัดโซนบนของเฟรม (รั้ว/หลังคา/เสาไฟ/ท้องฟ้า) ทิ้งก่อนเสมอ — validated
    บนคลิปจริงว่าโครงเสาไฟถูกจับเป็นเส้นทแยง (false positive sideline)
    ถ้าไม่ตัดโซนนี้ออก

    รองรับทั้งคลิปกลางวัน (fixed threshold) และกลางคืน (adaptive + gamma)
    """
    frame = frame.copy()
    frame[:int(frame.shape[0] * COURT_SURFACE_TOP_FRAC), :] = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ─── Auto-detect lighting: ถ้ามืดเกินไป ใช้ Gamma + Adaptive Threshold ───
    if not _auto_brightness_ok(gray):
        # Gamma correction: ยกแสงโซนมืดขึ้นก่อน
        frame_bright = _apply_gamma(frame, gamma=2.0)
        gray = cv2.cvtColor(frame_bright, cv2.COLOR_BGR2GRAY)
        # CLAHE: normalize local contrast (ทนต่อ spotlight ไม่สม่ำเสมอ)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        # Adaptive threshold: ทำงานได้แม้ contrast รวมต่ำ
        bright = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, blockSize=31, C=-10
        )
    else:
        _, bright = cv2.threshold(gray, BRIGHT_LINE_THRESHOLD, 255, cv2.THRESH_BINARY)

    edges = cv2.Canny(bright, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=40,
        minLineLength=int(frame.shape[1] * MIN_LINE_LENGTH_FRAC), maxLineGap=15)
    if lines is None:
        return []
    return [tuple(int(v) for v in l.flatten()) for l in lines]


def _line_angle_deg(line: tuple) -> float:
    x1, y1, x2, y2 = line
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180


def _line_intersection(l1: tuple, l2: tuple) -> tuple | None:
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return (px, py)


def _cluster_lines(lines: list, key_fn, merge_dist: float) -> list:
    """จัดกลุ่มเส้นที่ตำแหน่งใกล้กัน (chain-linkage ตาม key_fn)

    จำเป็นเพราะเส้นคอร์ทมีความหนา → Hough มักเจอขอบบน/ขอบล่างของ
    เส้นเดียวกันเป็นเส้นแยก 2 เส้น ต้อง merge ก่อน ไม่งั้นจะเข้าใจผิดว่า
    เป็นเส้นคนละเส้น (เช่น baseline ขอบบน/ล่าง ถูกจับเป็น baseline+service_line)
    """
    if not lines:
        return []
    sorted_lines = sorted(lines, key=key_fn)
    groups = [[sorted_lines[0]]]
    for l in sorted_lines[1:]:
        if abs(key_fn(l) - key_fn(groups[-1][-1])) <= merge_dist:
            groups[-1].append(l)
        else:
            groups.append([l])
    return groups


def _fit_line_through_points(group: list) -> tuple:
    """fit เส้นตรงผ่านจุดปลายทั้งหมดของกลุ่ม (least squares) →
    (x1,y1,x2,y2) ครอบคลุมช่วงจริงของกลุ่ม"""
    pts = np.array([(x1, y1) for x1, y1, x2, y2 in group]
                   + [(x2, y2) for x1, y1, x2, y2 in group], dtype=np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    t = (pts[:, 0] - x0) * vx + (pts[:, 1] - y0) * vy
    t_min, t_max = float(t.min()), float(t.max())
    return (x0 + vx * t_min, y0 + vy * t_min, x0 + vx * t_max, y0 + vy * t_max)


def find_court_corners(lines: list, frame_w: int | None = None,
                       frame_h: int | None = None) -> dict:
    """จำแนกเส้น → จุดตัด 4 จุด (baseline/service line × sidelines)

    heuristic: เส้นเกือบแนวนอน (angle ใกล้ 0/180) = baseline/service line
    เส้นเฉียงมาก (converging เพราะ perspective) = sidelines
    เส้นแนวนอนที่ต่ำสุดในภาพ (y เฉลี่ยมากสุด) = baseline (ใกล้กล้องสุด)

    ⚠️ ถ้าเส้นที่ classify เป็น sideline ดันเกือบขนานกับ baseline/service
    line (เช่น misclassify เส้นเงา/เส้นอื่นเป็น sideline) จุดตัดจะยิงออก
    นอกเฟรมไปไกลมาก (validated บนคลิปจริง 2026-07-09: พบ x เกิน -800px
    ในเฟรม 960px กว้าง) → sanity check ทิ้งผลถ้าจุดไหนหลุดขอบเฟรมเกินไป
    แทนที่จะปล่อยให้ homography คำนวณจากจุดผิด ๆ อย่างเงียบ ๆ
    """
    if not lines:
        return {}
    horizontal, diagonal = [], []
    for l in lines:
        angle = _line_angle_deg(l)
        norm_angle = min(angle, 180 - angle)
        if norm_angle < HORIZONTAL_ANGLE_MAX_DEG:
            horizontal.append(l)
        elif norm_angle > DIAGONAL_ANGLE_MIN_DEG:
            diagonal.append(l)

    h_groups = _cluster_lines(horizontal, key_fn=lambda l: (l[1] + l[3]) / 2,
                              merge_dist=20)
    h_fitted = [_fit_line_through_points(g) for g in h_groups]
    d_groups = _cluster_lines(diagonal, key_fn=lambda l: (l[0] + l[2]) / 2,
                              merge_dist=50)
    d_fitted = [_fit_line_through_points(g) for g in d_groups]
    if len(h_fitted) < 2 or len(d_fitted) < 2:
        return {}

    h_fitted.sort(key=lambda l: -((l[1] + l[3]) / 2))
    baseline = h_fitted[0]
    service_line = h_fitted[1]

    d_fitted.sort(key=lambda l: (l[0] + l[2]) / 2)
    left_line, right_line = d_fitted[0], d_fitted[-1]

    corners = {}
    for name, a, b in (
        ("near_baseline_left", baseline, left_line),
        ("near_baseline_right", baseline, right_line),
        ("service_line_left", service_line, left_line),
        ("service_line_right", service_line, right_line),
    ):
        p = _line_intersection(a, b)
        if p is not None:
            corners[name] = p

    if frame_w is not None and frame_h is not None:
        margin_x, margin_y = frame_w * FRAME_BOUNDS_MARGIN, frame_h * FRAME_BOUNDS_MARGIN
        for name, (u, v) in list(corners.items()):
            if not (-margin_x <= u <= frame_w + margin_x
                    and -margin_y <= v <= frame_h + margin_y):
                return {}  # จุดใดจุดหนึ่งหลุดขอบ → ทิ้งทั้งชุด (เส้นน่าจะ misclassify)
    return corners


def detect_net_band(frame: np.ndarray, x_col: int,
                    search_top: int = 0, search_bottom: int | None = None
                    ) -> float | None:
    """หาความสูงพิกเซลของแถบเน็ต (เข้มกว่าพื้นหลัง) ที่คอลัมน์ x_col

    คืนความสูง (px) ของช่วงมืดต่อเนื่องที่ยาวพอสมควร หรือ None ถ้าไม่ชัด

    ⚠️ search_top/search_bottom ต้องเป็นหน้าต่างแคบรอบตำแหน่งเน็ตที่คาด
    ไว้จริง (ดู image_point_from_world) — ถ้ากว้างเกินไป (เช่นครอบคลุม
    ท้องฟ้า/พื้นหลัง) heuristic นี้จะไปเจอบริเวณมืดอื่นแทนเน็ต (validated
    บนคลิปจริง 2026-07-09: ท้องฟ้ามืดกลางคืนยาวกว่าแถบเน็ตจริงเสมอ)
    """
    h = frame.shape[0]
    search_bottom = search_bottom or h
    x_col = int(np.clip(x_col, 0, frame.shape[1] - 1))
    search_top = max(0, search_top)
    search_bottom = min(h, search_bottom)
    col = frame[search_top:search_bottom, x_col]
    gray_col = cv2.cvtColor(col.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY).flatten()
    if len(gray_col) < 5:
        return None

    # Otsu บนคอลัมน์เดียว: แยกกลุ่มมืด (เน็ต) ออกจากพื้นหลังสว่างกว่าได้ดี
    # กว่า percentile ตายตัว เมื่อสัดส่วนแถบเน็ตในคอลัมน์เล็ก
    thresh_val, _ = cv2.threshold(gray_col.reshape(-1, 1), 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_mask = gray_col <= thresh_val

    # หา run ต่อเนื่องที่ยาวที่สุด
    best_len, cur_len = 0, 0
    for v in dark_mask:
        cur_len = cur_len + 1 if v else 0
        best_len = max(best_len, cur_len)
    if best_len < 5:
        return None
    return float(best_len)


def calibrate_court_from_frame(frame: np.ndarray) -> CourtCalibration:
    """หา CourtCalibration จากเฟรมเดียว (ใช้ calibrate_court() สำหรับ
    ความเสถียรกว่าด้วยหลายเฟรม)"""
    lines = detect_court_lines(frame)
    h, w = frame.shape[:2]
    corners = find_court_corners(lines, frame_w=w, frame_h=h)
    if len(corners) < 4:
        return CourtCalibration(None, None, False,
                                reason=f"พบจุดตัดคอร์ทแค่ {len(corners)}/4")

    homography = solve_ground_homography(corners)
    if homography is None:
        return CourtCalibration(None, None, False, reason="แก้ homography ไม่ได้")

    # ใช้ inverse homography คาดตำแหน่งเน็ต (X=0 กลางคอร์ท, Z=11.885m)
    # ในภาพ — แม่นกว่าเดา region กว้าง ๆ จาก service line (validated:
    # เดากว้างไปเจอท้องฟ้ามืดกลางคืนแทนเน็ตจริง)
    net_u, net_v = image_point_from_world(homography, (0.0, BASELINE_TO_NET_M))
    service_v = (corners["service_line_left"][1]
                + corners["service_line_right"][1]) / 2
    # หน้าต่างค้นหาแคบรอบตำแหน่งที่คาด: ครึ่งหนึ่งของระยะ net→service line
    # ในภาพ (กันไม่ให้กว้างจนโดนพื้นหลัง แต่พอกว้างพอรับ error ของ homography)
    half_window = max(15, abs(service_v - net_v) * 0.5)
    net_h_px = detect_net_band(
        frame, int(net_u),
        search_top=int(net_v - half_window),
        search_bottom=int(net_v + half_window))
    if net_h_px is None:
        return CourtCalibration(homography, None, False,
                                reason="หาแถบเน็ตไม่เจอ (มี homography แต่ไม่มี height ref)")

    f_px = estimate_focal_length_px(net_h_px, BASELINE_TO_NET_M)
    return CourtCalibration(homography, f_px, True)


def calibrate_court(frames: list) -> CourtCalibration:
    """รวมผลจากหลายเฟรม (median ของ focal length) ให้เสถียรกว่าเฟรมเดียว

    ⚠️ เก็บไว้เป็น utility ทั่วไป — validated บนคลิปจริงแล้วว่า
    `build_reference_frame()` + calibrate ครั้งเดียวให้ผลเสถียรกว่า
    (ดู auto_detect_height_cm ที่ใช้แนวทางนั้นเป็นหลัก)
    """
    results = [calibrate_court_from_frame(f) for f in frames]
    valid = [r for r in results if r.valid]
    if not valid:
        reasons = {r.reason for r in results if r.reason}
        return CourtCalibration(None, None, False,
                                reason=f"ไม่มีเฟรมไหน calibrate สำเร็จ: {reasons}")
    f_med = float(np.median([r.focal_length_px for r in valid]))
    # ใช้ homography จากเฟรมที่ f_px ใกล้ median ที่สุด (กัน outlier)
    best = min(valid, key=lambda r: abs(r.focal_length_px - f_med))
    return CourtCalibration(best.homography, f_med, True)


def build_reference_frame(frames: list) -> np.ndarray:
    """Temporal median ของหลายเฟรม (กล้องนิ่ง) → เอาผู้เล่น/วัตถุที่ขยับ
    ออกไป เหลือแต่พื้นสนาม+เส้นคอร์ทที่นิ่ง

    แนวคิดเดียวกับ background subtraction ใน multi_person.py แต่กลับด้าน
    (ดึง background ที่นิ่งออกมา แทนที่จะดึง foreground ที่ขยับ)

    validated บนคลิปจริง 2026-07-09: ลด noise จาก occlusion/แสงไฟกลางคืน
    ทำให้ f_px ที่ประมาณได้เสถียรขึ้นมาก (จากแกว่ง ~2x ต่อเฟรมเดี่ยว
    เหลือ <1% เมื่อ calibrate จาก reference frame เดียวกัน) — จุดที่ยัง
    ไม่แก้คือบาง sub-range ของคลิปยังหาจุดตัดไม่ได้เลย (ดู FRAME_BOUNDS_MARGIN
    sanity check + fallback หลาย sub-range ใน auto_detect_height_cm)
    """
    stack = np.stack(frames, axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


def _sample_frames(video_path: str, start_frac: float, end_frac: float,
                   n: int) -> list:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    lo = int(total * start_frac)
    hi = max(int(total * end_frac), lo + 1)
    idxs = np.linspace(lo, hi - 1, num=min(n, hi - lo), dtype=int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def estimate_height_from_pose(court_cal: CourtCalibration,
                             ts, ready_frames) -> float | None:
    """ใช้ CourtCalibration + PoseTimeseries (ช่วง ready) → body_height_cm

    วัด ankle→head เป็น pixel จริง (ไม่ normalized) ที่ตำแหน่งเท้าเฉลี่ย
    ช่วง ready แล้วแปลงผ่าน weak-perspective — คืน None ถ้า calibration
    ไม่ valid หรือหาตำแหน่งเท้า/หัวไม่ได้ในช่วง ready
    """
    if not court_cal.valid or court_cal.focal_length_px is None:
        return None

    from .config import L_ANKLE, NOSE, R_ANKLE
    lm = ts.landmarks[ready_frames]
    w, h = ts.meta.width, ts.meta.height

    ankle_y = np.nanmedian((lm[:, L_ANKLE, 1] + lm[:, R_ANKLE, 1]) / 2.0) * h
    ankle_x = np.nanmedian((lm[:, L_ANKLE, 0] + lm[:, R_ANKLE, 0]) / 2.0) * w
    head_y = np.nanmedian(lm[:, NOSE, 1]) * h
    if np.isnan(ankle_y) or np.isnan(head_y):
        return None

    pixel_height = abs(ankle_y - head_y)
    depth_m = court_cal.depth_of((ankle_x, ankle_y))
    if depth_m is None or depth_m <= 0:
        return None
    return estimate_height_cm(pixel_height, depth_m, court_cal.focal_length_px)


REFERENCE_FRAME_RANGES = ((0.0, 1.0), (0.0, 0.5), (0.5, 1.0), (0.2, 0.8))


def auto_detect_height_cm(video_path: str, ts, config,
                          sample_frames: int = 40
                          ) -> tuple[float | None, CourtCalibration]:
    """หา court calibration จาก reference frame (temporal median — ดู
    build_reference_frame) แล้วรวมกับ PoseTimeseries ที่ extract ไว้แล้ว
    → ประมาณส่วนสูงผู้เล่นอัตโนมัติ

    ลองหลาย sub-range ของคลิป (ทั้งคลิป → ครึ่งแรก → ครึ่งหลัง →
    ตรงกลาง 60%) เป็น fallback เพราะ validated บนคลิปจริงว่าบางช่วง
    ของคลิปหาจุดตัดคอร์ทไม่ได้ (คนบัง/แสงเปลี่ยน) แต่ช่วงอื่นสำเร็จ —
    ใช้ผลแรกที่ calibrate สำเร็จ (ไม่ mix ผลจากหลาย range เข้าด้วยกัน
    เพราะ homography ต้องมาจาก reference frame เดียวกันกับที่วัด net)

    คืน (height_cm หรือ None ถ้าล้มเหลว, CourtCalibration สำหรับ debug)
    ไม่ throw — ล้มเหลวแล้ว caller ควร fallback ไปใช้ body_height_cm
    ที่กรอกเอง/ค่า default ตามเดิม
    """
    last_cal = CourtCalibration(None, None, False, reason="ไม่มีเฟรมให้ลองเลย")
    # ⚠️ validated บนคลิปจริง: จำนวนเฟรมที่ sample ก็มีผลต่อว่า calibrate
    # สำเร็จไหม (ไม่ใช่แค่ sub-range) — ลองหลายค่าเพื่อเพิ่มโอกาสสำเร็จ
    for n in sorted({15, 25, sample_frames}):
        for lo, hi in REFERENCE_FRAME_RANGES:
            frames = _sample_frames(video_path, lo, hi, n)
            if not frames:
                continue
            ref_frame = build_reference_frame(frames)
            court_cal = calibrate_court_from_frame(ref_frame)
            last_cal = court_cal
            if court_cal.valid:
                break
        if last_cal.valid:
            break

    if not last_cal.valid:
        return None, last_cal

    from .calibration import _ready_frames
    ready = _ready_frames(ts)
    height_cm = estimate_height_from_pose(last_cal, ts, ready)
    if height_cm is None or not (100.0 <= height_cm <= 220.0):
        return None, CourtCalibration(last_cal.homography, last_cal.focal_length_px,
                                      False, reason=f"height นอกช่วงสมเหตุสมผล: {height_cm}")
    return height_cm, last_cal
