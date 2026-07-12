"""Full pinhole camera model + complete court line model — model-based
court registration (แทนที่การหาแค่ 4 มุมแบบ court_calibration.py เดิม)

แนวคิด: ไม่ต้องเจอเส้นครบทุกเส้นในเฟรมเดียว — ตั้งสมมติฐาน "ท่ากล้อง"
(ตำแหน่ง+มุม+focal length) แล้ว "ทาย" ว่าเส้นคอร์ททั้งสนามควรอยู่ตรงไหน
ในภาพ (ทำนายได้แม้ส่วนที่ไม่เห็น/โดนบัง) เทียบกับเส้นที่ detect เจอจริง
ในภาพ (edge map) แล้วปรับท่ากล้องให้ตรงกันที่สุด (คล้าย sports-field
registration ที่ใช้ในระบบวิเคราะห์ฟุตบอล/บาสเกตบอล)

ข้อดีเทียบกับ 4-corner approach เดิม:
- ใช้เส้นที่เห็นได้ "กี่เส้นก็ได้" เป็นหลักฐาน (ไม่ต้องครบ 4 เส้นเป๊ะ)
- เส้นที่ misclassify 1-2 เส้น ไม่ทำให้ทั้งระบบพังเพราะ score มาจาก
  หลักฐานรวมของทุกเส้น ไม่ใช่จุดตัดของเส้นที่ผิดแค่คู่เดียว
- ทำนาย/วาดส่วนคอร์ทที่ไม่เห็นในเฟรมได้ (far baseline, doubles line
  ที่โดนคนบัง ฯลฯ) เพราะโมเดลเป็นกล้อง 3 มิติเต็มรูปแบบ ไม่ใช่แค่
  ground-plane homography ระนาบเดียว
- แก้ปัญหา camera tilt (pitch) ที่ weak-perspective ใน court_calibration.py
  สมมติไว้ว่าไม่มี — pinhole model เต็มรูปแบบจัดการเรื่องนี้ถูกต้องเลย
"""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import minimize

# ITF standard dimensions (m)
SINGLES_HALF_WIDTH_M = 8.23 / 2
DOUBLES_HALF_WIDTH_M = 10.97 / 2
BASELINE_TO_SERVICE_M = 5.485
BASELINE_TO_NET_M = 11.885
FULL_COURT_LENGTH_M = 23.77
NET_HEIGHT_CENTER_M = 0.914
NET_HEIGHT_POST_M = 1.07

# เส้นคอร์ททั้งหมด: {name: ((X1,Z1,Y1), (X2,Z2,Y2))} — Y=ความสูงจากพื้น
# (ปกติ 0 ยกเว้นเส้น/จุดบนเน็ต) ระบบพิกัด: origin กึ่งกลาง near baseline,
# X=ข้าง(ขวา+), Z=ความลึกจากกล้อง(ไกลขึ้น+), Y=ความสูง(ขึ้น+)
COURT_LINES = {
    "near_baseline": ((-DOUBLES_HALF_WIDTH_M, 0, 0), (DOUBLES_HALF_WIDTH_M, 0, 0)),
    "far_baseline": ((-DOUBLES_HALF_WIDTH_M, FULL_COURT_LENGTH_M, 0),
                     (DOUBLES_HALF_WIDTH_M, FULL_COURT_LENGTH_M, 0)),
    "near_singles_sideline_left": ((-SINGLES_HALF_WIDTH_M, 0, 0),
                                   (-SINGLES_HALF_WIDTH_M, FULL_COURT_LENGTH_M, 0)),
    "near_singles_sideline_right": ((SINGLES_HALF_WIDTH_M, 0, 0),
                                    (SINGLES_HALF_WIDTH_M, FULL_COURT_LENGTH_M, 0)),
    "doubles_sideline_left": ((-DOUBLES_HALF_WIDTH_M, 0, 0),
                              (-DOUBLES_HALF_WIDTH_M, FULL_COURT_LENGTH_M, 0)),
    "doubles_sideline_right": ((DOUBLES_HALF_WIDTH_M, 0, 0),
                               (DOUBLES_HALF_WIDTH_M, FULL_COURT_LENGTH_M, 0)),
    "near_service_line": ((-SINGLES_HALF_WIDTH_M, BASELINE_TO_SERVICE_M, 0),
                          (SINGLES_HALF_WIDTH_M, BASELINE_TO_SERVICE_M, 0)),
    "far_service_line": ((-SINGLES_HALF_WIDTH_M,
                         FULL_COURT_LENGTH_M - BASELINE_TO_SERVICE_M, 0),
                         (SINGLES_HALF_WIDTH_M,
                         FULL_COURT_LENGTH_M - BASELINE_TO_SERVICE_M, 0)),
    "center_service_line": ((0, BASELINE_TO_SERVICE_M, 0),
                            (0, FULL_COURT_LENGTH_M - BASELINE_TO_SERVICE_M, 0)),
    "net_line": ((-DOUBLES_HALF_WIDTH_M, BASELINE_TO_NET_M, 0),
                (DOUBLES_HALF_WIDTH_M, BASELINE_TO_NET_M, 0)),
    "net_post_left": ((-DOUBLES_HALF_WIDTH_M, BASELINE_TO_NET_M, 0),
                      (-DOUBLES_HALF_WIDTH_M, BASELINE_TO_NET_M, NET_HEIGHT_POST_M)),
    "net_post_right": ((DOUBLES_HALF_WIDTH_M, BASELINE_TO_NET_M, 0),
                       (DOUBLES_HALF_WIDTH_M, BASELINE_TO_NET_M, NET_HEIGHT_POST_M)),
    "net_top": ((-DOUBLES_HALF_WIDTH_M, BASELINE_TO_NET_M, NET_HEIGHT_POST_M),
               (DOUBLES_HALF_WIDTH_M, BASELINE_TO_NET_M, NET_HEIGHT_POST_M)),
}

# เฉพาะช่วงคอร์ทใกล้กล้อง (มักเห็นในเฟรมจริงมากกว่า far court ที่อาจ
# เล็ก/มืดเกินจะ detect ได้) — ใช้เป็นชุดหลักสำหรับ scoring
NEAR_COURT_LINES = tuple(
    name for name in COURT_LINES if not name.startswith("far_")
)

# จุดที่ตั้งชื่อไว้ตายตัว — ให้ UI คลิกจุด (court_click_tool.html) เลือกจาก
# รายการนี้แทนการกรอกพิกัด world เอง ⚠️ รายการเดียวกันถูก mirror ไว้ใน
# court_click_tool.html (JS ฝั่ง browser อ่านไฟล์นี้ตรง ๆ ไม่ได้) — ถ้าแก้
# ที่นี่ต้องแก้ทั้งสองที่ให้ตรงกัน
NAMED_POINTS = {
    "near_baseline_left_doubles": (-DOUBLES_HALF_WIDTH_M, 0, 0),
    "near_baseline_right_doubles": (DOUBLES_HALF_WIDTH_M, 0, 0),
    "near_baseline_left_singles": (-SINGLES_HALF_WIDTH_M, 0, 0),
    "near_baseline_right_singles": (SINGLES_HALF_WIDTH_M, 0, 0),
    "near_service_left_singles": (-SINGLES_HALF_WIDTH_M, BASELINE_TO_SERVICE_M, 0),
    "near_service_right_singles": (SINGLES_HALF_WIDTH_M, BASELINE_TO_SERVICE_M, 0),
    "near_service_center_T": (0, BASELINE_TO_SERVICE_M, 0),
    "far_service_left_singles": (-SINGLES_HALF_WIDTH_M,
                                 FULL_COURT_LENGTH_M - BASELINE_TO_SERVICE_M, 0),
    "far_service_right_singles": (SINGLES_HALF_WIDTH_M,
                                  FULL_COURT_LENGTH_M - BASELINE_TO_SERVICE_M, 0),
    "far_service_center_T": (0, FULL_COURT_LENGTH_M - BASELINE_TO_SERVICE_M, 0),
    "far_baseline_left_doubles": (-DOUBLES_HALF_WIDTH_M, FULL_COURT_LENGTH_M, 0),
    "far_baseline_right_doubles": (DOUBLES_HALF_WIDTH_M, FULL_COURT_LENGTH_M, 0),
    "net_post_left_base": (-DOUBLES_HALF_WIDTH_M, BASELINE_TO_NET_M, 0),
    "net_post_left_top": (-DOUBLES_HALF_WIDTH_M, BASELINE_TO_NET_M, NET_HEIGHT_POST_M),
    "net_post_right_base": (DOUBLES_HALF_WIDTH_M, BASELINE_TO_NET_M, 0),
    "net_post_right_top": (DOUBLES_HALF_WIDTH_M, BASELINE_TO_NET_M, NET_HEIGHT_POST_M),
    "net_center_bottom": (0, BASELINE_TO_NET_M, 0),
    "net_center_top": (0, BASELINE_TO_NET_M, NET_HEIGHT_CENTER_M),
}


@dataclass
class CameraPose:
    """กล้องแบบ pinhole เต็มรูปแบบ — แทนที่ ground-plane homography +
    weak-perspective แยกกัน ด้วยโมเดลเดียวที่ถูกต้องทั้ง ground และ height

    สมมติฐาน: กล้องอยู่บนเส้นกึ่งกลางคอร์ท (ไม่มี lateral offset) และ
    roll = 0 (แก้ไปแล้วโดย calibration.py roll correction ก่อนหน้านี้)
    """
    height_m: float        # กล้องสูงจากพื้นกี่เมตร
    distance_m: float      # กล้องอยู่หลัง near baseline กี่เมตร
    tilt_deg: float         # มุมก้ม (+) / เงย (-) จากแนวนอน
    pan_deg: float          # มุมหันซ้าย(-)/ขวา(+) จากแกนกลางคอร์ท
    focal_length_px: float
    principal_point: tuple = None  # (cx, cy) px; None = กึ่งกลางเฟรม

    def project(self, world_point: tuple, frame_shape: tuple) -> tuple | None:
        """world (X, Z, Y) → image (u, v); None ถ้าจุดอยู่หลังกล้อง"""
        h, w = frame_shape[:2]
        cx, cy = self.principal_point or (w / 2.0, h / 2.0)

        X, Z, Y = world_point
        cam = np.array([X - 0.0, Z - (-self.distance_m), Y - self.height_m])

        pan = np.radians(self.pan_deg)
        Rp = np.array([[np.cos(pan), -np.sin(pan), 0],
                      [np.sin(pan), np.cos(pan), 0],
                      [0, 0, 1]])
        cam = Rp @ cam

        tilt = np.radians(self.tilt_deg)
        Rt = np.array([[1, 0, 0],
                      [0, np.cos(tilt), -np.sin(tilt)],
                      [0, np.sin(tilt), np.cos(tilt)]])
        cx_, cz_, cy_ = cam[0], cam[1], cam[2]
        cam = Rt @ np.array([cx_, cz_, cy_])
        cx_, cz_, cy_ = cam[0], cam[1], cam[2]

        if cz_ <= 0.05:
            return None
        u = self.focal_length_px * cx_ / cz_ + cx
        v = -self.focal_length_px * cy_ / cz_ + cy
        return (float(u), float(v))

    def to_debug_dict(self) -> dict:
        return {"height_m": round(self.height_m, 2),
                "distance_m": round(self.distance_m, 2),
                "tilt_deg": round(self.tilt_deg, 2),
                "pan_deg": round(self.pan_deg, 2),
                "focal_length_px": round(self.focal_length_px, 1)}


def project_court_lines(pose: CameraPose, frame_shape: tuple,
                        line_names: tuple = NEAR_COURT_LINES) -> dict:
    """ทำนายตำแหน่งเส้นคอร์ททุกเส้น (รวมส่วนที่อาจไม่เห็นในภาพจริง)
    ตามท่ากล้องที่กำหนด → {name: ((u1,v1),(u2,v2))} หรือ None ถ้าจุด
    ปลายอยู่หลังกล้องทั้งคู่"""
    out = {}
    for name in line_names:
        a, b = COURT_LINES[name]
        pa = pose.project(a, frame_shape)
        pb = pose.project(b, frame_shape)
        out[name] = (pa, pb) if (pa is not None and pb is not None) else None
    return out


# ---------- edge-alignment scoring ("ท่ากล้องนี้ตรงกับที่เห็นจริงแค่ไหน") ----------

COURT_SURFACE_TOP_FRAC = 0.35  # ตัดโซนรั้ว/หลังคา/เสาไฟ/ท้องฟ้าทิ้งเหมือน court_calibration.py


def build_edge_map(frame: np.ndarray,
                   court_surface_top_frac: float = COURT_SURFACE_TOP_FRAC
                   ) -> np.ndarray:
    """Canny edge map ของโซนพื้นสนาม — ใช้เทียบกับเส้นที่โมเดลทำนาย"""
    masked = frame.copy()
    masked[:int(frame.shape[0] * court_surface_top_frac), :] = 0
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(bright, 50, 150)
    # dilate เล็กน้อย กัน sample point พลาด edge ไปไม่กี่ pixel
    return cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)


MIN_EVIDENCE_POINTS = 40   # ต้องมีจุดตัวอย่างในเฟรมอย่างน้อยเท่านี้ถึงจะนับคะแนน
MIN_LINES_VISIBLE = 3      # และต้องมีอย่างน้อยกี่เส้นที่ clip เข้าเฟรมจริง (ยาวพอ)
MIN_CLIPPED_LENGTH_PX = 20  # เส้นที่ clip เหลือสั้นกว่านี้ไม่นับเป็น "เส้นที่เห็น"


def score_pose_against_edges(pose: CameraPose, edge_map: np.ndarray,
                             frame_shape: tuple,
                             line_names: tuple = NEAR_COURT_LINES,
                             sample_step_px: float = 8.0) -> float:
    """คะแนน 0-1: สัดส่วนจุดตัวอย่างบนเส้นที่โมเดลทำนาย ซึ่งตรงกับ edge จริง

    เส้นที่โมเดลทำนายแล้วอยู่นอกเฟรมทั้งหมด (ไม่ถูก clip เหลือ) ไม่นับ
    คะแนน (ไม่ช่วยไม่หักคะแนน) — ใช้เฉพาะส่วนที่ "ควรจะเห็นได้" จริง

    ⚠️ ต้องมีหลักฐานขั้นต่ำ (จำนวนจุด + จำนวนเส้นที่เห็นจริง) ไม่งั้น
    pose แบบ "เกือบทั้งคอร์ทอยู่นอกเฟรม เหลือจุดเดียวที่บังเอิญตรง edge"
    จะได้คะแนนเต็มแบบไม่มีความหมาย (validated บนคลิปจริง 2026-07-09:
    search เจอ pose ที่มีแค่ 2 จุดซ้ำกันในเฟรม แต่ได้ score=1.0)
    """
    h, w = frame_shape[:2]
    lines = project_court_lines(pose, frame_shape, line_names)
    total, hit, n_lines_visible = 0, 0, 0
    for seg in lines.values():
        if seg is None:
            continue
        p1, p2 = seg
        ok, c1, c2 = cv2.clipLine(
            (0, 0, w, h),
            (int(round(p1[0])), int(round(p1[1]))),
            (int(round(p2[0])), int(round(p2[1]))))
        if not ok:
            continue
        length = float(np.hypot(c2[0] - c1[0], c2[1] - c1[1]))
        if length < MIN_CLIPPED_LENGTH_PX:
            continue
        n_lines_visible += 1
        n = max(2, int(length / sample_step_px))
        xs = np.linspace(c1[0], c2[0], n).round().astype(int)
        ys = np.linspace(c1[1], c2[1], n).round().astype(int)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        total += int(valid.sum())
        hit += int(edge_map[ys[valid], xs[valid]].astype(bool).sum())

    if total < MIN_EVIDENCE_POINTS or n_lines_visible < MIN_LINES_VISIBLE:
        return 0.0
    return hit / total


DEFAULT_POSE_BOUNDS = {
    "height_m": (0.8, 2.2),
    "distance_m": (1.0, 14.0),
    "tilt_deg": (-15.0, 25.0),
    "pan_deg": (-20.0, 20.0),
    "focal_length_px": (400.0, 2200.0),
}
_REFINE_STEP_SCALE = {"height_m": 0.15, "distance_m": 0.5, "tilt_deg": 2.0,
                     "pan_deg": 2.0, "focal_length_px": 60.0}


def _refine_local_generic(pose: CameraPose, score: float, score_fn, bounds: dict,
                          n_iters: int, rng: np.random.Generator
                          ) -> tuple[CameraPose, float]:
    """Coordinate-wise perturbation รอบท่าเริ่มต้นหนึ่งท่า — ใช้ score_fn
    ใดก็ได้ (edge-based หรือ point-based) แทนที่จะ hardcode edge scoring"""
    for _ in range(n_iters):
        field = rng.choice(list(_REFINE_STEP_SCALE))
        delta = float(rng.normal(0, _REFINE_STEP_SCALE[field]))
        val = getattr(pose, field) + delta
        lo, hi = bounds[field]
        if not (lo <= val <= hi):
            continue
        candidate = CameraPose(
            height_m=pose.height_m, distance_m=pose.distance_m,
            tilt_deg=pose.tilt_deg, pan_deg=pose.pan_deg,
            focal_length_px=pose.focal_length_px,
            principal_point=pose.principal_point)
        setattr(candidate, field, val)
        cand_score = score_fn(candidate)
        if cand_score > score:
            pose, score = candidate, cand_score
    return pose, score


def _search_pose_generic(score_fn, bounds: dict, n_random: int, n_refine_iters: int,
                         n_multistart: int, rng: np.random.Generator
                         ) -> tuple[CameraPose, float]:
    """เอนจิน random-search + multi-start local-refine ที่ใช้ร่วมกันได้ทั้ง
    edge-based (search_camera_pose) และ point-based (search_camera_pose_from_points)
    scoring — ต่างกันแค่ score_fn ที่ส่งเข้ามา"""
    def sample_pose() -> CameraPose:
        return CameraPose(
            height_m=float(rng.uniform(*bounds["height_m"])),
            distance_m=float(rng.uniform(*bounds["distance_m"])),
            tilt_deg=float(rng.uniform(*bounds["tilt_deg"])),
            pan_deg=float(rng.uniform(*bounds["pan_deg"])),
            focal_length_px=float(rng.uniform(*bounds["focal_length_px"])),
        )

    candidates = []
    for _ in range(n_random):
        pose = sample_pose()
        candidates.append((score_fn(pose), pose))
    candidates.sort(key=lambda sp: -sp[0])

    best_pose, best_score = candidates[0][1], candidates[0][0]
    for score, pose in candidates[:n_multistart]:
        refined_pose, refined_score = _refine_local_generic(
            pose, score, score_fn, bounds, n_refine_iters, rng)
        if refined_score > best_score:
            best_pose, best_score = refined_pose, refined_score

    return best_pose, best_score


def search_camera_pose(edge_map: np.ndarray, frame_shape: tuple,
                       bounds: dict | None = None,
                       n_random: int = 400, n_refine_iters: int = 80,
                       n_multistart: int = 6,
                       rng: np.random.Generator | None = None
                       ) -> tuple[CameraPose | None, float]:
    """หาท่ากล้องที่เส้นคอร์ททำนายตรงกับ edge จริงมากที่สุด

    ขั้น 1: random search กว้าง ๆ ทั่วช่วงที่เป็นไปได้ทางกายภาพ
    ขั้น 2: local refine แบบ multi-start — ปัญหา focal_length ↔ distance
    ambiguity แบบคลาสสิกใน single-view calibration (กล้องใกล้+FOV กว้าง
    ให้ภาพคล้ายกล้องไกล+FOV แคบ) ทำให้ refine จากจุดเดียวติด local
    optimum ง่าย (validated: refine จากจุดเดียวได้ ~0.62 ขณะที่ true
    pose ได้ ~0.92) → refine จาก top-K ผู้เข้ารอบของ random search แล้ว
    เลือกผลดีที่สุดจากทุกจุดเริ่มต้น

    คืน (None, 0.0) ถ้า edge_map ไม่มี edge เลย (คลิปมืดสนิท/เสียหมด)
    """
    bounds = bounds or DEFAULT_POSE_BOUNDS
    rng = rng or np.random.default_rng()
    if edge_map.sum() == 0:
        return None, 0.0

    score_fn = lambda pose: score_pose_against_edges(pose, edge_map, frame_shape)
    return _search_pose_generic(score_fn, bounds, n_random, n_refine_iters,
                                n_multistart, rng)


# ---------- point-correspondence scoring ("คนคลิกจุดที่เห็นจริงในเฟรม") ----------
#
# ใช้แทน/เสริม edge-based search เมื่อการ detect เส้นอัตโนมัติไม่น่าเชื่อถือ
# (แสง/สิ่งกีดขวาง) หรือคอร์ทไม่เข้าเฟรมครบ (ซูมมาก, เห็นแค่บางส่วน) —
# ให้คนคลิกจุดที่เห็นได้จริงกี่จุดก็ได้ (>= 4) จับคู่กับพิกัดจริงบนคอร์ท
# แทนที่จะพึ่ง Hough-line detection หรือ Canny edge ทั้งเฟรม

@dataclass
class PointCorrespondence:
    """จุดที่คนคลิกในภาพ (image_xy) จับคู่กับพิกัดจริงบนคอร์ท (world_xyz,
    หน่วยเมตร ระบบพิกัดเดียวกับ COURT_LINES)"""
    image_xy: tuple
    world_xyz: tuple


def point_on_line(line_name: str, t: float) -> tuple:
    """จุดบนเส้นคอร์ทที่สัดส่วน t (0=ปลายแรก, 1=ปลายที่สอง) ของ
    COURT_LINES[line_name] — ใช้ระบุจุดที่คนคลิก เช่น จุดตัดเส้น (t=0/1)"""
    a, b = COURT_LINES[line_name]
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


MAX_REPROJECTION_PENALTY_PX = 2000.0  # โทษจุดที่ pose ทำให้หลุดหลังกล้อง


def reprojection_error_px(pose: CameraPose, correspondences: list,
                          frame_shape: tuple) -> float:
    """RMSE (พิกเซล) ระหว่างจุดที่ pose ทำนายกับจุดที่คนคลิกจริง"""
    errs = []
    for corr in correspondences:
        pred = pose.project(corr.world_xyz, frame_shape)
        if pred is None:
            errs.append(MAX_REPROJECTION_PENALTY_PX)
            continue
        du = pred[0] - corr.image_xy[0]
        dv = pred[1] - corr.image_xy[1]
        errs.append(du * du + dv * dv)
    return float(np.sqrt(np.mean(errs)))


def score_pose_against_points(pose: CameraPose, correspondences: list,
                              frame_shape: tuple) -> float:
    """แปลง reprojection RMSE เป็นคะแนน (ยิ่งมากยิ่งดี, 0-1) — ใช้กับ
    search engine เดียวกับ edge-based scoring ได้ตรง ๆ"""
    rmse = reprojection_error_px(pose, correspondences, frame_shape)
    return 1.0 / (1.0 + rmse)


MIN_POINT_CORRESPONDENCES = 4  # 5 พารามิเตอร์ที่ต้อง solve (height/distance/tilt/pan/focal)
_POSE_FIELDS = ("height_m", "distance_m", "tilt_deg", "pan_deg", "focal_length_px")

# จุดที่คลิกทั้งหมดอยู่ระดับความลึก (Z) เดียวกัน (เช่น คลิกแค่ 4 มุมเสาเน็ต
# ที่อยู่ Z เดียวกันหมด) เป็นสภาวะ degenerate จริง — validated เชิงประจักษ์
# (2026-07-09): ไม่มี noise เลยก็ยัง solve ได้ pose ผิดที่ score สูงพอกัน
# (0.95) เพราะจุดที่ Z เดียวกันไม่พอแยก focal length ↔ distance ได้
# (ไม่มี perspective foreshortening ให้เทียบข้ามระยะ) ต้องมีจุดอย่างน้อย
# อีก 1 ระดับความลึกที่ต่างจากกลุ่มหลัก — ยิ่งดีถ้ามี >= 2 จุดกระจาย
# ซ้าย-ขวาที่ระดับนั้น (validated: 1 จุดเดี่ยวช่วยได้บ้างแต่ noise
# sensitivity ยังสูง ~16%, 2 จุดกระจายกว้างลด noise sensitivity เหลือ <1%)
MIN_DEPTH_SPREAD_M = 3.0


def depth_spread_m(correspondences: list) -> float:
    """ช่วงความลึก (world Z, เมตร) ที่จุด correspondence ครอบคลุม — ใช้เช็ค
    ว่าจุดที่คลิกมามีความหลากหลายพอจะแก้ focal-length↔distance ambiguity
    ได้ไหม (ดู MIN_DEPTH_SPREAD_M)"""
    depths = [corr.world_xyz[1] for corr in correspondences]
    return max(depths) - min(depths) if depths else 0.0


def _polish_pose_gradient(pose: CameraPose, correspondences: list, frame_shape: tuple,
                          bounds: dict) -> tuple[CameraPose, float]:
    """L-BFGS-B polish รอบท่าที่ดีที่สุดจาก random+refine search — จุด
    correspondence ให้ objective (reprojection error) ที่ต่อเนื่อง/หา
    gradient ได้จริง (ต่างจาก edge-hit score ที่เป็น step function
    เพราะนับ pixel hit/miss) จึง refine ได้ละเอียดกว่า coordinate-wise
    hill climbing มาก (validated: coordinate hill-climb เพียงอย่างเดียว
    เหลือ rmse ~6px, เติม polish นี้ทำให้ sub-pixel ได้)"""
    def objective(x):
        candidate = CameraPose(height_m=x[0], distance_m=x[1], tilt_deg=x[2],
                               pan_deg=x[3], focal_length_px=x[4],
                               principal_point=pose.principal_point)
        return reprojection_error_px(candidate, correspondences, frame_shape)

    x0 = [getattr(pose, f) for f in _POSE_FIELDS]
    result = minimize(objective, x0, method="L-BFGS-B",
                      bounds=[bounds[f] for f in _POSE_FIELDS])
    polished = CameraPose(height_m=result.x[0], distance_m=result.x[1],
                          tilt_deg=result.x[2], pan_deg=result.x[3],
                          focal_length_px=result.x[4],
                          principal_point=pose.principal_point)
    return polished, score_pose_against_points(polished, correspondences, frame_shape)


def search_camera_pose_from_points(correspondences: list, frame_shape: tuple,
                                   bounds: dict | None = None,
                                   n_random: int = 4000, n_refine_iters: int = 200,
                                   n_multistart: int = 10,
                                   rng: np.random.Generator | None = None
                                   ) -> tuple[CameraPose | None, float]:
    """หาท่ากล้องจากจุดที่คนคลิกเอง (แทน/เสริม edge-based เมื่อ auto-detect
    เส้นคอร์ทไม่น่าเชื่อถือ หรือคอร์ทไม่เข้าเฟรมครบ)

    รวมจุดบนเน็ต (net_post_left/right, net_top — ความสูง 0.914-1.07m
    เหนือพื้น) เข้าไปด้วยช่วยตัด focal-length↔distance ambiguity ได้มาก
    เพราะเป็นจุดที่ไม่ได้อยู่ระนาบเดียวกับจุดบนพื้น (non-coplanar)

    คืน (None, 0.0) ถ้าจุดที่ให้มาน้อยกว่า MIN_POINT_CORRESPONDENCES
    """
    bounds = bounds or DEFAULT_POSE_BOUNDS
    rng = rng or np.random.default_rng()
    if len(correspondences) < MIN_POINT_CORRESPONDENCES:
        return None, 0.0

    score_fn = lambda pose: score_pose_against_points(pose, correspondences, frame_shape)
    best_pose, best_score = _search_pose_generic(
        score_fn, bounds, n_random, n_refine_iters, n_multistart, rng)

    polished_pose, polished_score = _polish_pose_gradient(
        best_pose, correspondences, frame_shape, bounds)
    if polished_score > best_score:
        best_pose, best_score = polished_pose, polished_score

    return best_pose, best_score
