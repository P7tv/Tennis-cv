"""ฟีเจอร์ "รูปทรงของวงสวิง" — เสริม hit detection ที่ตอนนี้ดูแต่ความเร็ว

ทำไมต้องมี
──────────
ฟีเจอร์ชุดเดิมใน hit_detection.py (FEATURE_COLS) ทุกตัวเป็นอย่างใดอย่างหนึ่ง:

    · ขนาดความเร็ว ณ เฟรมนั้นหรือค่าเฉลี่ยรอบ ๆ  (wrist_speed, speed_pre/post/std)
    · ระยะห่างของวัตถุ ณ เฟรมนั้น                  (ball_dist, racket_dist)
    · ตำแหน่งข้อมือเทียบตัว ณ เฟรมนั้น              (wrist_y_rel, wrist_x_rel)

พอแปลง velocity -> speed ทิศทางถูกทิ้งไปหมด เหลือ wrist_dir_change ตัวเดียวที่
ยังเห็นทิศ แต่มันดูแค่ ±2 เฟรม (~70ms) ซึ่งสั้นกว่าวงสวิงจริง (~1 วิ) มาก

ผลคือโมเดลตอบได้แค่ "ข้อมือเร็วไหม" ไม่เคยถูกถามว่า "มันวาดเป็นวงตีจริงหรือ
เปล่า" — การเดินเร็ว ปรับท่า หรือเก็บลูก ก็ทำให้ข้อมือมี local max ได้เหมือนกัน
นั่นคือที่มาของ false alarm (precision 0.367)

หลักการที่ใช้
────────────
1. kinetic chain — การตีจริงส่งแรงจากล่างขึ้นบน สะโพก -> ไหล่ -> ศอก -> ข้อมือ
   ลำดับการถึงจุดพีคนี้เป็นลายเซ็นเชิงชีวกลศาสตร์ที่การเคลื่อนไหวทั่วไปไม่มี
2. วงกวาด — ข้อมือกวาดเป็นส่วนโค้งรอบไหล่ กวาดกว้างและไปทางเดียว
3. การกลับทิศก่อนปะทะ — ต้องถอย (backswing) แล้วสวนกลับ
4. การหมุนลำตัว — ไหล่หมุนนำ/ตามสะโพก (X-factor)

หน่วยและ fps
────────────
· ระยะทางหารด้วยความกว้างไหล่ (px) เสมอ -> ไม่ขึ้นกับระยะกล้อง/ขนาดตัว
· ความเร็วคูณ _fps_ratio กลับไปเป็น "px ต่อเฟรมที่ 29.97fps"
· ความยาวหน้าต่างใช้ _f() -> ครอบเวลาจริงเท่ากันทุก fps
เหตุผลเดียวกับที่ hit_detection.py ทำ (ดูบล็อกหน่วยเวลาในไฟล์นั้น)

⚠️ ฟีเจอร์กลุ่มนี้แยก "ตีลม" ออกจาก "ตีโดนลูก" ไม่ได้ เพราะตีลมมีวงสวิงและ
   kinetic chain ครบทุกอย่าง สิ่งเดียวที่แยกได้คือลูกบอล ซึ่งเห็นแค่ 4.4-22%
   ของเฟรม (docs/BALL_DETECTION_ISSUE.md) -> นี่คือเพดานที่ฟีเจอร์กลุ่มนี้
   ยกไม่ได้ ไม่ว่าจะเพิ่มอีกกี่ตัว
"""

from __future__ import annotations

import numpy as np

from .config import (L_ELBOW, L_HIP, L_SHOULDER, L_WRIST, R_ELBOW, R_HIP,
                     R_SHOULDER, R_WRIST)
from .hit_detection import _f, _fps_ratio

# หน้าต่างที่ใช้มองวงสวิง (เฟรมที่ 29.97fps) — ~0.5 วิ ต่อข้าง
# เลือกจากข้อมูลจริง: ระยะ backswing_peak -> impact ของ dataset นี้อยู่ราว
# 8-25 เฟรม แล้วแต่ประเภทท่า ส่วน follow_through อยู่หลัง impact ราว 10-20
# เฟรม -> ±15 ครอบทั้งวงโดยไม่กินสโตรกข้างเคียง (GT ห่างกันอย่างน้อย 54 เฟรม)
SWING_WINDOW = 15

# ระยะข้ามเฟรมตอนวัดมุมกวาดสะสม (เฟรมที่ 29.97fps) — ดูเหตุผลในหัวข้อ 2
# ของ swing_shape_features() ยิ่งมากยิ่งทน noise แต่ถ้ามากเกินจะกลืนการ
# เปลี่ยนทิศจริงตอนสวิงเร็ว 3 เฟรม ~ 0.1 วิ ซึ่งสั้นกว่าช่วงกวาดจริงมาก
SWEEP_STRIDE = 3

EPS = 1e-6

SWING_FEATURE_COLS = [
    "chain_order",       # ลำดับพีค สะโพก->ไหล่->ศอก->ข้อมือ ถูกกี่คู่ (0-1)
    "chain_lag",         # เวลาจากพีคสะโพกถึงพีคข้อมือ (วินาที)
    "chain_gain",        # ความเร็วพีคข้อมือ / ความเร็วพีคสะโพก
    "sweep_total",       # มุมกวาดสะสมของข้อมือรอบไหล่ (เรเดียน)
    "sweep_ratio",       # กวาดสุทธิ / กวาดสะสม (1 = ไปทางเดียวเรียบ)
    "path_len_bw",       # ความยาวเส้นทางข้อมือ (หน่วยความกว้างไหล่)
    "straightness",      # ระยะตรง / ความยาวเส้นทาง
    "radius_cv",         # ความแปรปรวนของระยะไหล่-ข้อมือ (แขนเหยียดคงที่ = ต่ำ)
    "pre_reversal_cos",  # cos มุมระหว่างทิศช่วงต้นกับช่วงท้ายก่อนปะทะ (-1 = กลับทิศ)
    "torso_sweep",       # มุมที่แนวไหล่หมุนไปตลอดหน้าต่าง (เรเดียน)
    "x_factor_max",      # ผลต่างมุมไหล่-สะโพกสูงสุด (เรเดียน)
]


SWING3D_FEATURE_COLS = [
    "sweep3d_total",    # มุมกวาดของข้อมือรอบไหล่ ใน 3 มิติ (เรเดียน)
    "path3d_len",       # ความยาวเส้นทางข้อมือ 3D / ความกว้างไหล่
    "arm_ext_range",    # ช่วงการเหยียด-งอแขน (ไหล่->ข้อมือ) / ความกว้างไหล่
    "depth_range",      # ระยะที่ข้อมือเคลื่อนตามแกนลึก — 2D มองไม่เห็นเลย
    "torso3d_sweep",    # มุมที่แนวไหล่หมุนไปใน 3 มิติ
    "plane_flatness",   # เส้นทางข้อมือแบนเป็นระนาบแค่ไหน (0 = แบนสนิท)
]


def _xy(lm: np.ndarray, idx: int, width: float, height: float) -> np.ndarray:
    """landmark normalized -> พิกัด px รูป (T, 2)"""
    return np.stack([lm[:, idx, 0] * width, lm[:, idx, 1] * height], axis=1)


def _mid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2.0


def _speed(p: np.ndarray) -> np.ndarray:
    """ความเร็ว px/เฟรม จาก central difference — คืนความยาวเท่าเดิม ปลาย = NaN"""
    v = np.full(len(p), np.nan)
    if len(p) < 3:
        return v
    d = p[2:] - p[:-2]
    v[1:-1] = np.linalg.norm(d, axis=1) / 2.0
    return v


def _argmax_or_none(a: np.ndarray) -> int | None:
    """argmax ที่ทน NaN ล้วน"""
    if a.size == 0 or np.all(np.isnan(a)):
        return None
    return int(np.nanargmax(a))


def _unwrapped_angle(vec: np.ndarray) -> np.ndarray:
    """atan2 ต่อเฟรม แล้ว unwrap — ช่องที่เป็น NaN ถูกเว้นไว้

    ต้อง unwrap เฉพาะช่วงที่ต่อเนื่องกัน ไม่งั้น np.unwrap จะกระโดดข้าม NaN
    แล้วสร้างมุมสะสมปลอมขึ้นมา
    """
    ang = np.arctan2(vec[:, 1], vec[:, 0])
    ok = ~np.isnan(ang)
    if ok.sum() >= 2:
        ang[ok] = np.unwrap(ang[ok])
    return ang


def _nan_ptp(a: np.ndarray) -> float:
    """ค่าสูงสุด - ต่ำสุด ที่ทน NaN"""
    if np.all(np.isnan(a)):
        return 0.0
    return float(np.nanmax(a) - np.nanmin(a))


def swing_shape_features(
    landmarks: np.ndarray,
    frame: int,
    fps: float | None = None,
    width: float = 1.0,
    height: float = 1.0,
    side: str | None = None,
) -> dict:
    """ฟีเจอร์รูปทรงวงสวิงรอบเฟรมที่กำหนด

    landmarks  (T, 33, C) normalized 0-1 แบบเดียวกับ PoseTimeseries.landmarks
    frame      เฟรมที่สงสัยว่าเป็นจังหวะปะทะ
    side       "left"/"right" — ข้างที่ถือไม้ ถ้าไม่ระบุจะเลือกข้างที่เร็วกว่า

    คืน dict ที่มีคีย์ครบตาม SWING_FEATURE_COLS เสมอ (ค่า default เมื่อคำนวณ
    ไม่ได้) — เพื่อให้เอาไปทำ DataFrame แล้วไม่มีช่องหาย
    """
    out = {
        "chain_order": 0.0, "chain_lag": 0.0, "chain_gain": 1.0,
        "sweep_total": 0.0, "sweep_ratio": 0.0,
        "path_len_bw": 0.0, "straightness": 1.0, "radius_cv": 0.0,
        "pre_reversal_cos": 1.0,
        "torso_sweep": 0.0, "x_factor_max": 0.0,
    }
    if landmarks is None or len(landmarks) < 5:
        return out

    T = len(landmarks)
    W = _f(SWING_WINDOW, fps)
    lo, hi = max(0, frame - W), min(T, frame + W // 2 + 1)
    if hi - lo < 5:
        return out

    lm = landmarks[lo:hi]
    f_rel = frame - lo
    r = _fps_ratio(fps)

    l_sh = _xy(lm, L_SHOULDER, width, height)
    r_sh = _xy(lm, R_SHOULDER, width, height)
    l_hip = _xy(lm, L_HIP, width, height)
    r_hip = _xy(lm, R_HIP, width, height)

    # ช่วงที่ track หลุดจะเป็น NaN ล้วน — เช็คก่อนเรียก nanmedian เพื่อไม่ให้
    # numpy ยิง RuntimeWarning ออกมารกทุกคลิป (ผลลัพธ์เหมือนกัน)
    sw_px = np.abs(r_sh[:, 0] - l_sh[:, 0])
    if np.all(np.isnan(sw_px)):
        return out
    bw = float(np.nanmedian(sw_px))
    if not np.isfinite(bw) or bw < EPS:
        return out

    sh_c, hip_c = _mid(l_sh, r_sh), _mid(l_hip, r_hip)

    # ── เลือกข้างที่ถือไม้ ──────────────────────────────────────────────
    wr = {"left": _xy(lm, L_WRIST, width, height),
          "right": _xy(lm, R_WRIST, width, height)}
    el = {"left": _xy(lm, L_ELBOW, width, height),
          "right": _xy(lm, R_ELBOW, width, height)}
    if side not in ("left", "right"):
        sp = {k: _speed(v) for k, v in wr.items()}
        peak = {k: (np.nanmax(v) if not np.all(np.isnan(v)) else -np.inf)
                for k, v in sp.items()}
        side = "left" if peak["left"] > peak["right"] else "right"
    wrist, elbow = wr[side], el[side]

    # ── 1. kinetic chain ───────────────────────────────────────────────
    # พีคของแต่ละข้อต่อควรมาตามลำดับ สะโพก -> ไหล่ -> ศอก -> ข้อมือ
    # (proximal-to-distal sequencing) จำกัดการมองไว้ถึงเฟรมปะทะ เพราะหลัง
    # ปะทะเป็น follow-through ซึ่งมีพีคของตัวเองแล้วทำให้ลำดับเพี้ยน
    chain_end = min(len(lm), f_rel + 2)
    speeds = {name: _speed(p)[:chain_end] * r
              for name, p in (("hip", hip_c), ("sh", sh_c),
                              ("el", elbow), ("wr", wrist))}
    order = ["hip", "sh", "el", "wr"]
    peaks = {k: _argmax_or_none(v) for k, v in speeds.items()}
    if all(peaks[k] is not None for k in order):
        pairs = [(order[i], order[i + 1]) for i in range(3)]
        out["chain_order"] = sum(peaks[a] <= peaks[b] for a, b in pairs) / 3.0
        out["chain_lag"] = (peaks["wr"] - peaks["hip"]) / float(fps or 29.97)
        hip_peak = float(np.nanmax(speeds["hip"])) if not np.all(
            np.isnan(speeds["hip"])) else 0.0
        wr_peak = float(np.nanmax(speeds["wr"])) if not np.all(
            np.isnan(speeds["wr"])) else 0.0
        out["chain_gain"] = wr_peak / (hip_peak + EPS)

    # ── 2. วงกวาดของข้อมือรอบไหล่ ──────────────────────────────────────
    #
    # 🔴 วัดมุมสะสมแบบข้าม stride ไม่ใช่ทีละเฟรม — |dมุม| ทีละเฟรมสะสม noise
    # ของ pose estimator ไปด้วย ข้อมือมี jitter ~0.084 ส่วนของความกว้างไหล่
    # (loeuf_cv/augment.py วัดจากข้อมูลจริง) ซึ่งที่รัศมีแขนปกติคิดเป็นมุม
    # ~0.04 rad/เฟรม สุ่มทิศ -> คนที่ยืนเฉย ๆ 23 เฟรมสะสมได้เกือบ 1 rad
    # เทียบเท่าการกวาดจริง 57 องศา ทั้งที่ไม่ได้ขยับไปไหนเลย
    #
    # การกวาดจริงมีทิศทางต่อเนื่อง จึงรอดจากการ decimate ส่วน noise สุ่มทิศ
    # หักล้างกันเองภายใน stride (เจอจาก test ที่ให้คนยืนสั่นอยู่กับที่แล้ว
    # ได้ sweep_total สูงกว่าวงสวิงจริง)
    rel = wrist - sh_c
    ang = _unwrapped_angle(rel)
    ok = ~np.isnan(ang)
    if ok.sum() >= 3:
        a = ang[ok]
        stride = min(_f(SWEEP_STRIDE, fps), max(1, len(a) // 3))
        a_s = a[::stride]
        if len(a_s) < 2 or a_s[-1] != a[-1]:
            a_s = np.append(a_s, a[-1])      # อย่าทิ้งหางของช่วง
        total = float(np.sum(np.abs(np.diff(a_s))))
        out["sweep_total"] = total
        out["sweep_ratio"] = float(abs(a[-1] - a[0]) / (total + EPS))

    # ── 3. รูปทรงเส้นทาง ───────────────────────────────────────────────
    #
    # ⚠️ ข้อจำกัดที่รู้อยู่ ยังไม่แก้: การตัด NaN ออกแล้วต่อจุดที่เหลือเข้าหากัน
    # ทำให้ช่วงที่ track หลุดกลายเป็น "กระโดดครั้งเดียวไกล ๆ" ซึ่งบวกเข้า
    # ความยาวเส้นทางเป็นระยะปลอม (มีผลกับ path_len_bw / straightness)
    #
    # ไม่แก้ตอนนี้เพราะ hit_classifier.pkl ที่ใช้อยู่เทรนด้วยนิยามนี้ — เปลี่ยน
    # สูตรฟีเจอร์โดยไม่เทรนใหม่คือ train/serve skew (โปรเจกต์นี้โดนมาแล้ว 3 รอบ)
    # ถ้าจะแก้ ต้องทำพร้อมกันทั้งชุด: แก้สูตร -> เทรนใหม่ -> วัด LOPO ใหม่ ->
    # รัน benchmark ใหม่ ดู docs/SWING_SHAPE_FEATURES.md หัวข้อ "ข้อจำกัด"
    ok_w = ~np.isnan(wrist[:, 0])
    if ok_w.sum() >= 3:
        p = wrist[ok_w]
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        path = float(np.sum(seg))
        out["path_len_bw"] = path / bw
        out["straightness"] = float(
            np.linalg.norm(p[-1] - p[0]) / (path + EPS))
        rad = np.linalg.norm(rel[ok_w], axis=1)
        m = float(np.nanmean(rad))
        out["radius_cv"] = float(np.nanstd(rad) / (m + EPS)) if m > EPS else 0.0

    # ── 4. การกลับทิศก่อนปะทะ (backswing -> forward) ───────────────────
    # เทียบทิศเฉลี่ยของครึ่งแรกกับครึ่งหลังของช่วง "ก่อน" เฟรมปะทะ
    # ตีจริง: ถอยแล้วสวน -> cos ติดลบ · เดิน/ปรับท่า: ไปทางเดิม -> cos บวก
    half = f_rel // 2
    if f_rel >= 4 and half >= 2:
        d1 = wrist[half] - wrist[0]
        d2 = wrist[f_rel] - wrist[half]
        n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
        if np.isfinite(n1) and np.isfinite(n2) and n1 > EPS and n2 > EPS:
            out["pre_reversal_cos"] = float(np.dot(d1, d2) / (n1 * n2))

    # ── 5. การหมุนลำตัว ────────────────────────────────────────────────
    sh_ang = _unwrapped_angle(r_sh - l_sh)
    hip_ang = _unwrapped_angle(r_hip - l_hip)
    out["torso_sweep"] = _nan_ptp(sh_ang[:f_rel + 1])
    sep = np.abs(sh_ang - hip_ang)[:f_rel + 1]
    if not np.all(np.isnan(sep)):
        out["x_factor_max"] = float(np.nanmax(sep))

    return out


# ---------------------------------------------------------------------------
# เวอร์ชัน 3 มิติ — ใช้ world_landmarks ที่ไม่เคยถูกแตะเลย
# ---------------------------------------------------------------------------
#
# ทำไมถึงน่าจะดีกว่า 2D: ฟีเจอร์ข้างบนวัดบน "ภาพที่ถูกกล้องฉายแบน" วงสวิงเดียวกัน
# ถ่ายคนละมุมได้ค่าคนละอย่าง ส่วน world_landmarks ของ MediaPipe เป็นพิกัดเมตร
# ที่อ้างอิงจุดกึ่งกลางสะโพก -> ระยะกล้อง/ซูม/มุมกล้อง หายไปจากสมการ
#
# ตรวจแล้วว่าใช้ได้จริงบนข้อมูลชุดนี้ (ไม่ใช่ค่าขยะ):
#   ครอบคลุม 99.8% ของเฟรม (NaN 0.24%)
#   ความกว้างไหล่ median 0.335 m (ถูกตามกายวิภาค) · std ข้ามเฟรมแค่ 8%
#   ไหล่->ข้อมือ median 0.436 m
#   แกนลึกขยับจริง 0.98 m และเรียบ (เปลี่ยนเฟรมละ 0.029 m)
#
# ⚠️ ความลึกของ MediaPipe เป็นค่าที่ **โมเดลเดา** ไม่ใช่วัดด้วยเซนเซอร์ แกน z
# จึงอ่อนที่สุดในสามแกน — ต้องวัดผลจริง ห้ามสรุปว่าดีกว่าเพราะ "เป็น 3D"
#
# ยังหารด้วยความกว้างไหล่อยู่ ทั้งที่เป็นเมตรแล้ว — เพราะต้องการตัดผลของ
# **ขนาดตัวคน** ด้วย ไม่ใช่แค่ระยะกล้อง (คนสูงแขนยาวกว่าโดยธรรมชาติ)


def swing_shape_3d_features(
    world_landmarks: np.ndarray,
    frame: int,
    fps: float | None = None,
    side: str | None = None,
) -> dict:
    """ฟีเจอร์รูปทรงวงสวิงในพิกัด 3 มิติ (เมตร) จาก PoseTimeseries.world_landmarks"""
    out = {"sweep3d_total": 0.0, "path3d_len": 0.0, "arm_ext_range": 0.0,
           "depth_range": 0.0, "torso3d_sweep": 0.0, "plane_flatness": 0.0}
    if world_landmarks is None or len(world_landmarks) < 5:
        return out

    T = len(world_landmarks)
    W = _f(SWING_WINDOW, fps)
    lo, hi = max(0, frame - W), min(T, frame + W // 2 + 1)
    if hi - lo < 5:
        return out
    # หน้าต่างเอียงมาทางช่วงก่อนปะทะ (เหมือนเวอร์ชัน 2D) จึงไม่ต้องแยกก่อน/หลัง
    lm = world_landmarks[lo:hi]

    l_sh, r_sh = lm[:, L_SHOULDER, :3], lm[:, R_SHOULDER, :3]
    sh_len = np.linalg.norm(r_sh - l_sh, axis=1)
    if np.all(np.isnan(sh_len)):        # เช็คก่อนเรียก nanmedian กัน RuntimeWarning
        return out
    bw = float(np.nanmedian(sh_len))
    if not np.isfinite(bw) or bw < EPS:
        return out

    sh_c = (l_sh + r_sh) / 2.0
    wrist = lm[:, R_WRIST if side != "left" else L_WRIST, :3]

    rel = wrist - sh_c
    ok = np.isfinite(rel).all(axis=1)
    if ok.sum() < 3:
        return out
    v = rel[ok]

    # มุมกวาดสะสม — ข้าม stride ด้วยเหตุผลเดียวกับเวอร์ชัน 2D (กัน noise สะสม)
    stride = min(_f(SWEEP_STRIDE, fps), max(1, len(v) // 3))
    vs = v[::stride]
    if len(vs) >= 2:
        n = np.linalg.norm(vs, axis=1)
        good = n > EPS
        vs, n = vs[good], n[good]
        if len(vs) >= 2:
            cos = np.sum(vs[:-1] * vs[1:], axis=1) / (n[:-1] * n[1:])
            out["sweep3d_total"] = float(
                np.sum(np.arccos(np.clip(cos, -1.0, 1.0))))

    p = wrist[ok]
    out["path3d_len"] = float(np.sum(np.linalg.norm(np.diff(p, axis=0),
                                                   axis=1)) / bw)
    arm = np.linalg.norm(v, axis=1)
    out["arm_ext_range"] = float((arm.max() - arm.min()) / bw)
    out["depth_range"] = float((p[:, 2].max() - p[:, 2].min()) / bw)

    # ── ความแบนของระนาบสวิง — วัดได้เฉพาะใน 3 มิติ ──────────────────────
    # การตีเทนนิสเกิดในระนาบสวิงระนาบเดียวโดยประมาณ (swing plane เป็นแนวคิด
    # มาตรฐานทางชีวกลศาสตร์) ส่วนการเดิน/ปรับท่า/เอื้อมหยิบของ ไม่มีระนาบ
    # ค่านี้คือ singular value ที่เล็กที่สุด / ใหญ่ที่สุด ของเส้นทางข้อมือที่
    # ลบค่าเฉลี่ยแล้ว = ความหนาของเส้นทางเทียบกับความกว้าง (0 = แบนสนิท)
    if len(p) >= 4:
        s = np.linalg.svd(p - p.mean(axis=0), compute_uv=False)
        if s[0] > EPS:
            out["plane_flatness"] = float(s[-1] / s[0])

    sh_v = (r_sh - l_sh)[np.isfinite(r_sh - l_sh).all(axis=1)]
    if len(sh_v) >= 2:
        n = np.linalg.norm(sh_v, axis=1)
        good = n > EPS
        sh_v, n = sh_v[good], n[good]
        if len(sh_v) >= 2:
            cos = np.sum(sh_v[:-1] * sh_v[1:], axis=1) / (n[:-1] * n[1:])
            out["torso3d_sweep"] = float(
                np.sum(np.arccos(np.clip(cos, -1.0, 1.0))))

    return out
