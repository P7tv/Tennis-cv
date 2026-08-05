"""ฟีเจอร์จากไม้เทนนิส — ตัวที่ปะทะลูกจริง ๆ แต่เกือบไม่เคยถูกใช้

ทำไมของเดิมใช้ไม่ได้
────────────────────
hit_detection.py มีฟีเจอร์ชื่อ `racket_dist` อยู่แล้ว แต่มันคือ **ระยะจากไม้ถึง
ลูกบอล** และคำนวณเฉพาะเมื่อเห็นลูก:

    if racket_bboxes and racket_bboxes.get(f) and not np.isnan(ball_traj[f]).any():

ในคลิปจริงเห็นลูกลอยแค่ 4.4-22% ของเฟรม (docs/BALL_DETECTION_ISSUE.md) ส่วนไม้
เห็นถึง 15-100% (median ~67%) -> ข้อมูลไม้ส่วนใหญ่ถูกล็อกไว้หลังเงื่อนไขลูกบอล
ทั้งที่คำถาม "คนนี้กำลังเหวี่ยงไม้อยู่ไหม" ไม่ต้องใช้ลูกบอลเลย

🔴 ปัญหาคุณภาพข้อมูลที่ต้องกรองก่อน
──────────────────────────────────
วัดระยะจาก bbox ไม้ถึงข้อมือที่ใกล้ที่สุดในทุกคลิป พบว่าหลายคลิปได้ค่า
**19-25 เท่าของความกว้างไหล่** ทั้งที่ไม้ที่ถืออยู่ในมือควรห่างราว 0.5-1.5 เท่า:

    IMG_0300A4(VL)    25.3 เท่า   ทั้งที่ "เจอไม้" 100% ของเฟรม
    IMG_0301A4(SV1)   19.0 เท่า   เจอ 96%
    IMG_0300B53-VL     9.1 เท่า

= ตัวตรวจจับไปล็อกวัตถุอื่น (ไม้ของคนอื่น / ไม้ที่วางอยู่) ราว 5 จาก 16 คลิป
ถ้าเอามาใช้ตรง ๆ จะได้ข้อมูลเป็นพิษใน 1 ใน 3 ของคลิปโดยไม่มี error ให้เห็น

จึงต้องกรองด้วยระยะจากมือก่อนเสมอ (หลักการเดียวกับ filter_static_ball_bboxes
ที่คัด "ลูกที่นอนนิ่ง" ออกก่อนเข้า Kalman) — ทั้ง dataset เหลือ 47% ของ
detection ดิบ = 19,290 เฟรมที่เป็นไม้ของผู้เล่นจริง

⚠️ สิ่งที่ลองแล้วไม่ได้ผล: ใช้ระยะไม้-ข้อมือหา "มือถนัด" ของแต่ละคลิป —
ความมั่นใจออกมา 53-81% ไม่เด็ดขาดพอจะเอามาเป็นฟีเจอร์ (ค่า dominant_side ใน
cache ก็เชื่อไม่ได้ เพราะเป็น dropdown ที่คนเลือกเองใน webui ไม่ได้คำนวณ)
"""

from __future__ import annotations

import numpy as np

from .config import L_SHOULDER, L_WRIST, R_SHOULDER, R_WRIST
from .hit_detection import _f, _fps_ratio
from .swing_shape import EPS, SWING_WINDOW

# ไม้ที่อยู่ห่างข้อมือเกินกี่เท่าของความกว้างไหล่ = ไม่ใช่ไม้ของผู้เล่นคนนี้
# 3.0 เผื่อไว้กว้างพอสมควร (ความยาวไม้จริงราว 1.5-2 เท่าของความกว้างไหล่ บวก
# ความคลาดของ bbox) แต่ยังตัดเคส 9-25 เท่าที่เห็นในข้อมูลออกได้หมด
RACKET_GATE_BW = 3.0

RACKET_FEATURE_COLS = [
    "racket_seen_frac",    # สัดส่วนเฟรมในหน้าต่างที่เจอไม้ "ของผู้เล่นคนนี้"
    "racket_wrist_bw",     # ระยะไม้-ข้อมือ ณ เฟรมที่สงสัย
    "racket_speed_bw",     # ความเร็วศูนย์กลางไม้ (px/เฟรมที่ 29.97fps / ไหล่)
    "racket_speed_ratio",  # ความเร็วไม้ / ความเร็วข้อมือ = ผลของคันโยก
    "racket_decel",        # ความเร็วไม้หลังปะทะ / พีค = การเบรกจากถ่ายโมเมนตัม
    "racket_area_cv",      # ความแปรปรวนพื้นที่ bbox = การพลิกหน้าไม้
]


def _wrist_px(lm, f, idx, width, height):
    x, y = lm[f, idx, 0] * width, lm[f, idx, 1] * height
    return None if np.isnan(x) or np.isnan(y) else np.array([x, y])


def _gated_racket_track(racket_bboxes, lm, lo, hi, bw, width, height, side):
    """คืน (เฟรม, ศูนย์กลาง, พื้นที่, ระยะถึงข้อมือ) ของไม้ที่ผ่านด่านกรองแล้ว

    เลือก bbox ที่ใกล้ข้อมือที่สุดในเฟรมนั้น (ไม่ใช่ box แรกในลิสต์) แล้วรับไว้
    ต่อเมื่ออยู่ใกล้พอที่จะเป็นไม้ที่ถืออยู่จริง
    """
    idxs = ((R_WRIST,) if side == "right" else
            (L_WRIST,) if side == "left" else (L_WRIST, R_WRIST))
    out = []
    for f in range(lo, hi):
        boxes = racket_bboxes.get(f)
        if not boxes or f >= len(lm):
            continue
        wrists = [w for w in (_wrist_px(lm, f, i, width, height) for i in idxs)
                  if w is not None]
        if not wrists:
            continue
        best = None
        for x, y, w, h in boxes:
            c = np.array([x + w / 2.0, y + h / 2.0])
            d = min(float(np.linalg.norm(c - wr)) for wr in wrists)
            if best is None or d < best[2]:
                best = (c, float(w * h), d)
        if best is not None and best[2] <= RACKET_GATE_BW * bw:
            out.append((f, best[0], best[1], best[2]))
    return out


def racket_shape_features(
    racket_bboxes: dict | None,
    landmarks: np.ndarray,
    frame: int,
    fps: float | None = None,
    width: float = 1.0,
    height: float = 1.0,
    side: str | None = None,
    wrist_speed_px: float | None = None,
) -> dict:
    """ฟีเจอร์ของไม้รอบเฟรมที่สงสัยว่าเป็นจังหวะปะทะ

    คืน dict ที่มีคีย์ครบตาม RACKET_FEATURE_COLS เสมอ — เฟรมที่ไม่เจอไม้จะได้
    ค่า default ที่แปลว่า "ไม่มีข้อมูล" คู่กับ racket_seen_frac = 0 ซึ่งเป็น
    ตัวบอกโมเดลว่าไม่ต้องเชื่อฟีเจอร์ที่เหลือในกลุ่มนี้
    """
    out = {
        "racket_seen_frac": 0.0, "racket_wrist_bw": RACKET_GATE_BW,
        "racket_speed_bw": 0.0, "racket_speed_ratio": 0.0,
        "racket_decel": 1.0, "racket_area_cv": 0.0,
    }
    if not racket_bboxes or landmarks is None or len(landmarks) < 5:
        return out

    T = len(landmarks)
    W = _f(SWING_WINDOW, fps)
    lo, hi = max(0, frame - W), min(T, frame + W // 2 + 1)
    if hi - lo < 3:
        return out

    lm = landmarks
    sw = np.abs(lm[lo:hi, R_SHOULDER, 0] - lm[lo:hi, L_SHOULDER, 0]) * width
    if np.all(np.isnan(sw)):
        return out
    bw = float(np.nanmedian(sw))
    if not np.isfinite(bw) or bw < EPS:
        return out

    track = _gated_racket_track(racket_bboxes, lm, lo, hi, bw, width, height,
                               side)
    out["racket_seen_frac"] = len(track) / float(hi - lo)
    if not track:
        return out

    frames = np.array([t[0] for t in track])
    cent = np.array([t[1] for t in track])
    area = np.array([t[2] for t in track])
    dist = np.array([t[3] for t in track])

    # ระยะ ณ เฟรมที่สงสัย — ถ้าเฟรมนั้นไม่เจอไม้ ใช้ตัวที่ใกล้ที่สุดในเวลา
    out["racket_wrist_bw"] = float(
        dist[int(np.argmin(np.abs(frames - frame)))] / bw)

    if len(area) >= 2:
        m = float(np.mean(area))
        if m > EPS:
            out["racket_area_cv"] = float(np.std(area) / m)

    if len(track) >= 3:
        # ความเร็วต่อเฟรมจากคู่ที่ติดกัน — หารช่องว่างจริง ไม่ใช่สมมติว่าห่าง 1
        # เฟรม (ไม้หายเป็นช่วง ๆ ถ้าไม่หารจะได้ความเร็วสูงเกินจริงตรงรอยต่อ)
        gaps = np.diff(frames).astype(float)
        step = np.linalg.norm(np.diff(cent, axis=0), axis=1) / np.maximum(gaps, 1.0)
        # ข้ามรอยต่อที่ห่างเกินครึ่งหนึ่งของหน้าต่าง — ตรงนั้นเป็นการเดา ไม่ใช่ข้อมูล
        step = step[np.diff(frames) <= max(2, W // 2)]
        if step.size:
            r = _fps_ratio(fps)
            peak = float(np.max(step)) * r
            out["racket_speed_bw"] = peak / bw
            if wrist_speed_px:
                out["racket_speed_ratio"] = peak / (wrist_speed_px * r + EPS)
            # การเบรก: ความเร็วเฉลี่ยหลังพีค เทียบกับพีค — ตอนปะทะจริงโมเมนตัม
            # ถ่ายให้ลูก ไม้จึงช้าลงชัดกว่าการเหวี่ยงเปล่า
            i = int(np.argmax(step))
            post = step[i + 1:]
            out["racket_decel"] = float(
                np.mean(post) * r / (peak + EPS)) if post.size else 1.0
    return out
