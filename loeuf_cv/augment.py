"""Skeleton augmentation — สร้างตัวอย่างสังเคราะห์จาก pose ที่มีอยู่

ใช้กับ **hit detection** เป็นหลัก ไม่ใช่ stroke classification — เหตุผล:

  hit detection  โจทย์คือ "ช่วงนี้คือจังหวะปะทะไหม" ซึ่งตัดสินจากรูปทรงของ
                 กราฟความเร็ว (พุ่งขึ้นแล้วเบรกกะทันหัน) กฎฟิสิกส์ข้อนี้เป็น
                 สากล ไม่ขึ้นกับว่าใครตี -> คูณสเกลความเร็ว/ยืดหดเวลา = ได้
                 "จังหวะปะทะแบบใหม่ที่ถูกฟิสิกส์จริง" ความหลากหลายเพิ่มขึ้นจริง

  stroke type    โจทย์คือ "ท่านี้คือท่าอะไร" ซึ่งตัดสินจากรูปร่างท่าทางของคน
                 การ scale/ใส่ noise ให้โครงกระดูกของคนเดิม ไม่ได้สร้าง
                 "สไตล์การตีของคนใหม่" -> ความหลากหลายไม่เพิ่ม
                 ⚠️ วัดแล้วยืนยัน: FH มาจากคนเดียวจริง ๆ (20 จาก 21 ตัวอย่าง
                 เป็นของ pl006) พอ LOPO ตัดคนนั้นออก เหลือตัวอย่างเดียว —
                 augment เท่าไรก็ยังเป็นคนเดิม ดู docs/STROKE_CLASSIFICATION.md

ผลที่วัดได้จริงกับ hit classifier (Leave-One-Person-Out, 16 คลิป / 7 คน):

    สำเนา/คลิป   แถวฝึก    F1 ดีสุด   recall   precision
    0 (เดิม)      2,535     0.461     54.7%     39.8%
    3            11,566     0.544     57.0%     52.1%

ที่ threshold เดียวกับ production (0.60) ดีขึ้น **ทั้งสองด้านพร้อมกัน**
(recall 54.7% -> 64.0%, precision 39.8% -> 42.6%) ไม่ใช่การแลกกัน

ทุกฟังก์ชันรับ/คืน landmark array รูป (T, J, C) พิกัด normalized 0-1 เหมือน
PoseTimeseries.landmarks และไม่แก้ของเดิม (คืน array ใหม่เสมอ)

⚠️ หลักการที่ห้ามละเมิด: augment ที่ระดับ **landmark** แล้วปล่อยให้ feature
ถูกคำนวณใหม่ด้วยโค้ดจริง ห้าม augment ที่ระดับตัวเลข feature — ไม่งั้นเกิด
train/serve skew (โปรเจกต์นี้เคยโดนมาแล้ว 3 รอบ ดู
schema_builder/classifier.py::build_stroke_features)
"""

from __future__ import annotations

import numpy as np

from .config import L_ANKLE, L_HIP, L_SHOULDER, R_ANKLE, R_HIP, R_SHOULDER

# ---------------------------------------------------------------------------
# ขนาด noise — วัดจากข้อมูลจริง ไม่ได้ตั้งเอง
# ---------------------------------------------------------------------------
#
# วิธีวัด: fit Savitzky-Golay (หน้าต่าง 11 เฟรม, องศา 3) ให้ trajectory ของแต่ละ
# joint แล้วดู residual — แต่**นับเฉพาะเฟรมที่ joint นั้นเกือบนิ่ง** (ความเร็ว
# ต่ำกว่าเปอร์เซ็นไทล์ที่ 25) เพราะตอนสวิงเร็ว residual คือการเคลื่อนไหวจริงที่
# ตัวปรับเรียบตามไม่ทัน ไม่ใช่ noise — ถ้ารวมช่วงสวิงด้วยจะได้ค่าสูงเกินจริง
# เกือบ 3 เท่า (ข้อมือ 0.250 -> 0.084) แล้วตั้ง sigma ใหญ่จนกลบสัญญาณที่ต้องการ
#
# ผลจาก 16 คลิปจริง (หน่วย = ส่วนของความกว้างไหล่):
#   nose 0.049 · ไหล่ 0.049-0.062 · ศอก 0.064-0.093 · ข้อมือ 0.071-0.098
JITTER_SIGMA_CORE = 0.063    # ลำตัว/หัว/สะโพก/เข่า
JITTER_SIGMA_LIMB = 0.084    # ข้อมือ/ศอก — ปลายแขนสั่นกว่าเพราะเคลื่อนเร็ว

# joint ที่ถือว่าเป็น "ปลายแขน" (สั่นมากกว่า) — index ตาม MediaPipe BlazePose
_LIMB_JOINTS = (13, 14, 15, 16, 17, 18, 19, 20, 21, 22)

# คู่ซ้าย-ขวาของ BlazePose 33 จุด สำหรับ mirror
_LR_PAIRS = ((1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14),
             (15, 16), (17, 18), (19, 20), (21, 22), (23, 24), (25, 26),
             (27, 28), (29, 30), (31, 32))


def shoulder_width(landmarks: np.ndarray) -> float:
    """ความกว้างไหล่ median (หน่วย normalized) — ไม้บรรทัดของ noise/สเกล"""
    sw = np.abs(landmarks[:, R_SHOULDER, 0] - landmarks[:, L_SHOULDER, 0])
    if np.isnan(sw).all():
        return 0.0
    v = float(np.nanmedian(sw))
    return v if np.isfinite(v) else 0.0


def jitter_landmarks(landmarks: np.ndarray, rng: np.random.Generator,
                     scale: float = 1.0) -> np.ndarray:
    """เติม noise แบบเกาส์เซียนให้ทุก joint ทุกเฟรม

    ขนาด noise ผูกกับความกว้างไหล่ของคลิปนั้น -> คลิปที่ยืนไกลกล้อง (ตัวเล็กใน
    ภาพ) ได้ noise เล็กตามสัดส่วน เหมือนที่ pose estimator สั่นจริง

    scale: ตัวคูณ sigma (1.0 = เท่าที่วัดได้จริง) ใช้ทดลองว่า noise แรง/เบา
    ส่งผลยังไง — ค่าที่มากกว่า ~2 เริ่มกลบสัญญาณการตี
    """
    out = landmarks.copy()
    bw = shoulder_width(landmarks)
    if bw <= 0 or scale <= 0:
        return out
    sigma = np.full(out.shape[1], JITTER_SIGMA_CORE * bw * scale)
    for j in _LIMB_JOINTS:
        if j < len(sigma):
            sigma[j] = JITTER_SIGMA_LIMB * bw * scale
    noise = rng.normal(0.0, 1.0, size=out[..., :2].shape) * sigma[None, :, None]
    out[..., :2] += noise
    return out


def rescale_body(landmarks: np.ndarray, factor: float) -> np.ndarray:
    """ย่อ/ขยายโครงกระดูกรอบจุดกึ่งกลางสะโพก (คนตัวใหญ่ขึ้น/เล็กลง)

    ขยายรอบสะโพกไม่ใช่รอบจุด (0,0) เพื่อให้ผู้เล่น **ยังยืนที่เดิมในเฟรม** —
    ถ้าขยายรอบมุมภาพ ตัวจะลอยออกนอกเฟรมและ feature ตำแหน่งจะเพี้ยนหมด

    ผลที่ได้เทียบเท่า "คนตัวสูงขึ้น 10%" หรือ "กล้องเข้าใกล้ขึ้น 10%" ซึ่ง
    แยกกันไม่ออกอยู่แล้วในภาพ 2 มิติมุมเดียว
    """
    out = landmarks.copy()
    hip = np.nanmean(out[:, [L_HIP, R_HIP], :2], axis=1)      # (T, 2)
    ok = np.isfinite(hip).all(axis=1)
    if not ok.any():
        return out
    hip[~ok] = np.nanmean(hip[ok], axis=0)
    out[..., :2] = hip[:, None, :] + (out[..., :2] - hip[:, None, :]) * factor
    return out


def time_warp(landmarks: np.ndarray, factor: float
              ) -> tuple[np.ndarray, np.ndarray]:
    """ยืด/หดแกนเวลา — factor > 1 = สวิงช้าลง (เฟรมเยอะขึ้น)

    คืน (landmarks ใหม่, src_index) โดย src_index[i] = ตำแหน่งเฟรมเดิม
    (ทศนิยม) ที่เฟรมใหม่ i มาจาก ใช้ map เฟรมของเฉลย/ลูก/ไม้ให้ตรงกัน

    นี่คือตัวที่สร้าง "จังหวะปะทะแบบใหม่" จริง ๆ — คนตีเร็ว/ช้าต่างกันได้ตาม
    ธรรมชาติ และรูปทรงการเบรกยังถูกฟิสิกส์อยู่ ต่างจากการเติม noise ที่ให้แค่
    ความทนทาน
    """
    t = len(landmarks)
    if t < 2 or factor <= 0:
        return landmarks.copy(), np.arange(t, dtype=float)
    n_new = max(2, int(round(t * factor)))
    src = np.linspace(0, t - 1, n_new)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, t - 1)
    w = (src - lo)[:, None, None]
    out = landmarks[lo] * (1.0 - w) + landmarks[hi] * w
    return out, src


def warp_frame_index(frame: float, src_index: np.ndarray) -> int:
    """แปลงเลขเฟรม "ของเดิม" -> เลขเฟรมในไทม์ไลน์ที่ยืด/หดแล้ว

    src_index เพิ่มขึ้นเสมอ จึงใช้ searchsorted หาตำแหน่งได้ตรง ๆ
    """
    return int(np.clip(np.searchsorted(src_index, float(frame)),
                       0, len(src_index) - 1))


def resample_series(arr: np.ndarray, src_index: np.ndarray) -> np.ndarray:
    """ยืด/หด array ที่ index ตามเฟรม (เช่น ball_traj (T,2)) ให้ตรงกับ time_warp

    ใช้ interpolate เชิงเส้นแบบเดียวกับ time_warp เพื่อให้ ball/pose ยังตรงเฟรม
    กัน — ถ้าไม่ทำ ฟีเจอร์ระยะห่างลูก-ข้อมือจะเพี้ยนทั้งชุด
    NaN ถูกรักษาไว้ (เฟรมที่ไม่เจอลูกต้องยังเป็น "ไม่เจอ")
    """
    t = len(arr)
    if t < 2:
        return arr.copy()
    lo = np.floor(src_index).astype(int)
    hi = np.minimum(lo + 1, t - 1)
    w = src_index - lo
    while w.ndim < arr.ndim:
        w = w[..., None]
    out = arr[lo] * (1.0 - w) + arr[hi] * w
    # ถ้าปลายใดปลายหนึ่งเป็น NaN ผลลัพธ์ต้องเป็น NaN (อย่าให้ interp กลบไป)
    out[np.isnan(arr[lo]) | np.isnan(arr[hi])] = np.nan
    return out


def remap_bboxes(bboxes: dict | None, src_index: np.ndarray) -> dict | None:
    """ย้าย dict {frame -> list[bbox]} ไปยังไทม์ไลน์ที่ยืด/หดแล้ว"""
    if not bboxes:
        return bboxes
    out: dict[int, list] = {}
    for f, items in bboxes.items():
        out.setdefault(warp_frame_index(f, src_index), []).extend(items)
    return out


def mirror_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """สะท้อนซ้าย-ขวา — ได้ "คนถนัดมืออีกข้างที่ตีท่าเดิม"

    🔴 ห้ามใช้เพื่อแปลง FH เป็น BH เด็ดขาด — สะท้อนโฟร์แฮนด์ของคนถนัดขวาจะได้
    "โฟร์แฮนด์ของคนถนัดซ้าย" ไม่ใช่แบ็คแฮนด์ สองท่านี้ใช้กลไกร่างกายคนละแบบ
    (ข้างถนัด ฝ่ามือนำ vs ข้ามลำตัว หลังมือนำ) ไม่ใช่ภาพสะท้อนกัน
    ผู้เรียก **ต้องคง stroke_type เดิม** และสลับแค่ dominant_side
    """
    out = landmarks.copy()
    out[..., 0] = 1.0 - out[..., 0]
    for a, b in _LR_PAIRS:
        if a < out.shape[1] and b < out.shape[1]:
            tmp = out[:, a].copy()
            out[:, a] = out[:, b]
            out[:, b] = tmp
    return out


def body_height_frac(landmarks: np.ndarray) -> float:
    """สัดส่วนความสูงของคนในภาพ (จมูก->ข้อเท้า) — ใช้ตรวจว่า rescale สมเหตุผล"""
    ank = np.nanmean(landmarks[:, [L_ANKLE, R_ANKLE], 1], axis=1)
    sh = np.nanmean(landmarks[:, [L_SHOULDER, R_SHOULDER], 1], axis=1)
    d = np.abs(ank - sh)
    return float(np.nanmedian(d)) if not np.isnan(d).all() else 0.0
