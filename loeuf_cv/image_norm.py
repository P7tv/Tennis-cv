"""Image normalization — ลดความต่างระหว่างมือถือแต่ละรุ่น/เลนส์

รันต่อเฟรมก่อนเข้า pose model (และ ball detector ในอนาคต):
1. Gray-world white balance — สีเพี้ยนต่างยี่ห้อ → โทนกลางเดียวกัน
   (สำคัญกับ ball detector ที่พึ่งสีลูก มากกว่า pose)
2. Exposure normalization — ดึง mean luma เข้า target เดียวกัน
3. CLAHE — เฉพาะภาพมืด (luma < threshold) เพิ่ม local contrast

หมายเหตุ: VQ stats (luma/blur) ต้องวัดจากเฟรม "ดิบ" ก่อน normalize
เพื่อรายงานคุณภาพคลิปตามจริง — pose_extractor จัดลำดับให้แล้ว

สิ่งที่จงใจไม่ทำ: per-device lens undistortion (ต้องมี calibration
target ต่อรุ่น — ไม่ realistic กับ user ทั่วไป) → ใช้ guideline
"ผู้เล่นอยู่กลางเฟรม + เลนส์หลัก 1x" + VQ edge-guard แทน
"""

import cv2
import numpy as np

TARGET_LUMA = 110.0
CLAHE_LUMA_THRESHOLD = 60.0
MAX_CHANNEL_GAIN = 2.0


def gray_world_white_balance(frame_bgr: np.ndarray) -> np.ndarray:
    """สมมติฉากโดยรวมเป็นสีกลาง → scale แต่ละ channel ให้ mean เท่ากัน"""
    means = frame_bgr.reshape(-1, 3).mean(axis=0)
    gray = float(means.mean())
    if gray < 1.0:
        return frame_bgr
    gains = np.clip(gray / np.maximum(means, 1.0), 1.0 / MAX_CHANNEL_GAIN,
                    MAX_CHANNEL_GAIN)
    return cv2.convertScaleAbs(frame_bgr * gains[None, None, :])


def normalize_exposure(frame_bgr: np.ndarray,
                       target_luma: float = TARGET_LUMA) -> np.ndarray:
    luma = float(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).mean())
    if luma < 1.0:
        return frame_bgr
    gain = np.clip(target_luma / luma, 0.5, 2.5)
    if abs(gain - 1.0) < 0.1:
        return frame_bgr
    return cv2.convertScaleAbs(frame_bgr, alpha=gain)


def apply_clahe(frame_bgr: np.ndarray) -> np.ndarray:
    """เพิ่ม local contrast บน L channel (ใช้เฉพาะภาพมืด)"""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def normalize_frame(frame_bgr: np.ndarray, raw_luma: float) -> np.ndarray:
    """Preprocessing มาตรฐานต่อเฟรม (raw_luma = luma ก่อน normalize)"""
    out = gray_world_white_balance(frame_bgr)
    out = normalize_exposure(out)
    if raw_luma < CLAHE_LUMA_THRESHOLD:
        out = apply_clahe(out)
    return out
