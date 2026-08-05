"""หาจังหวะ "เสียงกระทบ" จากแทร็กเสียงของคลิป

ทำไมถึงน่าสนใจ
──────────────
คลิปทุกไฟล์มีเสียงติดมาด้วย (AAC 44.1 kHz) ซึ่งมีความละเอียดเวลาดีกว่าภาพ
**1,470 เท่า** (44,100 ตัวอย่าง/วินาที เทียบกับ 30 เฟรม/วินาที) และเสียงลูก
กระทบไม้เป็น transient ที่คมมาก

ทดสอบเบื้องต้นแล้วพบสัญญาณจริง — AUC ของพลังงานที่เฟรมเฉลย เทียบกับเฟรมสุ่ม:

    1,000-3,000 Hz    0.811
    8,000-20,000 Hz   0.813
    ทั้งย่านรวมกัน       0.526   <- ต้องกรองความถี่ ไม่งั้นเสียงลม/เสียงคุยกลบหมด

(อ้างอิง: ฟีเจอร์ภาพที่ดีที่สุดของเราคือ sweep_total ได้ 0.832)

⚠️ สิ่งที่เสียงอย่างเดียวทำไม่ได้
────────────────────────────
  1. แยกไม่ออกว่าใครตี — ดริลที่มีคนป้อนลูกจะมีเสียงตีทั้งสองฝั่ง
  2. เสียงลูกกระทบพื้นก็เป็น transient เหมือนกัน
  3. เสียงเดินทาง 343 m/s ผู้เล่นห่างกล้อง 10-20 m -> เสียงมาช้ากว่าภาพ
     30-60 ms = 1-2 เฟรมที่ 30fps เป็นความคลาดเชิงระบบที่ต้องหักออก

จึงออกแบบให้ใช้ **ร่วมกับภาพ** ไม่ใช่แทนภาพ:
  ภาพบอกว่า "ประมาณตอนไหน" (ตอนนี้คลาด median 2 เฟรม) -> จำกัดหน้าต่างค้นหา
  เสียงบอกว่า "ตอนไหนเป๊ะ" ภายในหน้าต่างนั้น
การจำกัดหน้าต่างด้วยภาพแก้ปัญหาข้อ 1 และ 2 ไปในตัว
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# ย่านความถี่ที่ใช้ (Hz) — เลือกจากการวัด AUC ต่อย่านบนข้อมูลจริง
# เสียงลูกกระทบไม้/พื้นเป็น broadband transient ที่พลังงานเด่นในย่านกลาง-สูง
# ส่วนเสียงลม เสียงพูด และเสียงรถ อยู่ต่ำกว่า 1 kHz เป็นหลัก
ONSET_BAND_HZ = (1000.0, 16000.0)

# ความยาวหน้าต่าง STFT (วินาที) — สั้นเพื่อความละเอียดเวลา
# 512 ตัวอย่างที่ 44.1 kHz = 11.6 ms ซึ่งละเอียดกว่า 1 เฟรมวิดีโอ (33 ms) ราว 3 เท่า
STFT_WINDOW_S = 0.0116
STFT_HOP_S = 0.0029          # ก้าวละ ~2.9 ms (ซ้อนกัน 75%)

# หน้าต่างที่ใช้หาค่ากลางสำหรับ whitening (วินาที) — ต้องยาวกว่า transient มาก
# เพื่อให้ประมาณ "เสียงพื้นหลัง" ได้ ไม่ใช่ไปหักตัว transient เอง
WHITEN_WINDOW_S = 0.5

SPEED_OF_SOUND_M_S = 343.0


def extract_audio(video: Path, out_wav: Path, sr: int = 44100) -> bool:
    """ดึงแทร็กเสียงเป็น wav โมโนด้วย ffmpeg — คืน False ถ้าคลิปไม่มีเสียง"""
    import subprocess
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vn", "-ac", "1",
         "-ar", str(sr), "-f", "wav", str(out_wav)],
        capture_output=True, text=True)
    return r.returncode == 0 and out_wav.is_file() and out_wav.stat().st_size > 44


def onset_envelope(wav_path: Path, band: tuple[float, float] = ONSET_BAND_HZ):
    """คืน (เวลาเป็นวินาที, ค่าความ "สะดุด" ของเสียง) ต่อกรอบ STFT

    วิธี: spectral flux ที่ผ่านการ whitening
      1. STFT แล้วเอาเฉพาะย่านที่สนใจ
      2. หารแต่ละ bin ด้วยค่ากลางเคลื่อนที่ของตัวเอง = whitening
         ขั้นนี้สำคัญที่สุด — ทำให้ transient เบา ๆ ในคลิปที่เสียงดังโดยรวม
         ยังโผล่ขึ้นมาได้ และกันไม่ให้คลิปที่ลมแรงกลบทุกอย่าง
      3. ผลต่างทางบวกระหว่างกรอบ = "พลังงานที่เพิ่งเพิ่มขึ้น" = การกระทบ
    """
    from scipy.io import wavfile
    from scipy.signal import stft

    sr, x = wavfile.read(wav_path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float64)
    peak = np.abs(x).max()
    if peak > 0:
        x /= peak

    nper = max(64, int(round(STFT_WINDOW_S * sr)))
    hop = max(16, int(round(STFT_HOP_S * sr)))
    f, t, Z = stft(x, sr, nperseg=nper, noverlap=nper - hop, boundary=None)
    mag = np.abs(Z)

    sel = (f >= band[0]) & (f <= band[1])
    if not sel.any():
        return t, np.zeros(len(t))
    mag = mag[sel]

    # whitening ด้วยค่ากลางเคลื่อนที่ต่อ bin
    w = max(3, int(round(WHITEN_WINDOW_S / STFT_HOP_S)) | 1)
    pad = w // 2
    padded = np.pad(mag, ((0, 0), (pad, pad)), mode="edge")
    # ใช้ค่าเฉลี่ยเคลื่อนที่แทน median เพราะเร็วกว่ามากและผลต่างกันน้อยตรงนี้
    # (median กันค่าโดดได้ดีกว่า แต่ transient สั้นมากจนแทบไม่ดันค่าเฉลี่ย)
    kern = np.ones(w) / w
    base = np.apply_along_axis(lambda r: np.convolve(r, kern, mode="valid"),
                               1, padded)
    whitened = mag / (base + 1e-9)

    flux = np.diff(whitened, axis=1, prepend=whitened[:, :1])
    flux[flux < 0] = 0.0
    env = flux.sum(axis=0)
    return t, env


def sound_delay_frames(distance_m: float, fps: float) -> float:
    """เสียงมาช้ากว่าภาพกี่เฟรม เมื่อแหล่งเสียงอยู่ห่างกล้องเท่านี้

    ที่ 30fps: 10 m = 0.87 เฟรม · 20 m = 1.75 เฟรม — ไม่ใหญ่แต่ใหญ่พอที่จะ
    กินงบความคลาดทั้งหมดของเรา (ตอนนี้ median 2 เฟรม)
    """
    return distance_m / SPEED_OF_SOUND_M_S * fps


def audio_cache_path(video: Path, cache_dir: Path | None = None) -> Path:
    """ที่เก็บ wav ที่ดึงมา — แยกตามขนาดไฟล์ต้นทางกันหยิบของเก่ามาใช้"""
    base = cache_dir or Path(__file__).resolve().parent.parent / ".audio_cache"
    try:
        tag = video.stat().st_size
    except OSError:
        tag = 0
    return base / f"{video.stem}_{tag}.wav"


# ต้องมี event อย่างน้อยเท่านี้ถึงจะประมาณค่าชดเชยเวลาเดินทางของเสียงได้
# ค่ากลางจาก 1-2 ตัวไม่มีความหมาย และถ้าประมาณผิดจะเลื่อนทั้งคลิปผิดตาม
MIN_EVENTS_FOR_OFFSET = 3

# เลื่อนได้ไกลสุดกี่เฟรม — เท่ากับครึ่งความกว้างหน้าต่างที่ให้ค้นหา
DEFAULT_WINDOW_FRAMES = 6.0


def refine_hits_with_audio(events: list[dict], video_path, fps: float,
                           window_frames: float = DEFAULT_WINDOW_FRAMES,
                           cache_dir: Path | None = None,
                           total_frames: int | None = None) -> list[dict]:
    """ปรับเฟรมปะทะของ event ที่เลือกแล้ว ด้วยเสียงกระทบ

    วัดผลจริง (131 ลูกที่ระบบตรวจเจอ · หน้าต่างมาจากคำทำนาย ไม่ใช่จากเฉลย):

                        |Δ| median   ภายใน 1   ภายใน 2   ภายใน 4
      ภาพอย่างเดียว          2.0       45.0%     56.5%     78.6%
      ภาพ + เสียง            1.0       61.8%     76.3%     85.5%

    🔴 การชดเชยเวลาเดินทางของเสียงคำนวณจาก **คำทำนายของภาพ** ไม่ใช่จากเฉลย:
    ค่าชดเชยของคลิป = ค่ากลางของ (เฟรมที่เสียงเลือก - เฟรมที่ภาพทำนาย)
    ใช้ได้เพราะความคลาดของภาพไม่มีทิศทางชัด (median +0.5 เฟรม) ส่วนความช้าของ
    เสียงเป็นค่าคงที่ทิศเดียว (วัดได้ +0 ถึง +3 เฟรม และ **เป็นบวกทั้ง 16 คลิป**
    ตรงกับที่ฟิสิกส์ทำนายจากระยะ 10-30 m) จึงแยกสองอย่างนี้ออกจากกันได้

    ไม่มี ffmpeg / คลิปไม่มีเสียง / event น้อยเกินไป -> คืนของเดิม ไม่ throw
    """
    if not events or not video_path or fps <= 0:
        return events
    if len(events) < MIN_EVENTS_FOR_OFFSET:
        return events

    video = Path(video_path)
    if not video.is_file():
        # ⚠️ ต้องเตือนดัง ๆ ไม่ใช่คืนเงียบ ๆ — เคสที่เจอจริงคือผู้เรียกส่ง
        # relative path มาหลังย้าย CWD ทำให้ benchmark ออกมาเหมือนเดิมทุกหลัก
        # โดยไม่มีสัญญาณอะไรบอกว่าการปรับด้วยเสียงไม่ได้ทำงานเลย
        print(f"⚠️ ปรับด้วยเสียงไม่ได้ — ไม่พบไฟล์วิดีโอ: {video}")
        return events
    wav = audio_cache_path(video, cache_dir)
    try:
        if not wav.is_file() and not extract_audio(video, wav):
            return events
        times, env = onset_envelope(wav)
    except Exception as e:
        print(f"ข้ามการปรับด้วยเสียง ({video.name}): {e}")
        return events
    if len(times) == 0:
        return events

    picks = []
    for e in events:
        f, _ = best_onset_near(times, env, e["frame"], fps, window_frames)
        picks.append(f)
    have = [(p - e["frame"]) for p, e in zip(picks, events) if p is not None]
    if len(have) < MIN_EVENTS_FOR_OFFSET:
        return events

    offset = float(np.median(have))
    # 🔴 ต้องจำกัดเฟรมปลายทางให้อยู่ในคลิปด้วย ไม่ใช่จำกัดแค่ระยะเลื่อน —
    # แทร็กเสียงมักยาวกว่าวิดีโอเล็กน้อย เฟรมที่เสียงชี้จึงหลุดท้ายคลิปได้
    # เจอจริง: builder พังทั้งคลิปด้วย "index 72 is out of bounds for size 72"
    # แล้ว GT ทั้งคลิปถูกนับเป็น FN
    last = (total_frames - 1) if total_frames else None
    for e, p in zip(events, picks):
        if p is None:
            continue
        f = int(round(p - offset))
        if last is not None:
            f = max(0, min(f, last))
        else:
            f = max(0, f)
        if f != e["frame"] and abs(f - e["frame"]) <= window_frames:
            e["frame_before_audio"] = e["frame"]
            e["frame"] = f
            e["timestamp_sec"] = round(f / fps, 2)
    return sorted(events, key=lambda x: x["frame"])


def best_onset_near(times: np.ndarray, env: np.ndarray, frame: int, fps: float,
                    window_frames: float) -> tuple[int | None, float]:
    """หาจุดที่เสียง "สะดุด" แรงที่สุดในหน้าต่างรอบเฟรมที่กำหนด

    คืน (เฟรมของวิดีโอที่ตรงกับจุดนั้น, ความแรง) — None ถ้าไม่มีข้อมูลในช่วง
    """
    if len(times) == 0:
        return None, 0.0
    lo = (frame - window_frames) / fps
    hi = (frame + window_frames) / fps
    m = (times >= lo) & (times <= hi)
    if not m.any():
        return None, 0.0
    idx = np.flatnonzero(m)
    j = idx[int(np.argmax(env[idx]))]
    return int(round(times[j] * fps)), float(env[j])
