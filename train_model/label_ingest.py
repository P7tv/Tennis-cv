"""แปลง client-labeled stroke JSON (เช่น dataset/sessions/set2/*.json) เป็น
โครงสร้างที่ build_training_data.py ใช้ต่อได้ — 1 ไฟล์ label ต่อ 1 คลิป
(อาจมีหลาย stroke), format ดู docs/CLIENT_LABEL_GUIDE.md / docs/LABEL_SPEC.md
(หมายเหตุ: ไฟล์เหล่านี้เป็น JSON ต่อคลิป ต่างจาก stroke_labels.csv ที่ spec
เดิมอธิบายไว้ — ใช้กับคลิป session ยาวที่มีหลาย stroke ในคลิปเดียว)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


# frame fields เท่าที่มีในไฟล์ label จริง — SV ไม่มี unit_turn/recovery_frame
# (ตรงกับ keyframe spec: "No unit_turn in serve")
KEYFRAME_FIELD_MAP = {
    "unit_turn": "unit_turn_frame",
    "backswing_peak": "backswing_peak_frame",
    "impact": "impact_frame",
    "follow_through_peak": "follow_through_peak_frame",
    "recovery_position": "recovery_frame",
}


@dataclass
class PlayerMeta:
    player_id: str
    name: str
    height: float | None
    dominant_side: str


@dataclass
class StrokeLabel:
    clip_id: str
    stroke_no: int
    player_id: str
    stroke_type: str
    fps: float
    usable: bool
    keyframes: dict[str, int | None]  # ชื่อ B-block -> frame index หรือ None


@dataclass
class SessionLabel:
    label_path: Path
    video_path: Path
    players: dict[str, PlayerMeta]
    strokes: list[StrokeLabel]


def _extract_keyframes(stroke: dict) -> dict[str, int | None]:
    return {
        name: stroke.get(field)
        for name, field in KEYFRAME_FIELD_MAP.items()
    }


def load_session_label(label_path: Path) -> SessionLabel:
    """อ่าน label json 1 ไฟล์ + resolve path วิดีโอ (คาดว่าอยู่โฟลเดอร์เดียวกัน)"""
    data = json.loads(label_path.read_text(encoding="utf-8"))

    players = {
        p["player_id"]: PlayerMeta(
            player_id=p["player_id"],
            name=p.get("name", p["player_id"]),
            height=p.get("height"),
            dominant_side=p.get("dominant_side", "right"),
        )
        for p in data.get("players", [])
    }

    strokes = []
    for s in data.get("strokes", []):
        if not s.get("usable", True):
            continue
        if s.get("impact_frame") is None:
            continue  # impact คือ keyframe บังคับ — ไม่มีก็ใช้ไม่ได้
        strokes.append(StrokeLabel(
            clip_id=s.get("clip_id", label_path.stem),
            stroke_no=s.get("stroke_no", 0),
            player_id=s.get("player_id"),
            stroke_type=s.get("stroke_type"),
            fps=s.get("fps", 30.0),
            usable=True,
            keyframes=_extract_keyframes(s),
        ))

    clip_path = data.get("strokes", [{}])[0].get("clip_path") if data.get("strokes") else None
    video_path = _resolve_video_path(label_path, clip_path, data.get("video_id"))

    return SessionLabel(
        label_path=label_path,
        video_path=video_path,
        players=players,
        strokes=strokes,
    )


def _resolve_video_path(label_path: Path, clip_path: str | None, video_id: str | None) -> Path:
    """หาไฟล์วิดีโอที่อยู่โฟลเดอร์เดียวกับ label json — ชื่อไฟล์จริงบนดิสก์
    อาจสะกด/extension ไม่ตรงกับ clip_path/video_id เป๊ะ (เช่น case, .MOV vs .mov)
    เลยลอง candidate หลายแบบ + fallback หา stem เดียวกัน"""
    folder = label_path.parent
    candidates = [c for c in (clip_path, video_id) if c]
    for name in candidates:
        p = folder / name
        if p.exists():
            return p
        for existing in folder.iterdir():
            if existing.is_file() and existing.name.lower() == name.lower():
                return existing

    stem_guess = label_path.stem.split("_stroke_labels_")[0]
    for existing in folder.iterdir():
        if existing.is_file() and existing.stem.lower() == stem_guess.lower() \
                and existing.suffix.lower() in (".mp4", ".mov", ".avi"):
            return existing

    # ชุด set3/set4 พบว่า clip_path ใน json บางไฟล์เขียน suffix ผิดจากไฟล์จริง
    # (เช่น json บอก "IMG_0300B6(SV1).mp4" แต่ไฟล์จริงชื่อ "IMG_0300B6-SV1.mp4",
    # บางไฟล์ยังสลับ stroke type ผิดไปเลยเช่น "(VL1)" ทั้งที่ไฟล์จริงคือ "(SV1)")
    # — stem_guess (เลข IMG_XXXX จากชื่อไฟล์ label เอง) ยังเชื่อถือได้กว่า ใช้
    # prefix-match แทน แต่ยอมใช้เฉพาะตอนแมตช์ได้ไฟล์เดียวชัดเจน กันจับผิดไฟล์เวลา
    # มีหลายคลิปขึ้นต้นเลขเดียวกันในโฟลเดอร์เดียวกัน
    prefix_matches = [
        existing for existing in folder.iterdir()
        if existing.is_file() and existing.suffix.lower() in (".mp4", ".mov", ".avi")
        and existing.stem.lower().startswith(stem_guess.lower())
    ]
    if len(prefix_matches) == 1:
        return prefix_matches[0]

    raise FileNotFoundError(
        f"หาไฟล์วิดีโอของ {label_path.name} ไม่เจอ (ลองแล้ว: {candidates}, stem={stem_guess})"
    )


def find_session_labels(sessions_dir: Path) -> list[Path]:
    """สแกนหา label json ทั้งหมดใต้ sessions_dir (recursive) — รองรับหลายชุด
    ในอนาคต (dataset/sessions/set2/, dataset/sessions/set3/, ...) โดยไม่ผูก
    กับชื่อโฟลเดอร์ใดโฟลเดอร์หนึ่ง"""
    return sorted(sessions_dir.glob("**/*_stroke_labels_*.json"))
