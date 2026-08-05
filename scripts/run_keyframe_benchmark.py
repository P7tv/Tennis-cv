#!/usr/bin/env python3
"""วัด keyframe accuracy ของ pipeline production เทียบ ground truth ของลูกค้า

เกณฑ์รับงาน: acceptance accuracy (impact + backswing_peak) >= 0.80 ที่ ±1 เฟรม

ต่างจาก scripts/run_benchmark.py เดิม (ซึ่งรองรับแค่คลิป single-stroke +
stroke_labels.csv + StrokePipeline) — ตัวนี้วัด build_loeuf_schema() ที่เป็น
path ที่ส่งลูกค้าจริง บนคลิป session ที่มีหลาย stroke ต่อคลิป

3 stage แยกกันเพราะ tracking (YOLO+MediaPipe) คือส่วนที่แพงที่สุด
รันครั้งเดียวต่อคลิปแล้ว cache ไว้ ป้อนทั้ง 2 โหมด:

    track    วิดีโอ → tracks/ball/racket/ball_traj  (แพง, ~2 ชม. สำหรับ 13 คลิป)
    predict  mode A: detect_hit_events() → build_loeuf_schema()
             mode B: hit_events จาก GT impact → build_loeuf_schema()  (oracle)
    score    1:1 matching + keyframe_accuracy + เขียนรายงาน

Mode A = จริงตามที่ลูกค้าได้ (รวม error ของ hit detection)
Mode B = เพดานของ keyframe logic ล้วน ๆ (ตัด error ของ hit detection ออก)
ส่วนต่าง A-B บอกว่าควรไปแก้ hit detection หรือ keyframe logic ก่อน

ตัวอย่าง:
    python scripts/run_keyframe_benchmark.py track --only IMG_0284
    python scripts/run_keyframe_benchmark.py predict --mode both --only IMG_0284
    python scripts/run_keyframe_benchmark.py score
    python scripts/run_keyframe_benchmark.py all --limit 3
"""

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows console บาง terminal default เป็น cp1252 ไม่รองรับตัวอักษรไทย/emoji
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2

from loeuf_cv.benchmark_keyframes import (
    BENCH_KEYFRAME_COLS, MATCH_TOLERANCE_FRAMES, NOT_DETECTED_PRED,
    detection_metrics, label_rows_from_session, match_strokes,
    pred_strokes_from_schema, render_keyframe_markdown, score_mode,
    score_phase_mode, sensitivity_without_anomalies,
)
from loeuf_cv import hit_detection
from loeuf_cv.config import PipelineConfig
from loeuf_cv.hit_detection import detect_hit_events, extract_ball_trajectory_kalman
from loeuf_cv.schema_builder.builder import build_loeuf_schema
from train_model.label_ingest import find_session_labels, load_session_label
from webui.yolo_track import track_players_with_yolo

ROOT = Path(__file__).resolve().parent.parent
# v2: เก็บ ball_bboxes ดิบไว้ด้วย เพื่อให้ทดลองอัลกอริทึม ball tracking
#     (เช่น filter_static_ball_bboxes) ได้โดยไม่ต้องรัน YOLO ใหม่ 112 นาที
CACHE_VERSION = 2
DEFAULT_MODEL = "runs/pose/tennis_ball_racket_pose_finetune_best.pt"

# ข้อจำกัดเชิงโค้ดที่ทราบล่วงหน้า — ใส่ในรายงานเพื่อไม่ให้ตัวเลขต่ำถูกอ่านผิด
KNOWN_QUIRKS = [
    "🔴 **`backswing_peak` (B2) เป็นค่าประมาณจากจังหวะ impact ไม่ใช่ผลตรวจจับ"
    "การเคลื่อนไหว** — schema_builder/keyframes.py ใช้ offset คงที่ (groundstroke"
    " `impact-3`, slice `impact-5`) ส่วนเสิร์ฟหาเฟรมที่ข้อมือสูงสุดในช่วง"
    " `impact-35..-9` เหตุผล: GT ของลูกค้าวาง B2 ห่าง impact แค่ median 3 เฟรม"
    " (0 จาก 78 groundstroke ที่ห่าง ≥20 เฟรม) ไม่ตรงกับนิยาม 'จุดถอยหลังสุด'"
    " เชิงพื้นที่ — ทดลองนิยามเชิงจลนศาสตร์แล้วแพ้ค่าคงที่ทุกตัว (velocity"
    " reversal 0.144 · argmax|x−x_imp| 1.5s ของเดิม 0.053 vs ค่าคงที่ ~0.49)"
    " → **ต้องยืนยันนิยาม B2 กับลูกค้า** ถ้าต้องการจุดถอยหลังสุดจริงต้อง re-label",
    "✅ กิ่ง slice ของ B2 (`impact-5`) **พิสูจน์ข้ามคนได้แล้ว** (2026-07-30) —"
    " ชุด label เพิ่ม IMG_0266-SL ทำให้ SL มาจาก 2 คน (earth 11 + prem 10)"
    " เดิมมาจากคนเดียวจึง Leave-One-Person-Out ให้ 0.00 โดยโครงสร้าง ไม่ใช่เพราะ"
    " โมเดลแย่ — ตอนนี้วัดได้จริง Mode B backswing SL = 0.571",
    "`unit_turn` (B1) **ไม่ได้ detect** — ตั้งเป็นจุดกึ่งกลางคงที่"
    " `backswing_peak - 0.5s` (คอมเมนต์ `# Approximation` ในโค้ด) ระยะ 15 เฟรมนี้"
    " ตรงกับ GT median ของ `backswing_peak - unit_turn` พอดี (n=59) → ความแม่นของ"
    " B1 ผูกกับความแม่นของ B2 ทั้งหมด",
    "`recovery_position` (B5) ใช้เกณฑ์ `vel_mag < 0.5` บนพิกัด normalized 0-1"
    " (schema_builder/keyframes.py) = ครึ่งความกว้างภาพต่อเฟรม ซึ่งเข้าเงื่อนไข"
    " แทบทุกครั้งที่เฟรมแรกหลัง follow-through → accuracy 0.000 ทุกท่า (median"
    " พลาด 17-35 เฟรม) ด้วยเหตุผลนี้ ไม่ใช่เพราะ tracking — **ยังไม่ได้แก้**",
    "`trophy_position` (B6) ออกเฉพาะเมื่อ classifier ตอบ SV"
    " (schema_builder/builder.py) → accuracy ผูกกับ stroke classifier ด้วย"
    " ไม่ใช่ความแม่นของ keyframe เพียว ๆ — ตัวเลข classifier ที่เชื่อถือได้คือ"
    " **Leave-One-Person-Out 0.727** ไม่ใช่ leave-one-clip-out 0.803 (คนเดียวกัน"
    " อยู่หลายคลิป: earth 4 คลิป, poom 3 คลิป → group ด้วยคลิปยังมี leakage"
    " ระดับบุคคล)",
    "BL2/BL3 (ball speed, trajectory clearance) ยังเป็น null เสมอ เพราะไม่มี"
    " court calibration — ไม่กระทบตัวเลข keyframe แต่เป็นข้อจำกัดของ output รวม",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _set_name(label_path: Path) -> str:
    """set2 / set3 / set4 / sessions_deferred/set4"""
    parent = label_path.parent
    if parent.parent.name == "sessions_deferred":
        return f"sessions_deferred/{parent.name}"
    return parent.name


def _pick_target_track(tracks):
    """ผู้เล่นหลัก = track ที่มี pose coverage สูงสุด (ลอกจาก
    train_model/build_training_data.py:71 — ไม่ import เพราะไฟล์นั้น mutate
    sys.path + import ของหนักตอน module scope)"""
    def coverage(t):
        vis = t.pose.visibility
        return float((vis.mean(axis=1) > 0).sum())
    return max(tracks, key=coverage)


@contextmanager
def scratch_cwd():
    """detect_hit_events() append `hit_candidates.csv` ลง CWD เสมอ
    (loeuf_cv/hit_detection.py:567) — ย้าย CWD ไป temp กันไฟล์โผล่ในรีโป

    ⚠️ os.chdir เป็น process-global → ห้ามรัน stage นี้แบบ parallel
    """
    old = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="kf_bench_scratch_")
    try:
        os.chdir(tmp)
        yield
    finally:
        os.chdir(old)
        shutil.rmtree(tmp, ignore_errors=True)


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=ROOT, capture_output=True, text=True,
                              timeout=10).stdout.strip() or "?"
    except Exception:
        return "?"


def _slim_schema(schema: dict) -> dict:
    """เก็บแค่ที่ scoring ต้องใช้ — ตัด visualization (path ต่อเฟรม) กับ metric
    ออก ไม่งั้น cache ของคลิป 10.8 นาทีบวมมาก"""
    return {
        "session_metadata": schema.get("session_metadata", {}),
        "strokes": [{
            "stroke_root": s.get("stroke_root", {}),
            "keyframe": s.get("keyframe", {}),
            "debug_pose_confidence":
                s.get("stroke_metadata", {}).get("debug_pose_confidence"),
        } for s in schema.get("strokes", [])],
    }


class Manifest:
    """สถานะต่อ (clip, stage) — เขียนทุกคลิปเพื่อให้ Ctrl-C เสียแค่คลิปเดียว"""

    def __init__(self, path: Path):
        self.path = path
        self.data = {}
        if path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                print(f"⚠️ manifest เสียหาย เริ่มใหม่: {path}")

    def entry(self, clip: str, stage: str) -> dict:
        return self.data.get(clip, {}).get(stage, {})

    def is_done(self, clip: str, stage: str, video: Path) -> bool:
        e = self.entry(clip, stage)
        if e.get("status") != "ok" or e.get("cache_version") != CACHE_VERSION:
            return False
        try:
            st = video.stat()
        except OSError:
            return False
        return (e.get("video_size") == st.st_size
                and abs(e.get("video_mtime", 0) - st.st_mtime) < 1)

    def record(self, clip: str, stage: str, video: Path, **extra):
        try:
            st = video.stat()
            vsize, vmtime = st.st_size, st.st_mtime
        except OSError:
            vsize, vmtime = None, None
        self.data.setdefault(clip, {})[stage] = {
            "cache_version": CACHE_VERSION,
            "video_size": vsize, "video_mtime": vmtime,
            "updated": datetime.now().isoformat(timespec="seconds"),
            **extra,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(self.path)


def _collect_sessions(dataset_root: Path, only: str | None, limit: int | None):
    """คืน [(label_path, set_name, session)] เรียงคลิปสั้นก่อน เพื่อให้ pilot
    เจอผลเร็วและคลิปยาว (IMG_0294 10.8 นาที) ไปอยู่ท้ายสุด"""
    # ต้อง glob จาก dataset/ ไม่ใช่ dataset/sessions/ ไม่งั้นได้ 12 ไม่ใช่ 13
    out = []
    for p in find_session_labels(dataset_root):
        if only and only.lower() not in p.stem.lower():
            continue
        try:
            session = load_session_label(p)
        except FileNotFoundError as e:
            print(f"  ⚠️ ข้าม {p.name}: {e}")
            continue
        out.append((p, _set_name(p), session))

    def size(item):
        try:
            return item[2].video_path.stat().st_size
        except OSError:
            return 0
    out.sort(key=size)
    return out[:limit] if limit else out


# ---------------------------------------------------------------------------
# stage 1: track
# ---------------------------------------------------------------------------

def stage_track(args, sessions, manifest: Manifest):
    cache = Path(args.cache_dir) / "tracking"
    print(f"\n=== stage: track ({len(sessions)} คลิป) ===")

    for i, (label_path, set_name, session) in enumerate(sessions, 1):
        clip = f"{set_name}/{session.video_path.stem}"
        out = cache / f"{set_name}/{session.video_path.stem}.pkl"

        if not args.force and manifest.is_done(clip, "track", session.video_path) \
                and out.exists():
            print(f"[{i}/{len(sessions)}] {clip} — cached ข้าม")
            continue

        print(f"[{i}/{len(sessions)}] {clip} — tracking...", flush=True)
        t0 = time.time()
        try:
            cap = cv2.VideoCapture(str(session.video_path))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or session.strokes[0].fps
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            player = next(iter(session.players.values()))
            config = PipelineConfig(dominant_side=player.dominant_side,
                                    subject_height_cm=player.height)

            tracks, ball_bboxes, racket_bboxes, racket_keypoints = \
                track_players_with_yolo(
                    str(session.video_path), max_players=args.max_players,
                    config=config, model_path=args.model,
                    base_model=args.base_model)
            if not tracks:
                raise RuntimeError("ไม่เจอผู้เล่นในคลิปนี้")

            track = _pick_target_track(tracks)
            ball_traj, ball_measured = extract_ball_trajectory_kalman(
                ball_bboxes, len(track.pose.landmarks), return_measured=True,
                fps=fps)

            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "wb") as f:
                pickle.dump({
                    "cache_version": CACHE_VERSION,
                    "track": track, "n_tracks": len(tracks),
                    "racket_bboxes": racket_bboxes,
                    "racket_keypoints": racket_keypoints,
                    "ball_bboxes": ball_bboxes,
                    "ball_traj": ball_traj,
                    "ball_measured": ball_measured,
                    "fps": fps, "duration_sec": n_frames / fps if fps else 0.0,
                    "video_meta": {"width": width, "height": height,
                                   "total_frames": n_frames},
                    "dominant_side": player.dominant_side,
                    "subject_height_cm": player.height,
                }, f)

            dur = round(time.time() - t0, 1)
            manifest.record(clip, "track", session.video_path, status="ok",
                            duration_s=dur, n_tracks=len(tracks),
                            n_pose_frames=len(track.pose.landmarks))
            print(f"     ok ({dur}s, {len(tracks)} track, "
                  f"{len(track.pose.landmarks)} เฟรม)")
        except Exception as e:
            manifest.record(clip, "track", session.video_path, status="error",
                            error=str(e), traceback=traceback.format_exc(),
                            duration_s=round(time.time() - t0, 1))
            print(f"     ERROR: {e} — ข้ามคลิปนี้")


# ---------------------------------------------------------------------------
# stage 2: predict
# ---------------------------------------------------------------------------

_LOPO_MAP_CACHE: dict | None = None


def _lookup_person(mapping: dict, clip: str) -> str | None:
    """หา player_id ของคลิป โดยทนกับ "ชื่อชุด" ที่เขียนคนละแบบ

    train_hit_classifier_from_cache.py สร้างคีย์จาก `pkl.parent.name` = ชื่อ
    โฟลเดอร์ชั้นเดียว ("set4/IMG_0294(SV1)") ส่วน benchmark ใช้ path เต็มจาก
    dataset root ("sessions_deferred/set4/IMG_0294(SV1)") — คลิปเดียวกันแต่
    คีย์ไม่ตรง ถ้าไม่รองรับจะ error เฉพาะคลิปที่อยู่ลึกกว่าหนึ่งชั้น
    """
    if clip in mapping:
        return mapping[clip]
    tail = "/".join(Path(clip).parts[-2:])          # <โฟลเดอร์แม่>/<ชื่อคลิป>
    if tail in mapping:
        return mapping[tail]
    stem = Path(clip).name
    hits = {v for k, v in mapping.items() if Path(k).name == stem}
    return hits.pop() if len(hits) == 1 else None


LOPO_LABEL = "LOPO (โมเดลไม่เคยเห็นคนในคลิปที่วัด)"
INSAMPLE_LABEL = "in-sample (โมเดลเทรนจากคลิปเดียวกับที่วัด)"


def _eval_mode_label(args) -> str:
    return LOPO_LABEL if getattr(args, "hit_classifier_dir", None) else INSAMPLE_LABEL


def _eval_mode_from_manifest(manifest, args) -> str:
    """โหมดที่ใช้ตอน 'ทำนาย' จริง — อ่านจาก manifest ก่อนเสมอ

    🔴 ทำไมไม่ดูจาก args ตรง ๆ: การเลือกโมเดล LOPO เกิดตอน stage predict แต่
    รายงานถูกสร้างตอน stage score ซึ่งเป็นคนละคำสั่งกัน ถ้าอ่านจาก args ของ
    score รายงานจะติดป้ายตามแฟล็กที่พิมพ์ตอน score ไม่ใช่ตามสิ่งที่เกิดขึ้นจริง
    -> รัน predict --hit-classifier-dir แล้ว score เปล่า ๆ จะได้รายงานที่เขียนว่า
    "in-sample" ทั้งที่วัดแบบ LOPO (และกลับกันซึ่งอันตรายกว่า คืออ้าง LOPO
    ทั้งที่ปนเปื้อน) เจอจริงตอนรันชุด swing features

    manifest บันทึกไว้ตอน predict จึงเป็นแหล่งความจริง ส่วน args ใช้เป็นตัวสำรอง
    สำหรับ manifest เก่าที่ยังไม่มีคีย์นี้
    """
    seen = {e.get("hit_model_eval")
            for clip in manifest.data.values()
            for stage, e in clip.items()
            if stage == "predict_a" and e.get("status") == "ok"}
    seen.discard(None)
    if len(seen) == 1:
        return seen.pop()
    if len(seen) > 1:
        return "ปนกันหลายโหมด ⚠️ ให้รัน predict --force ใหม่ทั้งชุด"
    return _eval_mode_label(args) + " (เดาจากแฟล็ก — manifest ไม่ได้บันทึกไว้)"


def _use_lopo_model(args, clip: str) -> None:
    """สลับ hit classifier ให้เป็นตัวที่ **ไม่เคยเห็นคนในคลิปนี้**

    ⚠️ ทำไมต้องมี: hit_classifier.pkl ตัวจริงเทรนจาก 16 คลิปเดียวกับที่ใช้วัด
    benchmark → ตัวเลขที่ได้มีส่วนที่มาจาก "การจำคลิป" ไม่ใช่ความสามารถจริง
    วัดแล้ว: โมเดลเดิม LOPO recall 54.7% แต่ benchmark 86.6% (ช่องว่าง 31.9 pp)
    ส่วนโมเดลที่ผ่าน augmentation 68.0% vs 80.8% (12.8 pp) — เอาสองตัวนี้มา
    เทียบกันบน benchmark ที่ปนเปื้อน ตัวที่ "จำเก่งกว่า" จะชนะเสมอ ซึ่งตรงข้าม
    กับสิ่งที่เราต้องการ

    ไม่แตะโค้ด production: ใช้ตัวแปรสภาพแวดล้อม LOEUF_HIT_CLASSIFIER ที่
    _load_hit_classifier() รองรับอยู่แล้ว + ล้าง cache ระดับโมดูลก่อนทุกคลิป
    (ไม่งั้นคลิปที่ 2 เป็นต้นไปจะยังใช้โมเดลของคลิปแรก)
    """
    global _LOPO_MAP_CACHE
    d = getattr(args, "hit_classifier_dir", None)
    if not d:
        return
    d = Path(d)
    if _LOPO_MAP_CACHE is None:
        with open(d / "clip_person.json", encoding="utf-8") as f:
            _LOPO_MAP_CACHE = json.load(f)

    person = _lookup_person(_LOPO_MAP_CACHE, clip)
    if person is None:
        raise RuntimeError(
            f"ไม่รู้ว่า {clip} เป็นของใคร — clip_person.json ไม่มีคีย์นี้ "
            f"(สร้างใหม่ด้วย train_hit_classifier_from_cache.py --lopo-out)")
    model = d / f"{person}.pkl"
    if not model.is_file():
        raise RuntimeError(f"ไม่พบโมเดล LOPO ของ {person} ที่ {model}")

    os.environ["LOEUF_HIT_CLASSIFIER"] = str(model)
    hit_detection._hit_clf_cache = None

    # ตัวปรับเฟรมปะทะต้องสลับตามคนเดียวกัน ไม่งั้นตัวเลขปนเปื้อนที่ขั้นนี้แทน
    rd = getattr(args, "impact_refiner_dir", None)
    if rd:
        rm = Path(rd) / f"{person}.pkl"
        if not rm.is_file():
            raise RuntimeError(f"ไม่พบ refiner LOPO ของ {person} ที่ {rm}")
        os.environ["LOEUF_IMPACT_REFINER"] = str(rm)
    else:
        # ไม่ระบุ = ปิดการปรับไปเลย ดีกว่าเผลอใช้ตัวที่เทรนจากทุกคน
        os.environ["LOEUF_IMPACT_REFINER"] = ""
    hit_detection._impact_refiner_cache = None
    print(f"    [LOPO] {clip} -> โมเดลที่ไม่เคยเห็น {person}"
          f"{' (+refiner)' if rd else ''}")


def _build_for_mode(cached: dict, session, mode: str, ml_threshold=None):
    """คืน (schema, hit_events) — โหมด A ให้ pipeline หา impact เอง,
    โหมด B ป้อน GT impact เข้าไป (oracle)

    ทั้ง 2 โหมดเรียก build_loeuf_schema ตัวเดียวกัน → รูป output/กฎ trophy
    เหมือนกันเป๊ะ ส่วนต่าง A-B จึงเป็นผลของ hit detection ล้วน ๆ
    """
    track = cached["track"]
    fps = cached["fps"]
    vm = cached["video_meta"]
    config = PipelineConfig(dominant_side=cached["dominant_side"],
                            subject_height_cm=cached["subject_height_cm"])

    # คำนวณ ball trajectory ใหม่จาก bbox ดิบทุกครั้ง ไม่ใช้ cached["ball_traj"]
    # — นี่คือเหตุผลที่ cache v2 เก็บ ball_bboxes ไว้: เปลี่ยนอัลกอริทึม ball
    # tracking แล้ววัดผลได้ทันทีโดยไม่ต้องรัน YOLO ใหม่ 2 ชั่วโมง
    # (cached["ball_traj"] เก็บไว้เพื่อ backward-compat กับ cache v1 เท่านั้น)
    if cached.get("ball_bboxes") is not None:
        ball_traj, ball_measured = extract_ball_trajectory_kalman(
            cached["ball_bboxes"], len(track.pose.landmarks),
            return_measured=True, fps=fps)
    else:
        ball_traj, ball_measured = cached["ball_traj"], None

    if mode == "a":
        with scratch_cwd():
            hit_events = detect_hit_events(
                ball_traj, [track], fps, vm["width"], vm["height"],
                racket_bboxes=cached["racket_bboxes"],
                ball_measured=ball_measured,
                **({"ml_prob_threshold": ml_threshold}
                   if ml_threshold is not None else {}))
    else:
        gt = sorted((s for s in session.strokes
                     if s.keyframes.get("impact") is not None),
                    key=lambda s: s.keyframes["impact"])
        # builder อ่านแค่ frame / player_id / bounce_court_* (optional)
        hit_events = [{"frame": int(s.keyframes["impact"]),
                       "player_id": track.track_id, "confidence": 1.0}
                      for s in gt]

    schema = build_loeuf_schema(
        [track], hit_events, fps, vm, config,
        racket_keypoints=cached["racket_keypoints"],
        ball_traj=ball_traj)
    return schema, hit_events


def stage_predict(args, sessions, manifest: Manifest):
    modes = ["a", "b"] if args.mode == "both" else [args.mode]
    tracking_dir = Path(args.cache_dir) / "tracking"
    print(f"\n=== stage: predict (mode {'+'.join(modes)}) ===")

    for i, (label_path, set_name, session) in enumerate(sessions, 1):
        clip = f"{set_name}/{session.video_path.stem}"
        pkl = tracking_dir / f"{set_name}/{session.video_path.stem}.pkl"
        if not pkl.exists():
            print(f"[{i}/{len(sessions)}] {clip} — ยังไม่มี tracking cache ข้าม")
            continue

        with open(pkl, "rb") as f:
            cached = pickle.load(f)

        _use_lopo_model(args, clip)

        for mode in modes:
            stage = f"predict_{mode}"
            out = Path(args.cache_dir) / f"preds/mode_{mode}/{set_name}/{session.video_path.stem}.json"
            if not args.force and manifest.is_done(clip, stage, session.video_path) \
                    and out.exists():
                print(f"[{i}/{len(sessions)}] {clip} mode {mode} — cached ข้าม")
                continue

            t0 = time.time()
            try:
                schema, hit_events = _build_for_mode(cached, session, mode,
                                                    ml_threshold=getattr(args, "ml_threshold", None))
                slim = _slim_schema(schema)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(slim, ensure_ascii=False, indent=1),
                               encoding="utf-8")

                n_gt = len(session.strokes)
                n_pred = len(slim["strokes"])
                # mode B ป้อน GT impact เข้าไปตรง ๆ → จำนวน stroke ที่ออกมาต้อง
                # เท่า GT เป๊ะ ถ้าไม่เท่าคือ builder ทิ้ง stroke เงียบ ๆ
                # (`if not track: continue`) = บั๊ก ไม่ใช่ผลของโมเดล
                mismatch = (mode == "b" and n_pred != n_gt)
                warn = (f"  ⚠️ mode B ได้ {n_pred} stroke แต่ GT มี {n_gt}"
                        if mismatch else "")
                manifest.record(clip, stage, session.video_path, status="ok",
                                n_pred=n_pred, n_gt=n_gt,
                                mode_b_count_mismatch=mismatch,
                                n_hit_events=len(hit_events),
                                # บันทึกไว้ให้ stage score อ่าน — ไม่งั้นรายงาน
                                # จะติดป้ายตามแฟล็กของคำสั่ง score แทนที่จะเป็น
                                # สิ่งที่เกิดขึ้นจริงตอนทำนาย
                                hit_model_eval=_eval_mode_label(args),
                                duration_s=round(time.time() - t0, 1))
                print(f"[{i}/{len(sessions)}] {clip} mode {mode} — "
                      f"{n_pred} stroke (GT {n_gt}){warn}")
            except Exception as e:
                manifest.record(clip, stage, session.video_path, status="error",
                                error=str(e), traceback=traceback.format_exc())
                print(f"[{i}/{len(sessions)}] {clip} mode {mode} — ERROR: {e}")


# ---------------------------------------------------------------------------
# stage 3: score
# ---------------------------------------------------------------------------

def _load_preds(cache_dir: Path, mode: str, set_name: str, stem: str):
    p = cache_dir / f"preds/mode_{mode}/{set_name}/{stem}.json"
    if not p.exists():
        return None
    return pred_strokes_from_schema(json.loads(p.read_text(encoding="utf-8")))


def stage_score(args, sessions, manifest: Manifest):
    cache_dir = Path(args.cache_dir)
    print(f"\n=== stage: score ({len(sessions)} คลิป) ===")

    all_rows, per_clip = [], []
    preds_a, preds_a_matched, preds_b = {}, {}, {}
    rows_matched = []
    n_anomalies = 0
    # คลิปที่ track/predict ล้ม — GT ของมันจะกลายเป็น FN ทั้งก้อน ต้องโชว์ให้เห็น
    # ไม่ใช่ปล่อยให้ดูเหมือนโมเดลตรวจไม่เจอ
    errored_clips = []

    for label_path, set_name, session in sessions:
        clip_id = f"{set_name}/{session.video_path.stem}"
        for stage in ("track", "predict_a", "predict_b"):
            e = manifest.entry(clip_id, stage)
            if e.get("status") == "error":
                errored_clips.append({"clip": clip_id, "stage": stage,
                                      "error": e.get("error")})

        raw = json.loads(label_path.read_text(encoding="utf-8")).get("strokes", [])
        rows = label_rows_from_session(session, set_name, raw_strokes=raw)
        all_rows.extend(rows)
        n_anomalies += sum(1 for r in rows if r["anomalies"])

        stem = session.video_path.stem
        gt_impacts = [r["impact"] for r in rows]

        # ---- Mode A: 1:1 matching ----
        pa = _load_preds(cache_dir, "a", set_name, stem)
        if pa is None:
            # ไม่มี prediction เลย (tracking/predict ล้ม) → GT ทุกตัวเป็น FN
            for r in rows:
                preds_a[r["clip_path"]] = NOT_DETECTED_PRED
            per_clip.append({"clip": f"{set_name}/{stem}", "n_gt": len(rows),
                             "n_pred": 0, "tp": 0, "fn": len(rows), "fp": 0,
                             "duration_sec": 0.0, "deltas": []})
        else:
            pred_impacts = [p["impact_frame"] for p in pa]
            m = match_strokes(gt_impacts, pred_impacts, args.match_tolerance)
            matched_gt = {gi: pi for gi, pi, _ in m["matches"]}
            for gi, r in enumerate(rows):
                if gi in matched_gt:
                    preds_a[r["clip_path"]] = pa[matched_gt[gi]]
                    preds_a_matched[r["clip_path"]] = pa[matched_gt[gi]]
                    rows_matched.append(r)
                else:
                    preds_a[r["clip_path"]] = NOT_DETECTED_PRED
            per_clip.append({
                "clip": f"{set_name}/{stem}", "n_gt": len(rows),
                "n_pred": len(pa), "tp": len(m["matches"]),
                "fn": len(m["fn"]), "fp": len(m["fp"]),
                # ความยาววิดีโอ (ไม่ใช่เวลาประมวลผล) — ใช้คิด FP ต่อนาที
                "duration_sec": _duration_from_cache(cache_dir, set_name, stem),
                "deltas": [d for _, _, d in m["matches"]],
                "fn_nearest_delta": m["fn_nearest_delta"],
            })

        # ---- Mode B: จับคู่ด้วย "ค่า" impact frame ไม่ใช่ลำดับ index ----
        # mode B ป้อน GT impact เข้า build_loeuf_schema ตรง ๆ → predicted impact
        # ต้องเท่ากับ GT เป๊ะ จึงจับคู่ด้วยค่าได้เลย
        # ⚠️ ห้ามจับคู่ด้วย index: builder มี `if not track: continue` ที่ทิ้ง
        # stroke กลางทางได้ ซึ่งจะทำให้ index หลังจากนั้นเลื่อนทั้งแถวแบบเงียบ ๆ
        # แล้วเทียบ keyframe ผิด stroke (ตัวเลขจะเพี้ยนโดยไม่มีอะไรเตือน)
        pb = _load_preds(cache_dir, "b", set_name, stem)
        if pb is not None:
            by_impact = {p["impact_frame"]: p for p in pb
                         if p["impact_frame"] is not None}
            for r in rows:
                p = by_impact.get(r["impact"])
                if p is not None:
                    preds_b[r["clip_path"]] = p

    # ---- scoring ----
    mode_a = score_mode(all_rows, preds_a, args.accuracy_tolerance)
    mode_a_matched = (score_mode(rows_matched, preds_a_matched,
                                 args.accuracy_tolerance)
                      if rows_matched else None)
    mode_b = score_mode(all_rows, preds_b, args.accuracy_tolerance) if preds_b else None
    sensitivity = sensitivity_without_anomalies(all_rows, preds_a,
                                                args.accuracy_tolerance)
    # เกณฑ์ที่ลูกค้าเขียนไว้เอง — "อยู่ในช่วงการเคลื่อนไหวเดียวกันถือว่าใช้ได้"
    # (cv_schema_table_loeuf > Keyframe selection) ดู score_phase_mode()
    mode_a_phase = score_phase_mode(all_rows, preds_a)
    mode_b_phase = score_phase_mode(all_rows, preds_b) if preds_b else None
    det = detection_metrics(per_clip)

    total_min = round(sum(c.get("duration_sec", 0.0) for c in per_clip) / 60.0, 1)
    results = {
        "mode_a": mode_a, "mode_a_matched": mode_a_matched, "mode_b": mode_b,
        "mode_a_phase": mode_a_phase, "mode_b_phase": mode_b_phase,
        "detection": det, "sensitivity": sensitivity,
        "n_anomalies": n_anomalies, "quirks": KNOWN_QUIRKS,
        "errored_clips": errored_clips,
        "meta": {
            "n_label_files": len(sessions), "n_gt_strokes": len(all_rows),
            "annotators": "atikan", "fps": "29.97",
            "total_minutes": total_min,
            "checkpoint": args.model, "git_sha": _git_sha(),
            "run_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "accuracy_tolerance": args.accuracy_tolerance,
            "match_tolerance": args.match_tolerance,
            # ⚠️ ตัวเลขจะต่างกันมากระหว่างสองโหมดนี้ (acceptance 0.506 vs 0.308)
            # ต้องระบุไว้ในรายงานเสมอ ไม่งั้นอ่านสลับกันแล้วเข้าใจผิดทั้งฉบับ
            "hit_model_eval": _eval_mode_from_manifest(manifest, args),
        },
    }

    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_keyframe_markdown(results), encoding="utf-8")
    results_json = report.with_name("benchmark_keyframe_results.json")
    results_json.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                            encoding="utf-8")

    acc = mode_a["acceptance_accuracy"]
    tag = ("✅ ≥0.80" if acc is not None and acc >= 0.80
           else "⚠️ ต่ำกว่าเป้า 0.80" if acc is not None else "")
    pacc = mode_a_phase["acceptance_accuracy"]
    ptag = ("✅ ≥0.80" if pacc is not None and pacc >= 0.80
            else "⚠️ ต่ำกว่าเป้า 0.80" if pacc is not None else "")
    print(f"\nGT {len(all_rows)} stroke / {len(sessions)} คลิป")
    if errored_clips:
        print(f"⚠️ {len(errored_clips)} คลิป/stage ล้ม — GT ของคลิปนั้นถูกนับเป็น FN:")
        for e in errored_clips:
            print(f"   {e['clip']} [{e['stage']}]: {e['error']}")
    print(f"detection: recall {det['recall']} precision {det['precision']} "
          f"(TP {det['tp']} FN {det['fn']} FP {det['fp']})")
    print("acceptance accuracy (impact+backswing) — เกณฑ์ลูกค้า "
          f"\"ช่วงเดียวกัน\", Mode A end-to-end: {pacc} {ptag}")
    if mode_b_phase:
        print(f"  Mode B oracle: {mode_b_phase['acceptance_accuracy']}")
    print(f"เกณฑ์ ±{args.accuracy_tolerance} เฟรม (เข้มกว่าที่ลูกค้าขอ), "
          f"Mode A end-to-end: {acc} {tag}")
    if mode_a_matched:
        print(f"  matched-only: {mode_a_matched['acceptance_accuracy']}")
    if mode_b:
        print(f"  Mode B oracle: {mode_b['acceptance_accuracy']}")
    print(f"report → {report}\nresults → {results_json}")


def _duration_from_cache(cache_dir: Path, set_name: str, stem: str) -> float:
    pkl = cache_dir / f"tracking/{set_name}/{stem}.pkl"
    if not pkl.exists():
        return 0.0
    try:
        with open(pkl, "rb") as f:
            return float(pickle.load(f).get("duration_sec", 0.0))
    except Exception:
        return 0.0


def stage_dry_run(sessions):
    """ตรวจฝั่ง label อย่างเดียว — ไม่แตะวิดีโอ/YOLO"""
    from collections import Counter

    print(f"\n=== dry-run: label เท่านั้น ({len(sessions)} คลิป) ===")
    all_rows = []
    for label_path, set_name, session in sessions:
        raw = json.loads(label_path.read_text(encoding="utf-8")).get("strokes", [])
        rows = label_rows_from_session(session, set_name, raw_strokes=raw)
        all_rows.extend(rows)
        anomalies = sum(1 for r in rows if r["anomalies"])
        print(f"  {set_name}/{session.video_path.stem}: {len(rows)} stroke"
              f"{f' ({anomalies} anomaly)' if anomalies else ''}")

    print(f"\nรวม {len(all_rows)} stroke")
    print("type:", dict(Counter(r["stroke_type"] for r in all_rows)))
    print("\nGT coverage ต่อ keyframe:")
    for name in BENCH_KEYFRAME_COLS:
        n = sum(1 for r in all_rows if r.get(name) is not None)
        print(f"  {name:22} {n:3}/{len(all_rows)}")
    flags = Counter(f for r in all_rows for f in r["anomalies"])
    if flags:
        print("\nanomalies:", dict(flags))
    keys = [r["clip_path"] for r in all_rows]
    assert len(set(keys)) == len(keys), "คีย์ stroke ซ้ำ!"
    print(f"\nคีย์ unique ครบ {len(keys)} ตัว ✅")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Loeuf CV — keyframe accuracy benchmark vs ground truth")
    ap.add_argument("stage", choices=("track", "predict", "score", "all", "dry-run"))
    ap.add_argument("--dataset-root", default="dataset")
    ap.add_argument("--cache-dir", default="dataset/benchmark_cache")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="YOLO pose checkpoint (ball+racket)")
    ap.add_argument("--base-model", default="yolo11m.pt")
    ap.add_argument("--max-players", type=int, default=1)
    ap.add_argument("--mode", default="both", choices=("a", "b", "both"))
    ap.add_argument("--only", default=None, help="substring ของชื่อไฟล์ label")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="ไม่สนใจ cache")
    ap.add_argument("--match-tolerance", type=int, default=MATCH_TOLERANCE_FRAMES,
                    help="±N เฟรม สำหรับจับคู่ stroke (คนละตัวกับ accuracy)")
    ap.add_argument("--accuracy-tolerance", type=int, default=1,
                    help="±N เฟรม สำหรับตัดสินว่า keyframe ถูก")
    ap.add_argument("--report", default="docs/BENCHMARK_KEYFRAME.md")
    ap.add_argument("--impact-refiner-dir", default=None,
                    help="โฟลเดอร์ refiner ชุด LOPO (สร้างด้วย "
                         "train_impact_refiner.py --lopo-out) — ไม่ระบุ = ปิด"
                         "การปรับเฟรม ไม่ใช่ใช้ตัวที่รากโปรเจกต์")
    ap.add_argument("--hit-classifier-dir", default=None,
                    help="โฟลเดอร์โมเดล LOPO (จาก train_hit_classifier_from_cache"
                         ".py --lopo-out) — แต่ละคลิปจะใช้โมเดลที่ไม่เคยเห็นคน"
                         "ในคลิปนั้น = ตัวเลข end-to-end ที่ไม่ปนข้อมูลที่เคยเห็น")
    ap.add_argument("--ml-threshold", type=float, default=None,
                    help="ทับ ML_PROB_THRESHOLD (ต้องส่งเป็น argument — "
                         "การแก้ตัวแปรโมดูลไม่มีผลเพราะเป็น default argument "
                         "ที่ผูกค่าตอนนิยามฟังก์ชัน)")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        print(f"❌ ไม่พบ {dataset_root}")
        return 1

    sessions = _collect_sessions(dataset_root, args.only, args.limit)
    if not sessions:
        print("❌ ไม่เจอ label session เลย")
        return 1

    if args.stage == "dry-run":
        stage_dry_run(sessions)
        return 0

    manifest = Manifest(Path(args.cache_dir) / "manifest.json")
    if args.stage in ("track", "all"):
        stage_track(args, sessions, manifest)
    if args.stage in ("predict", "all"):
        stage_predict(args, sessions, manifest)
    if args.stage in ("score", "all"):
        stage_score(args, sessions, manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
