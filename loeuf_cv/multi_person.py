"""Multi-person pose extraction — background-subtraction blob tracking +
crop-zoom + per-blob single-person pose

แก้ 2 ปัญหาที่พบจากคลิปจริง (2026-07-09 diagnostic session):
1. หลายคนในเฟรม → แยก track ด้วยตำแหน่ง blob ต่อเนื่องเฟรมต่อเฟรม
   (centroid nearest-neighbor matching, ไม่ต้องมี person-detector model แยก)
2. คนไกล/ตัวเล็กในเฟรม → crop+upscale รอบ blob ก่อนส่งเข้า pose model
   validated บนคลิปจริง: whole-frame detection mean_visibility≈0.06
   (false positive, bbox เกินขอบเฟรม) → crop+zoom บริเวณเดียวกัน
   mean_visibility≈0.67 (skeleton ถูกต้อง)

ไม่ใช้ person-detector แยก (YOLO ฯลฯ): กล้องเทนนิสอยู่นิ่ง (ทริพอด/
มือถือตั้ง) → background subtraction พอสำหรับหา "คนขยับอยู่ตรงไหน"
แล้วให้ MediaPipe Pose (โมเดลเดิมที่ผ่านการทดสอบแล้วทั้งระบบ) ทำงาน
เฉพาะบริเวณ crop นั้น — ไม่เพิ่ม dependency ใหม่เลย

ข้อจำกัดที่รู้: ถ้าคนตัวเล็กในเฟรมจริง ๆ (เช่น <80px สูงใน 540p) การ
crop+zoom ช่วยเรื่อง "หาเจอ" ได้มาก แต่ไม่ได้สร้างรายละเอียดที่ไม่มีอยู่
จริงขึ้นมาใหม่ — คุณภาพ skeleton ยังจำกัดตามความละเอียดต้นทาง
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .config import PipelineConfig
from .pose_extractor import FrameStats, PoseTimeseries, VideoMeta, read_video_meta

MIN_BLOB_AREA = 400                # px^2 — กรอง noise (ใบไม้ไหว, เงา)
MIN_ASPECT_RATIO = 0.6             # h/w ขั้นต่ำ — คนยืนสูงกว่ากว้าง
TRACK_MATCH_MAX_DIST_FRAC = 0.12   # ระยะ centroid สูงสุด (สัดส่วนความกว้างเฟรม)
MAX_TRACK_GAP_FRAMES = 15          # blob หายได้ไม่เกินกี่เฟรมก่อนปิด track
MIN_TRACK_FRAMES = 10              # track สั้นกว่านี้ทิ้ง (noise)
CROP_MARGIN_FRAC = 0.4             # margin รอบ blob ก่อน crop
UPSCALE_TARGET_PX = 700            # ขยาย crop ให้สูงประมาณนี้ก่อนเข้าโมเดล
BG_HISTORY = 200
BG_VAR_THRESHOLD = 25

# ⚠️ validated บนคลิปจริง (2026-07-10): คนยืนนิ่งข้างสนาม (ผู้ชม/เก็บบอล)
# ให้ blob ที่ track ต่อเนื่องได้ง่าย/นานกว่าผู้เล่นตัวจริงที่ขยับตลอด
# ระหว่างแรลลี่ (ซึ่ง track มักหลุดสั้น ๆ บ่อยจากการบัง/ระยะไกล ทำให้
# กระจายเป็นหลาย track ID) — การเลือกด้วย "track ไหนยาวสุด" อย่างเดียว
# จึงเลือกคนยืนนิ่งผิดคนได้ (พิสูจน์แล้วจากคลิปจริง: track คนยืนนิ่ง
# y_std=2.3px ต่อเนื่อง 211 เฟรม ชนะ track ผู้เล่นจริงที่กระจายเป็น 13
# ชิ้น y_std 15-52px ต่อชิ้น) → กรอง track ที่แทบไม่ขยับทิ้งก่อนคัดเลือก
MIN_MOVEMENT_STD_FRAC = 0.01       # y-centroid std ขั้นต่ำ (สัดส่วนความสูงเฟรม)
MIN_CANDIDATE_SEPARATION_FRAC = 0.15  # ระยะเฉลี่ยขั้นต่ำระหว่าง track ถึงนับเป็นคนละคน (สัดส่วนความกว้างเฟรม)
MAX_DUPLICATE_OVERLAP_FRAC = 0.2   # overlap ของเฟรมที่ track ได้พร้อมกัน สูงกว่านี้ = active พร้อมกันจริง = คนละคนแน่นอน


@dataclass
class PlayerTrack:
    track_id: int
    role: str                  # "near" / "far" / "other" — ตาม apparent size
    pose: PoseTimeseries
    n_frames_tracked: int      # จำนวนเฟรมที่มี blob จริง (ไม่รวม gap)


def _boxes_should_merge(a: tuple, b: tuple, max_gap_frac: float = 0.3) -> bool:
    """สอง box น่าจะเป็นคนเดียวกันที่แตกเป็น 2 ชิ้น (เช่น หัว/ไหล่ กับ
    ลำตัว/ขา แยกกันเพราะ background subtraction เจอช่องว่างตรงกลาง) —
    ถ้า x-range overlap กัน (อยู่แนวตั้งเดียวกัน) และช่องว่างแนวตั้ง
    ระหว่างกันเล็กเทียบกับขนาด box"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x_overlap = min(ax + aw, bx + bw) - max(ax, bx)
    if x_overlap <= 0:
        return False
    if ay + ah <= by:
        gap = by - (ay + ah)
    elif by + bh <= ay:
        gap = ay - (by + bh)
    else:
        return True  # overlap กันอยู่แล้วในแนวตั้งด้วย
    return gap <= max_gap_frac * min(ah, bh)


def _merge_boxes(a: tuple, b: tuple) -> tuple:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = min(ax, bx), min(ay, by)
    x1, y1 = max(ax + aw, bx + bw), max(ay + ah, by + bh)
    return (x0, y0, x1 - x0, y1 - y0)


def _merge_split_person_boxes(boxes: list) -> list:
    """รวม box ที่น่าจะเป็นคนเดียวกันแตกเป็นหลายชิ้นเข้าด้วยกัน — validated
    บนคลิปจริง (2026-07-10 บ่าย): คนคนเดียวขยับเร็ว/แสงไม่สม่ำเสมอ ทำให้
    background subtraction เห็นเป็น 2 blob แยกกัน (หัว/ไหล่ vs ลำตัว/ขา)
    ในเฟรมเดียวกัน — ถ้าไม่รวมจะกลายเป็น track คนละ ID ที่ active
    "พร้อมกัน" ปลอม ๆ ทำให้ระบบ dedup เข้าใจผิดว่าเป็นคนละคนจริง"""
    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                if _boxes_should_merge(merged[i], merged[j]):
                    merged[i] = _merge_boxes(merged[i], merged[j])
                    del merged[j]
                    changed = True
                    break
            if changed:
                break
    return merged


def detect_blobs(bg_sub, frame: np.ndarray,
                 learning_rate: float = 0.0) -> list[tuple]:
    """คืน list ของ (x, y, w, h) ของ blob ที่ดูเป็นคน (moving + คนยืน)"""
    mask = bg_sub.apply(frame, learningRate=learning_rate)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= MIN_BLOB_AREA:
            boxes.append((x, y, w, h))
    boxes = _merge_split_person_boxes(boxes)
    return [b for b in boxes if b[3] >= b[2] * MIN_ASPECT_RATIO]


def match_tracks(active_centroids: dict, boxes: list, frame_w: int) -> dict:
    """greedy nearest-centroid matching เฟรมต่อเฟรม

    active_centroids: {track_id: (cx, cy)}
    คืน {box_index: track_id} เฉพาะคู่ที่อยู่ในระยะ max_dist
    """
    max_dist = TRACK_MATCH_MAX_DIST_FRAC * frame_w
    box_centroids = [(x + w / 2.0, y + h / 2.0) for x, y, w, h in boxes]
    pairs = []
    for bi, bc in enumerate(box_centroids):
        for tid, tc in active_centroids.items():
            d = float(np.hypot(bc[0] - tc[0], bc[1] - tc[1]))
            if d <= max_dist:
                pairs.append((d, bi, tid))
    pairs.sort(key=lambda p: p[0])
    assigned, used_tracks, used_boxes = {}, set(), set()
    for d, bi, tid in pairs:
        if bi in used_boxes or tid in used_tracks:
            continue
        assigned[bi] = tid
        used_boxes.add(bi)
        used_tracks.add(tid)
    return assigned


def crop_landmarks_to_frame(norm_in_crop: np.ndarray, crop_box: tuple,
                            frame_w: int, frame_h: int) -> np.ndarray:
    """แปลงพิกัด normalized ใน crop → normalized เทียบเฟรมเต็ม

    crop_box = (x0, y0, cw, ch) พิกัดพิกเซลใน "เฟรมเต็ม" ของ crop
    (ก่อน upscale — upscale factor cancel ออกจากสมการนี้พอดี เพราะ
    landmark.x/y ที่ mediapipe คืนมา normalized เทียบขนาดภาพที่ประมวลผล
    อยู่แล้ว ไม่ว่าจะ upscale เท่าไหร่)
    """
    x0, y0, cw, ch = crop_box
    out = norm_in_crop.copy()
    out[:, 0] = (out[:, 0] * cw + x0) / frame_w
    out[:, 1] = (out[:, 1] * ch + y0) / frame_h
    return out


def _crop_and_run_pose(pose_model, frame: np.ndarray, box: tuple):
    """crop รอบ box + margin, upscale, รัน pose → (result, crop_box_px)"""
    fh, fw = frame.shape[:2]
    x, y, w, h = box
    mx, my = int(w * CROP_MARGIN_FRAC), int(h * CROP_MARGIN_FRAC)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(fw, x + w + mx), min(fh, y + h + my)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
        return None, None
    scale = max(1.0, UPSCALE_TARGET_PX / crop.shape[0])
    crop_big = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
    result = pose_model.process(cv2.cvtColor(crop_big, cv2.COLOR_BGR2RGB))
    return result, (x0, y0, x1 - x0, y1 - y0)


def _mean_centroid(t: dict) -> tuple[float, float]:
    return float(np.mean(t["centroid_xs"])), float(np.mean(t["centroid_ys"]))


def _blob_frames(t: dict) -> set:
    return t.get("tracked_frame_idxs") or set(t["frames"].keys())


def _likely_same_person(t_a: dict, t_b: dict, min_sep: float) -> bool:
    """track สองอันเป็นคนเดียวกันที่ track หลุดแล้วได้ ID ใหม่ ถ้าตำแหน่ง
    เฉลี่ยใกล้กัน **และ** ไม่เคย active (blob ถูก track) พร้อมกันเลย —
    ต้องเช็คคู่กันเสมอ (ดู select_candidate_tracks docstring)"""
    cx_a, cy_a = _mean_centroid(t_a)
    cx_b, cy_b = _mean_centroid(t_b)
    if np.hypot(cx_a - cx_b, cy_a - cy_b) >= min_sep:
        return False
    frames_a, frames_b = _blob_frames(t_a), _blob_frames(t_b)
    overlap = len(frames_a & frames_b) / max(1, min(len(frames_a), len(frames_b)))
    return overlap <= MAX_DUPLICATE_OVERLAP_FRAC


def select_candidate_tracks(tracks: dict, frame_width: int, frame_height: int,
                            max_players: int) -> list[list[tuple[int, dict]]]:
    """จัดกลุ่ม track fragment ที่น่าจะเป็นคนเดียวกัน (track หลุดแล้วได้
    ID ใหม่) เข้าด้วยกัน แล้วเลือกกลุ่มที่มีข้อมูลรวมเยอะสุด (สูงสุด
    max_players คน) — คืน list ของ cluster โดยแต่ละ cluster คือ list
    ของ (tid, track_dict) ทั้งหมดที่เป็นคนเดียวกัน

    ⚠️ เดิม (ก่อน 2026-07-10 บ่าย) เลือกแค่ track เดียวที่ยาวสุดต่อคน
    แล้วทิ้ง fragment ที่เหลือทั้งหมด — validated บนคลิปจริงว่าเสียข้อมูล
    ไปมาก (ผู้เล่นจริงคนหนึ่ง track ขาดเป็น 7 ท่อนเพราะ blob กระพริบเกิน
    MAX_TRACK_GAP_FRAMES ซ้ำ ๆ ระหว่างแรลลี่ — เก็บแค่ท่อนแรก 268/780
    เฟรม (34%) ทั้งที่รวมทุกท่อนได้ pose สำเร็จถึง 516/780 เฟรม (66%))
    → เปลี่ยนจาก "เลือก 1 track ต่อคน" เป็น "จัดกลุ่ม fragment ของคน
    เดียวกันด้วย union-find แล้ว merge" แทน (ดู extract_multi_person)

    เกณฑ์จัดกลุ่ม fragment (validate แล้วทั้งคู่บนคลิปจริง):
    1. ต้อง track ได้อย่างน้อย MIN_TRACK_FRAMES เฟรม
    2. ต้องมีการขยับจริง (MIN_MOVEMENT_STD_FRAC) — กันคนยืนนิ่งข้างสนาม
       (ผู้ชม/เก็บบอล) ที่ track ต่อเนื่องได้ง่ายกว่าผู้เล่นจริงที่ขยับ
       ตลอด (validated: คนยืนนิ่ง track ได้ 211 เฟรมต่อเนื่อง y_std=2.3px
       vs ผู้เล่นจริงกระจายเป็นหลาย track สั้น ๆ y_std 15-52px ต่อชิ้น)
    3. สอง fragment รวมกลุ่มเป็นคนเดียวกัน ถ้าตำแหน่งเฉลี่ยใกล้กัน
       (MIN_CANDIDATE_SEPARATION_FRAC) **และ** ไม่เคย active พร้อมกันเลย
       (MAX_DUPLICATE_OVERLAP_FRAC) — สองเงื่อนไขต้องจริงพร้อมกัน

       ⚠️ ใช้แค่ระยะห่างอย่างเดียวไม่พอ — validated บนคลิปจริงอีกเคส:
       สองคนยืนใกล้กันจริงตลอดคลิป (overlap ของเฟรมที่ track ได้พร้อมกัน
       สูงถึง 0.69) มี mean centroid ใกล้กันมากเหมือนกรณี "คนเดียวกัน"
       ทุกประการ แต่เป็นคนละคนจริง (ยืนยันแล้วด้วยตาว่าแยก skeleton
       ถูกต้อง) — ถ้าเช็คแค่ระยะห่างจะเผลอรวมคนสองคนเป็นคนเดียว จึงต้อง
       เช็ค temporal overlap ควบคู่ไปด้วยเสมอ: overlap สูง = active
       พร้อมกันจริง = ต้องเป็นคนละคนแน่นอน (ห้าม merge)

    ⚠️ ข้อควรระวังตอน implement — ลองมาแล้ว 2 วิธี ทั้งคู่พังคนละแบบ
    (validate จริงบนคลิปจริงทั้งคู่ 2026-07-10 บ่าย):

    1. **union-find (single-linkage)** — union แค่คู่ที่ตรงเงื่อนไขโดยตรง
       พัง: track สองอันที่ overlap กันสูงจริง (คนละคนแน่นอน) โดน merge
       เข้า cluster เดียวกันได้ผ่าน fragment ตัวกลางที่บังเอิญ compatible
       กับทั้งสองฝั่งแยกกัน (transitive merge ผิดคน)

    2. **complete-linkage** (ต้อง compatible กับ**สมาชิกทุกคน**ใน cluster)
       — แก้ปัญหาข้อ 1 ได้ แต่พังอีกแบบ: fragment ของคนเดียวกันที่ควรรวม
       เข้า cluster หลัก ดันไม่ผ่านเพราะ fail กับสมาชิกเก่าตัวใดตัวหนึ่ง
       (เช่น ตำแหน่งขยับไปนิดหน่อยตามเวลา) ทั้งที่ compatible กับ track
       หลัก (anchor) สบาย ๆ — ยืนยันด้วยตาจากภาพจริง: track ที่ควรเป็น
       "near" ท่อนที่หลุดไป ดันถูกจัดเป็น "far" แยกต่างหาก (คนละคนปลอม)

    **ทางแก้ที่ใช้จริง: average-linkage** — เทียบ candidate กับ "จุดรวม"
    (aggregate: centroid ทุกจุด + เฟรมที่ track ได้ทุกเฟรม) ของสมาชิกที่
    อยู่ใน cluster แล้วทั้งหมด ไม่ใช่เทียบทีละคน (complete) หรือเทียบแค่
    คู่ที่เจอ (single) — ทนต่อ outlier สมาชิกตัวเดียวได้ (ไม่ fail ทั้ง
    cluster เพราะสมาชิกตัวเดียว) และยังกันไม่ให้ merge ผิดคนได้เหมือนเดิม
    (เพราะ overlap เทียบกับ union ของเฟรมทั้ง cluster ซึ่งยังคง detect
    overlap สูงกับคนละคนได้อยู่)
    """
    min_movement_std = MIN_MOVEMENT_STD_FRAC * frame_height
    candidates = {tid: t for tid, t in tracks.items()
                 if len(t["frames"]) >= MIN_TRACK_FRAMES
                 and float(np.std(t["centroid_ys"])) >= min_movement_std}

    min_sep = MIN_CANDIDATE_SEPARATION_FRAC * frame_width
    # ประมวลผลจาก track ที่มีข้อมูลเยอะสุดก่อน (แกนหลักของแต่ละ cluster
    # ควรเป็น track ที่น่าเชื่อถือที่สุด)
    ordered = sorted(candidates.items(), key=lambda kv: -len(kv[1]["frames"]))
    clusters: list[list[tuple[int, dict]]] = []
    aggregates: list[dict] = []  # จุดรวมของแต่ละ cluster (นับถึงตอนนี้)

    for tid, t in ordered:
        placed_idx = next((i for i, agg in enumerate(aggregates)
                           if _likely_same_person(t, agg, min_sep)), None)
        if placed_idx is not None:
            clusters[placed_idx].append((tid, t))
            agg = aggregates[placed_idx]
            agg["centroid_xs"].extend(t["centroid_xs"])
            agg["centroid_ys"].extend(t["centroid_ys"])
            agg["tracked_frame_idxs"] |= _blob_frames(t)
        else:
            clusters.append([(tid, t)])
            aggregates.append({
                "centroid_xs": list(t["centroid_xs"]),
                "centroid_ys": list(t["centroid_ys"]),
                "tracked_frame_idxs": set(_blob_frames(t)),
            })

    ranked = sorted(clusters, key=lambda c: -sum(len(t["frames"]) for _, t in c))
    return ranked[:max_players]


def _build_background_model(video_path: str):
    """pass แรก: warm up background subtractor ทั้งคลิป (ไม่รัน pose)
    เพราะไม่มีเฟรมสนามเปล่าแยกต่างหาก — MOG2 ปรับตัวช้าพอที่คนซึ่งขยับ
    ตลอดคลิปจะยังถูกจัดเป็น foreground แม้ warm-up ผ่านคลิปทั้งอัน

    ⚠️ ข้อจำกัดที่รู้ (2026-07-11, ยังไม่แก้ — ดู CHANGELOG 0.3.6/0.3.7):
    คลิปฝึกซ้อมแบบป้อนบอลที่ผู้เล่นกลับไปยืนตำแหน่ง/ท่าเตรียมรับเดิมซ้ำ
    ทุกแต้ม ทำให้ MOG2 ตัดสินตำแหน่ง/ท่านั้นเป็น background ได้ (หลุด
    track 0.2-1.2s ทุกครั้งที่กลับไปจุดเดิม, พบ 3-4 ครั้งต่อคลิปยาว
    70-120s) — ลองแก้ 3 แนวทางแล้ว ทุกแนวทาง revert กลับมาใช้ตัวนี้ต่อ
    เพราะ net regression:
    1. segment-based warm-up (สร้าง MOG2 ใหม่ทุก N เฟรม) — model แต่ละ
       ช่วงเริ่มจากศูนย์ ทำให้หลุดกระจายที่อื่นมากขึ้นแทน
    2. causal continuous adaptation (learning_rate=-1 ไม่ freeze) — ลด
       dropout จุดเดิมได้จริงในคลิปที่มีปัญหา แต่พังคลิปอื่นที่เดิมไม่มี
       ปัญหาเลย (coverage 92.7%→63.4%)
    3. hybrid: MOG2 เดิม + frame-diff rescue fallback เฉพาะ track ที่
       active แต่ไม่ถูก blob จับ — ตัว rescue ทำงานถูกที่ตั้งใจ (ดึงคน
       ยืนนิ่งกลับมาได้จริง) แต่ box ที่ได้จาก frame-diff มีรูปร่าง/
       ตำแหน่งต่างจาก MOG2 พอที่ select_candidate_tracks (clustering ที่
       tune ตาม characteristic ของ MOG2 โดยเฉพาะ) แยกเป็นคนละ track แทน
       จะ stitch — ผู้เล่นเป้าหมายเองเสีย coverage มากกว่าเดิม (92.7%→76.5%)
    ยังไม่มี fix ที่ net-positive จริง — ต้อง fix clustering ให้ทนกับ
    box จากแหล่งอื่นก่อน หรือย้ายไปใช้ SAM2 ที่ไม่มีปัญหาคลาสนี้เลย"""
    cap = cv2.VideoCapture(video_path)
    bg = cv2.createBackgroundSubtractorMOG2(
        history=BG_HISTORY, varThreshold=BG_VAR_THRESHOLD, detectShadows=False)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        bg.apply(frame)
    cap.release()
    return bg


def extract_multi_person(video_path: str, config: PipelineConfig,
                         max_players: int = 2,
                         progress_callback=None) -> list[PlayerTrack]:
    """แยกคนหลายคนในคลิป → PoseTimeseries ต่อคน

    จัดกลุ่ม track fragment ที่เป็นคนเดียวกัน (track หลุดแล้วได้ ID ใหม่
    บ่อยระหว่างแรลลี่) แล้ว stitch เข้าเป็น PoseTimeseries เดียวต่อเนื่อง
    ต่อคน (ดู select_candidate_tracks) เลือกคนที่มีข้อมูลรวมเยอะสุด
    สูงสุด max_players คน พร้อม role ("near"/"far"/"other" ตามขนาด
    apparent เฉลี่ย — คนที่ตัวใหญ่กว่า = ใกล้กล้องกว่า)
    """
    import mediapipe as mp

    meta = read_video_meta(video_path)
    bg = _build_background_model(video_path)

    tracks: dict = {}
    next_id = 0
    pose_model = mp.solutions.pose.Pose(
        static_image_mode=True, model_complexity=config.model_complexity,
        min_detection_confidence=config.min_detection_confidence)

    def _new_track(frame_idx: int) -> dict:
        return {"frames": {}, "centroid": None, "last_seen": frame_idx,
                "centroid_xs": [], "centroid_ys": [], "tracked_frame_idxs": set()}

    def _ingest_box(tid: int, box: tuple, frame_idx: int, norm_frame: np.ndarray) -> None:
        x, y, w, h = box
        tracks[tid]["centroid"] = (x + w / 2.0, y + h / 2.0)
        tracks[tid]["last_seen"] = frame_idx
        tracks[tid]["centroid_xs"].append(x + w / 2.0)
        tracks[tid]["centroid_ys"].append(y + h / 2.0)
        # ⚠️ นับทุกเฟรมที่ blob ได้จริง ไม่ใช่แค่เฟรมที่ pose สำเร็จ —
        # validated บนคลิปจริง (2026-07-10): ตอนสองคนยืนใกล้กัน pose
        # model มักสำเร็จแค่คนเดียวสลับกันไปมาต่อเฟรม ทำให้เฟรมที่
        # pose "สำเร็จ" ของสองคนที่ track พร้อมกันจริงดูเหมือนไม่
        # overlap เลย (ผิดพลาด) ต้องนับจาก blob tracking ตรง ๆ
        tracks[tid]["tracked_frame_idxs"].add(frame_idx)

        result, crop_box = _crop_and_run_pose(pose_model, norm_frame, box)
        if result is not None and result.pose_landmarks:
            lm = np.array([[p.x, p.y, p.z] for p in result.pose_landmarks.landmark])
            vis = np.array([p.visibility for p in result.pose_landmarks.landmark])
            wlm = np.array([[p.x, p.y, p.z]
                           for p in result.pose_world_landmarks.landmark])
            lm[:, :2] = crop_landmarks_to_frame(lm[:, :2], crop_box,
                                                meta.width, meta.height)
            tracks[tid]["frames"][frame_idx] = (lm, wlm, vis, h)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if config.normalize_frames:
            from .image_norm import normalize_frame
            raw_luma = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
            norm_frame = normalize_frame(frame, raw_luma)
        else:
            norm_frame = frame

        boxes = detect_blobs(bg, frame, learning_rate=0.0)
        active_centroids = {
            tid: t["centroid"] for tid, t in tracks.items()
            if frame_idx - t["last_seen"] <= MAX_TRACK_GAP_FRAMES}
        assigned = match_tracks(active_centroids, boxes, meta.width)

        for bi, box in enumerate(boxes):
            if bi in assigned:
                tid = assigned[bi]
            else:
                tid = next_id
                next_id += 1
                tracks[tid] = _new_track(frame_idx)
            _ingest_box(tid, box, frame_idx, norm_frame)

        if progress_callback:
            progress_callback(frame_idx, meta.frame_count)
        frame_idx += 1

    cap.release()
    pose_model.close()
    total_frames = frame_idx

    kept_clusters = select_candidate_tracks(tracks, meta.width, meta.height, max_players)

    def _primary_tid(cluster):
        # ใช้ fragment ที่มีเฟรม pose สำเร็จเยอะสุดในกลุ่มเป็น track_id
        # ตัวแทน (สำหรับ label/report เท่านั้น — pose ข้างในเป็นของรวม
        # ทุก fragment แล้ว)
        return max(cluster, key=lambda kv: len(kv[1]["frames"]))[0]

    heights = []
    for cluster in kept_clusters:
        all_heights = [f[3] for _, t in cluster for f in t["frames"].values()]
        heights.append((_primary_tid(cluster), float(np.median(all_heights))))
    heights.sort(key=lambda kv: -kv[1])
    role_by_id = {tid: ("near" if rank == 0 else "far" if rank == 1 else "other")
                 for rank, (tid, _) in enumerate(heights)}

    result_tracks = []
    for cluster in kept_clusters:
        tid = _primary_tid(cluster)
        T = total_frames
        landmarks = np.full((T, 33, 3), np.nan)
        world_landmarks = np.full((T, 33, 3), np.nan)
        visibility = np.zeros((T, 33))
        height_frac = np.zeros(T)
        n_frames_total = 0
        # รวม frame data จากทุก fragment ในกลุ่ม (คนเดียวกัน track ขาด
        # เป็นหลายท่อน) เข้า PoseTimeseries เดียวต่อเนื่อง — ท่อนไม่ควร
        # overlap กันเวลาอยู่แล้ว (เงื่อนไขจัดกลุ่มบังคับ overlap ต่ำ)
        for _, t in cluster:
            for fi, (lm, wlm, vis, h) in t["frames"].items():
                landmarks[fi] = lm
                world_landmarks[fi] = wlm
                visibility[fi] = vis
                height_frac[fi] = h / meta.height
            n_frames_total += len(t["frames"])

        pose_ts = PoseTimeseries(
            landmarks=landmarks, world_landmarks=world_landmarks,
            visibility=visibility,
            timestamps_ms=np.arange(T) / meta.fps * 1000.0,
            meta=meta,
            frame_stats=FrameStats(person_height_frac=height_frac))
        result_tracks.append(PlayerTrack(
            track_id=tid, role=role_by_id[tid], pose=pose_ts,
            n_frames_tracked=n_frames_total))

    order = {"near": 0, "far": 1, "other": 2}
    result_tracks.sort(key=lambda pt: order.get(pt.role, 9))
    return result_tracks
