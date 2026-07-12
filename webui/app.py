"""Loeuf CV — Inference Test UI (เครื่องมือทดสอบภายใน ไม่ใช่ deliverable)

Upload คลิป → รัน multi-person tracking + pose extraction → ดู overlay
skeleton + กำหนดว่า track ไหนคือผู้เล่น → ดู track stats — แทนที่การรัน
สคริปต์ทีละคำสั่งด้วยมือแบบที่ทำมาตลอด session การ debug (2026-07-10)

รัน: streamlit run webui/app.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from loeuf_cv.config import PipelineConfig
from loeuf_cv.multi_person import PlayerTrack, extract_multi_person
from loeuf_cv.occlusion_fill import kinematic_fill
from webui.overlay import NOT_A_PLAYER_PREFIX, render_overlay_video

st.set_page_config(page_title="Loeuf CV — Inference Test", layout="wide")
st.title("🎾 Loeuf CV — Inference Test UI")
st.caption("เครื่องมือทดสอบภายใน สำหรับ inspect ผล multi-person tracking ก่อนส่งเข้า pipeline จริง — ไม่ใช่ deliverable ที่ส่งลูกค้า")

if "tracks" not in st.session_state:
    st.session_state.tracks = None
    st.session_state.video_path = None

with st.sidebar:
    st.header("1. Upload คลิป")
    uploaded = st.file_uploader("วิดีโอ", type=["mp4", "mov", "avi"])

    st.header("2. Parameters")
    max_players = st.slider("max_players", 1, 6, 2,
                            help="จำนวนคนสูงสุดที่จะ track (รวมคนป้อนบอล/คนรอด้วย — เลือกไม่ใช่ผู้เล่นได้ทีหลัง)")
    model_complexity = st.selectbox("model_complexity", [0, 1, 2], index=1)
    height_input = st.number_input("subject_height_cm (0 = ไม่ระบุ)", min_value=0, value=0, step=1)
    dominant_side = st.selectbox("มือถนัด", ["right", "left"])
    normalize_frames = st.checkbox("normalize_frames (white balance/exposure)", value=False)

    run_btn = st.button("▶️ รัน inference", type="primary", disabled=uploaded is None)

if uploaded is not None and run_btn:
    tmp_dir = tempfile.mkdtemp()
    video_path = str(Path(tmp_dir) / uploaded.name)
    with open(video_path, "wb") as f:
        f.write(uploaded.getbuffer())

    config = PipelineConfig(
        normalize_frames=normalize_frames,
        model_complexity=model_complexity,
        dominant_side=dominant_side,
        subject_height_cm=height_input or None,
    )

    with st.spinner("กำลังรัน multi-person tracking + pose extraction... (อาจใช้เวลาหลายสิบวินาทีถึงหลายนาที ขึ้นกับความยาวคลิป)"):
        tracks = extract_multi_person(video_path, config, max_players=max_players)

    st.session_state.tracks = tracks
    st.session_state.video_path = video_path
    st.session_state.overlay_path = None
    if not tracks:
        st.warning("ไม่เจอ track ที่ผ่านเกณฑ์เลย (ลองเพิ่ม max_players หรือเช็คว่าคลิปมีคนขยับจริงไหม)")
    else:
        st.success(f"เสร็จแล้ว — เจอ {len(tracks)} track")

tracks = st.session_state.tracks
if tracks:
    video_path = st.session_state.video_path
    n_total_frames = len(tracks[0].pose.visibility)

    st.header("2. Track Review")
    rows = []
    for t in tracks:
        vis = t.pose.visibility
        tracked_mask = vis.mean(axis=1) > 0
        n = int(tracked_mask.sum())
        rows.append({
            "track_id": t.track_id,
            "role (auto)": t.role,
            "coverage_%": round(100 * n / n_total_frames, 1),
            "n_frames_tracked": n,
            "mean_visibility": round(float(vis[tracked_mask].mean()) if n else 0.0, 3),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption("⚠️ ตัวเลข coverage อย่างเดียวบอกไม่ได้ว่า track ถูกคน — เช็คต่อด้วย overlay video ด้านล่างเสมอ "
              "(เคยเจอจริง: track เกาะคนยืนนิ่งข้างสนามแทนคู่แข่ง โดยที่ coverage สูงและดูปกติทุกอย่าง)")

    st.subheader("กำหนดว่า track ไหนคือผู้เล่น")
    st.caption("แยกผู้เล่นออกจากคนป้อนบอล/คนยืนรอ — ระบบตรวจจับ \"คนขยับ\" เหมือนกันหมด แยกไม่ออกเองว่าใครสำคัญ")
    label_options = [f"{NOT_A_PLAYER_PREFIX} (คนป้อนบอล/อื่น ๆ)"] + [f"Player {i + 1}" for i in range(len(tracks))]
    player_labels = {}
    cols = st.columns(len(tracks))
    for i, (col, t) in enumerate(zip(cols, tracks)):
        with col:
            default_idx = i + 1 if t.role in ("near", "far") else 0
            player_labels[t.track_id] = st.selectbox(
                f"track {t.track_id} ({t.role}, {rows[i]['coverage_%']}%)",
                label_options, index=min(default_idx, len(label_options) - 1),
                key=f"label_{t.track_id}")

    st.header("3. Overlay Video")
    apply_fill = st.checkbox("ใช้ kinematic gap-fill (occlusion_fill — เติมข้อมือ/ข้อเท้าที่หายช่วงสั้น)", value=True)
    if st.button("🎬 Render overlay"):
        with st.spinner("กำลัง render วิดีโอ..."):
            display_tracks = tracks
            if apply_fill:
                display_tracks = [
                    PlayerTrack(t.track_id, t.role, kinematic_fill(t.pose), t.n_frames_tracked)
                    for t in tracks
                ]
            st.session_state.overlay_path = render_overlay_video(video_path, display_tracks, player_labels)

    if st.session_state.get("overlay_path"):
        st.video(st.session_state.overlay_path)
        st.caption("สี = track_id/label ตามตารางด้านบน · จุดทึบ = confidence ≥ 0.5 · จุดเทาจาง = confidence ต่ำ (เหมือน low_confidence flag ใน schema)")
