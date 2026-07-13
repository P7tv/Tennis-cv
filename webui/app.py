"""Loeuf CV — Inference Test UI (เครื่องมือทดสอบภายใน ไม่ใช่ deliverable)

Upload คลิป → รัน multi-person tracking + pose extraction → ดู overlay
skeleton + กำหนดว่า track ไหนคือผู้เล่น → ดู track stats — แทนที่การรัน
สคริปต์ทีละคำสั่งด้วยมือแบบที่ทำมาตลอด session การ debug (2026-07-10)

รัน: streamlit run webui/app.py
"""

import os
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
    tracking_method = st.radio("Tracking Method", ["Classical CV (MOG2)", "SAM 2 Click-to-Track", "YOLO11 + BoT-SORT 🆕"])
    if tracking_method == "SAM 2 Click-to-Track":
        sam2_chunk_size = st.slider(
            "SAM 2 chunk_size (frames)", min_value=10, max_value=150, value=100, step=10,
            help="เฟรมต่อ chunk ที่ SAM 2 โหลดพร้อมกัน — ลดถ้า RAM ไม่พอ\n"
                 "30 เฟรม ≈ 377 MB · 100 เฟรม ≈ 1.26 GB · 150 เฟรม ≈ 1.88 GB")
    else:
        sam2_chunk_size = 30
    max_players = st.slider("max_players", 1, 6, 2,
                            help="จำนวนคนสูงสุดที่จะ track (รวมคนป้อนบอล/คนรอด้วย — เลือกไม่ใช่ผู้เล่นได้ทีหลัง)")
    model_complexity = st.selectbox("model_complexity", [0, 1, 2], index=1)
    height_input = st.number_input("subject_height_cm (0 = ไม่ระบุ)", min_value=0, value=0, step=1)
    dominant_side = st.selectbox("มือถนัด", ["right", "left"])
    normalize_frames = st.checkbox("normalize_frames (white balance/exposure)", value=False)

    # UI for run button depends on method
    if tracking_method == "Classical CV (MOG2)":
        run_btn = st.button("▶️ รัน inference (MOG2)", type="primary", disabled=uploaded is None)
    elif tracking_method == "YOLO11 + BoT-SORT 🆕":
        run_btn = st.button("▶️ Run YOLO11 + BoT-SORT", type="primary", disabled=uploaded is None)
    else:
        run_btn = False

# Temp dir on D: drive workspace to avoid exhausting C: drive (only ~2GB free)
_TMP_DIR = Path(__file__).parent / ".tmp"
_TMP_DIR.mkdir(exist_ok=True)

if uploaded is not None:
    if "video_path" not in st.session_state or st.session_state.get("uploaded_name") != uploaded.name:
        import shutil
        
        # GC old files
        old_video_path = st.session_state.get("video_path")
        if old_video_path and os.path.exists(old_video_path):
            try:
                shutil.rmtree(os.path.dirname(old_video_path), ignore_errors=True)
            except Exception:
                pass
                
        old_overlay = st.session_state.get("overlay_path")
        if old_overlay and os.path.exists(old_overlay):
            try:
                os.unlink(old_overlay)
            except Exception:
                pass
                
        tmp_dir = tempfile.mkdtemp(dir=str(_TMP_DIR))
        video_path = str(Path(tmp_dir) / uploaded.name)
        with open(video_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.session_state.video_path = video_path
        st.session_state.uploaded_name = uploaded.name
        st.session_state.tracks = None

    video_path = st.session_state.video_path

    config = PipelineConfig(
        normalize_frames=normalize_frames,
        model_complexity=model_complexity,
        dominant_side=dominant_side,
        subject_height_cm=height_input or None,
    )

    if tracking_method == "YOLO11 + BoT-SORT 🆕":
        if run_btn:
            from webui.yolo_track import track_players_with_yolo
            prog = st.progress(0, text="YOLO Tracking...")
            def _yolo_cb(cur, tot):
                prog.progress(min(cur / max(tot, 1), 1.0), text=f"Processing frame {cur}/{tot}")
            with st.spinner("Running YOLO11 + BoT-SORT..."):
                try:
                    tracks, ball_bboxes, racket_bboxes = track_players_with_yolo(video_path, max_players=max_players, config=config, progress_callback=_yolo_cb)
                    st.session_state.tracks = tracks
                    st.session_state.ball_bboxes = ball_bboxes
                    st.session_state.racket_bboxes = racket_bboxes
                    prog.progress(1.0, text="Done!")
                except Exception as e:
                    st.error(f"YOLO error: {e}")
                    st.stop()

    elif tracking_method == "SAM 2 Click-to-Track":
        import cv2
        from streamlit_image_coordinates import streamlit_image_coordinates
        
        st.header("1.5 คลิกเลือกผู้เล่น")
        st.write("คลิกจุดที่ตัวผู้เล่นเป้าหมาย (Player 1) ในภาพด้านล่างเพื่อเริ่ม Track ด้วย SAM 2")
        cap = cv2.VideoCapture(video_path)
        ok, frame = cap.read()
        cap.release()
        
        if ok:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            value = streamlit_image_coordinates(frame_rgb, key="sam2_click")
            
            if value is not None:
                st.write(f"คลิกล่าสุดที่พิกัด: x={value['x']}, y={value['y']}")
                
            run_btn = st.button("▶️ รัน SAM 2 Tracking", type="primary", disabled=(value is None))
            
            if run_btn and value is not None:
                with st.spinner("กำลังรัน SAM 2... (อาจใช้เวลาหลายนาที ขึ้นอยู่กับความยาวคลิป)"):
                    import json
                    import subprocess
                    
                    import os
                    prompts = [{"frame_idx": 0, "x": value["x"], "y": value["y"], "positive": True}]
                    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, dir=str(_TMP_DIR)) as f:
                        json.dump(prompts, f)
                        prompts_file = f.name
                    
                    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, dir=str(_TMP_DIR)) as f:
                        bboxes_file = f.name
                    
                    python_exe = str(Path(__file__).parent / ".venv-sam2" / "Scripts" / "python.exe")
                    cli_script = str(Path(__file__).parent / "run_sam2_cli.py")
                    ckpt = str(Path(__file__).parent / "checkpoints" / "sam2.1_hiera_tiny.pt")
                    
                    try:
                        result = subprocess.run([
                            python_exe, cli_script,
                            "--video", video_path,
                            "--prompts", prompts_file,
                            "--out", bboxes_file,
                            "--checkpoint", ckpt,
                            "--chunk-size", str(sam2_chunk_size),
                        ], capture_output=True, text=True)
                        if result.returncode != 0:
                            st.error("SAM 2 CLI failed")
                            with st.expander("Error details"):
                                st.code(result.stderr or result.stdout or "(no output)")
                            st.stop()
                        
                        with open(bboxes_file, "r") as f:
                            bboxes_str = json.load(f)
                        bboxes = {int(k): v for k, v in bboxes_str.items()}
                    finally:
                        try:
                            os.unlink(prompts_file)
                        except Exception:
                            pass
                        try:
                            os.unlink(bboxes_file)
                        except Exception:
                            pass
                    
                    from webui.sam2_track import extract_pose_from_sam2_track
                    pose_ts = extract_pose_from_sam2_track(video_path, bboxes, config)
                    
                    tracks = [PlayerTrack(track_id=1, role="player", pose=pose_ts, n_frames_tracked=len(bboxes))]
                    st.session_state.tracks = tracks
                    st.session_state.overlay_path = None
                    st.success(f"SAM 2 เสร็จสิ้น — เจอ 1 track ({len(bboxes)} frames)")
                    st.rerun()

    elif tracking_method == "Classical CV (MOG2)" and run_btn:
        with st.spinner("กำลังรัน multi-person tracking + pose extraction... (อาจใช้เวลาหลายสิบวินาทีถึงหลายนาที ขึ้นกับความยาวคลิป)"):
            tracks = extract_multi_person(video_path, config, max_players=max_players)

        st.session_state.tracks = tracks
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
            
            # GC old overlay before rendering a new one
            old_overlay = st.session_state.get("overlay_path")
            if old_overlay and os.path.exists(old_overlay):
                try:
                    os.unlink(old_overlay)
                except Exception:
                    pass
            
            try:
                ball_bboxes = st.session_state.get("ball_bboxes")
                racket_bboxes = st.session_state.get("racket_bboxes")
                st.session_state.overlay_path = render_overlay_video(
                    video_path, display_tracks, player_labels,
                    ball_bboxes=ball_bboxes, racket_bboxes=racket_bboxes
                )
            except Exception as e:
                st.error("เกิดข้อผิดพลาดในการ render วิดีโอ (FFMPEG error)")
                with st.expander("รายละเอียด Error"):
                    st.code(str(e))
                st.stop()

    if st.session_state.get("overlay_path"):
        st.video(st.session_state.overlay_path)
        st.caption("สี = track_id/label ตามตารางด้านบน · จุดทึบ = confidence ≥ 0.5 · จุดเทาจาง = confidence ต่ำ (เหมือน low_confidence flag ใน schema)")
