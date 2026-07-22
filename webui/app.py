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
        import glob
        import os
        
        # ค้นหาโมเดล .pt ทั้งหมดในโปรเจกต์
        available_models = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"]
        custom_models = glob.glob("runs/**/*.pt", recursive=True) + glob.glob("*.pt")
        for m in custom_models:
            m_norm = os.path.normpath(m)
            if m_norm not in available_models and not any(m_norm.endswith(x) for x in ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"]):
                available_models.append(m_norm)
                
        base_models = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"]
        base_model_path = st.selectbox("Base Model (สำหรับหาคน)", base_models, index=2, help="เลือกโมเดลหาคน (n = เร็วสุด, m = แม่นสุด)")
        
        yolo_model_path = st.selectbox("Custom Ball Model", available_models, help="เลือกโมเดลลูกเทนนิสที่คุณ Train เองจากในรายการ")
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
            import importlib
            import webui.yolo_track
            importlib.reload(webui.yolo_track)
            from webui.yolo_track import track_players_with_yolo
            
            prog = st.progress(0, text="YOLO Tracking...")
            def _yolo_cb(cur, tot):
                prog.progress(min(cur / max(tot, 1), 1.0), text=f"Processing frame {cur}/{tot}")
            with st.spinner("Running YOLO11 + BoT-SORT..."):
                try:
                    tracks, ball_bboxes, racket_bboxes, racket_keypoints = track_players_with_yolo(
                        video_path,
                        max_players=max_players,
                        config=config,
                        progress_callback=_yolo_cb,
                        model_path=yolo_model_path,
                        base_model=base_model_path
                    )
                    st.session_state.tracks = tracks
                    st.session_state.ball_bboxes = ball_bboxes
                    st.session_state.racket_bboxes = racket_bboxes
                    st.session_state.racket_keypoints = racket_keypoints
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

    # ─────────────────────────────────────────────────────────
    # Section 2.3: Court Calibration
    # ─────────────────────────────────────────────────────────
    st.header("2.3 Court Calibration")
    st.caption("ใช้หา homography สนาม (image → เมตรจริง) เพื่อแสดงตำแหน่งเด้งลูกจริงบนสนาม")
    
    if "court_homography" not in st.session_state:
        st.session_state.court_homography = None
    if "court_cal_status" not in st.session_state:
        st.session_state.court_cal_status = None
    
    cal_col1, cal_col2 = st.columns([2, 1])
    with cal_col1:
        if st.button("🎾 Auto-Calibrate Court"):
            with st.spinner("สแกนหาเส้นสนามอัตโนมัติ..."):
                from loeuf_cv.court_calibration import auto_detect_height_cm
                _, court_cal = auto_detect_height_cm(
                    video_path, tracks[0].pose, config, sample_frames=25
                )
                if court_cal.valid:
                    st.session_state.court_homography = court_cal.homography
                    st.session_state.court_cal_status = ("✅", f"Auto-calibrate สำเร็จ")
                else:
                    st.session_state.court_cal_status = ("⚠️", f"Auto-calibrate ไม่สำเร็จ: {court_cal.reason}")
    
    if st.session_state.court_cal_status:
        icon, msg = st.session_state.court_cal_status
        if icon == "✅":
            st.success(msg)
        else:
            st.warning(msg)
    
    # Manual 4-point picker (fallback ถ้า auto fail)
    with st.expander("✏️ Manual Court Calibration (คลิกกำหนด 4 มุมสนามเอง)", expanded=(st.session_state.court_homography is None)):
        st.info("ใช้กรณีที่ Auto-Calibrate ไม่สำเร็จ หรือเส้นคอร์ทหลุดกรอบวิดีโอ")
        
        cal_mode = st.radio(
            "เลือกจุดอ้างอิงที่จะคลิก:",
            [
                "4 มุมมาตรฐาน (ซ้าย-ขวา)",
                "กึ่งกลาง และ ฝั่งขวา (ใช้เมื่อฝั่งซ้ายหลุดกรอบ)",
                "กึ่งกลาง และ ฝั่งซ้าย (ใช้เมื่อฝั่งขวาหลุดกรอบ)",
                "เส้นเสิร์ฟ และ เส้นเน็ต (ใช้เมื่อมุมหลังหลุดกรอบทั้งซ้ายขวา)"
            ]
        )
        
        if cal_mode == "4 มุมมาตรฐาน (ซ้าย-ขวา)":
            _pt_names = ["near_baseline_left", "near_baseline_right", "service_line_left", "service_line_right"]
            st.caption("คลิกตามลำดับ: 1.มุม baseline ซ้าย, 2.มุม baseline ขวา, 3.มุม service line ซ้าย, 4.มุม service line ขวา")
        elif cal_mode == "กึ่งกลาง และ ฝั่งขวา (ใช้เมื่อฝั่งซ้ายหลุดกรอบ)":
            _pt_names = ["center_baseline", "near_baseline_right", "center_service_line", "service_line_right"]
            st.caption("คลิกตามลำดับ: 1.จุดกึ่งกลาง baseline, 2.มุม baseline ขวา, 3.จุดกึ่งกลาง service line, 4.มุม service line ขวา")
        elif cal_mode == "กึ่งกลาง และ ฝั่งซ้าย (ใช้เมื่อฝั่งขวาหลุดกรอบ)":
            _pt_names = ["near_baseline_left", "center_baseline", "service_line_left", "center_service_line"]
            st.caption("คลิกตามลำดับ: 1.มุม baseline ซ้าย, 2.จุดกึ่งกลาง baseline, 3.มุม service line ซ้าย, 4.จุดกึ่งกลาง service line")
        else:
            _pt_names = ["service_line_left", "service_line_right", "net_left", "net_right"]
            st.caption("คลิกตามลำดับ: 1.มุม service line ซ้าย, 2.มุม service line ขวา, 3.จุดตัดขอบสนามเดี่ยวซ้ายกับเส้นใต้เน็ต, 4.จุดตัดขอบสนามเดี่ยวขวากับเส้นใต้เน็ต")
            
        import cv2 as _cv2
        _cap = _cv2.VideoCapture(video_path)
        total_frames = int(_cap.get(_cv2.CAP_PROP_FRAME_COUNT))
        cal_frame_idx = st.slider("🎞️ เลื่อนหาเฟรมที่เห็นเส้นสนามชัดเจน (ไม่มีคนบัง)", 0, max(0, total_frames-1), 0, key=f"cal_frame_slider")
        _cap.set(_cv2.CAP_PROP_POS_FRAMES, cal_frame_idx)
        _ok, _frame = _cap.read()
        _cap.release()
        
        if _ok:
            try:
                from streamlit_image_coordinates import streamlit_image_coordinates
                
                mode_key = cal_mode.split()[0]
                if f"court_manual_pts_{mode_key}" not in st.session_state:
                    st.session_state[f"court_manual_pts_{mode_key}"] = []
                
                pts_list = st.session_state[f"court_manual_pts_{mode_key}"]
                
                # วาดจุดที่คลิกไปแล้วลงบนภาพ
                _disp_frame = _frame.copy()
                for i, p in enumerate(pts_list):
                    _cv2.circle(_disp_frame, p, 8, (0, 0, 255), -1) # จุดสีแดง
                    _cv2.circle(_disp_frame, p, 10, (255, 255, 255), 2) # ขอบขาว
                    _cv2.putText(_disp_frame, str(i+1), (p[0]+15, p[1]-15), _cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                
                # ถัาครบ 4 จุด วาดเส้นเชื่อม
                if len(pts_list) == 4:
                    import numpy as np
                    pts_arr = np.array(pts_list, np.int32).reshape((-1, 1, 2))
                    _cv2.polylines(_disp_frame, [pts_arr], isClosed=True, color=(0, 255, 0), thickness=3)

                _frame_rgb = _cv2.cvtColor(_disp_frame, _cv2.COLOR_BGR2RGB)
                
                _clicked = streamlit_image_coordinates(_frame_rgb, key=f"court_picker_{mode_key}")
                
                if _clicked is not None:
                    pt = (_clicked["x"], _clicked["y"])
                    if len(pts_list) < 4:
                        if not pts_list or pt != pts_list[-1]:
                            pts_list.append(pt)
                            st.rerun()
                
                st.markdown("### จุดที่เลือก:")
                for idx, name in enumerate(_pt_names):
                    if idx < len(pts_list):
                        st.write(f"✅ **{idx+1}. {name}:** {pts_list[idx]}")
                    else:
                        st.write(f"⏳ **{idx+1}. {name}:** (รอคลิก...)")
                
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    if st.button("✅ ยืนยันพิกัด (Apply)", key=f"apply_{mode_key}", type="primary", disabled=len(pts_list) < 4):
                        from loeuf_cv.court_calibration import solve_ground_homography
                        img_pts = dict(zip(_pt_names, pts_list[:4]))
                        H = solve_ground_homography(img_pts)
                        if H is not None:
                            st.session_state.court_homography = H
                            st.session_state.court_cal_status = ("✅", "Manual calibration สำเร็จ")
                            st.rerun()
                        else:
                            st.error("แก้ homography ไม่ได้ ลองคลิกใหม่")
                with col_m2:
                    if st.button("↩️ ย้อนกลับ 1 จุด (Undo)", key=f"undo_{mode_key}", disabled=len(pts_list) == 0):
                        pts_list.pop()
                        st.rerun()
                with col_m3:
                    if st.button("🗑️ ล้างทั้งหมด (Reset)", key=f"reset_{mode_key}", disabled=len(pts_list) == 0):
                        st.session_state[f"court_manual_pts_{mode_key}"] = []
                        st.rerun()
            except ImportError:
                st.warning("ต้องติดตั้ง streamlit-image-coordinates ก่อน: pip install streamlit-image-coordinates")
    
    st.header("2.5 Hit Event Log")
    st.write("วิเคราะห์การเคลื่อนที่ของลูกเทนนิสและพิกัดผู้เล่นเพื่อหาจังหวะการตี (Hit Events)")
    
    # Store hit events in session state
    if "hit_events" not in st.session_state:
        st.session_state.hit_events = None
        
    if st.button("📊 วิเคราะห์ Hit Events"):
        ball_bboxes = st.session_state.get("ball_bboxes")
        if ball_bboxes:
            with st.spinner("กำลังคำนวณ Kinematics และค้นหา Hit Events..."):
                import sys, importlib
                import loeuf_cv.hit_detection
                importlib.reload(loeuf_cv.hit_detection)
                from loeuf_cv.hit_detection import extract_ball_trajectory_kalman, detect_hit_events
                import cv2
                # force reload 
                # Get video metadata
                cap = cv2.VideoCapture(video_path)
                fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                # Extract and process
                traj = extract_ball_trajectory_kalman(ball_bboxes, n_frames)
                racket_bboxes = st.session_state.get("racket_bboxes")
                hits = detect_hit_events(traj, tracks, fps, fw, fh, racket_bboxes=racket_bboxes)
                # Enrich with bounce detection
                from loeuf_cv.bounce_detection import add_bounce_to_hits
                court_H = st.session_state.get("court_homography")
                hits = add_bounce_to_hits(hits, traj, court_H, fps)
                
                st.session_state.hit_events = hits
                st.session_state.ball_traj = traj
                st.success(f"วิเคราะห์เสร็จสิ้น: พบการตีทั้งหมด {len(hits)} ครั้ง")
        else:
            st.warning("ไม่มีข้อมูลลูกเทนนิส (โปรดใช้ YOLO11 + BoT-SORT ที่มี Custom Model ในการตรวจจับลูกเทนนิส)")
            
    if st.session_state.hit_events is not None:
        if len(st.session_state.hit_events) > 0:
            import pandas as pd
            df_hits = pd.DataFrame(st.session_state.hit_events)
            st.dataframe(df_hits, use_container_width=True)
            
            # Mini Court Map
            has_bounce_data = "bounce_court_x_m" in df_hits.columns and df_hits["bounce_court_x_m"].notna().any()
            if has_bounce_data:
                st.subheader("🎾 Bounce Position Map")
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import matplotlib.patches as patches
                
                fig, ax = plt.subplots(figsize=(5, 8), facecolor="#1a472a")
                ax.set_facecolor("#1a6b35")
                
                # Draw court outline (half court: 0 – 11.885m depth, ±4.115m width)
                court_w, court_h = 8.23, 23.77
                half_w = court_w / 2
                ax.add_patch(patches.Rectangle((-half_w, 0), court_w, court_h,
                    linewidth=2, edgecolor="white", facecolor="none"))
                # Service line
                ax.axhline(5.485, color="white", linewidth=1)
                ax.axhline(court_h - 5.485, color="white", linewidth=1)
                # Net
                ax.axhline(court_h / 2, color="white", linewidth=3)
                # Centre line
                ax.axvline(0, ymin=5.485/court_h, ymax=(court_h-5.485)/court_h,
                           color="white", linewidth=1)
                
                # Plot bounces
                valid_bounces = df_hits.dropna(subset=["bounce_court_x_m", "bounce_court_z_m"])
                for _, row in valid_bounces.iterrows():
                    color = "#FFD700" if row.get("bounce_in_court") else "red"
                    ax.scatter(row["bounce_court_x_m"], row["bounce_court_z_m"],
                               c=color, s=60, zorder=5, alpha=0.85)
                    ax.annotate(str(int(row["frame"])),
                                (row["bounce_court_x_m"], row["bounce_court_z_m"]),
                                textcoords="offset points", xytext=(4, 4),
                                fontsize=6, color="white")
                
                ax.set_xlim(-half_w - 1, half_w + 1)
                ax.set_ylim(-1, court_h + 1)
                ax.set_aspect("equal")
                ax.set_xlabel("Width (m)", color="white")
                ax.set_ylabel("Depth (m)", color="white")
                ax.tick_params(colors="white")
                ax.set_title(f"🎾 Bounce Positions ({len(valid_bounces)} bounces)", color="white")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                
                if not st.session_state.get("court_homography") is not None:
                    st.caption("⚠️ ไม่มี Court Homography — แสดงพิกัด pixel แทน meter ได้ทำ Calibration ก่อนจะได้พิกัดจริง")
            
            csv = df_hits.to_csv(index=False).encode('utf-8')
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("💾 ดาวน์โหลดเป็น CSV", data=csv, file_name="hit_events.csv", mime="text/csv")
            with col2:
                import json
                from loeuf_cv.schema_builder.builder import build_loeuf_schema
                from loeuf_cv.config import PipelineConfig
                import cv2
                
                cap = cv2.VideoCapture(video_path)
                fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                video_meta = {"width": fw, "height": fh, "total_frames": n_frames}
                cfg = PipelineConfig()
                
                try:
                    racket_keypoints = st.session_state.get("racket_keypoints")
                    loeuf_json_data = build_loeuf_schema(tracks, st.session_state.hit_events, fps, video_meta, cfg,
                                                          racket_keypoints=racket_keypoints)
                    json_str = json.dumps(loeuf_json_data, indent=2, ensure_ascii=False).encode('utf-8')
                    st.download_button("📦 Export Loeuf Schema (JSON)", data=json_str, file_name=f"{loeuf_json_data['session_metadata']['session_id']}.json", mime="application/json")
                except Exception as e:
                    st.error(f"Failed to build JSON: {e}")
                    
        else:
            st.info("ไม่พบจังหวะการตีที่เข้าข่ายในคลิปนี้")

    st.header("3. Overlay Video")
    apply_fill = st.checkbox("ใช้ kinematic gap-fill (occlusion_fill — เติมข้อมือ/ข้อเท้าที่หายช่วงสั้น)", value=True)
    draw_court = st.checkbox("วาดเส้นสนามจำลอง (Auto-detected Court)", value=False)
    
    # checkbox to draw hit events
    draw_hits = st.checkbox("แสดงเอฟเฟกต์ 💥 HIT! บนวิดีโอ (ถ้ามี Hit Events)", value=True)
    
    if st.button("🎬 Render overlay"):
        with st.spinner("กำลัง render วิดีโอ..."):
            display_tracks = tracks
            if apply_fill:
                display_tracks = [
                    PlayerTrack(t.track_id, t.role, kinematic_fill(t.pose), t.n_frames_tracked)
                    for t in tracks
                ]
            
            # Auto-detect court if requested
            court_homography = None
            if draw_court and tracks:
                from loeuf_cv.court_calibration import auto_detect_height_cm
                st.toast("กำลังสแกนหาเส้นสนามจำลอง...")
                # We just need the court_cal output
                _, court_cal = auto_detect_height_cm(video_path, tracks[0].pose, None, sample_frames=15)
                if court_cal.valid:
                    court_homography = court_cal.homography
                    st.toast("สแกนเส้นสนามสำเร็จ!")
                else:
                    st.toast(f"สแกนเส้นสนามไม่สำเร็จ: {court_cal.reason}")
            
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
                racket_keypoints = st.session_state.get("racket_keypoints")
                hit_events = st.session_state.get("hit_events") if draw_hits else None

                st.session_state.overlay_path = render_overlay_video(
                    video_path, display_tracks, player_labels,
                    ball_bboxes=ball_bboxes, racket_bboxes=racket_bboxes,
                    racket_keypoints=racket_keypoints,
                    court_homography=court_homography,
                    hit_events=hit_events
                )
            except Exception as e:
                st.error("เกิดข้อผิดพลาดในการ render วิดีโอ (FFMPEG error)")
                with st.expander("รายละเอียด Error"):
                    st.code(str(e))
                st.stop()

    if st.session_state.get("overlay_path"):
        st.video(st.session_state.overlay_path)
        st.caption("สี = track_id/label ตามตารางด้านบน · จุดทึบ = confidence ≥ 0.5 · จุดเทาจาง = confidence ต่ำ (เหมือน low_confidence flag ใน schema)")
