# TrackNetV2 Integration Design Spec

## 1. Objective
Integrate TrackNetV2 (specialized deep neural network for tennis ball tracking using 3-frame convolutional heatmaps) into the Loeuf Tennis CV pipeline as an optional, high-precision adapter module to reliably detect flying tennis balls, eliminate static ball false positives, and accurately measure ball trajectory, bounce points, and line calling.

## 2. Architecture & File Structure

```text
loeuf_cv/
└── tracknet/
    ├── __init__.py           # Exports track_ball_with_tracknet, TrackNetV2
    ├── model.py              # PyTorch TrackNetV2 Architecture (9-channel in -> 1-channel heatmap out)
    ├── weights.py            # Pre-trained weights downloader & loader
    └── tracker.py            # 3-frame sliding window inference & heatmap peak extraction
```

### Module Responsibilities:
1. `loeuf_cv/tracknet/model.py`:
   - Input: Tensor `(B, 9, H, W)` where $H=360, W=640$, concatenating 3 consecutive RGB frames ($t-1, t, t+1$).
   - Architecture: VGG-based Encoder-Decoder with Skip Connections and Batch Normalization.
   - Output: Heatmap `(B, 1, H, W)` with Sigmoid activation representing ball center probability.

2. `loeuf_cv/tracknet/weights.py`:
   - Manages downloading and caching pre-trained weights (`checkpoints/tracknetv2_tennis.pt`).
   - If offline or download fails, provides graceful error reporting without crashing the pose pipeline.

3. `loeuf_cv/tracknet/tracker.py`:
   - Implements `track_ball_with_tracknet(video_path: str, conf_threshold: float = 0.5, batch_size: int = 16, device: str = "cuda") -> BallObservations`.
   - Extracts coordinates $(x, y)$ from heatmap Gaussian blobs via `cv2.findContours` or `np.argmax`.
   - Returns `BallObservations(positions, confidence, fps)` matching `loeuf_cv/ball.py`.

## 3. Integration with Existing Pipeline & WebUI

1. **Adapter Interface (`loeuf_cv/ball.py`)**:
   - `BallObservations` is consumed by `fuse_impact(ts, keyframes, ball, config)` for precise impact timestamp refinement.
   - `build_ball_path(ball, ts, keyframes)` formats the VZ2 visualization payload.
   - `build_ball_block(ball)` generates the schema BL1–BL10 output.

2. **WebUI Integration (`webui/app.py` & `webui/yolo_track.py`)**:
   - Sidebar toggle: `use_tracknet = st.sidebar.checkbox("🎾 Use TrackNetV2 for Ball Tracking (High Accuracy)", value=False)`.
   - If checked: runs TrackNetV2 to extract ball positions; if unchecked or fallback: uses YOLO ball tracking.

## 4. Error Handling & Fallback
- If CUDA is unavailable: automatically falls back to CPU or standard YOLO ball tracking.
- If weights are missing and internet is unavailable: alerts user and gracefully falls back to YOLO ball tracking.
- Null invariant: If ball is undetected in specific frames, outputs `NaN` / `null` with `not_detectable` flag, preserving schema compliance.
