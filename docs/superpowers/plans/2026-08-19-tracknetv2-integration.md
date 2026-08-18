# TrackNetV2 Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate TrackNetV2 as a modular, high-accuracy ball tracking adapter that produces `BallObservations` for impact fusion, bounce detection, and trajectory visualization.

**Architecture:** A standalone package `loeuf_cv/tracknet/` containing the PyTorch neural network model, pre-trained weights loader, and sliding-window heatmap tracker that plugs directly into `loeuf_cv/ball.py` and provides a WebUI toggle.

**Tech Stack:** PyTorch, Torchvision, OpenCV, NumPy, Streamlit.

## Global Constraints
- Input resolution: $640 \times 360$ (RGB), concatenated 3-frame window `(B, 9, 360, 640)`.
- Output: Heatmap `(B, 1, 360, 640)` converted to normalized $(x, y)$ in $[0, 1]$.
- Compatible with `BallObservations` dataclass in `loeuf_cv/ball.py`.
- Must not break existing 132 pytest test cases.

---

### Task 1: TrackNetV2 Model Architecture (`loeuf_cv/tracknet/model.py`)

**Files:**
- Create: `loeuf_cv/tracknet/__init__.py`
- Create: `loeuf_cv/tracknet/model.py`
- Test: `tests/test_tracknet.py`

**Interfaces:**
- Produces: `TrackNetV2(nn.Module)`: takes `torch.Tensor` of shape `(B, 9, 360, 640)` and returns `torch.Tensor` of shape `(B, 1, 360, 640)`.

- [ ] **Step 1: Write the failing test**
Create `tests/test_tracknet.py`:
```python
import torch
import pytest
from loeuf_cv.tracknet.model import TrackNetV2

def test_tracknet_forward_pass_shape():
    model = TrackNetV2()
    dummy_input = torch.randn(2, 9, 360, 640)
    output = model(dummy_input)
    assert output.shape == (2, 1, 360, 640)
    assert (output >= 0.0).all() and (output <= 1.0).all()
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_tracknet.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'loeuf_cv.tracknet')

- [ ] **Step 3: Implement TrackNetV2 PyTorch model**
Create `loeuf_cv/tracknet/model.py`:
```python
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class TrackNetV2(nn.Module):
    def __init__(self, in_channels=9, out_channels=1):
        super().__init__()
        # Encoder
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = ConvBlock(128, 256)
        self.conv6 = ConvBlock(256, 256)
        self.conv7 = ConvBlock(256, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv8 = ConvBlock(256, 512)
        self.conv9 = ConvBlock(512, 512)
        self.conv10 = ConvBlock(512, 512)

        # Decoder
        self.upsample1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv11 = ConvBlock(512, 256)
        self.conv12 = ConvBlock(256, 256)
        self.conv13 = ConvBlock(256, 256)

        self.upsample2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv14 = ConvBlock(256, 128)
        self.conv15 = ConvBlock(128, 128)

        self.upsample3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv16 = ConvBlock(128, 64)
        self.conv17 = ConvBlock(64, 64)

        self.conv18 = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)

        x = self.upsample1(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)

        x = self.upsample2(x)
        x = self.conv14(x)
        x = self.conv15(x)

        x = self.upsample3(x)
        x = self.conv16(x)
        x = self.conv17(x)

        x = self.conv18(x)
        return self.sigmoid(x)
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_tracknet.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add loeuf_cv/tracknet/ tests/test_tracknet.py
git commit -m "feat(tracknet): implement TrackNetV2 neural network architecture"
```

---

### Task 2: Weights Loader & Heatmap Coordinate Extractor (`loeuf_cv/tracknet/weights.py`, `tracker.py`)

**Files:**
- Create: `loeuf_cv/tracknet/weights.py`
- Create: `loeuf_cv/tracknet/tracker.py`
- Test: `tests/test_tracknet.py`

**Interfaces:**
- `load_tracknet_model(model_path: str | None, device: str) -> TrackNetV2`
- `extract_ball_coordinates(heatmap: np.ndarray, threshold: float = 0.5) -> tuple[float, float, float] | None`
- `track_ball_with_tracknet(video_path: str, ...) -> BallObservations`

- [ ] **Step 1: Write test for coordinate extraction and tracker**
Add to `tests/test_tracknet.py`:
```python
import numpy as np
from loeuf_cv.tracknet.tracker import extract_ball_coordinates

def test_extract_ball_coordinates():
    # Mock Gaussian heatmap centered at (100, 200) on 360x640 heatmap
    hm = np.zeros((360, 640), dtype=np.float32)
    hm[200, 100] = 1.0
    res = extract_ball_coordinates(hm, threshold=0.5)
    assert res is not None
    x_norm, y_norm, conf = res
    assert pytest.approx(x_norm, abs=0.01) == 100 / 640.0
    assert pytest.approx(y_norm, abs=0.01) == 200 / 360.0
    assert conf >= 0.5
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_tracknet.py -k test_extract_ball_coordinates -v`
Expected: FAIL

- [ ] **Step 3: Implement tracker coordinate extraction and model loading**
Implement `loeuf_cv/tracknet/tracker.py` and `loeuf_cv/tracknet/weights.py`.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_tracknet.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add loeuf_cv/tracknet/ tests/test_tracknet.py
git commit -m "feat(tracknet): add weights loader, coordinate extractor, and batch tracker"
```

---

### Task 3: WebUI Toggle & Ball Path Integration (`webui/app.py`, `webui/yolo_track.py`)

**Files:**
- Modify: `webui/yolo_track.py`
- Modify: `webui/app.py`

**Interfaces:**
- Add `use_tracknet: bool = False` argument to `track_players_with_yolo`.
- If `use_tracknet` is True, invoke `track_ball_with_tracknet` to replace or refine ball trajectory.
- Add checkbox in `webui/app.py` sidebar.

- [ ] **Step 1: Add parameter and branch to `webui/yolo_track.py`**
- [ ] **Step 2: Add sidebar UI control to `webui/app.py`**
- [ ] **Step 3: Verify with end-to-end integration test**
- [ ] **Step 4: Run full test suite to ensure 0 regressions**
Run: `pytest tests/`
Expected: 133+ passed, 0 failed.

- [ ] **Step 5: Commit**
```bash
git add webui/app.py webui/yolo_track.py
git commit -m "feat(webui): add TrackNetV2 toggle and ball tracking integration"
```
