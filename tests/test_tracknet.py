import torch
import pytest
import numpy as np

def test_tracknet_forward_pass_shape():
    from loeuf_cv.tracknet.model import TrackNetV2
    model = TrackNetV2()
    # Batch=2, Channels=9 (3 RGB frames), H=360, W=640
    dummy_input = torch.randn(2, 9, 360, 640)
    output = model(dummy_input)
    
    assert output.shape == (2, 1, 360, 640), f"Expected (2, 1, 360, 640) but got {output.shape}"
    assert (output >= 0.0).all() and (output <= 1.0).all(), "Heatmap outputs must be activated by sigmoid [0, 1]"

def test_extract_ball_coordinates():
    from loeuf_cv.tracknet.tracker import extract_ball_coordinates
    # Mock Gaussian heatmap centered at (100, 200) on 360x640 heatmap
    hm = np.zeros((360, 640), dtype=np.float32)
    hm[200, 100] = 1.0
    res = extract_ball_coordinates(hm, threshold=0.5)
    assert res is not None
    x_norm, y_norm, conf = res
    assert pytest.approx(x_norm, abs=0.01) == 100 / 640.0
    assert pytest.approx(y_norm, abs=0.01) == 200 / 360.0
    assert conf >= 0.5
