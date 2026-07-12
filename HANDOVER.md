# Loeuf CV - Project Handover & Next Steps

This document summarizes the progress made regarding the SAM 2 memory crash issue and outlines the exact next steps for the next session or AI assistant.

## 1. What Has Been Accomplished (Completed)
- **Problem Fixed**: The Apple Silicon Memory Crash during SAM 2 tracking (loading >300 frames) has been mitigated.
- **Solution Implemented**: 
  - `webui/sam2_track.py` was rewritten to process videos in 150-frame chunks with a 1-frame overlap.
  - The tracking mask is successfully passed across chunk boundaries using `predictor.add_new_mask(..., mask=last_mask.squeeze() > 0)`.
  - Added zero-padding to frame extraction (`f"{i:05d}.jpg"`) to fix lexicographical sorting bugs.
- **Verification**: `scripts/test_sam2_chunk.py` was created and successfully tracked a synthetic 100-frame video across 30-frame chunks without crashing.
- **Git State**: All changes (including ignoring large model files) were squashed and pushed to the branch `feature/sam2-chunking`.

## 2. Next Immediate Steps (Priority 1 Remaining)

The core logic is done, but it needs to be connected to the user interface and the official SAM 2 repository.

### Step 2.1: UI Integration for SAM 2 (`webui/app.py`)
Currently, `app.py` uses the old `multi_person.py` background-subtraction tracking. We need to integrate the new `sam2_track.py` so we can test the tracking accuracy on real tennis videos.
- **Task**: 
  1. Modify `webui/app.py` (Streamlit UI).
  2. Add an interface (e.g., using `streamlit-image-coordinates` or a simple text input for X,Y coordinates) to let the user click on a player in the first frame of the uploaded video.
  3. Send the clicked coordinates (`ClickPrompt`) to the `track_player_with_sam2` function in `sam2_track.py`.
  4. Display the resulting tracked bounding boxes as an overlay video to verify accuracy.

### Step 2.2: Switch to Official SAM 2 Repo
Currently, the project uses a pip fork of SAM 2 (`sam2==1.1.0` in `requirements-sam2.txt`). 
- **Task**: 
  1. Change the dependency to the official Meta repository to ensure long-term stability and access to official updates.
  2. The new install command or requirement should point to `git+https://github.com/facebookresearch/sam2.git`.

## 3. Instructions for the Next AI Assistant
1. **Context**: You are working on a tennis tracking application (`Loeuf CV`). The user is currently on branch `feature/sam2-chunking`.
2. **Action**: Read this `HANDOVER.md` file. Start by executing Step 2.1 to integrate SAM 2 with the Streamlit UI (`webui/app.py`), then proceed to Step 2.2.
3. **Execution**: Do not re-implement the chunking logic in `sam2_track.py` as it is already working perfectly. Focus entirely on the UI and the dependency switch.
