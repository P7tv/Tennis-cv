# Pseudo-Labeling Pipeline Design

## Purpose
To automatically generate training labels (pseudo-labels) for the Stroke Classifier by using the existing Hit Classifier to detect impact frames in unlabeled videos, and extracting the stroke type from the video filenames.

## Scope
- Target videos: All `.mov` and `.mp4` files in `_incoming/new_data/extracted/260816_set`, EXCLUDING any file containing "rally" in its name.
- Result location: `dataset/sessions/260816_set_auto` (videos and generated `.json` files).

## Process Flow

1. **File Migration & Filtering**
   - Read files from `_incoming/new_data/extracted/260816_set`.
   - Skip files containing "rally" (e.g., `IMG_03064-rally.mov`).
   - Copy the valid files to `dataset/sessions/260816_set_auto`.

2. **Feature Tracking (Cache Generation)**
   - Run `python tools/run_keyframe_benchmark.py track --dataset-root dataset` to ensure all copied videos have their YOLO/MediaPipe features extracted and cached.

3. **Hit Detection**
   - For each copied video, load the cached features.
   - Run the Hit Classifier (`checkpoints/hit_classifier.pkl`) over the cached features to detect `impact_frame` indices.
   - Run the Impact Refiner (`checkpoints/impact_refiner.pkl`) if necessary to snap to the precise audio-based or visual impact frame.

4. **Label Generation**
   - Parse the `stroke_type` from the filename (e.g., `IMG_03061-FH.mov` -> `FH`).
   - Construct a JSON object adhering to the standard label schema.
   - Generate default values for other fields:
     - `label_tier`: "core"
     - `usable`: true
     - `annotator_id`: "auto-pseudo"
     - `label_date`: current date
     - `impact_frame`: from Hit Detection
     - `stroke_type`: from filename
   - Save the JSON alongside the video file (e.g., `IMG_03061-FH_stroke_labels_auto-pseudo_2026-08-18.json`).

## Expected Outcome
The `dataset/sessions/260816_set_auto` folder will contain 16 videos and 16 perfectly formatted JSON label files. The system can then be fed into `tools/retrain_with_new_data.py` to seamlessly upgrade the Stroke Classifier without manual labeling effort.
