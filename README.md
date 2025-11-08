#  Occluded Person Reconstruction Pipeline (CSRT + Optical Flow + Pose Estimation)

This project reconstructs and visualizes human motion even when the person is **temporarily occluded** (e.g., by a bus).  
It combines **object tracking (CSRT + Optical Flow)**, **pose estimation (MediaPipe)**, and **background inpainting**.

---

##  Project Overview

| Stage | Script | Description |
|--------|---------|-------------|
| 1️ | `1_preprocess_bg.py` | Extract frames from the video and build a static background image using median blending. |
| 2️ | `2_Person_tracker_Data.py` | Tkinter GUI for semi-automatic tracking using CSRT before/after occlusion and Optical Flow during occlusion. Saves bounding boxes per frame to CSV. |
| 3️ | `3_pose_estimation.py` | Uses MediaPipe Pose to estimate and interpolate poses for each tracked bounding box, reconstructing occluded poses. |
| 4️ | `4_Pose_estimation_inpainted.py` | Same as step 3, but replaces occluded frames with the reconstructed **background image** (inpainted output). |

---

## Requirements

Install all dependencies before running:

```bash
pip install opencv-python mediapipe pillow numpy pandas
```

---

## Step-by-Step Workflow

### 1. Background Preprocessing (`1_preprocess_bg.py`)

- Loads your raw video (`Bus_crossing.mp4`).
- Extracts all frames and builds a **median background** using frames 60–110 (where the person isn’t visible).
- Saves:
  - `frames/` — extracted frames  
  - `background.jpg` — clean background image

**Output:**
```
frames/
background.jpg
```

---

### 2. Person Tracking + Occlusion Handling (`2_Person_tracker_Data.py`)

#### What it does
- Interactive Tkinter GUI for tracking one person across an occlusion.
- Uses:
  - **CSRT tracker** before and after occlusion.
  - **Lucas–Kanade Optical Flow** during occlusion (to estimate motion when person is hidden).

#### How it works
1. **Load video** → `Select Video`
2. **Set frame indices** (before and after occlusion) → `Set Frames & Play`
3. **Pause automatically** at `before_frame`  
   → draw a bounding box → click `Init Pre-Tracker`
4. When playback pauses after occlusion → draw new box → click `Init Post-Tracker`
5. Continue playback → click `Save Tracking CSV` at the end.

#### Output
- Generates a CSV file named `tracker_<video_name>.csv` containing:

```
frame_idx, x, y, w, h, method
```

where `method` ∈ {`CSRT-pre`, `OpticalFlow`, `CSRT-post`}

#### Algorithms used
- **CSRT Tracker** — robust appearance-based correlation filter tracker for visible frames.  
- **Optical Flow (Lucas–Kanade)** — tracks motion of feature points during occlusion.  
- The system switches:
  - CSRT → Optical Flow at `bus_enter_frame`
  - Optical Flow → CSRT at `bus_occlude_frame`

---

###  3. Pose Estimation (`3_pose_estimation.py`)

#### What it does
- Loads the CSV tracking data.
- Runs **MediaPipe Pose** inside each bounding box.
- Interpolates keypoints between visible frames to **fill occlusion gaps**.

#### Output
- `bus_crossing_pose_reconstructed.mp4` — original video + bounding boxes + skeleton overlay.
- Console output:
  ```
  Extracting poses from visible frames...
  Predicting missing poses...
  Rendering final video...
  ```

---

### 4. Pose Estimation + Inpainting (`4_Pose_estimation_inpainted.py`)

#### What it does
- Enhances the previous step by **replacing occluded frames** with the clean `background.jpg`.
- Reconstructs and overlays predicted poses and bounding boxes on top of the background.
- Performs **bounding-box + pose interpolation** across occlusion frames.

#### Output
- `bus_crossing_pose_reconstructed_inpainted.mp4` — final reconstruction video with occlusion removed.

---

## Algorithm Summary

| Stage | Technique | Purpose |
|--------|------------|----------|
| Background Median | `np.median()` blending | Creates clean background image without moving subjects |
| CSRT Tracker | Correlation-filter-based tracking | Tracks object when visible |
| Optical Flow (Lucas–Kanade) | Motion of pixel patches | Predicts motion during occlusion |
| Pose Estimation | MediaPipe Pose | Estimates human joint positions |
| Linear Interpolation | Numpy vector math | Smoothly fills missing frames |
| Inpainting | Background replacement | Removes occluding object visually |

---

## Output Files Summary

| File | Description |
|------|--------------|
| `background.jpg` | Clean median background |
| `tracker_<video>.csv` | Frame-wise bounding boxes |
| `bus_crossing_pose_reconstructed.mp4` | Pose reconstruction without inpainting |
| `bus_crossing_pose_reconstructed_inpainted.mp4` | Final inpainted output |

---

## Example Results

- **Green boxes:** Pre-occlusion tracking (CSRT)  
- **Yellow boxes:** During occlusion (Optical Flow)  
- **Blue boxes:** Post-occlusion tracking (CSRT)  
- **Red dots / green lines:** Estimated human pose.

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|--------|-----|
| Tracker drifts during occlusion | Too few visible points | Increase `maxCorners` or adjust optical flow window |
| Blank background | Wrong frame range in `1_preprocess_bg.py` | Choose frames where person is absent |
| Pose missing | MediaPipe failed to detect | Lower occlusion or interpolate between more frames |
| GUI crashes | Tkinter closed early | Run with Python ≥ 3.9 and keep window active |
