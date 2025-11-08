import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# === CONFIG ===
video_path = "road_dataset2.mp4"
csv_path = input("Enter path to CSV file with bounding boxes: ")
output_path = "road_dataset2_pose_reconstructed.mp4"

# === LOAD CSV & VIDEO ===
df = pd.read_csv(csv_path)
cap = cv2.VideoCapture(video_path)
FPS = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# === MEDIA PIPE POSE ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)

# === VIDEO WRITER ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, FPS, (width, height))

# === Convert CSV to dict for quick lookup ===
bbox_dict = {int(row.frame_idx): (row.x, row.y, row.w, row.h) for _, row in df.iterrows()}
frame_indices = sorted(bbox_dict.keys())

# === Pose utilities ===
def get_pose_keypoints(frame, bbox):
    """Crop frame by bbox and run pose detection, return joint coords (in original frame scale)."""
    x, y, w, h = map(int, bbox)
    cropped = frame[y:y+h, x:x+w]
    if cropped.size == 0:
        return None

    rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)
    if not result.pose_landmarks:
        return None

    keypoints = []
    for lm in result.pose_landmarks.landmark:
        keypoints.append([x + lm.x * w, y + lm.y * h])
    return np.array(keypoints)


def draw_pose(frame, keypoints):
    """Draw pose skeleton given 2D keypoints."""
    connections = mp_pose.POSE_CONNECTIONS
    for i, j in connections:
        if i < len(keypoints) and j < len(keypoints):
            p1, p2 = keypoints[i], keypoints[j]
            cv2.line(frame, tuple(map(int, p1)), tuple(map(int, p2)), (0, 255, 0), 2)
    for p in keypoints:
        cv2.circle(frame, tuple(map(int, p)), 3, (0, 0, 255), -1)


# === STORAGE ===
poses = {}  # {frame_idx: keypoints}

# === 1️⃣ Extract poses for frames with bounding boxes ===
print(" Extracting poses from visible frames...")
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
for idx in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break
    if idx in bbox_dict:
        bbox = bbox_dict[idx]
        joints = get_pose_keypoints(frame, bbox)
        if joints is not None:
            poses[idx] = joints

cap.release()

# === 2️⃣ Predict missing poses via linear interpolation ===
print("📈 Predicting missing poses between visible frames...")
predicted_poses = poses.copy()

known_frames = sorted(list(poses.keys()))
for i in range(len(known_frames) - 1):
    f1, f2 = known_frames[i], known_frames[i + 1]
    k1, k2 = poses[f1], poses[f2]
    gap = f2 - f1 - 1
    if gap > 0:
        # Compute per-frame velocity
        velocity = (k2 - k1) / (gap + 1)
        for j in range(1, gap + 1):
            predicted_poses[f1 + j] = k1 + velocity * j

# === 3️⃣ Playback + Render video with bbox and predicted skeleton ===
print(" Rendering final video...")
cap = cv2.VideoCapture(video_path)
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw bbox if available
    if frame_idx in bbox_dict:
        x, y, w, h = map(int, bbox_dict[frame_idx])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Draw pose (actual or predicted)
    if frame_idx in predicted_poses:
        draw_pose(frame, predicted_poses[frame_idx])

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
pose.close()
print(f" Saved reconstructed pose video to {output_path}")