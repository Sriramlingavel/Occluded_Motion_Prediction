import cv2
import numpy as np
import os

# === STEP 1: LOAD VIDEO ===
video_path = 'road_dataset2.mp4'
cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video Info:")
print(f"  FPS: {fps} | Resolution: {width}x{height}")
print(f"  Total Frames: {total_frames} (~{total_frames/fps:.1f} seconds)")

# === STEP 2: CREATE OUTPUT FOLDER TO SAVE FRAMES ===
frames_dir = "frames"
os.makedirs(frames_dir, exist_ok=True)
print(f"\n✓ Saving all frames to folder: '{frames_dir}/'")

# === STEP 3: EXTRACT AND SAVE ALL FRAMES ===
print("\nExtracting all frames...")

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
for i in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        break
    frame_filename = os.path.join(frames_dir, f"frame_{i:04d}.jpg")
    cv2.imwrite(frame_filename, frame)

print(f"✓ Extracted {i+1} frames to '{frames_dir}/'")

# === STEP 4: ASK USER FOR BUS OCCLUSION RANGE ===
print("\nNow, to build the background:")
print("Please specify when the bus appears and disappears in the video.")
print("For example:  bus enters at frame 27 and leaves at frame 44.")

bus_start = int(input("Enter bus appearance start frame number: "))
bus_end = int(input("Enter bus disappearance end frame number: "))

# === STEP 5: SELECT FRAMES THAT DON'T INCLUDE THE BUS ===
frames_for_bg = []
print(f"\nBuilding background using frames excluding [{bus_start}-{bus_end}] ...")

for i in range(total_frames):
    if bus_start <= i <= bus_end:
        continue  # Skip frames where bus is present
    frame_path = os.path.join(frames_dir, f"frame_{i:04d}.jpg")
    frame = cv2.imread(frame_path)
    if frame is not None:
        frames_for_bg.append(frame)

# Use median blending to create clean background
background = np.median(frames_for_bg, axis=0).astype(np.uint8)

cv2.imwrite('background.jpg', background)
print(f"✓ Background model created from {len(frames_for_bg)} frames")
print("✓ Background image saved as 'background.jpg'")

cap.release()
print("\n✅ All frames extracted and background saved successfully.")
