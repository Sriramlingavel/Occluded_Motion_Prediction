"""
tk_csrt_tracker_native.py
Same as your version but displays the video at native size (1:1 pixel mapping)
so bounding box drawing aligns perfectly.

Dependencies:
pip install opencv-python pillow
"""

import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import os
import csv

# -------------------- Globals --------------------
cap = None
video_path = None
total_frames = 0
orig_W = orig_H = None

before_frame = None
after_frame = None
frames_set = False

panel = None
status_label = None
frame_label = None

current_frame_idx = 0
current_frame_bgr = None
current_display_img = None

# drawing
drawing = False
sx = sy = ex = ey = 0

# bounding boxes stored in original-video coords
bbox_before = None
bbox_after = None

# trackers
pre_tracker = None
post_tracker = None
pre_tracking_active = False
post_tracking_active = False

paused = False
playing = False

tracking_data = []  # stores per-frame (idx, x, y, w, h, method)


# -------------------- Helpers --------------------
def reset_state(full=False):
    global cap, video_path, total_frames, orig_W, orig_H
    global before_frame, after_frame, frames_set
    global current_frame_idx, current_frame_bgr, current_display_img
    global bbox_before, bbox_after
    global pre_tracker, post_tracker, pre_tracking_active, post_tracking_active
    global paused, playing

    if cap and cap.isOpened():
        cap.release()
    cap = None
    video_path = None
    total_frames = 0
    orig_W = orig_H = None

    before_frame = None
    after_frame = None
    frames_set = False

    current_frame_idx = 0
    current_frame_bgr = None
    current_display_img = None

    bbox_before = None
    bbox_after = None

    pre_tracker = None
    post_tracker = None
    pre_tracking_active = False
    post_tracking_active = False

    paused = False
    playing = False

    status_label.config(text="Status: Idle")
    frame_label.config(text="Frame: 0/0")
    clear_canvas()

def clear_canvas():
    global panel
    if panel:
        panel.config(image='')

def open_video():
    global cap, video_path, total_frames, orig_W, orig_H, panel

    path = filedialog.askopenfilename(title="Select video file",
                                      filetypes=[("Video files","*.mp4 *.avi *.mkv *.mov")])
    if not path:
        return
    reset_state()
    try:
        cap_local = cv2.VideoCapture(path)
        if not cap_local.isOpened():
            messagebox.showerror("Error","Cannot open video")
            return

        w = int(cap_local.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap_local.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap_local.get(cv2.CAP_PROP_FRAME_COUNT))

        # Commit globals
        globals()['cap'] = cap_local
        globals()['video_path'] = path
        globals()['orig_W'] = w
        globals()['orig_H'] = h
        globals()['total_frames'] = total

        # Resize window to video size dynamically
        panel.place_configure(width=w, height=h)
        root.geometry(f"{w + 260}x{h + 80}")

        status_label.config(text=f"Loaded: {path} ({w}x{h}, {total} frames)")
        frame_label.config(text=f"Frame: 0/{total}")

    except Exception as e:
        messagebox.showerror("Error", str(e))

def set_frames():
    global before_frame, after_frame, frames_set, current_frame_idx, paused, playing
    if cap is None:
        messagebox.showwarning("No video","Please select a video first")
        return
    try:
        b = int(entry_before.get())
        a = int(entry_after.get())
    except:
        messagebox.showwarning("Invalid","Enter integer frame indices")
        return
    if not (0 <= b < total_frames and 0 <= a < total_frames):
        messagebox.showwarning("Out of range","Frame indices must be within video length")
        return
    if b >= a:
        messagebox.showwarning("Invalid","before_frame must be less than after_frame")
        return
    before_frame = b
    after_frame = a

    global bus_enter_frame, bus_occlude_frame, mc_leave_frame
    try:
        bus_enter_frame = int(input("Enter one frame before bus enters the video: "))
        bus_occlude_frame = int(input("Enter one frame before bus occludes: "))
        mc_leave_frame = int(input("Enter one frame before the MC leaves the video: "))
    except ValueError:
        messagebox.showerror("Invalid input", "Enter valid integers for frame indices")
        return
    frames_set = True
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    current_frame_idx = 0
    paused = False
    playing = True
    status_label.config(text=f"Frames set: before={before_frame}, after={after_frame}. Playing...")
    root.after(10, play_loop)

def play_loop():
    global cap, current_frame_idx, current_frame_bgr, paused, playing
    global pre_tracking_active, post_tracking_active
    global pre_tracker, post_tracker
    global optical_flow_active, prev_gray, prev_points, bbox_before

    if cap is None or not frames_set or not playing:
        return
    
    if not root.winfo_exists():
        return

    if paused:
        root.after(100, play_loop)
        return

    ret, frame = cap.read()
    if not ret:
        status_label.config(text="Status: Video ended")
        playing = False
        return

    current_frame_bgr = frame.copy()
    current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    frame_to_display = frame.copy()
    global bus_enter_frame, bus_occlude_frame, mc_leave_frame

    # ------------- CSRT Pre-Tracker (before occlusion) -------------
    if pre_tracking_active and pre_tracker is not None and current_frame_idx < bus_enter_frame:
        ok, r = pre_tracker.update(frame_to_display)
        if ok:
            x, y, w, h = [int(v) for v in r]
            cv2.rectangle(frame_to_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            tracking_data.append((current_frame_idx, x, y, w, h, "CSRT-pre"))
            bbox_before = (x, y, w, h)
        else:
            cv2.putText(frame_to_display, "Pre-tracker lost", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # ------------- Optical Flow (during occlusion 19–26) -------------

    elif pre_tracking_active and bus_enter_frame <= current_frame_idx <= bus_occlude_frame:
        gray = cv2.cvtColor(frame_to_display, cv2.COLOR_BGR2GRAY)

        # Initialize points once at the start of occlusion
        if not globals().get("optical_flow_active", False):
            optical_flow_active = True
            prev_gray = cv2.cvtColor(current_frame_bgr, cv2.COLOR_BGR2GRAY)
            x, y, w, h = bbox_before
            prev_points = cv2.goodFeaturesToTrack(prev_gray[y:y+h, x:x+w], maxCorners=50,
                                                  qualityLevel=0.3, minDistance=5)
            if prev_points is not None:
                prev_points[:, 0, 0] += x
                prev_points[:, 0, 1] += y

        if prev_points is not None:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray,
                                                              prev_points, None,
                                                              winSize=(15, 15),
                                                              maxLevel=2)
            good_new = next_points[status == 1]
            good_old = prev_points[status == 1]

            # Estimate average motion
            dx = np.mean(good_new[:, 0] - good_old[:, 0])
            dy = np.mean(good_new[:, 1] - good_old[:, 1])
            x, y, w, h = bbox_before
            x, y = int(x + dx), int(y + dy)
            bbox_before = (x, y, w, h)
            cv2.rectangle(frame_to_display, (x, y), (x + w, y + h), (0, 200, 255), 2)
            tracking_data.append((current_frame_idx, x, y, w, h, "OpticalFlow"))

            cv2.putText(frame_to_display, "OpticalFlow tracking", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            prev_gray = gray.copy()
            prev_points = good_new.reshape(-1, 1, 2)

    # ------------- Resume Post-Tracker after occlusion -------------
    elif post_tracking_active and post_tracker is not None and bus_occlude_frame < current_frame_idx <= mc_leave_frame:
        ok, r = post_tracker.update(frame_to_display)
        if ok:
            x, y, w, h = [int(v) for v in r]
            cv2.rectangle(frame_to_display, (x, y), (x + w, y + h), (255, 0, 0), 2)
            tracking_data.append((current_frame_idx, x, y, w, h, "CSRT-post"))

        else:
            cv2.putText(frame_to_display, "Post-tracker lost", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # ---------------- Pause at before_frame or after_frame ----------------
    if current_frame_idx >= before_frame and not pre_tracking_active and not post_tracking_active:
        paused = True
        status_label.config(text=f"Paused at before_frame {before_frame}. Draw bbox and click 'Init Pre-Tracker'.")
        show_frame(frame_to_display)
        return

    if current_frame_idx >= after_frame and pre_tracking_active and not post_tracking_active:
        paused = True
        status_label.config(text=f"Paused at after_frame {after_frame}. Draw bbox and click 'Init Post-Tracker'.")
        show_frame(frame_to_display)
        return

    show_frame(frame_to_display)
    frame_label.config(text=f"Frame: {current_frame_idx}/{total_frames}")

    root.after(int(1000 / max(1, cap.get(cv2.CAP_PROP_FPS) or 24)), play_loop)

def show_frame(frame_bgr):
    global panel, current_display_img
    if not hasattr(root, 'winfo_exists') or not root.winfo_exists():
        return  # Root window was closed, skip rendering

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    current_display_img = frame_rgb.copy()
    try:
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
        panel.imgtk = imgtk
        panel.config(image=imgtk)
    except Exception as e:
        print("Skipped frame render (UI closed):", e)


# -------------------- Mouse drawing --------------------
def on_mouse_down(event):
    global drawing, sx, sy, ex, ey
    if not paused:
        return
    drawing = True
    sx, sy = event.x, event.y
    ex, ey = sx, sy

def on_mouse_move(event):
    global ex, ey
    if not drawing:
        return
    ex, ey = event.x, event.y
    temp = current_display_img.copy()
    cv2.rectangle(temp, (sx, sy), (ex, ey), (255,0,0), 2)
    imgtk = ImageTk.PhotoImage(image=Image.fromarray(temp))
    panel.imgtk = imgtk
    panel.config(image=imgtk)

def on_mouse_up(event):
    global drawing, bbox_before, bbox_after, sx, sy, ex, ey
    if not paused:
        return
    drawing = False
    ex, ey = event.x, event.y
    x1, y1 = min(sx, ex), min(sy, ey)
    x2, y2 = max(sx, ex), max(sy, ey)
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    if current_frame_idx >= before_frame and not pre_tracking_active:
        bbox_before = (x1, y1, w, h)
        status_label.config(text=f"Selected BEFORE bbox: {bbox_before}. Click 'Init Pre-Tracker'.")
    elif current_frame_idx >= after_frame and pre_tracking_active and not post_tracking_active:
        bbox_after = (x1, y1, w, h)
        status_label.config(text=f"Selected AFTER bbox: {bbox_after}. Click 'Init Post-Tracker'.")

# -------------------- Tracker init --------------------
def init_pre_tracker():
    global pre_tracker, pre_tracking_active, paused, playing
    if current_frame_bgr is None or bbox_before is None:
        messagebox.showwarning("Error","Draw bounding box first")
        return
    pre_tracker = cv2.TrackerCSRT_create()
    pre_tracker.init(current_frame_bgr, bbox_before)
    pre_tracking_active = True
    paused = False
    playing = True
    status_label.config(text="Pre-tracker initialized. Resuming playback.")
    root.after(10, play_loop)

def init_post_tracker():
    global post_tracker, post_tracking_active, paused, playing, bbox_after

    if current_frame_bgr is None:
        messagebox.showwarning("Error", "No frame loaded to initialize tracker.")
        return

    # Validate bbox_after and fall back if missing
    if bbox_after is None or not isinstance(bbox_after, (tuple, list)) or len(bbox_after) != 4:
        messagebox.showwarning("Warning", "bbox_after missing or invalid. Using last known bbox_before.")
        bbox_after = bbox_before
    bbox_after = tuple(map(int, bbox_after))


    post_tracker = cv2.TrackerCSRT_create()
    try:
        post_tracker.init(current_frame_bgr, bbox_after)
    except Exception as e:
        messagebox.showerror("Tracker Init Failed", f"Could not initialize post-tracker:\n{str(e)}")
        return

    post_tracking_active = True
    paused = False
    playing = True
    status_label.config(text="Post-tracker initialized. Resuming playback.")
    root.after(10, play_loop)

def restart():
    if video_path:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for var in ['before_frame','after_frame','frames_set','bbox_before','bbox_after',
                'pre_tracker','post_tracker','pre_tracking_active','post_tracking_active',
                'paused','playing','current_frame_idx']:
        globals()[var] = None if var in ('pre_tracker','post_tracker') else 0 if 'idx' in var else False
    status_label.config(text="Reset. Re-enter frames and Set Frames to replay.")
    frame_label.config(text=f"Frame: 0/{total_frames}")
    clear_canvas()


def save_tracking_csv():
    global tracking_data, video_path
    if not tracking_data:
        messagebox.showwarning("No Data", "No tracking data to save.")
        return
    base = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = f"tracker_{base}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "x", "y", "w", "h", "method"])
        writer.writerows(tracking_data)
    messagebox.showinfo("Saved", f"Tracking data saved to {csv_path}")

def replay_from_csv():
    global cap, video_path
    if not root.winfo_exists():
        return
    if not video_path:
        messagebox.showwarning("Warning", "Load a video first.")
        return

    base = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = f"tracker_{base}.csv"
    if not os.path.exists(csv_path):
        messagebox.showwarning("Missing File", f"{csv_path} not found.")
        return

    # Load all frame bboxes
    frame_boxes = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_boxes[int(row["frame_idx"])] = (
                int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"]), row["method"]
            )

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    status_label.config(text=f"Replaying from {csv_path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if idx in frame_boxes:
            x, y, w, h, method = frame_boxes[idx]
            color = (0,255,0) if "pre" in method else (0,200,255) if "Optical" in method else (255,0,0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, method, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        try:
            show_frame(frame)
        except tk.TclError:
            return  # Window closed, stop replay
        root.update()
        cv2.waitKey(int(1000 / max(1, cap.get(cv2.CAP_PROP_FPS) or 24)))
    status_label.config(text="Replay complete.")

# -------------------- GUI layout --------------------
root = tk.Tk()
root.title("CCTV CSRT Tracker (Before/After Occlusion)")

panel = tk.Label(root)
panel.place(x=10, y=10)

ctrl_x = 1030  # temporary default width, will resize dynamically
tk.Label(root, text="Controls", font=("Helvetica", 12, "bold")).place(x=ctrl_x, y=10)

btn_open = tk.Button(root, text="Select Video", width=20, command=open_video)
btn_open.place(x=ctrl_x, y=45)

tk.Label(root, text="Before-occ stop frame:").place(x=ctrl_x, y=90)
entry_before = tk.Entry(root, width=10)
entry_before.place(x=ctrl_x+130, y=90)
entry_before.insert(0, "1")

tk.Label(root, text="After-occ stop frame:").place(x=ctrl_x, y=120)
entry_after = tk.Entry(root, width=10)
entry_after.place(x=ctrl_x+130, y=120)
entry_after.insert(0, "50")

btn_set = tk.Button(root, text="Set Frames & Play", width=20, command=set_frames)
btn_set.place(x=ctrl_x, y=155)

btn_init_pre = tk.Button(root, text="Init Pre-Tracker", width=20, command=init_pre_tracker)
btn_init_pre.place(x=ctrl_x, y=200)

btn_init_post = tk.Button(root, text="Init Post-Tracker", width=20, command=init_post_tracker)
btn_init_post.place(x=ctrl_x, y=235)

btn_restart = tk.Button(root, text="Restart", width=20, command=restart)
btn_restart.place(x=ctrl_x, y=270)

btn_save_csv = tk.Button(root, text="Save Tracking CSV", width=20, command=save_tracking_csv)
btn_save_csv.place(x=ctrl_x, y=305)

btn_replay_csv = tk.Button(root, text="Replay from CSV", width=20, command=replay_from_csv)
btn_replay_csv.place(x=ctrl_x, y=340)

status_label = tk.Label(root, text="Status: Idle", anchor="w", justify="left")
status_label.place(x=ctrl_x, y=375)

frame_label = tk.Label(root, text="Frame: 0/0")
frame_label.place(x=ctrl_x, y=400)


panel.bind("<Button-1>", on_mouse_down)
panel.bind("<B1-Motion>", on_mouse_move)
panel.bind("<ButtonRelease-1>", on_mouse_up)

reset_state()
root.mainloop()
