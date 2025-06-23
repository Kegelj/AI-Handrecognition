import cv2
import random
import string
import os
import time
from pathlib import Path
from pynput.keyboard import Controller, Key
from ultralytics import YOLO

# === Class ID to gesture name mapping ===
class_names = {
    1: "index",
    2: "index_pinky",
    3: "pinky",
    4: "thumb",
    5: "thumb_index",
    6: "piu"
}

# === Gesture to key mapping ===
key_map = {
    "index": Key.up,
    "pinky": Key.left,
    "thumb": Key.right,
    "piu": Key.space
}

# === Generate a random string for filenames ===
def rand_string(length):
    return "".join(random.choice(string.ascii_letters + string.digits) for _ in range(length))

# === Get number of frames in a video ===
def length_of_video(video_name):
    video_path = Path(__file__).resolve().parents[1] / "project_assets/videos" / video_name
    cap = cv2.VideoCapture(str(video_path))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return length

# === Extract frames from video, skipping every N frames ===
def extracting_frames(video_name, save_path, skip_frames=5):
    print(" Extracting video frames...")

    file_name = os.path.splitext(video_name)[0]
    total_frames = length_of_video(video_name)
    if total_frames == 0:
        print(" Video length is 0. Aborting.")
        return 0

    video_path = Path(__file__).resolve().parents[1] / "project_assets/videos" / video_name
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()

    count = 0
    rand_suffix = rand_string(3)

    # Save initial test frame
    test_file_path = f"{save_path}{file_name}_{rand_suffix}_{count}_TEST.jpg"
    cv2.imwrite(test_file_path, frame)
    if os.path.isfile(test_file_path):
        print(" Test frame saved. Continuing extraction...")

    # Main extraction loop
    count = 1
    while ret:
        ret, frame = cap.read()
        if ret and count % skip_frames == 0:
            filename = f"{save_path}{file_name}_{rand_suffix}_{count}.jpg"
            cv2.imwrite(filename, frame)
            print(f" Saved frame {count}: {filename}")
        count += 1

    cap.release()
    print(" All frames extracted successfully.")
    return 0

# === Live tracking with YOLO + keyboard control (throttled to 15 FPS) ===
def live_tracking_yolo():
    model = YOLO("model_output/epoch41.pt")
    keyboard = Controller()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" Failed to open webcam.")
        return

    print(" Hand gesture control started. Press 'q' to quit.")
    last_piu_time = 0  # Last time 'piu' (space) was triggered

    try:
        while True:
            loop_start = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False)[0]
            keys_pressed = set()
            current_time = time.time()

            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < 0.5:
                    continue

                label = class_names.get(cls_id, f"class {cls_id}")

                # Combine gestures → multiple keys
                if label == "index_pinky":
                    keys_pressed.update([Key.left, Key.up])
                elif label == "thumb_index":
                    keys_pressed.update([Key.right, Key.up])
                elif label == "piu":
                    if current_time - last_piu_time > 1.0:
                        keys_pressed.add(Key.space)
                        last_piu_time = current_time
                else:
                    key = key_map.get(label)
                    if key:
                        keys_pressed.add(key)

                # Draw bounding box + label
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Simulate key presses
            for key in [Key.up, Key.left, Key.right, Key.space]:
                if key in keys_pressed:
                    keyboard.press(key)
                else:
                    keyboard.release(key)

            cv2.imshow(" YOLO Hand Gesture Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Throttle to ~15 FPS
            elapsed = time.time() - loop_start
            time.sleep(max(0, (1 / 15) - elapsed))

    finally:
        cap.release()
        cv2.destroyAllWindows()

# === Entry point ===
if __name__ == "__main__":
    live_tracking_yolo()
