import cv2
import random
import string
import os
from pathlib import Path
from pynput.keyboard import Controller, Key
from ultralytics import YOLO

# === YOLO-Klassennamen ===
class_names = {
    0: "nop",
    1: "index",
    2: "index_pinky",
    3: "pinky",
    4: "thumb",
    5: "thumb_index",
    6: "offen"
}

# === Tastenzuordnung für Handzeichen ===
key_map = {
    "index": Key.up,       # alternativ: 'w'
    "pinky": Key.left,     # alternativ: 'a'
    "thumb": Key.right,    # alternativ: 'd'
    "offen": Key.space     # entspricht Leertaste
}

# === Zufälliger Dateiname-Generator ===
def rand_string(length):
    return "".join(random.choice(string.ascii_letters + string.digits) for _ in range(length))

# === Länge des Videos ermitteln ===
def length_of_video(video_name):
    video_path = Path(__file__).resolve().parents[1] / "project_assets/videos" / video_name
    cap = cv2.VideoCapture(str(video_path))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return length

# === Frames extrahieren ===
def extracting_frames(video_name, save_path, skip_frames=5):
    print("*******EXTRACTING PHASE********")

    file_name_without_ext = os.path.splitext(video_name)[0]
    length = length_of_video(video_name)
    if length == 0:
        print("Length is 0, exiting extracting phase.")
        return 0

    video_path = Path(__file__).resolve().parents[1] / "project_assets/videos" / video_name
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()

    count = 0
    random_string = rand_string(3)

    test_file_path = f"{save_path}{file_name_without_ext}_{random_string}_{count}_TEST.jpg"
    cv2.imwrite(test_file_path, frame)
    if os.path.isfile(test_file_path):
        print("Saving Test Frame was Successful\nContinuing Extraction Phase")

    count = 1
    while ret:
        ret, frame = cap.read()
        if ret and count % skip_frames == 0:
            filename = f"{save_path}{file_name_without_ext}_{random_string}_{count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Frame {count} saved: {filename}")
        count += 1

    print("Videos fully saved.")
    cap.release()
    return 0

# === Live Tracking mit YOLO ===
def live_tracking_yolo():
    model = YOLO("model_output/best.pt")
    keyboard = Controller()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Konnte Kamera nicht öffnen.")
        return

    print("🚀 Kamera gestartet. Drücke 'q' zum Beenden.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False)[0]
            keys_pressed = set()

            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < 0.5:
                    continue

                label = class_names.get(cls_id, f"Klasse {cls_id}")
                key = key_map.get(label)
                if key:
                    keys_pressed.add(key)

                # Bounding Box zeichnen
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Alle relevanten Tasten überprüfen
            for key in [Key.up, Key.left, Key.right, Key.space]:
                if key in keys_pressed:
                    keyboard.press(key)
                else:
                    keyboard.release(key)

            cv2.imshow("YOLO Handzeichen", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

# === Hauptprogramm ===
if __name__ == "__main__":
    live_tracking_yolo()
