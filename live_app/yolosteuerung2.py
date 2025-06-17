import cv2
import random
import string
import os
from pathlib import Path
from pynput.keyboard import Controller
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

# === Zufälliger Dateiname-Generator ===
def rand_string(length):
    return "".join(random.choice(string.ascii_letters + string.digits) for _ in range(length))

# === Länge des Videos ermitteln ===
def length_of_video(video_name):
    video_path = Path(__file__).resolve().parents[1] / "project_assets/videos" / video_name
    cap = cv2.VideoCapture(str(video_path))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
    else:
        print("Videos fully saved.")
    cap.release()
    return 0

# === Live Tracking mit YOLO Steuerung ===
def live_tracking_yolo():
    model = YOLO("model_output/handzeichen_best.pt")
    keyboard = Controller()

    cap = cv2.VideoCapture(0)
    print("🚀 Kamera gestartet. Drücke 'q' zum Beenden.")

    last_center = None  # (x, y) der vorherigen Handposition
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False)[0]
            keys_pressed = set()

            max_conf = 0
            best_box = None
            best_label = None

            # Beste Box mit höchster Konfidenz nehmen
            for box in results.boxes:
                conf = float(box.conf[0])
                if conf > max_conf:
                    max_conf = conf
                    best_box = box
                    best_label = class_names.get(int(box.cls[0]), "unknown")

            if best_box is not None:
                # Box-Mitte berechnen
                x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                # Bewegung analysieren (verglichen mit letzter Position)
                if last_center is not None:
                    dx = (cx - last_center[0]) / img_w
                    dy = (cy - last_center[1]) / img_h

                    if dx < -0.03:  # nach links bewegt
                        keys_pressed.add('a')
                    elif dx > 0.03:  # nach rechts bewegt
                        keys_pressed.add('d')

                    if dy < -0.03:  # nach oben bewegt
                        keys_pressed.add('w')

                last_center = (cx, cy)

                # Space-Taste bei "offen" oder "index"
                if best_label in ["offen", "index"]:
                    keys_pressed.add('space')

                # Bounding Box & Label anzeigen
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, best_label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Tasten setzen
            for key in ['w', 'a', 'd', 'space']:
                if key in keys_pressed:
                    keyboard.press(key)
                else:
                    keyboard.release(key)

            cv2.imshow("YOLO Handsteuerung", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


# === Hauptprogramm ===
if __name__ == "__main__":
    live_tracking_yolo()
