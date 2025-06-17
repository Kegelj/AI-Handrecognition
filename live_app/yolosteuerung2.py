import cv2
import random
import string
import os
from pathlib import Path
import time
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

# === Zufälliger Dateiname-Generator ===
def rand_string(length):
    return "".join(random.choice(string.ascii_letters + string.digits) for _ in range(length))

# === Live Tracking mit YOLO Steuerung ===
def live_tracking_yolo():
    model = YOLO("model_output/best.pt")
    keyboard = Controller()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FPS, 20)
    print("🚀 Kamera gestartet. Drücke 'q' zum Beenden.")

    cv2.namedWindow("YOLO Handsteuerung", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO Handsteuerung", 640, 480)

    last_center = None
    press_duration = 0.5  # Sekunden
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Zeit bis Taste losgelassen wird
    key_expiry = {Key.left: 0, Key.right: 0, Key.up: 0, Key.space: 0}
    key_state = {Key.left: False, Key.right: False, Key.up: False, Key.space: False}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            now = time.time()
            results = model(frame, verbose=False)[0]
            best_box, best_label, max_conf = None, None, 0

            for box in results.boxes:
                conf = float(box.conf[0])
                if conf > max_conf:
                    max_conf = conf
                    best_box = box
                    best_label = class_names.get(int(box.cls[0]), "unknown")

            if best_box is not None:
                x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                if last_center is not None:
                    dx = (cx - last_center[0]) / img_w
                    dy = (cy - last_center[1]) / img_h
                    if dx < -0.03:
                        key_expiry[Key.left] = max(key_expiry[Key.left], now + press_duration)
                    elif dx > 0.03:
                        key_expiry[Key.right] = max(key_expiry[Key.right], now + press_duration)
                    if dy < -0.03:
                        key_expiry[Key.up] = max(key_expiry[Key.up], now + press_duration)
                last_center = (cx, cy)

                if best_label in ["offen", "index"]:
                    key_expiry[Key.space] = max(key_expiry[Key.space], now + press_duration)

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, best_label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Tastenstatus prüfen und ändern
            for key in key_expiry:
                if now < key_expiry[key]:
                    if not key_state[key]:
                        keyboard.press(key)
                        key_state[key] = True
                else:
                    if key_state[key]:
                        keyboard.release(key)
                        key_state[key] = False

            cv2.imshow("YOLO Handsteuerung", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        for key in key_state:
            if key_state[key]:
                keyboard.release(key)

# === Hauptprogramm ===
if __name__ == "__main__":
    live_tracking_yolo()
