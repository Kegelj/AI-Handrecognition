import cv2
import mediapipe as mp
from pathlib import Path

# 🔧 Parameter
PADDING_RATIO = 0.15
WIDEN_RATIO = 0.2
THUMB_I_DIR = Path("training/alle_daten/thumb_i")

# 📦 Feste Klassen-ID für "thumb"
CLASS_ID = 4  # wie im Hauptskript

# 🤖 MediaPipe Hands initialisieren
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

def adjust_bbox_width_only(x_min, y_min, x_max, y_max, width, height,
                         padding_ratio=0.15, widen_ratio=0.2):
    box_w = x_max - x_min
    box_h = y_max - y_min
    pad_h = box_h * padding_ratio / 2
    widen = box_w * widen_ratio / 2
    x1 = int(max(x_min - widen, 0))
    y1 = int(max(y_min - pad_h, 0))
    x2 = int(min(x_max + widen, width))
    y2 = int(min(y_max + pad_h, height))
    return x1, y1, x2, y2

for image_path in THUMB_I_DIR.rglob("*.jpg"):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"⚠️ Fehler beim Laden: {image_path}")
        continue

    height, width = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_coords = [lm.x * width for lm in hand_landmarks.landmark]
            y_coords = [lm.y * height for lm in hand_landmarks.landmark]

            x_min = min(x_coords)
            y_min = min(y_coords)
            x_max = max(x_coords)
            y_max = max(y_coords)

            x1, y1, x2, y2 = adjust_bbox_width_only(x_min, y_min, x_max, y_max, width, height,
                                                    padding_ratio=PADDING_RATIO, widen_ratio=WIDEN_RATIO)

            xc = (x1 + x2) / 2 / width
            yc = (y1 + y2) / 2 / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height

            label_path = image_path.with_suffix(".txt")
            with open(label_path, "w") as f:
                f.write(f"{CLASS_ID} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                print(f"✅ Label geschrieben: {label_path} für thumb_i")

print("🎉 Fertig! Alle Labels für 'thumb_i' erstellt.")
