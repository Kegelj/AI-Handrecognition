import cv2
import mediapipe as mp
from pathlib import Path

IMAGE_DIR = Path("extracted_pngs")
PADDING = 0.1  # 10 %

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

def make_square(x_min, y_min, x_max, y_max, width, height):
    box_w = x_max - x_min
    box_h = y_max - y_min
    side = max(box_w, box_h) * (1 + PADDING)

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2

    x1 = int(max(cx - side / 2, 0))
    y1 = int(max(cy - side / 2, 0))
    x2 = int(min(cx + side / 2, width))
    y2 = int(min(cy + side / 2, height))
    return x1, y1, x2, y2

for image_path in IMAGE_DIR.glob("*.png"):
    img = cv2.imread(str(image_path))
    if img is None:
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

            # Quadrat um die Hand berechnen
            x1, y1, x2, y2 = make_square(x_min, y_min, x_max, y_max, width, height)

            # YOLO-Format: normierte xc, yc, w, h
            xc = (x1 + x2) / 2 / width
            yc = (y1 + y2) / 2 / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height

            label_path = image_path.with_suffix(".txt")
            with open(label_path, "w") as f:
                f.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
