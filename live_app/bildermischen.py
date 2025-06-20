import cv2
import os
import random
from pathlib import Path

# === Pfade ===
path_x_root = Path("training/bilderx")       # Hauptordner mit Unterordnern voller Handbilder
path_y = Path("training/bildery")            # große Hintergrundbilder (1920x1080)
output_root = Path("training/endtime_bilder")
output_root.mkdir(parents=True, exist_ok=True)

# === Hintergrundbilder laden ===
images_y = list(path_y.glob("*.png")) + list(path_y.glob("*.jpg")) + list(path_y.glob("*.jpeg"))
if not images_y:
    raise RuntimeError(" Keine Hintergrundbilder in 'bildery' gefunden!")

# === Alle Unterordner durchgehen ===
for subfolder in path_x_root.iterdir():
    if not subfolder.is_dir():
        continue

    # Zielordner erstellen
    output_dir = output_root / subfolder.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Alle Handbilder im aktuellen Unterordner
    images_x = list(subfolder.glob("*.png")) + list(subfolder.glob("*.jpg")) + list(subfolder.glob("*.jpeg"))
    print(f" Verarbeite Ordner: {subfolder.name} mit {len(images_x)} Bildern")

    for idx, hand_img_path in enumerate(images_x):
        # Handbild laden
        hand_img = cv2.imread(str(hand_img_path), cv2.IMREAD_UNCHANGED)
        if hand_img is None:
            continue

        # Zufälliges Hintergrundbild wählen
        bg_img_path = random.choice(images_y)
        bg_img = cv2.imread(str(bg_img_path))
        if bg_img is None:
            continue

        bg_h, bg_w = bg_img.shape[:2]
        hand_h, hand_w = hand_img.shape[:2]
        if hand_h > bg_h or hand_w > bg_w:
            continue

        # Zufällige Position
        max_x = bg_w - hand_w
        max_y = bg_h - hand_h
        x_offset = random.randint(0, max_x)
        y_offset = random.randint(0, max_y)

        # Einfügen
        if hand_img.shape[2] == 4:
            alpha = hand_img[:, :, 3] / 255.0
            for c in range(3):
                bg_img[y_offset:y_offset+hand_h, x_offset:x_offset+hand_w, c] = (
                    alpha * hand_img[:, :, c] +
                    (1 - alpha) * bg_img[y_offset:y_offset+hand_h, x_offset:x_offset+hand_w, c]
                )
        else:
            bg_img[y_offset:y_offset+hand_h, x_offset:x_offset+hand_w] = hand_img

        # Speichern
        out_path = output_dir / f"{subfolder.name}_{idx:04d}.jpg"
        cv2.imwrite(str(out_path), bg_img)

print(" Alle Bilder aus allen Unterordnern verarbeitet.")
