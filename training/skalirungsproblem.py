import cv2
import numpy as np
from pathlib import Path

# --- Konfiguration ---
IMAGE_DIR = Path("training/alle_daten_alt")
RESULT_DIR = Path("training/scaled_daten")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = 256
INTERPOLATION = cv2.INTER_CUBIC

def smooth_downscale(image, target_size, step_factor=0.75):
    """
    Skaliert ein Bild sanft in mehreren Schritten auf target_size x target_size.
    """
    h, w = image.shape[:2]
    while min(h, w) * step_factor > target_size:
        new_w, new_h = int(w * step_factor), int(h * step_factor)
        image = cv2.resize(image, (new_w, new_h), interpolation=INTERPOLATION)
        w, h = new_w, new_h
        print(f"  Zwischenschritt: {w}x{h}")
    final_img = cv2.resize(image, (target_size, target_size), interpolation=INTERPOLATION)
    return final_img

# Alle Bilddateien (jpg, jpeg, png) rekursiv durchsuchen
image_paths = list(IMAGE_DIR.rglob("*.[jp][pn]g"))
print(f"Gefundene Bilder: {len(image_paths)}")

for image_path in image_paths:
    # Relativer Pfad zum Ursprungsverzeichnis
    rel_path = image_path.relative_to(IMAGE_DIR)
    print(f"\nVerarbeite: {rel_path}")

    # Bild laden
    img = cv2.imread(str(image_path))
    if img is None:
        print("  Bild konnte nicht geladen werden.")
        continue

    # Sanft runterskalieren
    final_img = smooth_downscale(img, TARGET_SIZE, step_factor=0.75)

    # Zielverzeichnis in scaled_data anlegen (gleicher Unterordner wie original)
    target_dir = RESULT_DIR / rel_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    # Neuen Dateinamen setzen
    output_path = target_dir / f"{image_path.stem}_scaled_256.jpg"
    cv2.imwrite(str(output_path), final_img)
    print(f"  Gespeichert: {output_path.relative_to(RESULT_DIR)}")

print("\nAlle Bilder wurden sanft skaliert und in scaled_data in identischer Ordnerstruktur gespeichert!")
