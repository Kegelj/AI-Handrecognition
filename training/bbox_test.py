import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model

# --- Pfade ---
MODEL_PATH = "model_output/handbox_model_epoch_10.h5"
IMAGE_DIR = Path("training/realtest")
RESULT_DIR = Path("training/resultat_bbox")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (256, 256)
INTERPOLATION = cv2.INTER_CUBIC  # Sanft

# --- Sanftes Runterskalieren ---
def smooth_downscale(image, target_size=256, step_factor=0.75):
    h, w = image.shape[:2]
    while min(h, w) * step_factor > target_size:
        new_w, new_h = int(w * step_factor), int(h * step_factor)
        image = cv2.resize(image, (new_w, new_h), interpolation=INTERPOLATION)
        w, h = new_w, new_h
        print(f"  Zwischenschritt: {w}x{h}")
    final_img = cv2.resize(image, (target_size, target_size), interpolation=INTERPOLATION)
    return final_img

# Modell laden
model = load_model(MODEL_PATH, compile=False)
print("Modell erfolgreich geladen.")

# Alle Bilder im Ordner durchgehen
image_paths = list(IMAGE_DIR.glob("*.jpg")) + list(IMAGE_DIR.glob("*.png"))
print(f"Gefundene Bilder: {[p.name for p in image_paths]}")

for image_path in image_paths:
    print(f"\nVerarbeite: {image_path.name}")
    original_img = cv2.imread(str(image_path))
    if original_img is None:
        print("  Bild konnte nicht geladen werden.")
        continue

    # Sanftes Runterskalieren auf 256x256
    scaled_img = smooth_downscale(original_img, target_size=IMG_SIZE[0])

    # Vorverarbeiten für Modell
    img_rgb = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb / 255.0
    input_tensor = np.expand_dims(img_normalized, axis=0)

    # Vorhersage
    pred = model.predict(input_tensor, verbose=0)[0]
    class_prob = pred[0]

    if class_prob < 0.5:
        print("  Kein Objekt erkannt.")
        continue

    # Bounding Box berechnen im skalierten Bild
    x_center, y_center, w_box, h_box = pred[1:]
    h, w = IMG_SIZE

    x_min = int((x_center - w_box / 2) * w)
    x_max = int((x_center + w_box / 2) * w)
    y_min = int((y_center - h_box / 2) * h)
    y_max = int((y_center + h_box / 2) * h)

    # Bounding Box zeichnen
    cv2.rectangle(scaled_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.putText(scaled_img, f"Prob: {class_prob:.2f}", (x_min, y_min - 10),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Ergebnis speichern
    output_path = RESULT_DIR / image_path.name
    success = cv2.imwrite(str(output_path), scaled_img)
    if success:
        print(f"  Gespeichert: {output_path.name}")
    else:
        print(f"  Fehler beim Speichern: {output_path}")

print("\nAlle Bilder sanft auf 256x256 skaliert, Bounding Box eingezeichnet und gespeichert.")
