import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model

# --- Pfade ---
MODEL_PATH = "model_output/handbox_model_epoch_120.h5"
IMAGE_DIR = Path("training/realtest")
RESULT_DIR = Path("training/resultat_bbox")
IMG_SIZE = (128, 128)

# Zielordner anlegen
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Modell laden
model = load_model(MODEL_PATH, compile=False)
print("Modell erfolgreich geladen.")

# Alle Bilder im Ordner durchgehen
image_paths = list(IMAGE_DIR.glob("*.jpg")) + list(IMAGE_DIR.glob("*.png"))
print(f"Gefundene Bilder: {[p.name for p in image_paths]}")

for image_path in image_paths:
    print(f"Verarbeite: {image_path.name}")
    original_img = cv2.imread(str(image_path))
    if original_img is None:
        print("  Bild konnte nicht geladen werden.")
        continue

    # Bild vorverarbeiten
    img_resized = cv2.resize(original_img, IMG_SIZE)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb / 255.0
    input_tensor = np.expand_dims(img_normalized, axis=0)

    # Vorhersage
    pred = model.predict(input_tensor)[0]
    class_prob = pred[0]

    # Wenn class_prob < 0.5, kein Objekt erkannt
    if class_prob < 0.5:
        print("  Kein Objekt erkannt.")
        continue

    # Bounding Box ausgeben
    x_center, y_center, w_box, h_box = pred[1:]
    h, w = original_img.shape[:2]

    # Koordinaten umrechnen
    x_min = int((x_center - w_box / 2) * w)
    x_max = int((x_center + w_box / 2) * w)
    y_min = int((y_center - h_box / 2) * h)
    y_max = int((y_center + h_box / 2) * h)

    # Bounding Box zeichnen
    cv2.rectangle(original_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.putText(original_img, f"Prob: {class_prob:.2f}", (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Ergebnis speichern
    output_path = RESULT_DIR / image_path.name
    success = cv2.imwrite(str(output_path), original_img)
    if success:
        print(f"  Gespeichert: {output_path.name}")
    else:
        print(f"  Fehler beim Speichern: {output_path}")

print("Alle Bilder verarbeitet und gespeichert.")
