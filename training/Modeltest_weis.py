import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model

# Pfade
MODEL_PATH = "model_output/best_hand_sign_model.h5"
IMAGE_DIR = Path("training/realtest")
RESULT_DIR = Path("training/resultat")
IMG_SIZE = (128, 128)
NUM_CLASSES = 6

# Labels für Klassen
CLASS_LABELS = {
    0: "index",
    1: "index_pinky",
    2: "pinky",
    3: "thumb",
    4: "nichts erkannt",
    5: "tumb_index"
}

# Zielordner anlegen
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Modell laden
model = load_model(MODEL_PATH, compile=False)

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

    # Vorhersage (Klasse + BBox)
    pred_class, pred_bbox = model.predict(input_tensor)
    pred_class_id = int(np.argmax(pred_class[0]))
    pred_class_prob = float(np.max(pred_class[0]))

    # Label-Text generieren
    label_text = f"{CLASS_LABELS[pred_class_id]} ({pred_class_prob:.2f})"
    cv2.putText(original_img, label_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Bounding Box zeichnen (nur wenn nicht "nichts erkannt")
    if pred_class_id != 4:
        # Umwandeln von (x_center, y_center, w, h) → (x_min, y_min, x_max, y_max)
        x_center, y_center, w_box, h_box = pred_bbox[0]
        x_min = x_center - w_box / 2
        x_max = x_center + w_box / 2
        y_min = y_center - h_box / 2
        y_max = y_center + h_box / 2

        # Koordinaten von [0,1] auf Bildgröße skalieren
        h, w = original_img.shape[:2]
        x_min = int(x_min * w)
        x_max = int(x_max * w)
        y_min = int(y_min * h)
        y_max = int(y_max * h)

        # Bounding Box zeichnen
        cv2.rectangle(original_img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    # Ergebnis speichern
    output_path = RESULT_DIR / image_path.name
    success = cv2.imwrite(str(output_path), original_img)
    if success:
        print(f"  Gespeichert: {output_path.name}")
    else:
        print(f"  Fehler beim Speichern: {output_path}")

print("Alle Bilder verarbeitet und gespeichert.")
