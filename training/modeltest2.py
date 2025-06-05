import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model

#  Pfade
MODEL_PATH = "model_output/best_hand_sign_model.h5" #path wo das model ist
IMAGE_DIR = Path("training/realtest") # path wo die daten sind die du testen willst
RESULT_DIR = Path("training/resultat") # path wo die bilder gespeichert werden
IMG_SIZE = (128, 128)  # Muss zum Modell passen
NUM_CLASSES = 6

#  Zielordner anlegen
RESULT_DIR.mkdir(parents=True, exist_ok=True)

#  Modell laden
model = load_model(MODEL_PATH, compile=False)

#  Alle Bilder im Ordner durchgehen
image_paths = list(IMAGE_DIR.glob("*.jpg")) + list(IMAGE_DIR.glob("*.png"))
print(f" Gefundene Bilder: {[p.name for p in image_paths]}")

for image_path in image_paths:
    print(f" Verarbeite: {image_path.name}")
    original_img = cv2.imread(str(image_path))
    if original_img is None:
        print("  Bild konnte nicht geladen werden.")
        continue

    #  Bild vorverarbeiten
    img_resized = cv2.resize(original_img, IMG_SIZE)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb / 255.0
    input_tensor = np.expand_dims(img_normalized, axis=0)

    #  Vorhersage (nur Klasse)
    pred_class, _ = model.predict(input_tensor)
    pred_class_id = int(np.argmax(pred_class[0]))
    pred_class_prob = float(np.max(pred_class[0]))

    #  Klasse auf Bild schreiben
    label = f"Class: {pred_class_id} ({pred_class_prob:.2f})"
    cv2.putText(original_img, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    #  Ergebnis speichern
    output_path = RESULT_DIR / image_path.name
    success = cv2.imwrite(str(output_path), original_img)
    if success:
        print(f" Gespeichert: {output_path.name}")
    else:
        print(f" Fehler beim Speichern: {output_path}")

print(" Alle Bilder verarbeitet und gespeichert.")
