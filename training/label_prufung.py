import cv2
import os
from pathlib import Path

# === Basisordner definieren ===
input_root = Path("training/alternativ_daten")
output_root = Path("training/alternativ_daten_labeltest")
output_root.mkdir(parents=True, exist_ok=True)

# === Bounding Box zeichnen ===
def draw_bbox(image, boxes):
    h, w = image.shape[:2]
    for box in boxes:
        class_id, x_center, y_center, bw, bh = box
        x1 = int((x_center - bw / 2) * w)
        y1 = int((y_center - bh / 2) * h)
        x2 = int((x_center + bw / 2) * w)
        y2 = int((y_center + bh / 2) * h)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image

# === Rekursiv durch alle Bilder gehen ===
for img_file in input_root.rglob("*.*"):
    if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    label_file = img_file.with_suffix(".txt")
    if not label_file.exists():
        continue

    # Bild laden
    img = cv2.imread(str(img_file))
    if img is None:
        continue

    # Labels laden
    with open(label_file, "r") as f:
        lines = f.readlines()
        boxes = [list(map(float, line.strip().split())) for line in lines]

    # BBox einzeichnen
    img_bbox = draw_bbox(img, boxes)

    # Zielpfad erstellen (gleiche Ordnerstruktur beibehalten)
    relative_path = img_file.relative_to(input_root)
    output_path = output_root / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Bild speichern
    cv2.imwrite(str(output_path), img_bbox)

print(f"  Alle Bilder mit BBox wurden gespeichert in: {output_root.resolve()}")
