import cv2
import os
from pathlib import Path

# === Ordner konfigurieren ===
image_dirs = [Path("training/images/train"), Path("training/images/val")]
label_dirs = [Path("training/labels/train"), Path("training/labels/val")]
output_dir = Path("training/resultat")
output_dir.mkdir(parents=True, exist_ok=True)

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

# === Durch Bilder und Labels iterieren ===
for img_dir, lbl_dir in zip(image_dirs, label_dirs):
    for img_file in img_dir.glob("*.*"):
        if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        label_file = lbl_dir / (img_file.stem + ".txt")
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

        # BBox zeichnen
        img_bbox = draw_bbox(img, boxes)

        # Speichern
        out_path = output_dir / img_file.name
        cv2.imwrite(str(out_path), img_bbox)

print(f" Alle Bilder mit BBox wurden gespeichert in: {output_dir.resolve()}")
