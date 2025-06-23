import cv2
import os
from pathlib import Path

# === Configure input and output directories ===
image_dirs = [Path("training/images/train"), Path("training/images/val")]
label_dirs = [Path("training/labels/train"), Path("training/labels/val")]
output_dir = Path("training/resultat")
output_dir.mkdir(parents=True, exist_ok=True)

# === Draw YOLO bounding boxes on an image ===
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

# === Iterate over all images and their label files ===
for img_dir, lbl_dir in zip(image_dirs, label_dirs):
    for img_file in img_dir.glob("*.*"):
        if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        label_file = lbl_dir / (img_file.stem + ".txt")
        if not label_file.exists():
            continue

        # Load image
        img = cv2.imread(str(img_file))
        if img is None:
            continue

        # Load YOLO labels
        with open(label_file, "r") as f:
            lines = f.readlines()
            boxes = [list(map(float, line.strip().split())) for line in lines]

        # Draw bounding boxes
        img_bbox = draw_bbox(img, boxes)

        # Save output image
        out_path = output_dir / img_file.name
        cv2.imwrite(str(out_path), img_bbox)

print(f" All images with bounding boxes saved to: {output_dir.resolve()}")
