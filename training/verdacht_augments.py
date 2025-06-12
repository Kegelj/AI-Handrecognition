import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import random

from scr.augments import MyImageAugmentor

# Parameter
IMG_SIZE = (256, 256)
DATA_DIR = Path("training/scaled_daten")

#  Zufälliges Bild raussuchen
all_imgs = sorted([p for p in DATA_DIR.rglob("*.jpg") if (p.with_suffix(".txt")).exists()])
random_img_path = random.choice(all_imgs)
txt_path = random_img_path.with_suffix(".txt")
print(f" Zufälliges Bild: {random_img_path}")

#  Bild und Label laden
img = cv2.imread(str(random_img_path))
img = cv2.resize(img, IMG_SIZE)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

with open(txt_path) as f:
    cls, x_c, y_c, w_box, h_box = map(float, f.readline().split())
    label = np.array([cls, x_c, y_c, w_box, h_box], dtype=np.float32)

#  Ursprungsbild mit BBox zeichnen (grün, fett)
img_orig_drawn = (img * 255).astype(np.uint8).copy()
h, w, _ = img_orig_drawn.shape
x_c_pix, y_c_pix = x_c * w, y_c * h
bw_pix, bh_pix = w_box * w, h_box * h
x1, y1 = int(x_c_pix - bw_pix / 2), int(y_c_pix - bh_pix / 2)
x2, y2 = int(x_c_pix + bw_pix / 2), int(y_c_pix + bh_pix / 2)
cv2.rectangle(img_orig_drawn, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Grün, fett

#  Tensoren
img_tf = tf.convert_to_tensor(img.astype(np.float32))
bbox_tf = tf.convert_to_tensor(label.astype(np.float32))

#  Augmentor-Instanz
augmentor = MyImageAugmentor(img_size=IMG_SIZE)

#  Augmentieren
aug_img_tf, aug_bbox_tf = augmentor.tf_augment(img_tf, bbox_tf)

#  In NumPy umwandeln
aug_img = aug_img_tf.numpy()
aug_bbox = aug_bbox_tf.numpy()

#  Augmentiertes Bild mit BBox zeichnen (grün, fett)
aug_img_drawn = (aug_img * 255).astype(np.uint8).copy()
h, w, _ = aug_img_drawn.shape
x_c_pix, y_c_pix = aug_bbox[1] * w, aug_bbox[2] * h
bw_pix, bh_pix = aug_bbox[3] * w, aug_bbox[4] * h
x1, y1 = int(x_c_pix - bw_pix / 2), int(y_c_pix - bh_pix / 2)
x2, y2 = int(x_c_pix + bw_pix / 2), int(y_c_pix + bh_pix / 2)
cv2.rectangle(aug_img_drawn, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Grün, fett

#  Vergleich anzeigen
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_orig_drawn)
plt.title(" Originalbild mit BBox")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(aug_img_drawn)
plt.title(" Augmentiertes Bild mit BBox")
plt.axis("off")

plt.tight_layout()
plt.savefig("augmented_comparison.png")
print(" Vergleichsbild gespeichert als augmented_comparison.png")
