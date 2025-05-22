import os
import random
import shutil
from pathlib import Path

# Parameter
SOURCE_DIR = Path("extracted_pngs")
DEST_DIR = Path("data_split")
TRAIN_RATIO = 0.8  # z. B. 80 % train, 20 % test

# Zielordner erstellen
train_dir = DEST_DIR / "train"
test_dir = DEST_DIR / "test"
for subdir in [train_dir, test_dir]:
    subdir.mkdir(parents=True, exist_ok=True)

# Alle PNG-Dateien finden (oder JPG, falls du das brauchst)
images = list(SOURCE_DIR.glob("*.png"))

# Shuffle für Zufallsauswahl
random.shuffle(images)

# Split
split_index = int(len(images) * TRAIN_RATIO)
train_images = images[:split_index]
test_images = images[split_index:]

# Hilfsfunktion zum Kopieren von Bild + passendem .txt
def copy_pair(img_path, target_dir):
    shutil.copy(img_path, target_dir)
    txt_path = img_path.with_suffix(".txt")
    if txt_path.exists():
        shutil.copy(txt_path, target_dir)

# Kopieren
for img in train_images:
    copy_pair(img, train_dir)
for img in test_images:
    copy_pair(img, test_dir)

print(f" {len(train_images)} Trainingsbeispiele, {len(test_images)} Testbeispiele kopiert.")
