import os
import shutil
from pathlib import Path
import random

# === Parameter ===
quellordner = Path("training/endtime_bilder")
img_exts = {".jpg", ".jpeg", ".png"}
split_ratio = 0.8  # 80% Train, 20% Val

# === Zielverzeichnisse ===
img_train = Path("training/images/train")
img_val = Path("training/images/val")
lbl_train = Path("training/labels/train")
lbl_val = Path("training/labels/val")

# Ordner erstellen
for p in [img_train, img_val, lbl_train, lbl_val]:
    p.mkdir(parents=True, exist_ok=True)

# === Dateien einsammeln (auch ohne txt!) ===
samples = []

for unterordner in quellordner.iterdir():
    if unterordner.is_dir():
        for datei in unterordner.iterdir():
            if datei.suffix.lower() in img_exts:
                txt_datei = datei.with_suffix('.txt')
                samples.append((datei, txt_datei if txt_datei.exists() else None))

# === Zufällig mischen und aufteilen ===
random.shuffle(samples)
split_index = int(len(samples) * split_ratio)
train_samples = samples[:split_index]
val_samples = samples[split_index:]

# === Dateien kopieren ===
def kopiere(sample_list, img_dst, lbl_dst):
    for img_path, lbl_path in sample_list:
        name = img_path.name
        shutil.copy(img_path, img_dst / name)

        if lbl_path:
            lbl_name = lbl_path.name
            shutil.copy(lbl_path, lbl_dst / lbl_name)

kopiere(train_samples, img_train, lbl_train)
kopiere(val_samples, img_val, lbl_val)

print(f" Fertig! {len(train_samples)} Trainingsbilder, {len(val_samples)} Validierungsbilder (inkl. Bilder ohne Label).")
