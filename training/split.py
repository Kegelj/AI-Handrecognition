import os
import shutil
from pathlib import Path
import random

# === Parameter ===
quellordner = Path("training/alternativ_daten")
img_exts = {".jpg", ".jpeg", ".png"}
split_ratio = 0.8  # 80% Train, 20% Val

# === Zielverzeichnisse ===
img_train = Path("training/images/train")
img_val = Path("training/images/val")
lbl_train = Path("training/labels/train")
lbl_val = Path("training/labels/val")

# Zielordner anlegen
for p in [img_train, img_val, lbl_train, lbl_val]:
    p.mkdir(parents=True, exist_ok=True)

# === Nur Bilder mit gültigem Label einsammeln ===
samples = []

for unterordner in quellordner.iterdir():
    if unterordner.is_dir():
        for datei in unterordner.iterdir():
            # Nur Bilddateien beachten
            if datei.suffix.lower() in img_exts:
                txt_datei = datei.with_suffix('.txt')
                # .txt muss existieren und Inhalt haben
                if txt_datei.exists() and txt_datei.stat().st_size > 0:
                    samples.append((datei, txt_datei))

print(f" Gefundene gültige Bilder mit Label: {len(samples)}")

# === Zufällig mischen und aufteilen ===
random.shuffle(samples)
split_index = int(len(samples) * split_ratio)
train_samples = samples[:split_index]
val_samples = samples[split_index:]

# === Dateien kopieren ===
def kopiere(sample_list, img_dst, lbl_dst):
    for img_path, lbl_path in sample_list:
        shutil.copy(img_path, img_dst / img_path.name)
        shutil.copy(lbl_path, lbl_dst / lbl_path.name)

kopiere(train_samples, img_train, lbl_train)
kopiere(val_samples, img_val, lbl_val)

print(f" Fertig!")
print(f"   → Trainingsbilder:    {len(train_samples)}")
print(f"   → Validierungsbilder: {len(val_samples)}")
