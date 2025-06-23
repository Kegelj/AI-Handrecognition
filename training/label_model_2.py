import cv2
from pathlib import Path

# === Parameter ===
video_path = Path("PXL_20250618_120528794.mp4")  # ← Pfad zum Video anpassen
output_dir = Path("training/alternativ_daten/piu")   # ← Zielordner für Frames
frame_interval = 5                   # ← jedes 5. Frame

# === Ordner erstellen, falls nicht vorhanden ===
output_dir.mkdir(parents=True, exist_ok=True)

# === Video öffnen ===
cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    print(" Fehler beim Öffnen des Videos.")
    exit()

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        filename = output_dir / f"frame_{saved_count:05d}.jpg"
        cv2.imwrite(str(filename), frame)
        saved_count += 1

    frame_count += 1

cap.release()
print(f" Fertig! {saved_count} Frames wurden in '{output_dir}' gespeichert.")
