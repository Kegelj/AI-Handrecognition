import cv2
from pathlib import Path
from ultralytics import YOLO
import requests

# === Parameter ===
PADDING_RATIO = 0.05
WIDEN_RATIO = 0.05
BASE_DIR = Path("training/alternativ_daten")
MODEL_DIR = Path("model_output")
MODEL_PATH = MODEL_DIR / "roboflow_hand.pt"

# === Ordnername → Klassen-ID Mapping ===
FOLDER_TO_CLASS_ID = {
    "index": 1,
    "index_pinky": 2,
    "pinky": 3,
    "thumb": 4,
    "thumb_index": 5,
    "fronthand": 6,
    "backhand": 6
}

# === Modell automatisch herunterladen (nur einmal) ===
def download_hand_model():
    if MODEL_PATH.exists():
        print(f"✅ Modell vorhanden: {MODEL_PATH}")
        return

    print("⬇️ Lade Roboflow-Handmodell ...")
    url = "https://huggingface.co/The-Myth/hand-detection-yolov8/resolve/main/best.pt"
    response = requests.get(url)
    response.raise_for_status()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)

    print(f"✅ Modell gespeichert unter: {MODEL_PATH.resolve()}")

# === Box erweitern ===
def adjust_bbox_width_only(x_min, y_min, x_max, y_max, width, height,
                           padding_ratio=0.15, widen_ratio=0.2):
    box_w = x_max - x_min
    box_h = y_max - y_min
    pad_h = box_h * padding_ratio / 2
    widen = box_w * widen_ratio / 2
    x1 = int(max(x_min - widen, 0))
    y1 = int(max(y_min - pad_h, 0))
    x2 = int(min(x_max + widen, width))
    y2 = int(min(y_max + pad_h, height))
    return x1, y1, x2, y2

# === Hauptfunktion ===
def main():
    download_hand_model()
    model = YOLO(str(MODEL_PATH))
    print(f"✅ Modell geladen: {MODEL_PATH.name}")

    for image_path in BASE_DIR.rglob("*.jpg"):
        folder_name = image_path.parent.name

        # Suffix entfernen
        for suffix in ["_i", "_p", "_m"]:
            if folder_name.endswith(suffix):
                folder_name = folder_name[:-len(suffix)]
                break

        if folder_name.lower() in ["nop", "faust"]:
            print(f"⏭ Ignoriere: {folder_name} ({image_path.name})")
            continue

        class_id = FOLDER_TO_CLASS_ID.get(folder_name)
        if class_id is None:
            print(f"⚠️ Unbekannter Ordnername: {folder_name}")
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            print(f"❌ Fehler beim Laden: {image_path}")
            continue

        height, width = img.shape[:2]
        results = model(img)[0]

        if results.boxes:
            for box in results.boxes:
                x_min, y_min, x_max, y_max = box.xyxy[0].tolist()

                x1, y1, x2, y2 = adjust_bbox_width_only(
                    x_min, y_min, x_max, y_max, width, height,
                    padding_ratio=PADDING_RATIO, widen_ratio=WIDEN_RATIO
                )

                xc = (x1 + x2) / 2 / width
                yc = (y1 + y2) / 2 / height
                w = (x2 - x1) / width
                h = (y2 - y1) / height

                label_path = image_path.with_suffix(".txt")
                with open(label_path, "w") as f:
                    f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

                print(f"✅ Label geschrieben für: {image_path.name}")
                break
        else:
            print(f"⚠️ Keine Hand erkannt in: {image_path.name}")

    print("\n🏁 Fertig! Alle Labels wurden erzeugt.")

if __name__ == "__main__":
    main()
