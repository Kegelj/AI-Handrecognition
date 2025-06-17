from ultralytics import YOLO
from pathlib import Path
import shutil
import time

def main():
    # === Pfad zur YAML-Datei ===
    yaml_path = Path(__file__).parent / "handzeichen.yaml"

    # === Modell laden ===
    model = YOLO("model_output/yolov8n.pt")

    # === Training starten ===
    model.train(
        data=str(yaml_path),
        epochs=50,
        imgsz=640,
        batch=16,
        name="handzeichen_train",
        pretrained=True
    )

    # === Nach dem Training: CSV kopieren ===
    run_dir = Path("runs/detect/handzeichen_train")
    results_csv = run_dir / "results.csv"
    ziel_csv = Path("model_output/trainings_log.csv")

    time.sleep(1)
    if results_csv.exists():
        shutil.copy(results_csv, ziel_csv)
        print(f" Trainings-CSV gespeichert unter: {ziel_csv.resolve()}")
    else:
        print(" results.csv nicht gefunden – ist das Training fehlgeschlagen?")

if __name__ == "__main__":
    main()
