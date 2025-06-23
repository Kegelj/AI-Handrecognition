from ultralytics import YOLO
from pathlib import Path
import shutil
import time
import subprocess
import os

def main():
    # Pfad zur YAML-Datei
    yaml_path = Path(__file__).parent / "handzeichen.yaml"
    if not yaml_path.exists():
        print(f"YAML-Datei nicht gefunden: {yaml_path}")
        return

    # Modell laden (lädt automatisch von Ultralytics, wenn lokal nicht vorhanden)
    model = YOLO("yolov8n.pt")

    # Training starten und jede Epoche speichern
    model.train(
        data=str(yaml_path),
        epochs=50,
        imgsz=640,
        batch=16,
        name="handzeichen_train",
        pretrained=True,
        save_period=1
    )

    # Letztes Run-Verzeichnis ermitteln
    run_dirs = sorted(Path("runs/detect").glob("handzeichen_train*"), key=os.path.getmtime)
    if not run_dirs:
        print("Kein Run-Verzeichnis gefunden.")
        return

    run_dir = run_dirs[-1]
    results_csv = run_dir / "results.csv"
    ziel_csv = Path("model_output/trainings_log.csv")

    time.sleep(1)

    if results_csv.exists():
        shutil.copy(results_csv, ziel_csv)
        print(f"Trainings-CSV gespeichert unter: {ziel_csv.resolve()}")

        uuid_script = Path("runs/csv_uuid_ergenzung.py")
        if uuid_script.exists():
            try:
                subprocess.run(["python", str(uuid_script)], check=True)
                print("UUID-Ergänzungs-Skript erfolgreich ausgeführt.")
            except subprocess.CalledProcessError as e:
                print(f"Fehler beim Ausführen des UUID-Skripts: {e}")
        else:
            print("UUID-Ergänzungs-Skript nicht gefunden.")
    else:
        print("results.csv nicht gefunden. Wurde das Training abgebrochen oder ist fehlgeschlagen?")

if __name__ == "__main__":
    main()
