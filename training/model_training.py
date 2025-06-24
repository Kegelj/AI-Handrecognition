from ultralytics import YOLO
from pathlib import Path
import shutil
import time
import subprocess
import os

def main():
    # === Path to the YAML config file ===
    yaml_path = Path(__file__).parent / "handzeichen.yaml"
    if not yaml_path.exists():
        print(f" YAML file not found: {yaml_path}")
        return

    # === Load YOLO model (automatically downloads if not present) ===
    model = YOLO("yolov8n.pt")

    # === Start training (saves model after every epoch) ===
    model.train(
        data=str(yaml_path),
        epochs=50,
        imgsz=640,
        batch=16,
        name="handzeichen_train",
        pretrained=True,
        save_period=1  # Save model after every epoch
    )

    # === Locate latest training run directory ===
    run_dirs = sorted(Path("runs/detect").glob("handzeichen_train*"), key=os.path.getmtime)
    if not run_dirs:
        print(" No training run directory found.")
        return

    run_dir = run_dirs[-1]
    results_csv = run_dir / "results.csv"
    output_csv = Path("model_output/trainings_log.csv")

    # === Give time for file system to flush ===
    time.sleep(1)

    if results_csv.exists():
        shutil.copy(results_csv, output_csv)
        print(f" Training log saved to: {output_csv.resolve()}")

        # === Run optional post-processing script ===
        uuid_script = Path("runs/csv_uuid_ergenzung.py")
        if uuid_script.exists():
            try:
                subprocess.run(["python", str(uuid_script)], check=True)
                print(" UUID enrichment script executed successfully.")
            except subprocess.CalledProcessError as e:
                print(f" Error running UUID script: {e}")
        else:
            print(" UUID enrichment script not found.")
    else:
        print(" results.csv not found. Was training aborted or failed?")

if __name__ == "__main__":
    main()
