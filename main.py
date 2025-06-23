import sys
import os
import glob
from pathlib import Path

# Set paths for different folder locations
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, 'game'))
sys.path.append(os.path.join(base_dir, 'cloud_data', 'connectors'))
sys.path.append(os.path.join(base_dir, 'cloud_data', 'dags'))

import airflow_connector
import game


def cleanup_csv_stash(csv_directory):
    csv_path = os.path.join(csv_directory, "*.csv")
    print(csv_path)
    csv_files = glob.glob(csv_path)
    
    deleted_counter = 0

    for csv_file in csv_files:
        try:
            os.remove(csv_file)
            print(f"Deleted: {csv_file}")
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {csv_file}: {e}")
    
    print(f"Cleanup complete: {deleted_count} CSV files deleted")

def trigger_pipeline():

    try:
        print("Triggering Airflow Dag...")
        airflow_connector.airflow_run_dags("CSV_Import_and_Processing_Pipeline")
        print("DAG triggered successfully!")
        return True
    except:
        print(f"ERROR: DAG failed to trigger.")
        print("WARNING: Local CSV Files will be kept.")
        return False
    
def main():

    print("Starting Game...")

    try:
        game_object = game.Game()
        game_object.run()

        print("Game finished. Starting Pipeline...")

        pipeline_successull = trigger_pipeline()

        if pipeline_successull:
            print("Pipline successfully triggered!")
            #cleanup_csv_stash("cloud_data/airflow_data/data_game/")
        else:
            print("ERROR: Couldn't Execute Pipeline Process.")
            print("WARNING: CSV Files are kept locally.")

    except KeyboardInterrupt:
        print("\nGame interrupted by User")
    except Exception as e:
        print(f"Unexpected Error: {e}")

if __name__ == "__main__":
    main()
    
