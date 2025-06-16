import sys
sys.path.append("/opt")
from connectors import database_connector as dc
from pathlib import Path
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime



def upload_stagingdata():
    base_path = Path(__file__).resolve().parents[1] / "airflow_data"

    data_sources = {
        "data_bbox": {
            "table": "staging_bbox",
            "columns": [
                ('training',),
                ('epoch',),
                ('current_time_start',),
                ('current_time_end',),
                ('mean_squared_error',),
                ('mean_center_dist',),
                ('mean_size_error',),
                ('mean_overlap',),
                ('combined_score',),
                ('acc_all_conditions',),
                ('train_loss',),
                ('val_loss',),
                ('train_acc',),
                ('val_acc',),
                ('current_lr',),
                ('img_size',),
                ('batch_size',)
            ]
        },
        "data_game": {
            "table": "staging_gamedata",
            "columns": [
                ('game_id',),
                ('user_name',),
                ('user_input',),
                ('timestamp',)
            ]
        }
    }

    for folder_name, config in data_sources.items():
        folder_path = base_path / folder_name
        processed_path = folder_path / "processed"
        processed_path.mkdir(parents=True, exist_ok=True)

        for file_path in folder_path.glob("*.csv"):
            try:
                print(f" Processing file: {file_path.name}")

                columns_as_list = [col[0] for col in config["columns"]]

                dc.copy_to_db(
                    filepath=str(file_path),
                    table=config["table"],
                    columns=columns_as_list,
                    format="CSV",
                    header=True,
                    delimiter=","
                )


                new_path = processed_path / file_path.name
                file_path.rename(new_path)
                print(f" File moved to: {new_path}")

            except Exception as e:
                print(f" Error with file {file_path.name}: {e}")








