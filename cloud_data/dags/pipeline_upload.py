import logging
import sys
from pathlib import Path
sys.path.append("/")
#sys.path.append(str(Path(__file__).resolve().parents[1]))
from opt.connectors import database_connector as dc
from pathlib import Path
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime


logger = logging.getLogger(__name__)


def upload_stagingdata():
    #base_path = Path(__file__).resolve().parents[1] / "tmp/airflow_data"
    base_path = "/tmp/airflow_data/"
    logger.info(f"Defined Basepath: {base_path}")
    data_sources = {
        "data_bbox": {
            "table": "staging_bbox",
            "columns": [
                ('training',), ('epoch',), ('current_time_start',), ('current_time_end',),
                ('mean_squared_error',), ('mean_center_dist',), ('mean_size_error',),
                ('mean_overlap',), ('combined_score',), ('acc_all_conditions',),
                ('train_loss',), ('val_loss',), ('train_acc',), ('val_acc',),
                ('current_lr',), ('img_size',), ('batch_size',)
            ]
        },
        "data_game": {
            "table": "staging_gamedata",
            "columns": [
                ('game_id',), ('user_name',), ('user_input',), ('timestamp',)
            ]
        },
        "data_yolo": {
            "table": "staging_yolo",
            "columns": [
                ('epoch',), ('time',), ('train_box_loss',), ('train_cls_loss',), ('train_dfl_loss',),
                ('metrics_precision_B',), ('metrics_recall_B',), ('metrics_mAP50_B',), ('metrics_mAP50_95_B',),
                ('val_box_loss',), ('val_cls_loss',), ('val_dfl_loss',),
                ('lr_pg0',), ('lr_pg1',), ('lr_pg2',),
                ('yolo_training_id',)
            ]
        }
    }

    for folder_name, config in data_sources.items():
        folder_path = base_path + folder_name
        logger.info(f"Defined folderpath: {folder_path}")
        processed_path = folder_path + "/processed"
        processed_path = Path(processed_path)
        logger.info(f"Defined processed path: {processed_path}")
        #processed_path.mkdir(parents=True, exist_ok=True)

        for file_path in Path(folder_path).rglob("*.csv"):
            try:
                print(f" Processing file: {file_path.name}")
                logger.info(f" Processing file: {file_path.name}")

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
                logger.info(f" File moved to: {new_path}")

            except Exception as e:
                print(f" Error with file {file_path.name}: {e}")
                logger.info(f" Error with file {file_path.name}: {e}")
    print(base_path)

with DAG(
    'Upload_csvs_to_staging',
    start_date=datetime(2024, 6, 9),
    schedule_interval="5 4 * * sun",  # https://crontab.guru/#*/5_*_*_*_*
    catchup=False,
) as dag:

    upload_task = PythonOperator(
        task_id='Upload_csvs',
        python_callable=upload_stagingdata
    )
if __name__ == "__main__":
    upload_stagingdata()