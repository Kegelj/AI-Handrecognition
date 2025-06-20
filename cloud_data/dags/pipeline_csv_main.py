import sys
sys.path.append("/")
#sys.path.append(str(Path(__file__).resolve().parents[1]))
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import pipeline_upload as pu
import pipeline_process_staging as ps
import pipeline_delete as pd


def upload_data():
    pu.upload_stagingdata()

def process_data():
    ps.process_game_stagingdata()

def delete_data():
    pd.delete_processed()

with DAG(
    'CSV_Import_and_Processing_Pipeline',
    start_date=datetime(2024, 6, 9),
    schedule_interval="5 4 * * sun",  # https://crontab.guru/#*/5_*_*_*_*
    catchup=False,
) as dag:

    upload_task = PythonOperator(
        task_id='Upload_csvs',
        python_callable=upload_data
    )
    process_task = PythonOperator(
        task_id='Process_staging_table',
        python_callable=process_data
    )
    delete_task = PythonOperator(
        task_id='Delete_staging_data',
        python_callable=delete_data
    )

    # task dependencies
    upload_task >> process_task >> delete_task