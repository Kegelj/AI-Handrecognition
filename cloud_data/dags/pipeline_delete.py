import sys
sys.path.append("/opt")
from connectors import database_connector as dc

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime

def delete_processed():

    tables = ["staging_gamedata","staging_bbox"]

    for table in tables:
        print(f"Deleting processed data from {table}")
        statement = f"DELETE FROM {table} WHERE processed = True RETURNING *"
        print(f"Deleted the following entries: \n")
        print(dc.insert_manual(statement))
    print("Deleting complete")


with DAG(
    'Delete processed',
    start_date=datetime(2024, 6, 9),
    schedule_interval="5 4 * * sun",  # https://crontab.guru/#*/5_*_*_*_*
    catchup=False,
) as dag:

    process_task = PythonOperator(
        task_id='delete_processed',
        python_callable=delete_processed
    )
