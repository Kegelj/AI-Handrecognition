import logging
import sys
sys.path.append("/opt")
from connectors import database_connector as dc

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime


logger = logging.getLogger(__name__)

def delete_processed():

    tables = ["staging_gamedata","staging_bbox"]

    for table in tables:
        logger.info(f"Deleting processed data from {table}")
        logger.info(f"Deleted the following entries: \n")
        statement = f"DELETE FROM {table} WHERE processed = True RETURNING *"
        logger.info(dc.insert_manual(statement))
    logger.info("Deleting complete")


with DAG(
    'Delete_processed',
    start_date=datetime(2024, 6, 9),
    schedule_interval="5 4 * * sun",  # https://crontab.guru/#*/5_*_*_*_*
    catchup=False,
) as dag:

    process_task = PythonOperator(
        task_id='delete_processed',
        python_callable=delete_processed
    )
