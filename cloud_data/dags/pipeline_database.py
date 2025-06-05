import sys
sys.path.append("/opt")
from connectors import database_connector as dc

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime


def check_gamedata():

    statement = "SELECT * FROM testtable"
    check_values = dc.query(statement)

def placeholder():

    print("blabla")


with DAG(
    'Save_file_data_to_database',
    start_date=datetime(2024, 6, 9),
    schedule_interval="*/5 * * * *",  # https://crontab.guru/#*/5_*_*_*_*
    catchup=False,
) as dag:


    query_task = PythonOperator(
        task_id='query',
        python_callable=check_gamedata
    )

    insert_task = PythonOperator(
        task_id='insert',
        python_callable=placeholder
    )

    query_task >> insert_task