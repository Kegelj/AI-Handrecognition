import sys
sys.path.append("/opt")
from connectors import database_connector as dc

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime


def get_stagingdata():
    statement = "SELECT * FROM staging WHERE NOT processed "
    check_values = dc.query(statement)
