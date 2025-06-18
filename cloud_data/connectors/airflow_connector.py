import os
from dotenv import load_dotenv
import requests

load_dotenv()

endpoint = "http://localhost:8080/api/v1/"
username = os.getenv("AIRFLOW_USER")
password = os.getenv("AIRFLOW_PASSWORD")

def airflow_check(actions):

    operation = {
        "connection": "connections?limit=100",
        "version": "version",
    }

    action = endpoint + operation[actions]
    print(f"action: {action}")
    response = requests.get(action,auth=(username, password))
    print(response.status_code)
    print(response.json())


def airflow_check_dags():

    action = "dags"
    full_endpoint = endpoint + action
    response = requests.get(full_endpoint, auth=(username, password))
    if response.status_code == 200:
        response_json = response.json()
        print(response.json()['dags'][0]["dag_id"])

        dag_list_tuples = [(dag["dag_id"], dag["file_token"]) for dag in response_json["dags"]]
        # [('Delete_processed', 'Ii9...Sy4c'),..]
        print(dag_list_tuples)

def airflow_run_dags():

    action = "dags/Delete_processed/dagRuns"
    full_endpoint = endpoint + action
    data = {
        "conf": {},
        "dag_run_id": "Delete_processed",
        "execution_date": "2025-05-15T14:04:43.602Z",
    }
    response = requests.post(full_endpoint, auth=(username, password),data=data)
    print(response.status_code)

airflow_run_dags()