import os
from dotenv import load_dotenv
import requests
import json

load_dotenv()

endpoint = "http://localhost:8080/api/v1/"
username = os.getenv("AIRFLOW_USER")
password = os.getenv("AIRFLOW_PASSWORD")

def airflow_check(actions):

    action = "version"
    full_endpoint = endpoint + action
    response = requests.get(full_endpoint, auth=(username, password))
    if response.status_code == 200:
        print(response.json())
    else:
        print("Failed to check version")

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
    else:
        print("Error when checking Dags")

def airflow_run_dags(dag_id):

    action = f"dags/{dag_id}/dagRuns"
    full_endpoint = endpoint + action

    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "conf": {}
    }
    response = requests.post(full_endpoint,headers=headers,data=json.dumps(data), auth=(username, password))

    if response.status_code == 200:
        print("DAG successfully startet:", response.json())
    else:
        print("Error while starting DAGs:", response.status_code, response.text)

if __name__ == "__main__":
    airflow_run_dags("CSV_Import_and_Processing_Pipeline")
