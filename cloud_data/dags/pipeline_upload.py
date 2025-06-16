import sys
sys.path.append("/opt")
from connectors import database_connector as dc

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime


def upload_stagingdata():
"""
1) Dateien aus dem Ordner airflow_data/data_bbox und Ordner airflow_data/data_game einlesen.
2) Je nachdem welcher Ordner gerade bearbeitet wird, müssen unterschiedliche Parameter gesetzt werden
3) die Funktion dc.copy_to_db verwenden (uploaded file in die Staging Form:
    bbox:
        filepath = airflow_data/data_bbox/dateiname.csv
        tablename = staging_bbox
        columns = [('training',),
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
                     ('batch_size',)]
        format = "CSV"
        header = True
        delimiter = ","

    game_data:
        filepath = airflow_data/data_game/dateiname.csv
        tablename = staging_gamedata
        columns = [('game_id',), ('user_name',), ('user_input',), ('timestamp',)]
        format = "CSV"
        header = True
        delimiter = ","
4) upgeloadete Datei in den Ordner data_bbox/processed/dateiname.csv oder data_game/processed/dateinname.csv verschieben
"""

    dc.copy_to_db(filepath[1], tablename[1], columns[1], format="CSV", header=True, delimiter=",")
    dc.update(table[0], column[0], value[0], action=action[0])