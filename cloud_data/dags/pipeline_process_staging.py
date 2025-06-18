import logging
import sys
sys.path.append("/opt")
from connectors import database_connector as dc

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime

logger = logging.getLogger(__name__)

def process_game_stagingdata():

    table = "staging_gamedata"

    query = f"SELECT * FROM {table} WHERE processed=False"
    returned_values = dc.query(query) # get data from Staging table

    game_id = set([(game[1],game[2]) for game in returned_values]) # {('45BX05', 'Peter'), ('36B0WC', 'Max')}
    game_names = set([name[2] for name in returned_values]) # {'Max','Peter'}
    game_data = [(game_data[1],int(game_data[3]),float(game_data[4])) for game_data in returned_values]

    # Insert new Names from Staging Table to game_user Table
    user_table = "game_user"
    logger.info("Starting User import:")
    for name in game_names:
        insert_statements = (f"INSERT INTO {user_table} (name) SELECT '{name}' WHERE NOT EXISTS ( SELECT 1 FROM game_user WHERE name = '{name}');")
        try:
            dc.insert_manual(insert_statements)
            logger.info(f"User '{name}' added to Database")
        except:
            logger.info("Error while importing User data")
    logger.info("User import finished")

    # Insert game_id user_id into games table
    games_table = "games"
    user_ids = dc.query(f"SELECT name, id from {user_table}")
    user_ids_dict = dict(user_ids)
    games_table_content = {(game_code, user_ids_dict[name]) for game_code, name in game_id} # new set with user id instead of name

    logger.info("Starting import of new game entries:")
    for game_entry in games_table_content:
        insert_statement = f"INSERT INTO {games_table} (game_id,user_id) SELECT '{game_entry[0]}',{game_entry[1]} WHERE NOT EXISTS ( SELECT 1 FROM {games_table} WHERE game_id = '{game_entry[0]}');"
        dc.insert_manual(insert_statement)
        logger.info(f"Entry for game '{game_entry[0]}' imported")
    logger.info("Game entries imported successfully")

    # Insert game data into game_data table
    data_table = "game_data"
    columns = ["game_id","user_input","time_in_milliseconds"]

    logger.info(f"Starting import of game data")
    try:
        dc.insert(table=data_table,values=game_data,amount="multi",columns=columns, operation="insert_specific")
        logger.info("Import of game data successfully finished")
    except:
        logger.info("Error while importing game data")

    # Flag processed game data

    logger.info("All imports finished - Continuing flagging processed data")
    for game in game_id:
        dc.update("staging_gamedata","game_id",game[0],action="processed")
        logger.info(f"Set staging data for game '{game[0]}' as processed ")
    logger.info("Finished flagging processed data")


with DAG(
    'Process_Gaming_Staging_Data',
    start_date=datetime(2024, 6, 9),
    schedule_interval="5 4 * * sun",  # https://crontab.guru/#*/5_*_*_*_*
    catchup=False,
) as dag:

    process_task = PythonOperator(
        task_id='process_game_stagingdata',
        python_callable=process_game_stagingdata
    )