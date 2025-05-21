import os
import psycopg2
from dotenv import load_dotenv


class DatabaseError(Exception):
    pass


def connection():

    load_dotenv()
    db_name = os.getenv("POSTGRES_DBNAME")
    db_user = os.getenv("POSTGRES_USER")
    db_password = os.getenv("POSTGRES_PASSWORD")
    db_host = os.getenv("POSTGRES_ENDPOINT")
    db_port = os.getenv("POSTGRES_PORT")


    try:
        dbconn = psycopg2.connect(
            dbname = db_name,
            user = db_user,
            password = db_password,
            host = db_host,
            port = db_port
        )
        print(f"Database Connection to {db_name} successfully established")
        return dbconn

    except Exception as e:
        raise DatabaseError(f"Connection to Database {db_name} failed with Error: \n{str(e)}")


def query(statement):

    conn = connection()
    with conn.cursor() as curs:
        curs.execute(statement)
        return curs.fetchall()


# values=() or None, slist=[(),()] or None,type= "insert"
def insert(table, values=None, slist=None, type="insert"):

    check_columns_statement= f"SELECT column_name as column_count FROM information_schema.columns WHERE table_name = '{table}' and column_name != 'id'"
    return_value = query(check_columns_statement) #Output von query ist [(str,),(str,),...]

    columns_tuple = tuple([column[0] for column in return_value])
    columns_count = len(columns_tuple)
    value_placeholder = f"({('%s,' * (columns_count - 1) + '%s')})"

    if type == "insert":
        statement = f"INSERT INTO {table} {columns_tuple} VALUES {value_placeholder}"


    connection = connection()
    with connection.cursor() as cursor:

        if not slist and values:
            cursor.execute(statement,values)
            return cursor.statusmessage

        if type(slist) == list and len(slist) > 0:
            cursor.executemany(statement,slist)
            return cursor.statusmessage