import os
import psycopg2
from dotenv import load_dotenv


class DatabaseError(Exception):
    pass


def _connection():

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
            port = db_port)
        print(f"Database Connection to {db_name} successfully established")
        return dbconn

    except Exception as e:
        raise DatabaseError(f"Connection to Database {db_name} failed with Error: \n{str(e)}")


def query(statement):

    conn = _connection()
    with conn.cursor() as curs:
        curs.execute(statement)
        fetched_data = curs.fetchall()
    conn.close()
    return fetched_data


def insert(table: str, values: [tuple] =None, slist: [list[tuple],[tuple]]=None, type: str ="insert"):

    if not table or table == "":
        return ("Please provide a 'table' when using this function")

    if not values or len(values) < 1 and not slist:
        return ("Please provide a 'value' or 'list' when using this function")

    if slist and values:
        return ("Please provide either a 'value' or a 'list' when using this function")



    query_columns= f"SELECT column_name as column_count FROM information_schema.columns WHERE table_name = '{table}' and column_name != 'id' order by column_name"
    return_value = query(query_columns) #Output of query is [(str,),(str,),...] in alphabetical order


    columns_tuple = tuple([column[0] for column in return_value])
    columns_count = len(columns_tuple) # number of placeholder %s to generate
    value_placeholder = f"({('%s,' * (columns_count - 1) + '%s')})"

    if type == "insert":
        statement = f"INSERT INTO {table} {columns_tuple} VALUES {value_placeholder}"


    connection = _connection()
    with connection.cursor() as cursor:

        if not slist and values:
            cursor.execute(statement,values)
            return cursor.statusmessage

        if type(slist) == list and len(slist) > 0:
            cursor.executemany(statement,slist)
            return cursor.statusmessage

    connection.close()
