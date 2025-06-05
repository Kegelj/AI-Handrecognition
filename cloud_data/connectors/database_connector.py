import os
import psycopg2

if os.name == "nt":
    from dotenv import load_dotenv
    load_dotenv()


class DatabaseError(Exception):
    pass


def _connection():

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
        #print(f"Database Connection to {db_name} successfully established")
        return dbconn

    except Exception as e:
        raise DatabaseError(f"Connection to Database {db_name} failed with Error: \n{str(e)}")


def query(statement: str):

    conn = _connection()
    with conn.cursor() as curs:
        curs.execute(statement)
        fetched_data = curs.fetchall()
    conn.close()
    return fetched_data

# insert(table: str, values: [tuple]=None, slist: [list[tuple],[tuple]]=None, operation: str ="insert")
def insert(table, values=None, slist=None, operation="insert"):

    if not table or table == "":
        return ("Please provide a 'table' when using this function")
    if (not values or len(values) < 1) and not slist:
        return ("Please provide a 'value' or 'list' when using this function")
    if slist and values:
        return ("Please provide either a 'value' or a 'list' when using this function")



    query_columns= f"SELECT column_name as column_count FROM information_schema.columns WHERE table_name = '{table}' and column_name != 'id' order by column_name"
    return_value = query(query_columns) #Output of query is [(str,),(str,),...] in alphabetical order


    columns_tuple = tuple([column[0] for column in return_value])
    columns_string = "(" + ", ".join(columns_tuple) + ")" # Values in the correct form for Statement (value1,value2)
    columns_count = len(columns_tuple) # number of placeholder %s to generate
    value_placeholder = f"({('%s,' * (columns_count - 1) + '%s')})" # Creates (%s,%s,...) with the number of columns


    if operation == "insert":
        statement = f"INSERT INTO {table} {columns_string} VALUES {value_placeholder}"


    connection = _connection()
    with connection.cursor() as cursor:

        if not slist and values:
            cursor.execute(statement,values)
            connection.commit()
            return cursor.statusmessage

        if not values and slist:
            cursor.executemany(statement,slist)
            connection.commit()
            return cursor.statusmessage

    connection.close()
