import os
import psycopg2

if os.name == "nt":
    from dotenv import load_dotenv
    load_dotenv()


class DatabaseError(Exception):
    pass


def _connection():
    """
    Establishes a connection to a database using the Environment variables defined in the .env File
    :return: Returns a connection Object that can be used with .cursor
    """

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


def _placeholder(value_list, origin="db"):
    """
    Used as a helper function to get a nice string format of the provided list to be used in DB Statements.

    :param value_list: [('col1',),('col2')],["columnname1","columnname2"]
    :param input: db (default),data (for lists)
    :return: columns_count (int), columns_string (str),columns_tuple (tuple), value_placeholder ( (%s,%s) )
    """

    if origin == "db":
        columns_tuple = tuple([column[0] for column in value_list])
        columns_count = len(columns_tuple)
        columns_string = "(" + ", ".join(columns_tuple) + ")"  # Values in the correct form for Statement (value1,value2)
    if origin == "data":
        columns_tuple = value_list
        columns_count = len(columns_tuple[0])

    value_placeholder = f"({('%s,' * (columns_count - 1) + '%s')})"  # Creates (%s,%s,...) with the number of columns

    if origin == "db":
        return columns_count, columns_string, columns_tuple, value_placeholder
    if origin == "data":
        return columns_count, columns_tuple, value_placeholder

def _columns_placeholder(values): # ["column1","column2",...]

    columns_count = len(values)
    columns_string = "(" + ", ".join(values) + ")"  # Values in the correct form for Statement (value1,value2)
    column_placeholder = f"({('%s,' * (columns_count - 1) + '%s')})"  # Creates (%s,%s,...) with the number of columns
    return columns_string, column_placeholder

def query(statement: str):

    conn = _connection()
    with conn.cursor() as curs:
        curs.execute(statement)
        fetched_data = curs.fetchall()
    conn.close()
    return fetched_data


def check_columns(tablename: str):
    """
    Return the names of the columns of the provided table

    :param tablename: table name inside Database
    :return: [('column1',),('column2',),..]
    """

    query_columns = f"SELECT column_name as column_count FROM information_schema.columns WHERE table_name = '{tablename}' and column_name != 'id' and column_name != 'processed' order by ordinal_position" # ordinal_position
    return query(query_columns)


# insert(table: str, values: [tuple] or [list[tuple],[tuple]]=None,columns: [list]=None, operation: str ="insert_all")
def insert(table, values=None, amount="single",columns=None, operation="insert_all"):
    """
    Database function to execute Statements.
    You will need to provide a table name, values in the form [('Ivan',23)] or  [('Ovin',23),('Ivan',25),....].
    Optional: columns ["col1","col2",..] (column name for specific inserts - can only be used in combination with operation "insert_specific")
    operation:
        insert_all - you need to provide as many values as columns of the Tables
        insert_specific - Only specified columns will be filled with values (have to be the same number of items)
    :param table: String - Name of the used database table
    :param values: list [('Ivan',23)] or [('Ovin',23),('Ivan',25),....]
    :param columns: list ["col1","col2",..]
    :param operation: insert_all,insert_specific
    :return: Feedback if it worked
    """


    if not table or table == "":
        return ("Please provide a 'table' when using this function")

    if not values or (isinstance(values, (list, tuple)) and len(values) == 0):
        return "Please provide 'values' when using this function"

    if columns == None:
        table_columns = check_columns(table)
    else:
        table_columns = columns
    columns_count, columns_string,_, value_placeholder = _placeholder(table_columns)


    if operation == "insert_all":
        statement = f"INSERT INTO {table} {columns_string} VALUES {value_placeholder}"
        print(statement)

    if operation == "insert_specific":
        #_,_,value_placeholder = _placeholder(values, origin="data")
        columns_string,value_placeholder = _columns_placeholder(columns)
        statement = f"INSERT INTO {table} {columns_string} VALUES {value_placeholder}"

    connection = _connection()
    try:
        with connection.cursor() as cursor:
            if amount == "single":
                cursor.execute(statement, values)
                #print(cursor.mogrify(statement, values))
            else:
                cursor.executemany(statement, values)


            connection.commit()
            return cursor.statusmessage

    finally:
        connection.close()

def insert_manual(statement):

    try:
        connection = _connection()
        with connection.cursor() as cursor:
                cursor.execute(statement)
                #print(cursor.mogrify(statement))
        connection.commit()
        return cursor.statusmessage

    finally:
        connection.close()

def copy_to_db(filepath: str,table: str,columns: list,format="CSV",header=True,delimiter=";"):
    """
    Function to upload files to database tables using the COPY Statement.

    :param filepath: Path to the file to be uploaded
    :param table: Name of the table to be used for the upload
    :param columns: [str,str,..] List of strings with the name of the columns to be filled
    :param format: Defines the fileformat to be used (text, CSV, binary)
    :param header: Define if the file has a header
    :param delimiter: Define which seperater is used in the file
    :return: Message if the operation was successful or not
    """

    if not filepath or filepath == "" or not os.path.isfile(filepath):
        return ("Please provide a working filepath when using this function")
    if not table or table == "":
        return ("Please provide a table when using this function")
    if not columns or not isinstance(columns,(list,tuple)):
        return ("Please provide columns in list form when using this function")

    header = "HEADER, " if header else ""
    copy_statement = f"COPY {table}({','.join(columns)}) FROM STDIN WITH (FORMAT {format}, {header}DELIMITER '{delimiter}')"
    #print(copy_statement)

    try:
        connection = _connection()
        with connection.cursor() as cursor:
            with open(filepath, 'r') as file:
                cursor.copy_expert(copy_statement, file)
            connection.commit()
        print(f"File ({filepath}) copied succesfully to table {table}.")

    except (Exception, psycopg2.DatabaseError) as error:
        if connection:
            connection.rollback()
        print(f"Error while importing: {error}")

    finally:
        if connection:
            connection.close()


def update(table,column,value=False,action="processed"):


    returned_columns = check_columns(table)
    columns = _placeholder(returned_columns)

    if action == "processed":
        statement = f"UPDATE {table} SET processed=True WHERE {column} = '{value}'"
    else:
        return "Please provide a valid action"

    try:
        connection = _connection()
        with connection.cursor() as cursor:
            cursor.execute(statement)
            connection.commit()
            return cursor.statusmessage
    finally:
        connection.close()