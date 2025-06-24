FROM apache/airflow:2.0.2

USER root
WORKDIR /opt/airflow


RUN apt-get update && apt-get install -y gcc libpq-dev

COPY cloud_data/requirements.txt /requirements.txt

USER airflow
RUN pip install python-dotenv
RUN pip install --no-cache-dir -r /requirements.txt
