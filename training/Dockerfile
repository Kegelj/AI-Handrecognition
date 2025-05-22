FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    gcc \
    && apt-get clean

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY training/ ./training
COPY cloud_data/ ./cloud_data
COPY model_output/ ./model_output

CMD ["bash"]