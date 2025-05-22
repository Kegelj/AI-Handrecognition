FROM tensorflow/tensorflow:2.15.0-gpu

RUN apt-get update && apt-get install -y \
    gcc \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY training/ ./training
COPY cloud_data/ ./cloud_data
COPY model_output/ ./model_output

CMD ["bash"]
