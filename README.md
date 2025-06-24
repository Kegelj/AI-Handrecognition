# AI-Handrecognition

AI-Handrecognition/ 
├── .gitignore 
├── README.md 
├── main.py 
├── requirements.txt 
├── cloud_data/ 
│   ├── airflow_data/ 
│   │   ├── data_bbox/ 
│   │   │   └── processed/ 
│   │   ├── data_game/ 
│   │   │   └── processed/ 
│   │   └── data_yolo/ 
│   │       └── processed/ 
│   ├── connectors/ 
│   ├── dags/ 
│   ├── logs/ 
│   ├── plugins/ 
│   ├── SQLStatements/ 
├── game/ 
│   ├── assets/ 
│   └── logs/ 
├── live_app/ 
├── model_output/ 
├── powerbi/ 
├── runs/ 
├── training/ 
│   ├── alternativ_daten/ 
│   ├── images/ 
│   └── labels/ 


## Hand Gesture Recognition – Training 

This section describes the steps to prepare data, train a YOLO model, and test its performance for hand gesture recognition. 

### 1. Labeling Images 

**Script:** `label_with_mediapipe.py` 

- Recursively processes all subdirectories inside `training/alternativ_daten`. 
- Uses the name of each subdirectory as the class label. 
- For each image, generates a corresponding `.txt` file in YOLO format:   
  `class_id x_center y_center width height`  
- Bounding boxes are automatically generated using MediaPipe. 

### 2. Creating Train/Validation Split 

**Script:** `split.py` 

- Splits the labeled images and annotations into training (80%) and validation (20%) sets. 
- Files are moved into the following directory structure: 

training/ 
├── images/ 
│ ├── train/ 
│ └── val/ 
├── labels/ 
│ ├── train/ 
│ └── val/ 


### 3. Training the Model 

**Script:** `model_training.py` 

- Fine-tunes a YOLO model using the configuration defined in `handzeichen.yaml`. 
- The model is trained on the labeled dataset located in `training/images` and `training/labels`. 
- Training results and checkpoints are stored in the `runs/` directory. 

### 4. Testing the Trained Model 

**Script:** `model_test.py` 

- Loads the trained model. 
- Evaluates its performance on test images. 
- Verifies if the correct hand gestures are detected as expected. 




## Prerequisites 

To ensure a functional use of this project please go through the neccessary prerequisites mentioned below:

### Docker Desktop
You will need this to build the neccesary images for the used Docker Containers

#### Build and run Image
1) Go to the root folder of this project and run this command:
```
docker-compose -f cloud_data/docker-compose.yml up --build
```
2) switch to the cloud_data folder and run this command:
```
docker-compose up
```

### Google CLI

Download and install [gcloud-cli](https://cloud.google.com/sdk/docs/install?hl=de) and follow the instructions

### .env
Will be used by the database_connector and cloud_connector module to connect to the respective services.<br>
The following Varibales will be used and need to be set:<br>
> `POSTGRES_ENDPOINT`<br>
`POSTGRES_PORT` <br>
`POSTGRES_USER`<br>
`POSTGRES_PASSWORD`<br>
`POSTGRES_DBNAME`<br>
`GOOGLE_BUCKET_NAME`<br>
`AIRFLOW_USER`<br>
`AIRFLOW_PASSWORD`<br>

