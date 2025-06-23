# AI-Handrecognition

AI-Handrecognition/  
├── training/  
│   ├── Dockerfile  
│   └── train.py  
├── docker-compose.yml   
├── cloud_data/  
├── model_output/  
│   └── model.h5  
├── live_app/  
│   ├── live_control.py  
│   ├── Dockerfile  
│   ├── model.h5  
│   └──  hand_signal.db  
├── dist/  
│   └── live_control.exe  
├── powerbi/  
│   └── dashboard.pbix  
├── requirements.txt  
└── README.md  



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

