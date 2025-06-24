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


## Structure

**You can use the links to jump to different sections, but we recommend reading everything to ensure proper usage**

- [AI](#hand-gesture-recognition--training)

- [Prerequisites](#prerequisites)

- [Game](#game)

- [Visualization](#visualization)

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

## Real-Time Hand Gesture Control

**Script:** `game_control.py`

This script enables real-time control of a game using hand gestures detected via a YOLO model.

### Features

- Uses a webcam feed and a trained YOLO model (`model_output/epoch41.pt`) to detect hand gestures.
- Maps detected gestures to keyboard inputs using `pynput`.
- Supports multiple gestures and combined actions:
  - `index` → ↑ (move up)
  - `pinky` → ← (move left)
  - `thumb` → → (move right)
  - `index_pinky` → ↑ + ←
  - `thumb_index` → ↑ + →

### Functions

- `live_tracking_yolo()`:  
  Starts the webcam and runs the YOLO model in a live loop at ~15 FPS. Recognized gestures are converted into key presses. Bounding boxes and labels are displayed in an OpenCV window. The loop stops on keypress `'q'`.

- `extracting_frames(video_name, save_path, skip_frames=5)`:  
  Loads a video and saves frames every N frames as `.jpg` files for dataset creation or debugging.

- `length_of_video(video_name)`:  
  Returns the total number of frames in the given video.

### Dependencies

- OpenCV (`cv2`)
- Pynput (`pynput.keyboard`)
- Ultralytics YOLOv8
- A trained model saved under `model_output/` (e.g. `epoch41.pt`)

### Usage

```bash
python game_control.py
```


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



## Game

The core game is a side-scrolling survival shooter where players control their character using hand gestures detected by the YOLO model.

## How to run the Game

### Direct game execution 
```python
python game/game.py
```
### Game execution with Data extracting Pipeline (recommended)
```python
python main.py
```
### Game Features

**Gameplay Mechanics:**
- **Movement Control:** Use hand gestures (pinky = left, thumb = right, index = jump)
- **Auto-Shooting:** Automatic firing system, no manual shooting required
- **Enemy Types:** Fight against goats and squirrels with different health and damage values
- **Survival Focus:** Keep your health above 0 while eliminating as many enemies as possible

**Technical Implementation:**
- Built with pygame for smooth 60+ FPS gameplay
- Real-time data logging of player actions, positions, health, and kill counts
- Automatic CSV export after each game session for analysis
- Galaxy-themed UI with neon green accents matching the AI theme

**Game Controls:**
```python
# Gesture Controls (via YOLO detection)
index_finger → Jump (↑)
pinky_finger → Move Left (←) 
thumb → Move Right (→)

# Keyboard Controls (for testing/backup)
A/← → Move Left
D/→ → Move Right  
W/↑ → Jump
P → Pause
F → Fullscreen
```

### Data Collection:
Every player action gets logged with timestamps for later analysis:

Player position (x, y coordinates)
Health status at each action
Kill counts (goats vs squirrels)
Input gestures and timing
Game duration and survival metrics

The game automatically exports this data as CSV files which then get processed by the Airflow pipeline and visualized in the Power BI dashboard.


## Visualization

### PowerBI

This project includes a Power BI dashboard for analyzing gaming performance and hand gesture recognition data. 

### Dashboard Features

The dashboard provides real-time analytics and insights into player performance:

### **The Hall of Fame**

- **Player Rankings:** Top performers ranked by total score
- **Performance Metrics:** Highest scores, longest survival times, total games played
- **Interactive Leaderboard:** Dynamic ranking system with medals for top 3 players

#### **Best Rounds Analysis**
- **Performance Quadrants:** Players can be categorized in 4 Zones based on Performance
- **Scatter Plot Visualization:** Survival time vs. total kills correlation
- **Game Quality Assessment:** Identifies peak performance sessions

#### **Game Data Analysis**
- **Movement Heatmaps:** Player position tracking and hotspot analysis
- **Input Pattern Analysis:** Visualizes the inputs players took
- **Kill Tracking** Individual Kill tracking Goats vs Squirrels

- **Detailed Performance Breakdown:** Individual player statistics
- **Game-specific Analysis:** Deep dive into selected games