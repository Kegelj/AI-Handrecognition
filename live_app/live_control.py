import cv2
import random
import string
import os
from pathlib import Path
import mediapipe as mp
from pynput.keyboard import Controller


# Generating random Filename
def rand_string(length):
    rand_str = "".join(random.choice(string.ascii_lowercase + string.ascii_uppercase + string.digits) for i in range(length))
    return rand_str

# Check length of the Video
def length_of_video(video_name):
    video_path = Path(__file__).resolve().parents[1] / "project_assets/videos" / video_name
    cap = cv2.VideoCapture(str(video_path))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return length

# Extracting Frames from a Video
def extracting_frames(video_name, save_path, skip_frames=5):
    print("*******EXTRACTING PHASE********")
    
    # Get the Videoname without IVAN.MP4 / .MOV
    file_name_without_ext = os.path.splitext(video_name)[0]
    
    # Check the Video length
    length = length_of_video(video_name)
    if length == 0:
        print("Length is 0, exiting extracting phase.")
        return 0
    
    # Set the Video Path / Videos that are used should be in the 'videos' folder
    video_path = Path(__file__).resolve().parents[1] / "project_assets/videos" / video_name 

    # Connect the Video with our File
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    
    # Set our needed tools
    count = 0
    random_string = rand_string(3)

    # Name our Testframe
    test_file_path = f"{save_path}{file_name_without_ext}_{random_string}_{count}_TEST.jpg"

    # Save Testframe and check if it was sucessfull
    cv2.imwrite(test_file_path, frame)
    if os.path.isfile(test_file_path):
        print("Saving Test Frame was Successfull\nContinuing Extraction Phase")

    # We check the Videoconnection with a boolean 'ret' and save every 'skip_frames' frame
    count = 1
    while ret:
        ret, frame = cap.read()
        if ret and count % skip_frames == 0:
            cv2.imwrite(f"{save_path}{file_name_without_ext}_{random_string}_{count}.jpg", frame)
            count += 1
            print(f"Frame {count}")
        else:
            count += 1
    else:
        print("Videos fully saved.")
    cap.release()
    return 0

def is_thumb_up(hand_landmarks):
    return hand_landmarks.landmark[4].x < hand_landmarks.landmark[1].x

def is_index_finger_up(hand_landmarks):
    return hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
    
def is_pinky_finger_up(hand_landmarks):
    return hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y
    
# mp_hands, hands und mp_Draw brauchen wir dann nicht mehr
def live_tracking():
    keyboard = Controller()
    
    #Hier holen wir uns die Kamera... 0 spricht das erste Lokale Kamera Element an, meistens ist es die richtige... Ansonsten 1,2... probieren
    cap = cv2.VideoCapture(0)
    try:
        while True:
            success, image = cap.read()
            if not success:
                break
            
            # Das ist unser RGB Frame den du mit dem Model analysierst
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        
        # -> Hier solltest du das Model einfügen und dann die Outputs mit den Tasten unten Verbinden. Ich hab die Variable 'model' nur als Platzhalter erstellt

            model = True
            if model:
                keyboard.press('w')
            else:
                keyboard.release('w')
            if model:
                keyboard.press('d')
            else:
                keyboard.release('d')
            if model:
                keyboard.press('a')
            else:
                keyboard.release('a')  

            cv2.imshow("Handerkennung", image)
  
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except:
        print("ERROR")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    live_tracking()