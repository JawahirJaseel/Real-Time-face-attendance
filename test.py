import cv2
import pickle
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
import csv
import time
from datetime import datetime
from win32com.client import Dispatch  # Ensure pywin32 is installed

# Initialize the TTS engine
def speak(text):
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(text)

# Load Haar cascade classifier for face detection
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam (0 is the default webcam)
video = cv2.VideoCapture(0)

# Define file paths for the names and faces data
names_file_path = 'data/names.pkl'
faces_data_file_path = 'data/faces_data.pkl'

if not os.path.exists(names_file_path) or not os.path.exists(faces_data_file_path):
    print("Error: Necessary files not found. Ensure 'names.pkl' and 'faces_data.pkl' exist in 'data' folder.")
    exit()

# Load the labels (names) and face data from files
with open(names_file_path, 'rb') as f:
    LABELS = pickle.load(f)

with open(faces_data_file_path, 'rb') as f:
    FACES = pickle.load(f)

FACES = np.array(FACES)
if len(FACES) != len(LABELS):
    min_length = min(len(FACES), len(LABELS))
    FACES = FACES[:min_length]
    LABELS = LABELS[:min_length]
    print(f"Adjusted data lengths: Faces={len(FACES)}, Labels={len(LABELS)}")

if FACES.ndim == 3:
    FACES = FACES.reshape(len(FACES), -1)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

if not os.path.exists('Attendance'):
    os.makedirs('Attendance')

imgBackground = cv2.imread('background.jpg')
COL_NAMES = ['NAME', 'TIME']  # Column headers

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w]
        resized_img = cv2.resize(crop_img, (50, 50))
        flattened_img = resized_img.flatten().reshape(1, -1)
        output = knn.predict(flattened_img)

        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        attendance = [str(output[0]), str(timestamp)]
        file_path = f"Attendance/Attendance_{date}.csv"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, str(output[0]), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    imgBackground[162:162 + 480, 55:55 + 640] = frame
    cv2.imshow("Frame", imgBackground)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('o'):
        speak("Attendance Taken.")
        time.sleep(5)
        if not os.path.isfile(file_path):
            with open(file_path, "w", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)  # Write header

        with open(file_path, "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(attendance)  # Write attendance data
        print(f"Attendance recorded for {output[0]} at {timestamp}.")

    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
