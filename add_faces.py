import cv2
import pickle
import numpy as np
import os

# Load Haar cascade classifier for face detection
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error: Could not open webcam.")
    exit()

faces_data = []
frame_counter = 0

# Prompt for user name
name = input("Enter your name: ").strip()

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cropped_face = frame[y:y + h, x:x + w]
        resized_face = cv2.resize(cropped_face, (50, 50))  # Resize to 50x50
        flattened_face = resized_face.flatten()  # Flatten the face to a 1D array of 7500 features

        if len(faces_data) < 100 and frame_counter % 10 == 0:
            faces_data.append(flattened_face)

        # Display the number of collected faces
        cv2.putText(frame, f"Collected: {len(faces_data)}/100", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)

    frame_counter += 1
    cv2.imshow("Face Collection", frame)

    # Exit if 'q' is pressed or 100 faces are collected
    if cv2.waitKey(1) & 0xFF == ord('q') or len(faces_data) == 100:
        break

# Release webcam and close windows
video.release()
cv2.destroyAllWindows()

faces_data = np.array(faces_data)

# Ensure the 'data' directory exists
os.makedirs('data', exist_ok=True)

# Handle names
names_file_path = 'data/names.pkl'
if os.path.exists(names_file_path):
    with open(names_file_path, 'rb') as f:
        names = pickle.load(f)
    names.extend([name] * 100)
else:
    names = [name] * 100

with open(names_file_path, 'wb') as f:
    pickle.dump(names, f)

# Handle face data
faces_file_path = 'data/faces_data.pkl'
if os.path.exists(faces_file_path):
    with open(faces_file_path, 'rb') as f:
        existing_faces = pickle.load(f)

    # Ensure existing_faces is a numpy array
    existing_faces = np.array(existing_faces)

    # Check if existing_faces has the right shape, reshape if needed
    if existing_faces.ndim == 1:
        # We assume each face is a 50x50 image flattened into a 1D array of 7500 features
        if existing_faces.size == 100 * 50 * 50:
            existing_faces = existing_faces.reshape(100, 50 * 50)  # Reshape to (100, 7500)
        else:
            print("Existing faces data size is incorrect. Skipping reshaping.")
            existing_faces = np.array([])  # Empty array to avoid concatenation errors

    # Ensure new faces_data is reshaped correctly
    faces_data = faces_data.reshape(100, -1)

    # Concatenate the new faces with the existing ones (vstack to stack vertically)
    if existing_faces.size > 0:
        faces = np.vstack((existing_faces, faces_data))  # Stack the existing faces with new faces
    else:
        faces = faces_data

else:
    faces = faces_data

# Save the combined faces data
with open(faces_file_path, 'wb') as f:
    pickle.dump(faces, f)

print("Face data collection complete.")


