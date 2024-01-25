import cv2
import pickle
import numpy as np
import os
import time
from datetime import datetime

# Open a video capture object using the default camera (0)
video = cv2.VideoCapture(0)

# Load the Haar Cascade Classifier for face detection
facedetect = cv2.CascadeClassifier('Data/haarcascade_frontalface_default.xml')

# Initialize an empty list to store face data
faces_data = []

# Counter to keep track of the number of frames processed
i = 0

# Get user input for their name
name = input("Enter your name: ")

# Loop to capture video frames and detect faces
while True:
    # Capture a frame from the video
    ret, frame = video.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Crop the face region from the frame
        crop_img = frame[y:y+h, x:x+w, :]

        # Resize the cropped face image to 50x50 pixels
        resized_img = cv2.resize(crop_img, (50, 50))

        # Append the resized face image to the faces_data list every 5 frames
        if len(faces_data) <= 5 and i % 5 == 0:
            faces_data.append(resized_img)

        i = i + 1

        # Display the count of captured faces on the frame
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

    # Display the current frame with annotations
    cv2.imshow("Frame", frame)

    # Wait for a key press or until 5 faces are captured
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) == 5:
        break

# Release the video capture object and close all windows
video.release()
cv2.destroyAllWindows()

# Convert the list of face images to a NumPy array and reshape it
faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(5, -1)

# Check if 'names.pkl' is present in the 'Data/' directory
if 'names.pkl' not in os.listdir('Data/'):
    # If not present, create a list with the entered name repeated 5 times
    names = [name] * 5
    # Save the list to 'names.pkl'
    with open('Data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    # If 'names.pkl' is present, load the existing list
    with open('Data/names.pkl', 'rb') as f:
        names = pickle.load(f)
    # Append the entered name 5 times to the existing list
    names = names + [name] * 5
    # Save the updated list to 'names.pkl'
    with open('Data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

# Check if 'faces_data.pkl' is present in the 'Data/' directory
if 'faces_data.pkl' not in os.listdir('Data/'):
    # If not present, save the NumPy array 'faces_data' to 'faces_data.pkl'
    with open('Data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    # If 'faces_data.pkl' is present, load the existing array
    with open('Data/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    # Append the new array 'faces_data' to the existing array
    faces = np.append(faces, faces_data, axis=0)
    # Save the updated array to 'faces_data.pkl'
    with open('Data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)
