# Import necessary libraries
from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime

# Import text-to-speech functionality
from win32com.client import Dispatch

# Function to speak text
def speak(str1):
    speak=Dispatch(("SAPI.SpVoice"))
    speak.Speak(str1)

# Open a video capture object using the default camera (0)
video = cv2.VideoCapture(0)

# Load the Haar Cascade Classifier for face detection
facedetect = cv2.CascadeClassifier('Data/haarcascade_frontalface_default.xml')

# Load pre-trained face recognition data from pickle files
with open('Data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('Data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Print the shape of the Faces matrix
print('Shape of Faces matrix --> ', FACES.shape)

# Initialize a K-Nearest Neighbors classifier with 5 neighbors
knn = KNeighborsClassifier(n_neighbors=5)

# Train the KNN classifier with the loaded face data and labels
knn.fit(FACES, LABELS)

# Define column names for attendance CSV file
COL_NAMES = ['NAME', 'TIME']

# Start an infinite loop for real-time face recognition
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

        # Resize the cropped face image to 50x50 pixels and flatten it
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)

        # Predict the identity of the face using the trained KNN classifier
        output = knn.predict(resized_img)

        # Get current timestamp
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")

        # Check if an attendance file for the current date already exists
        exist = os.path.isfile("Attendance_" + date + ".csv")

        # Draw rectangles and text on the frame for visualization
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

        # Create an attendance record with predicted identity and timestamp
        attendance = [str(output[0]), str(timestamp)]

    # Display the current frame with annotations
    cv2.imshow("Frame", frame)

    # Wait for a key press
    k = cv2.waitKey(1)

    # If 'o' is pressed, announce attendance and save it to a CSV file
    if k == ord('o'):
        speak("Attendance Taken..")
        time.sleep(5)
        if exist:
            # If file exists, append attendance to it
            with open("Attendance_" + date + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendance)
            csvfile.close()
        else:
            # If file doesn't exist, create it and write column names and attendance
            with open("Attendance_" + date + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)
            csvfile.close()

    # If 'q' is pressed, exit the loop
    if k == ord('q'):
        break

# Release the video capture object and close all windows
video.release()
cv2.destroyAllWindows()
