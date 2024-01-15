# Author: Usman Javed Swati
# GitHub: swatiAi
# Date: 15/01/2024
#
# Copyright (c) 2024 Usman Swati
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import cv2
import os
import numpy as np
from picamera2 import Picamera2
import csv

id = 0

# Choosing Haar Cascade Classifier As Face Detector
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Using LBPH Face Recognizer and Reading Data From .yml file
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

# Reading Data From .csv File
names = {}
with open('names.csv', mode='r') as csvfile:
    csv_reader = csv.DictReader(csvfile)
    for row in csv_reader:
        id = int(row["id"])  # Convert the id to an integer
        names[id] = row["name"]

# Settings For The Camera
cam = Picamera2()
cam.preview_configuration.main.size = (640, 360)
cam.preview_configuration.main.format = "RGB888"
cam.preview_configuration.controls.FrameRate = 60
cam.preview_configuration.align()
cam.configure("preview")
cam.start()

while True:
    # For Capturing Frames From The Camera
    current_frame = cam.capture_array()

    # Conversion from Color to Greyscale
    frameGray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Detection of Faces from Greyscale Image Using Haar Classifier
    faces = face_detector.detectMultiScale(
        frameGray,  # Analyzing the grayscale image for facial features
        scaleFactor=1.1,  # Modifying the image scale by 10% at each iteration for improved detection
        minNeighbors=5,  # Specifying the minimum neighbors required for retaining a candidate rectangle
        minSize=(30, 30)  # Defining the minimum size of objects; smaller objects are disregarded
    )

    for (x, y, w, h) in faces:
        # Determine name position by shifting right and upwards, outside the top of the bounding box
        name_position = (x + 5, y - 5)

        # Adjust confidence position by shifting right and upwards, inside the bottom of the bounding box
        confidence_position = (x + 5, y + h - 5)

        # Form a bounding box around the identified face
        cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 255),
                      3)  # Parameters: frame, top-left coordinates, bottom-right coordinates, box color, thickness

        # Utilize the recognizer.predict() method to obtain the predicted label (id) and confidence score for the
        # facial region
        predicted_id, confidence = recognizer.predict(frameGray[y:y + h, x:x + w])

        # Retrieve the associated name from the dictionary; if not found, use "unknown"
        name = names.get(predicted_id, "unknown")

        # Consider it a perfect match if confidence is less than 100
        if confidence < 100:
            predicted_id = names[predicted_id]
            confidence = f"{100 - confidence:.0f}%"
        else:
            predicted_id = "Unknown Entity"
            confidence = f"{100 - confidence:.0f}%"

        # Display Name And Confidence Of The Recognized Face
        cv2.putText(current_frame, str(name), name_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(current_frame, str(confidence), confidence_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

    exit_text = "Press 'q' To Exit"
    exit_text_size = cv2.getTextSize(exit_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
    exit_text_pos = (current_frame.shape[1] - exit_text_size[0] - 10, current_frame.shape[0] - 10)
    cv2.putText(current_frame, exit_text, exit_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

    # Display the frame to the user
    cv2.imshow('YourFace', current_frame)

    # Wait For 30 ms For 'q' If Pressed Then Close Program
    key = cv2.waitKey(30) & 0xff

    # Press 'q' To Close The Program Or Wait For Count To Complete Then Close The Program
    if key == 113:  # 'q' key
        break

# Cleanup
cam.stop()
cv2.destroyAllWindows()
