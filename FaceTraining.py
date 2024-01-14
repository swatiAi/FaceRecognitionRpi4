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
import csv

# Using LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
path = 'dataset'


# Function to read the images in the dataset, convert them to grayscale values, and return samples
def getImagesAndLabels(path):
    faceSamples = []
    ids = []
    unique_ids = set()

    for file_name in os.listdir(path):
        if file_name.endswith(".jpg"):
            id = int(file_name.split(".")[1])
            img_path = os.path.join(path, file_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            faces = face_detector.detectMultiScale(img)

            for (x, y, w, h) in faces:
                faceSamples.append(img[y:y + h, x:x + w])
                ids.append(id)
                unique_ids.add(id)

    return faceSamples, ids, unique_ids


faces, ids, unique_ids = getImagesAndLabels(path)


# Function To Train The Recognizer Using .yml File
def trainRecognizer(faces, ids):
    recognizer.train(faces, np.array(ids))
    recognizer.write('trainer/trainer.yml')


print("\nTraining Face Data. Please wait...")
# Get face samples, their corresponding labels, and unique IDs
faces, ids, unique_ids = getImagesAndLabels(path)

# Prompt user for names corresponding to each ID
id_to_name = {}
for uid in unique_ids:
    name = input(f"Enter name for ID {uid}: ")
    id_to_name[uid] = name

# Write ID-name pairs to a CSV file
with open('names.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'name'])
    for uid, name in id_to_name.items():
        writer.writerow([uid, name])

# Train the recognizer
trainRecognizer(faces, ids)

# Print the number of unique faces trained
num_faces_trained = len(set(ids))
if num_faces_trained == 1:
    print("\nFace Training Phase Complete!")
    print("\nOnly 1 Face Was Trained.")
else:
    print("\nFace Training Phase Complete!")
    print("\nTotal {} Faces Were Trained.".format(num_faces_trained))
