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
from picamera2 import Picamera2

# Choosing Haar Cascade Classifier As Face Detector
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Settings For The Camera
camera = Picamera2()
camera.preview_configuration.main.size = (640, 480)
camera.preview_configuration.main.format = "RGB888"
camera.preview_configuration.controls.FrameRate = 60
camera.preview_configuration.align()
camera.configure("preview")
camera.start()


def capture_and_save(frame, user_id, capture_count):
    # Conversion from Color to Greyscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detection of Faces from Greyscale Image Using Haar Classifier
    faces = face_detector.detectMultiScale(
        frame_gray,  # Analyzing the grayscale image for facial features
        scaleFactor=1.1,  # Modifying the image scale by 10% at each iteration for improved detection
        minNeighbors=5,  # Specifying the minimum neighbors required for retaining a candidate rectangle
        minSize=(30, 30)  # Defining the minimum size of objects; smaller objects are disregarded
    )

    for (x, y, w, h) in faces:
        # Create an outline around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255),
                      3)  # 5 parameters - frame, topleftcoords,bottomrightcooords,boxcolor,thickness

        capture_count += 1  # For limiting the image capture

        # Save the resized greyscale face to 'dataset' folder
        file_path = os.path.join("dataset", f"User.{user_id}.{capture_count}.jpg")

        # Write the images
        cv2.imwrite(file_path, frame_gray[y:y + h, x:x + w])

    # Display the image counter
    cv2.putText(frame, f'Count: {capture_count}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)

    exit_text = "Press 'q' To Exit"
    exit_text_size = cv2.getTextSize(exit_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
    exit_text_pos = (frame.shape[1] - exit_text_size[0] - 10, frame.shape[0] - 10)
    cv2.putText(frame, exit_text, exit_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

    # Display the frame to the user
    cv2.imshow('YourFace', frame)

    return capture_count


def main():
    capture_count = 0

    user_id = input('\nEnter User ID: ')
    print("\nInitializing Face Capture! Please Look Straight Into The Camera!")

    while True:
        # For Capturing Frames From The Camera
        current_frame = camera.capture_array()

        # Save Faces In The Current Frame
        capture_count = capture_and_save(current_frame, user_id, capture_count)

        # Wait For 30 ms For 'q' If Pressed Then Close Program
        key = cv2.waitKey(30) & 0xff

        # Press 'q' To Close The Program Or Wait For Count To Complete Then Close The Program
        if key == 113:  # 'q' key
            break
        elif capture_count >= 10:
            print(f"\nYour Face ID : {user_id}.")
            print("\nThe Process Is Complete ! Thank You For Your Time :) ")
            break

    # Cleanup
    camera.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
