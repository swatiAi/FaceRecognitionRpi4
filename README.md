**Face Recognition System**

**Overview**

This project is a comprehensive face recognition system, involving data collection, model training, and face recognition processes. It is designed to be user-friendly and adaptable for various applications.

**Requirements**

- Raspberry Pi 4
- Raspbian Or Other Linux Distro
- Python
- numpy
- opencv-python
- picamera2
- opencv-contrib-python
- pillow

**Installation**

Clone or download the repository:

git clone [[https://github.com/swatiAi/FaceRecognitionRpi4](https://github.com/swatiAi/FaceRecognitionRpi4)]

**Usage**

**FaceDataset.py**

- **Purpose** : This script is used to create and manage a dataset of face images. It involves capturing images from a video source and storing them in a structured format.
- **Execution** :

python FaceDataset.py --user\_id [user\_id] --dataset\_path [path]

- **Key Features** :
  - Automated face detection and cropping using OpenCV.
  - Saving images with a unique identifier for everyone.
  - Customizable image count and storage location.

**FaceTraining.py**

- **Purpose** : This script trains a face recognition model using the collected dataset. It involves preprocessing the images, extracting features, and training a classifier.
- **Execution** :

python FaceTraining.py --dataset\_path [path] --model\_path [path]]

- **Key Features** :
  - Feature extraction using techniques like LBPH (Local Binary Patterns Histograms).
  - Classifier training, commonly with algorithms like SVM (Support Vector Machine) or neural networks.
  - Model serialization for later use in recognition tasks.

**FaceRecognition.py**

- **Purpose** : This script uses the trained model to recognize faces in real-time or from stored images.
- **Execution** :

python FaceRecognition.py --model\_path [path] --source [camera/file]

- **Key Features** :
  - Real-time face detection and recognition using the trained model.
  - Capability to recognize multiple faces simultaneously.
  - Options for source input: live camera feed or image files.

**Contributing**

Contributions to enhance or fix issues in the project are highly welcomed. Please adhere to the code structure and document your changes.

**License**

This project is licensed under the MIT License. See the LICENSE file for detailed terms and conditions.

**Contact**

For queries or suggestions, feel free to contact [usmansawati@gmail.com].
