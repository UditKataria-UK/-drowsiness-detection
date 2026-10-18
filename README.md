# Driver Drowsiness Detection 🚗😴

A real-time driver drowsiness detection system built using **PyTorch, CNN, MediaPipe Face Landmarker, and OpenCV**.

The system detects whether the driver's eyes are open or closed and triggers a voice alert when the eyes remain closed for a continuous period.

## Features

- CNN-based eye state classification
- Open / Closed eye detection
- Real-time webcam detection
- MediaPipe Face Landmarker for eye localization
- Prediction smoothing for more stable results
- Drowsiness detection using continuous eye closure
- Voice alert when drowsiness is detected
- Single-image prediction
- Model evaluation on a separate test dataset

## Tech Stack

- Python
- PyTorch
- Torchvision
- OpenCV
- MediaPipe
- Pillow

## Project Structure

```text
open_closed_eye/
│
├── models/
│   └── face_landmarker.task
│
├── train.py
├── test.py
├── predict.py
├── webcam.py
├── split_dataset.py
├── drowsiness_model.pth
├── requirements.txt
├── .gitignore
└── README.md

The training and testing datasets are not included in this repository.

How It Works
Webcam
   ↓
MediaPipe Face Landmarker
   ↓
Eye Region Detection
   ↓
Eye Cropping
   ↓
CNN
   ↓
Open / Closed
   ↓
Prediction Smoothing
   ↓
Continuous Eye Closure
   ↓
Drowsiness Detection
   ↓
Voice Alert

CNN Architecture
The model uses two convolutional blocks:
Input Image
128 × 128 × 3
      ↓
Conv2D (3 → 32)
      ↓
ReLU
      ↓
MaxPool
      ↓
Conv2D (32 → 64)
      ↓
ReLU
      ↓
MaxPool
      ↓
Flatten
      ↓
Linear
      ↓
ReLU
      ↓
Linear
      ↓
2 Classes

The two output classes are:
0 → Closed_Eyes
1 → Open_Eyes

Training
The CNN is trained using:
- Loss Function: Cross Entropy Loss
- Optimizer: Adam
- Learning Rate: 0.001
- Batch Size: 32
- Epochs: 10
- Input Size: 128 × 128
Run training:
python train.py

The trained model is saved as:
drowsiness_model.pth

Model Evaluation
Run:
python test.py

The model achieved:
99.97% test accuracy
on the test dataset used for this project.
Single Image Prediction
To test an individual eye image:
python predict.py

Update the image_path inside predict.py with the image you want to classify.
Example output:
Prediction: Closed_Eyes
😴 DROWSY

or:
Prediction: Open_Eyes
👀 ALERT

Real-Time Webcam Detection
Run:
python webcam.py

The webcam captures the driver's face and MediaPipe identifies the eye regions.
The CNN then classifies the eyes as:
Open
Closed

If the eyes remain closed continuously for the configured duration, the system displays:
DROWSY

and triggers a voice alert.
Press:
Q

to exit the webcam window.
Installation
Create and activate a virtual environment:
python -m venv .venv

Activate it on macOS/Linux:
source .venv/bin/activate

Install dependencies:
pip install -r requirements.txt

Important
The MediaPipe Face Landmarker model file is required:
models/face_landmarker.task

The training and testing datasets are intentionally excluded from the repository because of their size.
Future Improvements
- Improve robustness under different lighting conditions
- Add eye aspect ratio based verification
- Add better temporal modeling
- Add GUI/web interface
- Support longer-term drowsiness monitoring
- Improve performance across different camera angles
Author
Udit Kataria
Built as a Deep Learning and Computer Vision project using PyTorch.