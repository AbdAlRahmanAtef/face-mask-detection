# Face Mask Detection

This project involves the development of a real-time face mask detection system using computer vision and machine learning techniques. The system classifies individuals as wearing a mask or not, based on input from a webcam feed or images.

## Project Overview

Face mask detection is a binary classification problem solved using a Convolutional Neural Network (CNN) built on top of the MobileNetV2 architecture. The project integrates:

- Real-time webcam detection
- Pretrained neural networks for feature extraction
- Data augmentation for enhanced training performance
- Performance visualization of training and testing accuracy

## Features

- **Real-time Face Mask Detection**: Detects faces in a webcam stream and classifies them.
- **Deep Learning-Based Classification**: Utilizes MobileNetV2 for efficient and accurate feature extraction.
- **Data Augmentation**: Includes rotation, zooming, width/height shifting, and flipping.

## Prerequisites

Before running the project, ensure you have the following installed:

- Python
- TensorFlow
- OpenCV
- NumPy
- Matplotlib
- imutils
- scikit-learn

## Installation

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/AbdAlRahmanAtef/face-mask-detection.git
   cd face-mask-detection
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The dataset is organized into two categories:

- `with_mask`: Images of people wearing masks
- `without_mask`: Images of people not wearing masks

Ensure the dataset is in the correct directory structure as shown above.

## Training the Model

To train the face mask detection model:

1. Adjust the dataset path in the training script:
   ```python
   DIRECTORY = "path/to/your/dataset"
   ```
2. Run the training script:
   ```bash
   python train_mask_detector.py
   ```
3. The trained model will be saved as `mask_detector.h5`.

## Real-Time Mask Detection

To run the real-time detection system:

1. Ensure your webcam is connected.
2. Execute the detection script:
   ```bash
   python detect_mask_video.py
   ```

## Visualization

The training script generates plots for:

- Training and validation loss
- Training and validation accuracy

These are displayed using Matplotlib at the end of the training process.

## Future Enhancements

- Add support for detecting multiple classes of masks.
- Optimize the model for deployment on edge devices.
- Implement a mobile application interface.

## Acknowledgements

- [MobileNetV2](https://arxiv.org/abs/1801.04381): For efficient feature extraction.
- [OpenCV](https://opencv.org/): For face detection and webcam integration.
- The [face mask dataset](#): Include the source of your dataset if publicly available.

# face-mask-detection
