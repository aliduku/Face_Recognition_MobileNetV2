# Face Recognition with Deep Learning

This repository contains code to perform face recognition using deep learning with the Labeled Faces in the Wild dataset and MobileNetV2 architecture. It includes a Colab notebook for training the model and a Python script for testing the model with a webcam.

## Overview

The project aims to demonstrate face recognition using deep learning techniques. The MobileNetV2 architecture is used for feature extraction, and the model is trained on the Labeled Faces in the Wild dataset. The trained model can then be used to predict the identity of faces captured from a webcam.

## Description

- `Face_Recognition_MobileNetV2.ipynb`: A Jupyter Notebook hosted on Google Colab for training the face recognition model using the MobileNetV2 architecture. The notebook covers dataset loading, model building, training, evaluation, and visualization of results.

- `Face_Recognition_Webcam.py`: A Python script that uses the trained model to perform real-time face recognition using a webcam. Detected faces are labeled with predicted identities and confidence scores.

## Steps to Use the Colab Notebook

1. Open the `Face_Recognition_MobileNetV2.ipynb` notebook in Google Colab.
2. Run the notebook cells sequentially to train the face recognition model.
3. The notebook will provide training accuracy, loss, and a confusion matrix for evaluation.
4. Trained weights will be saved as `weights.h5`.

## Steps to Use the Webcam Face Recognition

1. Ensure you have the required packages installed by running `pip install opencv-python keras scikit-learn imbalanced-learn` in your environment.
2. Save the `Face_Recognition_Webcam.py` script in your local directory.
3. Download the trained `weights.h5` file from the Colab notebook and place it in the same directory.
4. Run the script using `python Face_Recognition_Webcam.py`.
5. The webcam feed will display with predicted identities and confidence scores for detected faces.

## Metrics and Results

The Colab notebook provides training history graphs, a confusion matrix, and a classification report for evaluation. Here are some key metrics from the training:

- Test Loss: 0.2032
- Test Accuracy: 0.9504

A normalized confusion matrix and a classification report with precision, recall, and F1-score for each class are also included.

## Acknowledgments

The code in this repository is based on the Labeled Faces in the Wild dataset and MobileNetV2 architecture. The implementation leverages the capabilities of Keras, OpenCV, and scikit-learn.

## Disclaimer

This project is intended for educational purposes and demonstrates the use of deep learning for face recognition. It's important to respect privacy and ethical considerations when applying face recognition technology.
