# Car-Plate-Recognition
This project implements a license plate recognition system using a Convolutional Recurrent Neural Network (CRNN) combined with Connectionist Temporal Classification (CTC) loss. The model is capable of recognizing alphanumeric characters on license plates from images.
## Requirements 
Ensure you have a compatible version of Python installed (e.g., Python 3.10 or later). Install the required libraries using the following command:  
```pip install -r requirements.txt```
## Dataset source:
European License Plates Dataset from Kaggle: https://www.kaggle.com/datasets/abdelhamidzakaria/european-license-plates-dataset 

## Project Structure

dataset_final/: Contains the dataset split into train, val, and test folders.

preprocessing_data.py: Generates a CSV file with image paths and corresponding labels for training, validation, and testing.

CRNN_model.py: Contains the implementation of the CRNN model architecture.

train.py: Script for training the CRNN model.

demo.py: Script for using the best_model to recoginze the characters of an input car plate image.

requirements.txt: Lists the dependencies needed for the project.

