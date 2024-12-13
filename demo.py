import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from CRNN_model import CRNNModel

# Define characters and number of classes
characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
num_classes = len(characters)


    
# Decode predictions for analysis
def decode_predictions(predictions, characters):
    blank_index = len(characters)  # Assuming blank token is the last index
    decoded_output = []
    for pred in predictions:
        pred_text = []
        prev_char = None
        for p in pred:
            if p == blank_index:  # Skip blank token
                prev_char = None
                continue
            if p != prev_char:  # Skip repeated characters
                pred_text.append(characters[p])
            prev_char = p
        decoded_output.append("".join(pred_text))
    return decoded_output


def predict_demo(model, img_path, characters):
    transform_test = transforms.Compose([
        transforms.Lambda(lambda img: Image.fromarray(img) if isinstance(img, np.ndarray) else img),
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    image = Image.open(img_path)
    img_L = image.convert("L")  
    image_tensor = transform_test(img_L).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        output = output.permute(1, 0, 2)  
        predicted_sequence = output.argmax(2).permute(1,0).tolist()
    
    
    decoded_text = decode_predictions(predicted_sequence, characters)[0]
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Display the image on the left
    axs[0].imshow(image, cmap='gray')
    axs[0].axis('off')  # Remove axes for a cleaner look

    # Add the predicted label on the right with adjusted horizontal position
    axs[1].text(0.2, 0.6, 'Predicted Label:', ha='center', va='center', fontsize=20, fontweight='bold')
    axs[1].text(0.2, 0.4, decoded_text, ha='center', va='center', fontsize=25, fontweight='bold')
    axs[1].axis('off')  # Remove axes

    # Adjust the layout to prevent overlapping
    plt.subplots_adjust(wspace=0.4)

    # Show the plot
    plt.show()

if __name__ == '__main__':

    model = CRNNModel(num_classes=num_classes + 1)
    model.load_state_dict(torch.load('model_weight/best_model_9.pth'))
    img_path = '...' # Path to your image file
    predict_demo(model, img_path, characters)