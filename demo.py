import torch
import cv2
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


# Define characters and number of classes
characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
num_classes = len(characters)


class CRNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CRNNModel, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.3)  

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.3) 

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.dropout3 = nn.Dropout(0.4)  

        self.pool_final = nn.AdaptiveAvgPool2d((1, None))  

        self.lstm1 = nn.LSTM(256, 256, bidirectional=True, batch_first=True)
        self.dropout_lstm1 = nn.Dropout(0.5)  
        self.lstm2 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.dropout_lstm2 = nn.Dropout(0.5)  

        self.fc_out = nn.Linear(512, num_classes )   

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)
        x = self.pool_final(x)
        b, c, h, w = x.size()
        x = x.squeeze(2).permute(0, 2, 1)  # [batch, width, channel]
        x, _ = self.lstm1(x)
        x = self.dropout_lstm1(x)
        x, _ = self.lstm2(x)
        x = self.dropout_lstm2(x)
        x = self.fc_out(x)
        return F.log_softmax(x, dim=2)
    
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
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.subplots_adjust(top=0.9, bottom=0.25)
    plt.figtext(0.5, 0.2, f'Predicted Label: {decoded_text}', ha='center', fontsize=25, fontweight='bold')
    plt.show()

if __name__ == '__main__':

    model = CRNNModel(num_classes=num_classes + 1)
    model.load_state_dict(torch.load('model_weight/best_model_9.pth'))
    img_path = 'test_data/SPR911.JPG'
    predict_demo(model, img_path, characters)