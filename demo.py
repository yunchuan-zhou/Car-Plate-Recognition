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

# Dataset Class
class LicensePlateDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]
        label = self.data.iloc[idx, 2]
        label_encoded = [characters.index(c) for c in label]
        image = Image.open(img_path).convert("L")  # Ensure grayscale image

        if self.transform:
            #image = np.array(image)  # Convert PIL image to ndarray
            image = self.transform(image)  

        return image, torch.tensor(label_encoded), len(label_encoded)
    
# Custom Collate Function
def custom_collate_fn(batch):
    images, labels, label_lengths = zip(*batch)
    max_label_length = max(label_lengths)

    # Pad labels to the maximum label length in the batch
    pad_index = len(characters)
    padded_labels = torch.full((len(labels), max_label_length), pad_index, dtype=torch.long)
    for i, label in enumerate(labels):
        padded_labels[i, :len(label)] = label

    # Stack images (they're already tensors from the Dataset)
    images = torch.stack(images, dim=0)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)

    return images, padded_labels, label_lengths

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


def demo(model, test_loader, characters):
    model.eval()
    correct_chars = 0
    total_chars = 0
    with torch.no_grad():
        for i, (images, labels, label_lengths) in enumerate(test_loader):
            images = images
            labels = labels
            label_lengths = label_lengths
            

            output = model(images)
            output = output.permute(1, 0, 2)  # [width, batch, classes]

            predicted_sequences = output.argmax(2).permute(1, 0).tolist()

            label_texts = [
                "".join([characters[l] for l in label if l < len(characters)])
                for label in labels.tolist()
            ]
            decoded_predictions = decode_predictions(predicted_sequences, characters)

            for pred_text, label_text in zip(decoded_predictions, label_texts):
                # Character accuracy
                correct_chars += sum(p == l for p, l in zip(pred_text, label_text))
                total_chars += len(label_text)
                # Word accuracy
                if pred_text == label_text:
                    print(f"Prediction: {pred_text}, Label: {label_text}")

        char_accuracy = correct_chars / total_chars
        print(f"Total Character Accuracy: {char_accuracy:.4f}")

if __name__ == '__main__':

    transform_test = transforms.Compose([
        transforms.Lambda(lambda img: Image.fromarray(img) if isinstance(img, np.ndarray) else img),
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    batch_size = 32
    test_dataset = LicensePlateDataset(csv_file='test.csv', transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    model = CRNNModel(num_classes=num_classes + 1)
    model.load_state_dict(torch.load('best_model_9.pth'))
    demo(model, test_loader, characters)