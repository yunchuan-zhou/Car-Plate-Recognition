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
            image = np.array(image)  # Convert PIL image to ndarray
            image = self.transform(image)  # Apply transformations (like resizing)

        return image, torch.tensor(label_encoded), len(label_encoded)

# Resize with Padding Transformation
class ResizeWithPadding:
    def __init__(self, target_height=32, target_width=128):
        self.target_height = target_height
        self.target_width = target_width

    def __call__(self, image):
        image = np.array(image)  # Convert PIL Image to NumPy array
        h, w = image.shape[:2]  # Height and width of the image

        # Calculate the scale factor to resize the image
        scale = self.target_height / h
        new_width = int(w * scale)

        if new_width > self.target_width:
            scale = self.target_width / w
            new_width = self.target_width
            resized_image = cv2.resize(image, (new_width, self.target_height))
        else:
            resized_image = cv2.resize(image, (new_width, self.target_height))

        padded_image = np.zeros((self.target_height, self.target_width), dtype=np.uint8)
        padded_image[:, :new_width] = resized_image  # Place resized image on the left

        return Image.fromarray(padded_image)


# Custom Collate Function
def custom_collate_fn(batch):
    images, labels, label_lengths = zip(*batch)
    max_label_length = max(label_lengths)

    # Pad labels to the maximum label length in the batch
    padded_labels = torch.zeros((len(labels), max_label_length), dtype=torch.long)
    for i, label in enumerate(labels):
        padded_labels[i, :len(label)] = label

    # Stack images (they're already tensors from the Dataset)
    images = torch.stack(images, dim=0)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)

    return images, padded_labels, label_lengths


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np

# CRNN Model (as defined earlier)
class CRNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CRNNModel, self).__init__()
        # self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm2d(32)
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(64)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.dropout2 = nn.Dropout(0.3)

        # self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # self.bn3 = nn.BatchNorm2d(128)
        # self.pool3 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        # self.dropout3 = nn.Dropout(0.3)

        # self.fc1 = nn.Linear(128 * 8, 64)

        # self.lstm1 = nn.LSTM(64, 256, bidirectional=True, batch_first=True, dropout=0.3)
        # self.lstm2 = nn.LSTM(512, 256, bidirectional=True, batch_first=True, dropout=0.3)

        # self.fc_out = nn.Linear(512, num_classes)
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
        # x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # x = self.dropout2(x)
        # x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        # x = self.dropout3(x)

        # b, c, h, w = x.size()
        # x = x.permute(0, 3, 1, 2)
        # x = x.reshape(b, w, c * h)

        # x = F.relu(self.fc1(x))
        # x, _ = self.lstm1(x)
        # x, _ = self.lstm2(x)
        # x = self.fc_out(x)
        # return F.log_softmax(x, dim=2)
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


def train_model(model, dataloader,val_loader, criterion, optimizer, num_epochs, characters):
    train_losses = []
    train_word_accuracies = []
    train_char_accuracies = []
    val_losses = []
    val_word_accuracies = []
    val_char_accuracies = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct_words = 0
        total_words = 0
        correct_chars = 0
        total_chars = 0
        matched_predictions = []
        mismatched_predictions = []

        for batch_idx, (images, labels, label_lengths) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.permute(1, 0, 2)
            input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long)

            loss = criterion(outputs, labels, input_lengths, label_lengths)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            predicted_sequences = outputs.argmax(2).permute(1, 0).tolist()
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
                    #matched_predictions.append((pred_text, label_text))
                    correct_words += 1
                #else:
                    #mismatched_predictions.append((pred_text, label_text))
                total_words += 1

        epoch_loss = running_loss / len(dataloader)
        word_accuracy = correct_words / total_words
        char_accuracy = correct_chars / total_chars

        train_losses.append(epoch_loss)
        train_word_accuracies.append(word_accuracy)
        train_char_accuracies.append(char_accuracy)

        print(f"Epoch [{epoch}/{num_epochs}] - Loss: {epoch_loss:.4f}, Word Accuracy: {word_accuracy:.4f}, Char Accuracy: {char_accuracy:.4f}")
        # print("\nMatched Predictions:")
        # for pred, label in matched_predictions[:10]:
        #     print(f"Prediction: {pred}, Label: {label}")

        # print("\nMismatched Predictions:")
        # for pred, label in mismatched_predictions[:10]:
        #     print(f"Prediction: {pred}, Label: {label}")

        # validation
        model.eval()
        running_loss = 0.0
        correct_words = 0
        total_words = 0
        correct_chars = 0
        total_chars = 0
        matched_predictions = []
        mismatched_predictions = []

        with torch.no_grad():
            for batch_idx, (images, labels, label_lengths) in enumerate(val_loader):
                outputs = model(images)
                outputs = outputs.permute(1, 0, 2)
                input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long)

                loss = criterion(outputs, labels, input_lengths, label_lengths)

                running_loss += loss.item()

                predicted_sequences = outputs.argmax(2).permute(1, 0).tolist()
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
                        #matched_predictions.append((pred_text, label_text))
                        correct_words += 1
                    #else:
                        #mismatched_predictions.append((pred_text, label_text))
                    total_words += 1

            epoch_loss = running_loss / len(val_loader)
            word_accuracy = correct_words / total_words
            char_accuracy = correct_chars / total_chars

            val_losses.append(epoch_loss)
            val_word_accuracies.append(word_accuracy)
            val_char_accuracies.append(char_accuracy)

    return train_losses, train_word_accuracies, train_char_accuracies, val_losses, val_word_accuracies, val_char_accuracies



def plot_metrics(train_losses, train_word_accuracies, train_char_accuracies,val_losses, val_word_accuracies, val_char_accuracies):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(18, 5))

    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Word accuracy plot
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_word_accuracies, label="Train Word Accuracy")
    plt.plot(epochs, val_word_accuracies, label="Val Word Accuracy")
    plt.title("Training Word Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Character accuracy plot
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_char_accuracies, label="Train Character Accuracy")
    plt.plot(epochs, val_char_accuracies, label="Val Character Accuracy")
    plt.title("Training Character Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Define transforms
    transform = transforms.Compose([
        transforms.Lambda(lambda img: Image.fromarray(img) if isinstance(img, np.ndarray) else img),
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    dataset = LicensePlateDataset(csv_file='labels_train.csv', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
    val_set = LicensePlateDataset(csv_file='labels_val.csv', transform=transform)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

    model = CRNNModel(num_classes=num_classes + 1)  # Add 1 for CTC blank index
    criterion = nn.CTCLoss(blank=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    train_losses, train_word_accuracies, train_char_accuracies , val_losses, val_word_accuracies, val_char_accuracies= train_model(
        model, dataloader,val_loader, criterion, optimizer, num_epochs=100, characters=characters
    )

    # Plot metrics
    plot_metrics(train_losses, train_word_accuracies, train_char_accuracies,val_losses, val_word_accuracies, val_char_accuracies)
