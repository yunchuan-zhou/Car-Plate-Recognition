# import torch
# import torch.nn as nn
# import pandas as pd
# from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms
# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np
# import cv2

# # Character set and CRNN model structure
# characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
# num_classes = len(characters) + 1  # +1 for the blank token in CTC

# def encode_label(label):
#     return [characters.index(c) for c in label]

# class LicensePlateDataset(Dataset):
#     def __init__(self, csv_file, transform=None):
#         self.data = pd.read_csv(csv_file)
#         self.transform = transform

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         img_path = self.data.iloc[idx, 1]
#         label = self.data.iloc[idx, 2]
#         label_encoded = encode_label(label)
#         image = Image.open(img_path).convert("L")
#         if self.transform:
#             image = self.transform(image)
#         return image, torch.tensor(label_encoded), len(label_encoded)

# class ResizeWithPadding:
#     def __init__(self, target_height=32, target_width=128):
#         self.target_height = target_height
#         self.target_width = target_width

#     def __call__(self, image):
#         image = np.array(image)
#         h, w = image.shape[:2]
#         scale = self.target_height / h
#         new_width = int(w * scale)
#         resized_image = cv2.resize(image, (new_width, self.target_height))
#         padded_image = np.zeros((self.target_height, self.target_width), dtype=np.uint8)
#         padded_image[:, :min(new_width, self.target_width)] = resized_image[:, :min(new_width, self.target_width)]
#         return Image.fromarray(padded_image)

# class CRNN(nn.Module):
#     def __init__(self, num_classes):
#         super(CRNN, self).__init__()
#         # Convolutional layers
#         self.cnn = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(32),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(64),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(128),
#             nn.MaxPool2d(2, 2)
#         )
#         # RNN layers
#         self.rnn = nn.LSTM(128, 32, bidirectional=True, batch_first=True)
#         self.fc = nn.Linear(64, num_classes)

#     def forward(self, x):
#         x = self.cnn(x)  # (N, C, H, W)
#         b, c, h, w = x.size()
#         x = x.permute(0, 3, 1, 2).contiguous()  # (N, W, C, H)
#         x = x.view(b, w, -1)  # Flatten (N, W, C*H)
#         x, _ = self.rnn(x)
#         x = self.fc(x)
#         return x

# def test_model(model, dataloader, criterion):
#     model.eval()
#     running_loss = 0.0
#     with torch.no_grad():
#         for images, labels, label_lengths in dataloader:
#             images = images.unsqueeze(1)  # Add channel dimension (N, C, H, W)
#             outputs = model(images)
#             outputs = outputs.log_softmax(2).permute(1, 0, 2)  # (T, N, C)
#             input_lengths = torch.full((images.size(0),), outputs.size(0), dtype=torch.long)
#             loss = criterion(outputs, labels, input_lengths, label_lengths)
#             running_loss += loss.item()
#     return running_loss / len(dataloader)

# def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
#     train_losses = []
#     val_losses = []

#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0

#         for images, labels, label_lengths in train_loader:
#             images = images.unsqueeze(1)  # Add channel dimension (N, C, H, W)
#             outputs = model(images)
#             outputs = outputs.log_softmax(2).permute(1, 0, 2)  # (T, N, C)
#             input_lengths = torch.full((images.size(0),), outputs.size(0), dtype=torch.long)

#             loss = criterion(outputs, labels, input_lengths, label_lengths)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()

#         train_loss = running_loss / len(train_loader)
#         val_loss = test_model(model, val_loader, criterion)

#         train_losses.append(train_loss)
#         val_losses.append(val_loss)

#         print(f"Epoch [{epoch + 1}/{num_epochs}]")
#         print(f"  Training Loss: {train_loss:.4f}")
#         print(f"  Validation Loss: {val_loss:.4f}")

#     return train_losses, val_losses

# def plot_losses(train_losses, val_losses):
#     plt.figure(figsize=(10, 6))
#     plt.plot(train_losses, label="Training Loss")
#     plt.plot(val_losses, label="Validation Loss")
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     plt.title("Training and Validation Loss")
#     plt.legend()
#     plt.grid()
#     plt.show()

# if __name__ == '__main__':
#     # Transformations
#     transform = transforms.Compose([
#         ResizeWithPadding(target_height=32, target_width=128),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5], std=[0.5])
#     ])

#     # Dataset and DataLoader
#     train_dataset = LicensePlateDataset(csv_file='labels_train.csv', transform=transform)
#     val_dataset = LicensePlateDataset(csv_file='labels_val.csv', transform=transform)
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

#     # Model, criterion, optimizer
#     model = CRNN(num_classes=num_classes)
#     criterion = nn.CTCLoss(blank=num_classes - 1)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#     # Train the model
#     num_epochs = 10
#     train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)

#     # Plot losses
#     plot_losses(train_losses, val_losses)

#     # Save the trained model
#     torch.save(model.state_dict(), 'crnn_model.pth')
#     print("Model training complete. Saved to 'crnn_model.pth'")
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
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label_encoded), len(label_encoded)


# Resize with Padding Transformation
# Resize with Padding Transformation
class ResizeWithPadding:
    def __init__(self, target_height=32, target_width=128):
        self.target_height = target_height
        self.target_width = target_width

    def __call__(self, image):
        image = np.array(image)
        h, w = image.shape[:2]

        # Calculate the scale factor to resize the image
        scale = self.target_height / h
        new_width = int(w * scale)

        if new_width > self.target_width:
            # If the resized width exceeds the target width, scale down further
            scale = self.target_width / w
            new_width = self.target_width
            resized_image = cv2.resize(image, (new_width, self.target_height))
        else:
            resized_image = cv2.resize(image, (new_width, self.target_height))

        # Create a padded image with the target dimensions
        padded_image = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        padded_image[:, :new_width] = resized_image  # Place resized image on the left

        return Image.fromarray(padded_image)



# Custom Collate Function
def custom_collate_fn(batch):
    images, labels, label_lengths = zip(*batch)
    max_label_length = max(label_lengths)
    padded_labels = torch.zeros((len(labels), max_label_length), dtype=torch.long)
    for i, label in enumerate(labels):
        padded_labels[i, :len(label)] = label

    images = torch.stack(images, dim=0)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)
    return images, padded_labels, label_lengths

def decode_predictions(predictions, characters):
    """
    Decodes model predictions using greedy decoding.
    Removes repeated characters and blanks (CTC blank index).
    """
    blank_index = len(characters)  # Assume the blank index is the last one
    decoded_output = []

    for pred in predictions:
        pred_text = []
        prev_char = None
        for p in pred:
            if p == blank_index:  # Skip blank token
                prev_char = None
                continue
            if p != prev_char:  # Avoid repeated characters
                pred_text.append(characters[p])
            prev_char = p
        decoded_output.append("".join(pred_text))
    return decoded_output


# CRNN Model
class CRNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CRNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        self.flatten = nn.Flatten(start_dim=2)
        self.fc = nn.Linear(128 * 8, 64)

        self.lstm1 = nn.LSTM(64, 256, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)

        self.fc_out = nn.Linear(512, num_classes + 1)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        x = self.pool1(nn.ReLU()(self.bn1(self.conv1(x))))
        x = self.pool2(nn.ReLU()(self.bn2(self.conv2(x))))
        x = self.pool3(nn.ReLU()(self.bn3(self.conv3(x))))

        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2)
        x = self.flatten(x)
        x = self.fc(x)

        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        x = self.fc_out(x)
        return self.softmax(x)


def train_model(model, dataloader, criterion, optimizer, num_epochs):
    """
    Train the CRNN model while logging matched and mismatched predictions.

    Args:
        model: PyTorch model to train.
        dataloader: DataLoader providing the training data.
        criterion: Loss function (CTCLoss).
        optimizer: Optimizer for backpropagation.
        num_epochs: Number of training epochs.

    Returns:
        train_losses: List of average loss for each epoch.
        train_accuracies: List of accuracy for each epoch.
    """
    model.train()
    train_losses = []
    train_accuracies = []

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch [{epoch}/{num_epochs}] - Starting training...")
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        mismatched_predictions = []
        matched_predictions = []

        for batch_idx, (images, labels, label_lengths) in enumerate(dataloader):
            outputs = model(images)
            outputs = outputs.permute(1, 0, 2)  # Shape: (seq_len, batch_size, num_classes)
            input_lengths = torch.full((images.size(0),), outputs.size(0), dtype=torch.long)

            loss = criterion(outputs, labels, input_lengths, label_lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Decode predictions and labels
            predicted_sequences = outputs.argmax(2).permute(1, 0).tolist()
            label_texts = [
                "".join([characters[l] for l in label if l < len(characters)])
                for label in labels.tolist()
            ]

            for pred_seq, label_text in zip(predicted_sequences, label_texts):
                pred_text = "".join(
                    [characters[p] if p < len(characters) else "?" for p in pred_seq]
                )
                if pred_text == label_text:
                    matched_predictions.append((pred_text, label_text))
                    correct_predictions += 1
                else:
                    mismatched_predictions.append((pred_text, label_text))
                total_predictions += 1

        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = correct_predictions / total_predictions
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        print(f"Epoch [{epoch}/{num_epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
        print(f"Total Predictions: {total_predictions}, Matched Predictions: {correct_predictions}, Mismatched Predictions: {len(mismatched_predictions)}")

        # Log matched predictions
        print(f"\nMatched Predictions (Epoch {epoch}):")
        for pred, label in matched_predictions[:15]:
            print(f"Prediction: {pred}, Label: {label}")

        # Log mismatched predictions
        print(f"\nMismatched Predictions (Epoch {epoch}):")
        for pred, label in mismatched_predictions[:15]:
            print(f"Prediction: {pred}, Label: {label}")

        # Print counts
        print(f"\nNumber of tests in epoch: {total_predictions}")
        print(f"Number of mismatched predictions in epoch: {len(mismatched_predictions)}\n")

    return train_losses, train_accuracies



# Plot Loss and Accuracy
def plot_metrics(train_losses, train_accuracies):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Accuracy")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


# Main Function
if __name__ == '__main__':
    transform = transforms.Compose([
        ResizeWithPadding(target_height=32, target_width=128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = LicensePlateDataset(csv_file='labels_train.csv', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)

    model = CRNNModel(num_classes=num_classes)
    criterion = nn.CTCLoss(blank=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, train_accuracies = train_model(model, dataloader, criterion, optimizer, num_epochs=150)
    plot_metrics(train_losses, train_accuracies)



