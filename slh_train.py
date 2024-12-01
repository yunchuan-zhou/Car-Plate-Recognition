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


import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

# Import your model configuration
from CRNN_GRU import characters, num_classes


# Dataset class
class LicensePlateDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]
        label = self.data.iloc[idx, 2]
        label_encoded = encode_label(label)  # Encode label to indices
        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label_encoded), len(label_encoded)


# Resize with padding
class ResizeWithPadding:
    def __init__(self, target_height=32, target_width=128):
        self.target_height = target_height
        self.target_width = target_width

    def __call__(self, image):
        image = np.array(image)
        h, w = image.shape[:2]

        # Resize the image while maintaining the aspect ratio
        scale = self.target_height / h
        new_width = int(w * scale)
        resized_image = cv2.resize(image, (new_width, self.target_height))

        # Pad the image to the target width
        padded_image = np.zeros((self.target_height, self.target_width), dtype=np.uint8)
        padded_image[:, :min(new_width, self.target_width)] = resized_image[:, :min(new_width, self.target_width)]

        # Convert to 3-channel RGB by stacking
        padded_image = np.stack([padded_image] * 3, axis=-1)

        return Image.fromarray(padded_image)


# Custom collation function for variable-length labels
def custom_collate_fn(batch):
    images, labels, label_lengths = zip(*batch)

    # Pad labels to the maximum length in this batch
    max_label_length = max(label_lengths)
    padded_labels = torch.zeros((len(labels), max_label_length), dtype=torch.long)
    for i, label in enumerate(labels):
        padded_labels[i, :len(label)] = label

    # Stack images and label lengths
    images = torch.stack(images, dim=0)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)

    return images, padded_labels, label_lengths


# Encode labels into indices
def encode_label(label):
    return [characters.index(c) for c in label]


def decode_output(output):
    """
    Decodes a single prediction or a list of predictions.
    """
    if isinstance(output, int):  # Handle single integers
        return characters[output] if output < len(characters) else ""
    elif isinstance(output, list):  # Handle list of indices
        return ''.join([characters[i] for i in output if i < len(characters)])
    else:
        raise TypeError(f"Expected int or list, but got {type(output)}")



# Calculate accuracy
# Function to calculate accuracy
def calculate_accuracy(predictions, labels):
    """
    Calculate the percentage of correct predictions.
    Each prediction is compared to its corresponding label.
    """
    correct = 0
    total = len(labels)

    for pred_seq, label_seq in zip(predictions, labels):
        pred_text = decode_output(pred_seq)
        label_text = decode_output(label_seq)
        if pred_text == label_text:
            correct += 1

    return correct / total if total > 0 else 0



# Model Definition
class VGG_GRU_CTC_Model(nn.Module):
    def __init__(self):
        super(VGG_GRU_CTC_Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x128 -> 16x64
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x64 -> 8x32
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1)),  # 8x32 -> 4x33
        )
        self.rnn = nn.GRU(input_size=256 * 4, hidden_size=256, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(256 * 2, num_classes)  # +1 for blank token

    def forward(self, x):
        x = self.features(x)  # Shape: (batch_size, 256, 4, 33)

        # Reshape for RNN
        batch_size, channels, height, width = x.size()
        x = x.permute(0, 3, 1, 2)  # (batch_size, width, channels, height)
        x = x.contiguous().view(batch_size, width, -1)  # (batch_size, seq_len, features)

        x, _ = self.rnn(x)  # (batch_size, seq_len, hidden_size*2)
        x = self.classifier(x)  # (batch_size, seq_len, num_classes)
        return x
import matplotlib.pyplot as plt

# Main Training Loop
if __name__ == '__main__':
    print("Initializing dataset and dataloader...")
    transform = transforms.Compose([
        ResizeWithPadding(target_height=32, target_width=128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = LicensePlateDataset(csv_file='labels_train.csv', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
    print("Dataset loaded successfully!")
    print(f"Number of samples: {len(dataset)}")

    model = VGG_GRU_CTC_Model()

    criterion = nn.CTCLoss(blank=num_classes - 1)  # Blank token index is the last one
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  # Adjust learning rate

    print("Starting training...")
    num_epochs = 150
    train_accuracies = []

    # Training Loop
    # Training Loop with Debugging Logs
    # Training Loop with Debugging Logs
    losses = []
    accuracies = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        mismatched_predictions = []  # Reset mismatched predictions for each epoch

        print(f"\nEpoch [{epoch + 1}/{num_epochs}] - Starting training...")

        for batch_idx, (images, labels, label_lengths) in enumerate(dataloader):
            outputs = model(images)  # Forward pass
            outputs = outputs.permute(1, 0, 2).log_softmax(2)  # Adjust shape for CTCLoss
            input_lengths = torch.full((images.size(0),), outputs.size(0), dtype=torch.long)

            # Compute loss
            loss = criterion(outputs, labels, input_lengths, label_lengths.clone().detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Decode predictions for accuracy calculation
            predicted_sequences = outputs.argmax(2).permute(1, 0).tolist()  # Ensure list of lists
            batch_accuracy = calculate_accuracy(predicted_sequences, labels.tolist())
            correct_predictions += batch_accuracy * len(labels)
            total_samples += len(labels)

            # Log mismatched predictions for this batch
            for pred_seq, label_seq in zip(predicted_sequences, labels.tolist()):
                pred_text = decode_output(pred_seq)
                label_text = decode_output(label_seq)
                if pred_text != label_text:
                    mismatched_predictions.append((pred_text, label_text))

            # Debugging log: print predictions vs. labels for the first batch in the epoch
            if batch_idx == 0:
                print("\nSample Predictions vs. Labels (Batch 0):")
                for pred_seq, label_seq in zip(predicted_sequences[:5], labels.tolist()[:5]):
                    pred_text = decode_output(pred_seq)
                    label_text = decode_output(label_seq)
                    print(f"Prediction: {pred_text}, Label: {label_text}")

        # Record epoch loss and accuracy
        epoch_loss = total_loss / len(dataloader)
        epoch_accuracy = correct_predictions / total_samples
        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)

        # Log the total number of predictions and mismatched predictions
        total_mismatches = len(mismatched_predictions)
        print(f"\nEpoch [{epoch + 1}/{num_epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
        print(f"Total Predictions: {total_samples}, Mismatched Predictions: {total_mismatches}")

        # Print mismatched predictions for this epoch (first 10 for brevity)
        if total_mismatches > 0:
            print(f"\nMismatched Predictions (Epoch {epoch + 1}):")
            for pred, label in mismatched_predictions[:30]:
                print(f"Prediction: {pred}, Label: {label}")

    # Plot Loss and Accuracy
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), losses, marker='o', linestyle='-', color='b')
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.5)

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), accuracies, marker='o', linestyle='-', color='g')
    plt.title("Training Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(alpha=0.5)

    plt.tight_layout()
    plt.show()

