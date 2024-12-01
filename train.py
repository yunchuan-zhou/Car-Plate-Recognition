import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import random

from CRNN_GRU import VGG_GRU_CTC_Model,characters,num_classes

class LicensePlateDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]  
        label = self.data.iloc[idx, 2]    
        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, label
    
def encode_label(label):
    # Convert string label to a list of integer indices based on characters
    return [characters.index(c) for c in label]


def decode_output(output):
    # Convert the indices back to characters
    return ''.join([characters[i] for i in output])

# class LicensePlateDataset(Dataset):
#     def __init__(self, csv_file, transform=None):
#         self.data = pd.read_csv(csv_file)
#         self.transform = transform

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         img_path = self.data.iloc[idx, 1]  
#         label = self.data.iloc[idx, 2]     
#         label_encoded = encode_label(label)  # Encode label to indices
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

#         # Resize the image while maintaining the aspect ratio
#         scale = self.target_height / h
#         new_width = int(w * scale)
#         resized_image = cv2.resize(image, (new_width, self.target_height))

#         # Pad the image to the target width
#         padded_image = np.zeros((self.target_height, self.target_width), dtype=np.uint8)
#         padded_image[:, :min(new_width, self.target_width)] = resized_image[:, :min(new_width, self.target_width)]

#         # Convert to 3-channel RGB by stacking
#         padded_image = np.stack([padded_image] * 3, axis=-1)

#         return Image.fromarray(padded_image)

    
# def custom_collate_fn(batch):
#     images, labels, label_lengths = zip(*batch)  # Unpack the batch

#     # Pad labels to the maximum length in this batch
#     max_label_length = max(label_lengths)
#     padded_labels = torch.zeros((len(labels), max_label_length), dtype=torch.long)
#     for i, label in enumerate(labels):
#         padded_labels[i, :len(label)] = label  # Pad each label to the right

#     # Stack images and label lengths
#     images = torch.stack(images, dim=0)  # Stack images into a batch tensor
#     label_lengths = torch.tensor(label_lengths, dtype=torch.long)

#     return images, padded_labels, label_lengths

    
# if __name__ == '__main__':

#     '''
#     # normalize the images
#     transform = transforms.Compose([
#         ResizeWithPadding(target_height=32, target_width=128),
#         #transforms.RandomRotation(5),  
#         #transforms.ColorJitter(brightness=0.2, contrast=0.2),  
#         transforms.ToTensor(),  
#         transforms.Normalize(mean=[0.5], std=[0.5])  
#     ])
#     dataset = LicensePlateDataset(csv_file='labels_train.csv', transform=transform)


#     batch_size = 32
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     # test and visualize the dataloader
#     images, labels = next(iter(dataloader))
#     fig, axes = plt.subplots(1, 5, figsize=(15, 5))
#     for i in range(5):
#         image = images[i].permute(1, 2, 0).numpy()  
#         image = (image * 0.5) + 0.5  
#         axes[i].imshow(image.squeeze(), cmap='gray')  
#         axes[i].set_title(f"Label: {labels[i]}")
#         axes[i].axis("off")
#     plt.show()
#     '''


#     # Test the transformation
#     transform = transforms.Compose([
#         ResizeWithPadding(target_height=32, target_width=128),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Adjust for 3 channels
#     ])

#     dataset = LicensePlateDataset(csv_file='labels_train.csv', transform=transform)
#     image, label, label_length = dataset[0]
#     print(f"Image shape: {image.shape}")  # Should output (3, 32, 128)

#     # Visualize the transformed image
#     plt.imshow(image.permute(1, 2, 0).numpy() * 0.5 + 0.5)  # Denormalize for visualization
#     plt.title(f"Label: {label}")
#     plt.axis("off")
#     plt.show()

    
#     # load data
#     transform = transforms.Compose([
#         ResizeWithPadding(target_height=32, target_width=128),
#         #transforms.RandomRotation(5),  
#         #transforms.ColorJitter(brightness=0.2, contrast=0.2),  
#         transforms.ToTensor(),  
#         transforms.Normalize(mean=[0.5], std=[0.5])  
#     ])

#     dataset = LicensePlateDataset(csv_file='labels_train.csv', transform=transform)
#     dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)


#     model = VGG_GRU_CTC_Model()

#     #   CTC loss
#     criterion = nn.CTCLoss(blank=num_classes-1)  # blank token is used to represent the "no character" label
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#     # train the model
#     # num_epochs = 10
#     # for epoch in range(num_epochs):
#     #     model.train()
#     #     running_loss = 0.0
#     #     for images, labels, label_lengths in dataloader:
            
#     #         outputs = model(images)
            
            
#     #         # CTC loss requires the output shape of (T, N, C) (time step, batch size, class)
#     #         # CTC loss requires the label shape of (N, S) (batch size, label max length)
#     #         # CTC loss  requires the label length (N,)
#     #         outputs = outputs.log_softmax(2)  
#     #         loss = criterion(outputs, labels, label_lengths, torch.full((images.size(0),), outputs.size(1), dtype=torch.long))

            
#     #         optimizer.zero_grad()
#     #         loss.backward()
#     #         optimizer.step()
            
#     #         running_loss += loss.item()

#     # Train the model
#     num_epochs = 1
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         for images, labels, label_lengths in dataloader:
#             outputs = model(images)  # Shape: (batch_size, sequence_length, num_classes)

#             # Permute outputs to (T, N, C) for CTCLoss
#             outputs = outputs.permute(1, 0, 2).log_softmax(2)  # Shape: (sequence_length, batch_size, num_classes)
#             print(f"Outputs shape before permute: {outputs.shape}")
#             # Calculate the input lengths (all are equal to sequence_length)
#             input_lengths = torch.full((images.size(0),), outputs.size(0), dtype=torch.long)

#             # Calculate CTC loss
#             loss = criterion(outputs, labels, input_lengths, label_lengths)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()

#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

        
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

#     # save the model
#     torch.save(model.state_dict(), 'vgg_gru_ctc_model.pth')

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


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
        self.classifier = nn.Linear(256 * 2, num_classes + 1)  # +1 for blank token

    def forward(self, x):
        x = self.features(x)  # Shape: (batch_size, 256, 4, 33)
        batch_size, channels, height, width = x.size()

        x = x.permute(0, 3, 1, 2)  # (batch_size, width, channels, height)
        x = x.contiguous().view(batch_size, width, -1)  # (batch_size, sequence_length, features)

        x, _ = self.rnn(x)  # (batch_size, sequence_length, hidden_size*2)
        x = self.classifier(x)  # (batch_size, sequence_length, num_classes + 1)
        return x

def visualize_training_data(dataset, num_samples=5):
    """
    Visualize training data with images and their corresponding labels.
    """
    # Randomly sample data points
    sampled_data = random.sample(range(len(dataset)), num_samples)
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))

    for i, idx in enumerate(sampled_data):
        image, label, _ = dataset[idx]
        decoded_label = ''.join([characters[c] for c in label.tolist()])  # Decode label indices to string

        # Prepare the image for visualization
        image = image.permute(1, 2, 0).numpy()  # Change channel order for matplotlib
        image = (image * 0.5) + 0.5  # De-normalize the image
        
        # Display the image and label
        axes[i].imshow(image.squeeze(), cmap='gray')
        axes[i].set_title(f"Label: {decoded_label}", fontsize=12)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()



# Main Training Loop
if __name__ == '__main__':
    transform = transforms.Compose([
        ResizeWithPadding(target_height=32, target_width=128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = LicensePlateDataset(csv_file='labels_train.csv', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)

    model = VGG_GRU_CTC_Model()

    criterion = nn.CTCLoss(blank=num_classes)  # blank token is used to represent the "no character" label
    optimizer = Adam(model.parameters(), lr=0.001)

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels, label_lengths in dataloader:
            outputs = model(images)

            # Permute outputs to (T, N, C) for CTCLoss
            outputs = outputs.permute(1, 0, 2).log_softmax(2)  # Shape: (sequence_length, batch_size, num_classes)

            # Calculate the input lengths (all equal to sequence_length)
            input_lengths = torch.full((images.size(0),), outputs.size(0), dtype=torch.long)

            # Compute loss
            loss = criterion(outputs, labels, input_lengths, label_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), 'vgg_gru_ctc_model.pth')

    # Use the visualization function
    transform = transforms.Compose([
        ResizeWithPadding(target_height=32, target_width=128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Load the dataset
    train_dataset = LicensePlateDataset(csv_file='labels_train.csv', transform=transform)

    # Visualize 5 random samples
    visualize_training_data(train_dataset, num_samples=5)

