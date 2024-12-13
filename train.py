import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from CRNN_model import CRNNModel


# Define characters and number of classes
characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
num_classes = len(characters)

# Dataset Class to load License Plate Images and Labels
class LicensePlateDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]
        label = self.data.iloc[idx, 2]
        # Convert label to list of character indices
        label_encoded = [characters.index(c) for c in label]
        image = Image.open(img_path).convert("L")  # convert to grayscale image

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

    # Stack images (tensors from the Dataset)
    images = torch.stack(images, dim=0)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)

    return images, padded_labels, label_lengths



# Decode predictions for analysis
def decode_predictions(predictions, characters):
    blank_index = len(characters)  # the blank token is the last index
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
 

def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs, characters,device):
    
    model.to(device)
    # Initialize lists to store training, validation and test results
    train_losses = []
    train_word_accuracies = []
    train_char_accuracies = []
    val_losses = []
    val_word_accuracies = []
    val_char_accuracies = []
    test_word_accuracies = []
    test_char_accuracies = []
    best_val_character_acc = 0.0
    

    for epoch in range(1, num_epochs + 1):

        # training part
        model.train()
        running_loss = 0.0
        correct_words = 0
        total_words = 0
        correct_chars = 0
        total_chars = 0

        for batch_idx, (images, labels, label_lengths) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # Forward pass
            outputs = model(images)
            # permute the outputs to match the shape of the labels
            outputs = outputs.permute(1, 0, 2)
            input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long).to(device)
            # Calculate loss
            loss = criterion(outputs, labels, input_lengths, label_lengths)
            # Backpropagation
            loss.backward()
            # Update weights
            optimizer.step()

            running_loss += loss.item()
            # get the predicted sequences
            predicted_sequences = outputs.argmax(2).permute(1, 0).tolist()
            # Convert label indices to text
            label_texts = [
                "".join([characters[l] for l in label if l < len(characters)])
                for label in labels.tolist()
            ]
            # Decode predicted sequences
            decoded_predictions = decode_predictions(predicted_sequences, characters)

            for pred_text, label_text in zip(decoded_predictions, label_texts):
                # Character accuracy
                correct_chars += sum(p == l for p, l in zip(pred_text, label_text))
                total_chars += len(label_text)

                # Word accuracy
                if pred_text == label_text:
                    correct_words += 1
                total_words += 1

        # save the training results
        t_epoch_loss = running_loss / len(train_loader)
        t_word_accuracy = correct_words / total_words
        t_char_accuracy = correct_chars / total_chars

        train_losses.append(t_epoch_loss)
        train_word_accuracies.append(t_word_accuracy)
        train_char_accuracies.append(t_char_accuracy)

        

        # validation
        model.eval()
        running_loss = 0.0
        correct_words = 0
        total_words = 0
        correct_chars = 0
        total_chars = 0

        with torch.no_grad():
            for batch_idx, (images, labels, label_lengths) in enumerate(val_loader):
                images = images.to(device)
                labels = labels.to(device)
                # Forward pass
                outputs = model(images)
                # permute the outputs to match the shape of the labels
                outputs = outputs.permute(1, 0, 2)
                input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long).to(device)
                # Calculate losss
                loss = criterion(outputs, labels, input_lengths, label_lengths)

                running_loss += loss.item()
                # get the predicted sequences
                predicted_sequences = outputs.argmax(2).permute(1, 0).tolist()
                # Convert label indices to text
                label_texts = [
                    "".join([characters[l] for l in label if l < len(characters)])
                    for label in labels.tolist()
                ]
                # Decode predicted sequences
                decoded_predictions = decode_predictions(predicted_sequences, characters)

                # Calculate accuracy
                for pred_text, label_text in zip(decoded_predictions, label_texts):
                    # Character accuracy
                    correct_chars += sum(p == l for p, l in zip(pred_text, label_text))
                    total_chars += len(label_text)
                    # Word accuracy
                    if pred_text == label_text:
                        correct_words += 1
                    total_words += 1

            # save the validation results
            v_epoch_loss = running_loss / len(val_loader)
            v_word_accuracy = correct_words / total_words
            v_char_accuracy = correct_chars / total_chars

            val_losses.append(v_epoch_loss)
            val_word_accuracies.append(v_word_accuracy)
            val_char_accuracies.append(v_char_accuracy)

           


        # test part
        model.eval()

        correct_words = 0
        total_words = 0
        correct_chars = 0
        total_chars = 0

        with torch.no_grad():
            for batch_idx, (images, labels, label_lengths) in enumerate(test_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                # permute the outputs to match the shape of the labels
                outputs = outputs.permute(1, 0, 2)
                input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long).to(device)
                

                predicted_sequences = outputs.argmax(2).permute(1, 0).tolist()
                # Convert label indices to text
                label_texts = [
                    "".join([characters[l] for l in label if l < len(characters)])
                    for label in labels.tolist()
                ]
                # Decode predicted sequences
                decoded_predictions = decode_predictions(predicted_sequences, characters)

                # Calculate accuracy
                for pred_text, label_text in zip(decoded_predictions, label_texts):
                    # Character accuracy
                    correct_chars += sum(p == l for p, l in zip(pred_text, label_text))
                    total_chars += len(label_text)

                    # Word accuracy
                    if pred_text == label_text:
                        correct_words += 1
                    total_words += 1

            test_word_accuracy = correct_words / total_words
            test_char_accuracy = correct_chars / total_chars

            test_word_accuracies.append(test_word_accuracy)
            test_char_accuracies.append(test_char_accuracy)

            # Save best model based on character accuracy on validation set
            if v_char_accuracy > best_val_character_acc:
                best_val_word_acc = v_word_accuracy
                best_val_character_acc = v_char_accuracy
                best_test_word_acc = test_word_accuracy
                best_test_character_acc = test_char_accuracy
                torch.save(model.state_dict(), "train_model.pth")

            # Print epoch results
            print(f"Epoch [{epoch}/{num_epochs}] - T_Loss: {t_epoch_loss:.4f}, T_Word_Acc: {t_word_accuracy:.4f}, T_Char_Acc: {t_char_accuracy:.4f}, V_Loss: {v_epoch_loss:.4f}, V_Word_Acc: {v_word_accuracy:.4f}, V_Char_Acc: {v_char_accuracy:.4f}")

    # print the best results
    print(f"Best Validation Character Accuracy: {best_val_character_acc:.4f}")
    print(f"Corresponding Validation Word Accuracy: {best_val_word_acc:.4f}")
    print(f"Corresponding Test Character Accuracy: {best_test_character_acc:.4f}")
    print(f"Corresponding Test Word Accuracy: {best_test_word_acc:.4f}")

    

    return train_losses, train_word_accuracies, train_char_accuracies, val_losses, val_word_accuracies, val_char_accuracies, test_word_accuracies, test_char_accuracies



def plot_metrics(train_losses, train_word_accuracies, train_char_accuracies,val_losses, val_word_accuracies, val_char_accuracies, test_word_accuracies, test_char_accuracies):
    epochs = range(1, len(train_losses) + 1)

    # First Figure: Training and Validation Loss and Accuracies
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
    plt.title("Word Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Character accuracy plot
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_char_accuracies, label="Train Character Accuracy")
    plt.plot(epochs, val_char_accuracies, label="Val Character Accuracy")
    plt.title("Character Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Second Figure: Test Accuracies
    plt.figure(figsize=(10, 5))

    # Test Word Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, test_word_accuracies, label="Test Word Accuracy")
    plt.title("Test Word Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Test Character Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_char_accuracies, label="Test Character Accuracy")
    plt.title("Test Character Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define transforms for training data
    transform_train = transforms.Compose([
        transforms.Lambda(lambda img: Image.fromarray(img) if isinstance(img, np.ndarray) else img),
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        # do data augmentation with probability of 0.5
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0)], p=0.5),
        transforms.RandomApply([transforms.RandomRotation(degrees=5)], p=0.5),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # Define transforms for validation and test data
    transform_val_test = transforms.Compose([
        transforms.Lambda(lambda img: Image.fromarray(img) if isinstance(img, np.ndarray) else img),
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])



    batch_size = 32

    # Training data loader
    train_dataset = LicensePlateDataset(csv_file='labels_train.csv', transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    # Validation and test data loaders
    val_set = LicensePlateDataset(csv_file='labels_val.csv', transform=transform_val_test)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    test_set = LicensePlateDataset(csv_file='labels_test.csv', transform=transform_val_test)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)


    # Initialize the model, loss function, and optimizer
    model = CRNNModel(num_classes=num_classes + 1)  # Add 1 for CTC blank index
    criterion = nn.CTCLoss(blank=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005)

    # Train the model
    train_losses, train_word_accuracies, train_char_accuracies ,\
    val_losses, val_word_accuracies, val_char_accuracies ,\
    test_word_accuracies, test_char_accuracies = train_model(
        model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs=200, characters=characters,device=device
    )


    

    # Plot the training results
    plot_metrics(train_losses, train_word_accuracies, train_char_accuracies,val_losses, val_word_accuracies, val_char_accuracies, test_word_accuracies, test_char_accuracies)
