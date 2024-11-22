import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2


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
    
class ResizeWithPadding:
    def __init__(self, target_height=32, target_width=128):
        self.target_height = target_height
        self.target_width = target_width

    def __call__(self, image):
        image = np.array(image)
        h, w = image.shape[:2]


        scale = self.target_height / h
        new_width = int(w * scale)
        resized_image = cv2.resize(image, (new_width, self.target_height))


        padded_image = np.zeros((self.target_height, self.target_width), dtype=np.uint8)
        padded_image[:, :min(new_width, self.target_width)] = resized_image[:, :min(new_width, self.target_width)]


        return Image.fromarray(padded_image)
    
if __name__ == '__main__':


    # normalize the images
    transform = transforms.Compose([
        ResizeWithPadding(target_height=32, target_width=128),
        #transforms.RandomRotation(5),  
        #transforms.ColorJitter(brightness=0.2, contrast=0.2),  
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.5], std=[0.5])  
    ])
    dataset = LicensePlateDataset(csv_file='labels_test.csv', transform=transform)


    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # test and visualize the dataloader
    images, labels = next(iter(dataloader))
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    for i in range(5):
        image = images[i].permute(1, 2, 0).numpy()  
        image = (image * 0.5) + 0.5  
        axes[i].imshow(image.squeeze(), cmap='gray')  
        axes[i].set_title(f"Label: {labels[i]}")
        axes[i].axis("off")
    plt.show()