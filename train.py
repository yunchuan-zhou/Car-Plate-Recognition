import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


class LicensePlateDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]  
        label = self.data.iloc[idx, 2]    
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
    

if __name__ == '__main__':


    # normalize the images
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),        
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])
    dataset = LicensePlateDataset(csv_file='labels_test.csv', transform=transform)


    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # test and visualize the dataloader
    images, labels = next(iter(dataloader))
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    for i in range(5):
        image = images[i].permute(1, 2, 0).numpy()  # convert tensor (C, H, W) to (H, W, C)
        image = (image * 0.5) + 0.5  # denormalize
        axes[i].imshow(image)
        axes[i].set_title(f"Label: {labels[i]}")
        axes[i].axis("off")
    plt.show()