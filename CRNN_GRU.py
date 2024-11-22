import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import string

# characters set
characters = string.ascii_uppercase + string.digits
num_classes = len(characters) + 1  




# VGG16 + GRU + CTC
class VGG_GRU_CTC_Model(nn.Module):
    def __init__(self, hidden_size=128, num_layers=2):
        super(VGG_GRU_CTC_Model, self).__init__()

        # USE VGG16
        vgg = models.vgg16(pretrained=True)
        self.features = nn.Sequential(*list(vgg.children())[:-1])  
        self.flatten = nn.Flatten()

        # GRU 
        self.gru = nn.GRU(input_size=25088,  # VGG16 output size
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          bidirectional=True)

        # output layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # bidirectional GRU

    def forward(self, x):
        # extract features
        x = self.features(x)  # (batch_size, 512, 7, 7)
        x = self.flatten(x)   # (batch_size, 512 * 7 * 7)

        # GRU input: (batch_size, seq_len, input_size)
        x = x.unsqueeze(1)  # (batch_size, 1, 512 * 7 * 7)
        
        # 
        x, _ = self.gru(x)  # output shape: (batch_size, seq_len, hidden_size * 2)

        # extract the last time step
        x = x[:, -1, :]  # (batch_size, hidden_size * 2)
        
        
        x = self.fc(x)
        
        return x




