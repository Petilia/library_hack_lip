import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 1380)
        self.fc2 = nn.Linear(1380, 768)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 1380)
        self.fc2 = nn.Linear(1380, 768)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MetaCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode_image = ImageEncoder()
        self.encode_text = TextEncoder()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image, text):
        #open_clip realization
        image_features = self.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)

        text_features = self.encode_text(text)
        text_features = F.normalize(text_features, dim=-1)

        return image_features, text_features, self.logit_scale.exp()