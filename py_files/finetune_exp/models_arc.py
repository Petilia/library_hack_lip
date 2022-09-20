import torch.nn as nn
from torch.nn import functional as F

class DumNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 128)
        self.fc2 = nn.Linear(128, 768)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x