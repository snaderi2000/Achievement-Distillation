import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SimpleCNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)  # SAME padding
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)

    def forward(self, x: th.Tensor) -> th.Tensor:
        # x assumed to be in NCHW format (B, C, H, W)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        return x