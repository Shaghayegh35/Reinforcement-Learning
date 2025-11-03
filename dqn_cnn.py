import torch
import torch.nn as nn

class QNetCNN(nn.Module):
    def __init__(self, in_channels: int, n_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64*7*7, 512), nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    def forward(self, x):
        x = x / 255.0 if x.dtype == torch.uint8 else x
        h = self.conv(x)
        h = torch.flatten(h, 1)
        return self.head(h)

