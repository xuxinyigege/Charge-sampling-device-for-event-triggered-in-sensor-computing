"""Small ResNet used for the face-recognition experiment."""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Two 3 x 3 convolutions with a 1 x 1 residual projection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        return self.projection(x) + out


class FaceResNet(nn.Module):
    """Face classifier for 128 x 128 grayscale crops."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Conv2d(1, 4, kernel_size=5, stride=3) # Computing in the GCCD array 
        self.block1 = ResidualBlock(4, 32, stride=2)
        self.block2 = ResidualBlock(32, 64, stride=2)
        self.block3 = ResidualBlock(64, 128, stride=2)
        self.fc1 = nn.Linear(128 * 36, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward_features_after_stem(self, stem_feature: torch.Tensor) -> torch.Tensor:
        out = self.relu(stem_feature)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        return out.flatten(1)

    def forward_after_stem(self, stem_feature: torch.Tensor) -> torch.Tensor:
        out = self.forward_features_after_stem(stem_feature)
        out = self.relu(self.fc1(out))
        return self.fc2(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_after_stem(self.stem(x))


def load_model(weights_path: str, num_classes: int = 10, device: str = "cpu") -> FaceResNet:
    """Load a state-dict checkpoint into the cleaned model definition."""

    model = FaceResNet(num_classes=num_classes).to(device)
    try:
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model
