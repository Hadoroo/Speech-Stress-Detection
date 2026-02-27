import torch
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(
        self,
        num_classes=2,
        in_channels=1,
        k1=32,
        k2=64,
        dropout1=0.2,
        dropout2=0.3
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, k1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout1),

            nn.Conv2d(k1, k2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(k2, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))
