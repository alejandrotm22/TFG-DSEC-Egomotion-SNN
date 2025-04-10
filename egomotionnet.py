import torch
import torch.nn as nn

class EgoMotionNet(nn.Module):
    def __init__(self):
        super(EgoMotionNet, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, stride=2, padding=1),  # (8, 240, 320)
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), # (16, 120, 160)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),# (32, 60, 80)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))                          # (32, 1, 1)
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),              # (32)
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 6)           # Output: [Vx, Vy, Vz, Wx, Wy, Wz]
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x