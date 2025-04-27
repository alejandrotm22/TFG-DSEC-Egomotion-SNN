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
    

class ODEgoMotionNet(nn.Module):
    def __init__(self):
        super(ODEgoMotionNet, self).__init__()
        
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
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 16), # Una capa más de profundidad
            nn.ReLU(),
            nn.Linear(16, 6)           # Output: [Vx, Vy, Vz, Wx, Wy, Wz]
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x

class EgoMotionNetV2(nn.Module):
    def __init__(self):
        super(EgoMotionNetV2, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=7, stride=2, padding=3),  # (32, H/2, W/2)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2), # (64, H/4, W/4)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),# (128, H/8, W/8)
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),# (128, H/8, W/8)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))                           # (128, 1, 1)
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),                # (128)
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6)             # Output: [Vx, Vy, Vz, Wx, Wy, Wz]
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x
    

class EgoMotionNetV3(nn.Module):
    def __init__(self):
        super(EgoMotionNetV3, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3),  # (32, H/2, W/2)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2), # (64, H/4, W/4)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=4, padding=1),# (128, H/8, W/8)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),# (128, H/8, W/8)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))                           # (128, 1, 1)
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),                # (128)
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6)             # Output: [Vx, Vy, Vz, Wx, Wy, Wz]
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x
    
class EgoMotionNetV4(nn.Module):
    def __init__(self):
        super(EgoMotionNetV4, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3),  # (32, H/2, W/2)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2), # (64, H/4, W/4)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=4, padding=1),# (128, H/8, W/8)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),# (128, H/8, W/8)
            nn.ReLU(),
            nn.MaxPool2d(2)                           # (128, 1, 1)
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),                # (128)
            nn.Linear(76800, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 6)             # Output: [Vx, Vy, Vz, Wx, Wy, Wz]
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x
    
class EgoMotionNetV5(nn.Module):
    def __init__(self):
        super(EgoMotionNetV5, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1),  # (16, 240, 320)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # (32, 120, 160)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # (64, 60, 80)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),# (128, 30, 40)
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))                           # (128, 1, 1)
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),              # (128)
            nn.Linear(128, 64),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6)           # Output: [Vx, Vy, Vz, Wx, Wy, Wz]
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x
    
class EgoMotionNetV6(nn.Module):
    def __init__(self):
        super(EgoMotionNetV6, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1),  # (16, 240, 320)
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # (32, 120, 160)
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # (64, 60, 80)
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),# (128, 30, 40)
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))                           # (128, 1, 1)
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),              # (128)
            nn.Linear(256, 128),
            nn.ReLU(),
           # nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6)           # Output: [Vx, Vy, Vz, Wx, Wy, Wz]
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x