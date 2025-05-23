import torch
import torch.nn as nn

class EgoMotionNet(nn.Module): # Modelo básico
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
    

class ODEgoMotionNet(nn.Module): # Modelo básico con una capa más
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
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8), # Una capa más de profundidad
            nn.ReLU(),
            nn.Linear(8, 6)           # Output: [Vx, Vy, Vz, Wx, Wy, Wz]
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x

class EgoMotionNetV2(nn.Module): # Modelo básico más profundo
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
    

class EgoMotionNetV3(nn.Module): # Modelo básico más inspirado en la bibliografía
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
    
class EgoMotionNetV4(nn.Module): # Modelo completamente de la bibliografía
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
    
class EgoMotionNetV5(nn.Module): # Modelo básico con más capas de salida y con kernel size 3 
    def __init__(self):
        super(EgoMotionNetV5, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1),  # (16, 240, 320)
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # (32, 120, 160)
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # (64, 60, 80)
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),# (128, 30, 40)
            #nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))                           # (128, 1, 1)
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),              # (128)
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6)           # Output: [Vx, Vy, Vz, Wx, Wy, Wz]
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x
    
class EgoMotionNetV6(nn.Module): # Modelo V5 con más capas de salida
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
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6)           # Output: [Vx, Vy, Vz, Wx, Wy, Wz]
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x
    
class EgoMotionNetV7(nn.Module): # Es como el V5 pero con stride=1
    def __init__(self):
        super(EgoMotionNetV7, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1),  # (16, 240, 320)
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # (32, 120, 160)
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # (64, 60, 80)
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),# (128, 30, 40)
            #nn.BatchNorm2d(128),
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
    
class TDEgoMotionNet(nn.Module): # Modelo básico con 2 capas fully connected más
    def __init__(self):
        super(TDEgoMotionNet, self).__init__()
        
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
            nn.Linear(64, 64), # Una capa más de profundidad
            nn.ReLU(),
            nn.Linear(64, 32), # Una capa más de profundidad
            nn.ReLU(),
            nn.Linear(32, 6)           # Output: [Vx, Vy, Vz, Wx, Wy, Wz]
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x

class EgoMotionNetV8(nn.Module): # Modelo básico cambiando el adaptative pool por un maxpool
    def __init__(self):
        super(EgoMotionNetV8, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, stride=2, padding=1),  # (8, 240, 320)
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), # (16, 120, 160)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),# (32, 60, 80)
            nn.ReLU(),
            nn.MaxPool2d(2)                         # (32, 1, 1)
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),              
            nn.Linear(38400, 16),
            nn.ReLU(),
            nn.Linear(16, 6)           # Output: [Vx, Vy, Vz, Wx, Wy, Wz]
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x
    
class EgoMotionNetV9(nn.Module): # Modelo básico más profundo con maxpooling después de cada convolución y sin adaptative pooling
    def __init__(self):
        super(EgoMotionNetV9, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, stride=1, padding=1),   # (8, 480, 640)
            nn.ReLU(),
            nn.MaxPool2d(2),                                       # (8, 240, 320)

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),  # (16, 240, 320)
            nn.ReLU(),
            nn.MaxPool2d(2),                                       # (16, 120, 160)

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # (32, 120, 160)
            nn.ReLU(),
            nn.MaxPool2d(2),                                       # (32, 60, 80)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # (64, 60, 80)
            nn.ReLU(),
            nn.MaxPool2d(2),                                       # (64, 30, 40)

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # (64, 30, 40)
            nn.ReLU(),
            nn.MaxPool2d(2),                                       # (64, 15, 20)
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),                                          # 64 * 15 * 20 = 19200
            nn.Linear(19200, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x
    
class EgoMotionNetV10(nn.Module): # Modelo V9 con una capa fully connected más
    def __init__(self):
        super(EgoMotionNetV10, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, stride=1, padding=1),   # (8, 480, 640)
            nn.ReLU(),
            nn.MaxPool2d(2),                                       # (8, 240, 320)

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),  # (16, 240, 320)
            nn.ReLU(),
            nn.MaxPool2d(2),                                       # (16, 120, 160)

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # (32, 120, 160)
            nn.ReLU(),
            nn.MaxPool2d(2),                                       # (32, 60, 80)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # (64, 60, 80)
            nn.ReLU(),
            nn.MaxPool2d(2),                                       # (64, 30, 40)

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # (64, 30, 40)
            nn.ReLU(),
            nn.MaxPool2d(2),                               # (256, 5, 6)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),                                           # 256 * 5 * 6 = 7680
            nn.Linear(19200, 512),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x
    
class EgoMotionNetV9_Intermedia(nn.Module): # Modelo V9 con max pooling cada 2 capas convolucionales
    def __init__(self):
        super(EgoMotionNetV9_Intermedia, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, stride=1, padding=1),   # (8, 480, 640)
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),  # (16, 480, 640)
            nn.ReLU(),
            nn.MaxPool2d(2),                                       # (16, 240, 320)

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # (32, 240, 320)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # (64, 240, 320)
            nn.ReLU(),
            nn.MaxPool2d(2),                                       # (64, 120, 160)

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # (64, 120, 160)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # (64, 120, 160)
            nn.ReLU(),
            nn.MaxPool2d(2),                                       # (64, 60, 80)
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),                                          # 64 * 60 * 80 = 307200
            nn.Linear(307200, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x