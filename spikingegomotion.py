import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, layer, surrogate

class SpikingEgoMotionNet(nn.Module):
    def __init__(self, T=4):
        super(SpikingEgoMotionNet, self).__init__()
        self.T = T

        self.conv1 = nn.Conv2d(2, 8, kernel_size=3, stride=2, padding=1)
        self.sn1 = neuron.IFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True)
        
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.sn2 = neuron.IFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True)
        
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.sn3 = neuron.IFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(32, 16)
        self.sn_fc1 = neuron.IFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True)
        
        self.fc2 = nn.Linear(16, 6)

    def forward(self, x):  # x: (N, 2, H, W)
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # (T, N, 2, H, W)

        outputs = []
        for t in range(self.T):
            xt = x_seq[t]  # (N, 2, H, W)

            xt = self.conv1(xt)
            xt = self.sn1(xt)

            xt = self.conv2(xt)
            xt = self.sn2(xt)

            xt = self.conv3(xt)
            xt = self.sn3(xt)

            xt = self.pool(xt)
            xt = xt.view(xt.size(0), -1)

            xt = self.fc1(xt)
            xt = self.sn_fc1(xt)

            xt = self.fc2(xt)
            outputs.append(xt)

        return torch.stack(outputs).mean(dim=0)  # (N, 6)


class ODSpikingEgoMotionNet(nn.Module):
    def __init__(self, T=4):
        super(ODSpikingEgoMotionNet, self).__init__()
        self.T = T

        self.conv1 = nn.Conv2d(2, 8, kernel_size=3, stride=2, padding=1)
        self.sn1 = neuron.IFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True)
        
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.sn2 = neuron.IFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True)
        
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.sn3 = neuron.IFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(32, 64)
        self.sn_fc1 = neuron.IFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True)

        self.fc2 = nn.Linear(64, 16)
        self.sn_fc2 = neuron.IFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True)
        
        self.fc3 = nn.Linear(16, 6)

    def forward(self, x):  # x: (N, 2, H, W)
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # (T, N, 2, H, W)

        outputs = []
        for t in range(self.T):
            xt = x_seq[t]  # (N, 2, H, W)

            xt = self.conv1(xt)
            xt = self.sn1(xt)

            xt = self.conv2(xt)
            xt = self.sn2(xt)

            xt = self.conv3(xt)
            xt = self.sn3(xt)

            xt = self.pool(xt)
            xt = xt.view(xt.size(0), -1)

            xt = self.fc1(xt)
            xt = self.sn_fc1(xt)

            xt = self.fc2(xt)
            xt = self.sn_fc2(xt)

            xt = self.fc3(xt)
            outputs.append(xt)

        return torch.stack(outputs).mean(dim=0)  # (N, 6)
    
class SpikingEgoMotionNetV5(nn.Module):
    def __init__(self, T=4):
        super(SpikingEgoMotionNetV5, self).__init__()
        self.T = T

        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1)   # (16, 240, 320)
        self.sn1 = neuron.IFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # (32, 120, 160)
        self.sn2 = neuron.IFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # (64, 60, 80)
        self.sn3 = neuron.IFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # (128, 30, 40)
        self.sn4 = neuron.IFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # (128, 1, 1)

        self.fc1 = nn.Linear(128, 64)
        self.sn_fc1 = neuron.IFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True)

        self.fc2 = nn.Linear(64, 32)
        self.sn_fc2 = neuron.IFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True)

        self.fc3 = nn.Linear(32, 6)

    def forward(self, x):  # x: (N, 2, H, W)
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # (T, N, 2, H, W)

        outputs = []
        for t in range(self.T):
            xt = x_seq[t]  # (N, 2, H, W)

            xt = self.conv1(xt)
            xt = self.sn1(xt)

            xt = self.conv2(xt)
            xt = self.sn2(xt)

            xt = self.conv3(xt)
            xt = self.sn3(xt)

            xt = self.conv4(xt)
            xt = self.sn4(xt)

            xt = self.pool(xt)
            xt = xt.view(xt.size(0), -1)

            xt = self.fc1(xt)
            xt = self.sn_fc1(xt)

            xt = self.fc2(xt)
            xt = self.sn_fc2(xt)

            xt = self.fc3(xt)
            outputs.append(xt)

        return torch.stack(outputs).mean(dim=0)  # (N, 6)

class SpikingEgoMotionNetV9(nn.Module):
    def __init__(self, T=4):
        super(SpikingEgoMotionNetV9, self).__init__()
        self.T = T

        self.conv1 = nn.Conv2d(2, 8, kernel_size=3, stride=1, padding=1)
        self.sn1 = neuron.IFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.sn2 = neuron.IFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.sn3 = neuron.IFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.sn4 = neuron.IFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True)
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.sn5 = neuron.IFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True)
        self.pool5 = nn.MaxPool2d(2)

        # Tamaño final del feature map: (64, 15, 20) => 19200
        self.fc1 = nn.Linear(64 * 15 * 20, 256)
        self.sn_fc1 = neuron.IFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True)

        self.fc2 = nn.Linear(256, 64)
        self.sn_fc2 = neuron.IFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True)

        self.fc3 = nn.Linear(64, 6)  # salida continua

    def forward(self, x):  # x: (N, 2, 480, 640)
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # (T, N, 2, H, W)

        outputs = []
        for t in range(self.T):
            xt = x_seq[t]  # (N, 2, H, W)

            xt = self.pool1(self.sn1(self.conv1(xt)))
            xt = self.pool2(self.sn2(self.conv2(xt)))
            xt = self.pool3(self.sn3(self.conv3(xt)))
            xt = self.pool4(self.sn4(self.conv4(xt)))
            xt = self.pool5(self.sn5(self.conv5(xt)))

            xt = xt.view(xt.size(0), -1)  # flatten (N, 19200)

            xt = self.sn_fc1(self.fc1(xt))
            xt = self.sn_fc2(self.fc2(xt))

            xt = self.fc3(xt)  # salida no spiking para regresión
            outputs.append(xt)

        return torch.stack(outputs).mean(dim=0)  # (N, 6)
