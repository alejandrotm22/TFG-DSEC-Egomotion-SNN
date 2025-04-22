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
