import torch
from torch.utils.data import Dataset, DataLoader

class EgoMotionDataset(Dataset):
    def __init__(self, optical_flows, ego_motions, transform=None):
        self.optical_flows = optical_flows.astype('float32')  # shape: (N, 2, H, W)
        self.ego_motions = ego_motions.astype('float32')      # shape: (N, 6)
        self.transform = transform

    def __len__(self):
        return len(self.optical_flows)

    def __getitem__(self, idx):
        x = torch.tensor(self.optical_flows[idx])  # Optical flow
        y = torch.tensor(self.ego_motions[idx])    # Ego-motion ground truth

        if self.transform:
            x = self.transform(x)

        return x, y
    
class FlowAugmentation:
    def __init__(self, noise_std=0.1, drop_prob=0.1):
        self.noise_std = noise_std
        self.drop_prob = drop_prob

    def __call__(self, flow):
        flow = self.add_flow_noise(flow, std=self.noise_std)
        flow = self.random_occlusion(flow, drop_prob=self.drop_prob)
        
        return flow  
      
    # Add Gaussian noise to the flow
    def add_flow_noise(self, flow, std=0.1):
        noise = torch.randn_like(flow) * std
        return flow + noise
    
    # Add random occlusion
    def random_occlusion(self, flow, drop_prob=0.1):
        mask = (torch.rand_like(flow[:1,:,:]) > drop_prob).float()
        return flow * mask
    
        