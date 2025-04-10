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