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
        flow = self.add_flow_noise(flow, scale=self.noise_std)
        flow = self.random_occlusion(flow, drop_prob=self.drop_prob)
        
        return flow  
      
    # # Add Gaussian noise to the flow
    # def add_flow_noise(self, flow, std=0.1):
    #     noise = torch.randn_like(flow) * std
    #     return flow + noise
    
    # def add_flow_noise(self, flow, std_scale=0.1, min_std=1e-4):
    #     # Magnitud del flujo por píxel (euclídea si el flujo tiene 2 canales: u, v)
    #     magnitude = torch.norm(flow, dim=1, keepdim=True)  # Shape: (B, 1, H, W)

    #     # Calcula desviación estándar proporcional, pero con un mínimo para evitar 0
    #     std = magnitude * std_scale
    #     std = torch.clamp(std, min=min_std)

    #     # Ruido gaussiano dependiente del flujo
    #     noise = torch.randn_like(flow) * std
    #     return flow + noise
    
    def add_flow_noise(self, flow, scale=0.1, min_std=1e-4):
        """
        Añade ruido gaussiano proporcional al valor absoluto del flujo (por canal).

        flow: Tensor de forma (B, C=2, H, W)
        scale: Factor de escala del ruido (ej: 0.05 para 5%)
        """
        # Desviación estándar proporcional al valor absoluto
        std = torch.clamp(flow.abs() * scale, min=min_std)

        # Ruido gaussiano por canal
        noise = torch.randn_like(flow) * std

        return flow + noise

    
    # Add random occlusion
    def random_occlusion(self, flow, drop_prob=0.1):
        mask = (torch.rand_like(flow[:1,:,:]) > drop_prob).float()
        return flow * mask
    
        