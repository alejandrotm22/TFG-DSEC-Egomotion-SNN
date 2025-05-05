import random
import torch
import torch.nn as nn

from egomotionnet import *
from egomotiondataset import EgoMotionDataset
from egomotiondataset import FlowAugmentation

import math
import os
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import wandb

wandb.init(
    project="TFG",
    name="EgoMotionNetV10",
    mode="online",
    config={
        "epochs": 80,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "optimizer": "Adam",
        "loss_fn": "MSELoss"
    }
)

# Enable GPU
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


################################
## DATASET LOADING/GENERATION ##
################################

# Cargar los datos
data_saved_path = "/home/alejandro/Escritorio/TFG/TFG-DSEC-Egomotion-SNN/dataset"
ego_motions = np.load(os.path.join(data_saved_path, "gt_ego_motion.npy"))
optical_flows = np.load(os.path.join(data_saved_path, "optical_flow.npy"))

print(f"gt_ego_motion shape: {ego_motions.shape}")
print(f"optical_flow shape: {optical_flows.shape}")

# Semilla para reproducibilidad
np.random.seed(42)

# Total de muestras
N = optical_flows.shape[0]

# Indices aleatorios mezclados
indices = np.random.permutation(N)

# Porcentajes
train_end = int(0.8 * N)

# Índices por split
train_idx = indices[:train_end]
val_idx = indices[train_end:]

# Divisiones manuales
opt_flow_train = optical_flows[train_idx]
ego_train = ego_motions[train_idx]

opt_flow_val = optical_flows[val_idx]
ego_val = ego_motions[val_idx]


# Crear datasets
# train_dataset = EgoMotionDataset(opt_flow_train, ego_train, transform=FlowAugmentation(noise_std=(opt_flow_train.std() * 0.05), drop_prob=0.05))
# train_dataset = EgoMotionDataset(opt_flow_train, ego_train, transform=FlowAugmentation(noise_std=0.1, drop_prob=0))
train_dataset = EgoMotionDataset(opt_flow_train, ego_train)
val_dataset   = EgoMotionDataset(opt_flow_val, ego_val)

# Crear dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=False, pin_memory=True)
val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False, pin_memory=True)

##############
## TRAINING ##
##############


model = EgoMotionNetV10().to(device)
wandb.watch(model, log="all", log_freq=10)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 80

early_stopping_patience = 15  # Épocas sin mejorar antes de parar
best_val_loss = math.inf
epochs_without_improvement = 0

for epoch in tqdm(range(num_epochs)):
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:  # Asegúrate de tener `train_loader`
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)

    avg_train_loss = train_loss / len(train_loader.dataset)

    # Validación
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:  # Asegúrate de tener `val_loader`
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)

    avg_val_loss = val_loss / len(val_loader.dataset)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss
    })

    # ---- EARLY STOPPING ----
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0

    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= early_stopping_patience:
        print(f"Early stopping triggered at epoch {epoch+1}!")
        break


wandb.finish()
