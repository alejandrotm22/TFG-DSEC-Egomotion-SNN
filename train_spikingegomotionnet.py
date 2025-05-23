import random
import torch
import torch.nn as nn

from spikingegomotion import *
from egomotiondataset import EgoMotionDataset
from egomotiondataset import FlowAugmentation

import math
import os
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import wandb
from spikingjelly.activation_based import functional

wandb.init(project="TFG",
    name="SpikingEgoMotionNetV9_test",
    mode="online",
    config={
    "architecture": "SpikingEgoMotionNet",
    "epochs": 30,
    "batch_size": 1,
    "T": 4,
    "lr": 1e-3
})

# Enable GPU
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


################################
## DATASET LOADING/GENERATION ##
################################

# Cargar los datos
data_saved_path = "/home/alejandro/Escritorio/TFG/TFG-DSEC-Egomotion-SNN/dataset"
ego_motions = np.load(os.path.join(data_saved_path, "gt_ego_motion.npy"))
optical_flows = np.load(os.path.join(data_saved_path, "optical_flow.npy"))
test_ego_motions = np.load(os.path.join(data_saved_path, "test_gt_ego_motion.npy"))
test_optical_flows = np.load(os.path.join(data_saved_path, "test_optical_flow.npy"))

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
train_dataset = EgoMotionDataset(opt_flow_train, ego_train)
val_dataset   = EgoMotionDataset(opt_flow_val, ego_val)
test_dataset  = EgoMotionDataset(test_optical_flows, test_ego_motions)

# Crear dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=False, pin_memory=True)
val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False, pin_memory=True)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False, pin_memory=True)

##############
## TRAINING ##
##############


model = SpikingEgoMotionNetV9(T=4).to(device)
wandb.watch(model, log="all", log_freq=10)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 80

early_stopping_patience = 15  # Épocas sin mejorar antes de parar
best_val_loss = math.inf
epochs_without_improvement = 0

for epoch in tqdm(range(epochs)):
    model.train()
    train_loss = 0.0

    for inputs, targets in train_loader:  # inputs: (N, 2, H, W)
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        functional.reset_net(model)  # reset neuronas

        outputs = model(inputs)  # (N, 6)
        loss = criterion(outputs.squeeze(), targets.squeeze())  # Evita warnings

        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)

    avg_train_loss = train_loss / len(train_loader.dataset)

    # Validación
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            functional.reset_net(model)

            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets.squeeze())
            val_loss += loss.item() * inputs.size(0)

    avg_val_loss = val_loss / len(val_loader.dataset)

    #  # ---- EARLY STOPPING ----
    # if avg_val_loss < best_val_loss:
    #     best_val_loss = avg_val_loss
    #     epochs_without_improvement = 0

    # else:
    #     epochs_without_improvement += 1

    # if epochs_without_improvement >= early_stopping_patience:
    #     print(f"Early stopping triggered at epoch {epoch+1}!")
    #     break

    # Evaluación final en el conjunto de test
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            functional.reset_net(model)  # Reset neuronas para cada muestra

            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets.squeeze())
            test_loss += loss.item() * inputs.size(0)

    avg_test_loss = test_loss / len(test_loader.dataset)
    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "test_loss": avg_test_loss
    })

wandb.finish()