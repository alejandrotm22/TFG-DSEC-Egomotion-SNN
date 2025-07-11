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
    name="EgoMotionNetV10_67",
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

def rms(x):
    return np.sqrt(np.mean(np.square(x)))

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
print(f"test_gt_ego_motion shape: {test_ego_motions.shape}")
print(f"test_optical_flow shape: {test_optical_flows.shape}")

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
#train_dataset = EgoMotionDataset(opt_flow_train, ego_train, transform=FlowAugmentation(noise_std=0.1, drop_prob=0.1))
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
    # Inicializar listas para guardar predicciones y GTs
    all_vel_preds_val = []
    all_vel_gts_val = []
    all_omega_preds_val = []
    all_omega_gts_val = []

    with torch.no_grad():
        for inputs, targets in val_loader:  # Asegúrate de tener `val_loader`
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)

            # Separar velocidad y rotación
            outputs_np = outputs.squeeze().cpu().numpy()
            targets_np = targets.squeeze().cpu().numpy()

            vel_pred = outputs_np[:3]    # x, y, z
            omega_pred = outputs_np[3:]  # rot x, y, z

            vel_gt = targets_np[:3]
            omega_gt = targets_np[3:]

            # Guardar para RMS
            all_vel_preds_val.append(vel_pred)
            all_vel_gts_val.append(vel_gt)
            all_omega_preds_val.append(omega_pred)
            all_omega_gts_val.append(omega_gt)

    rms_vel_val = rms(np.array(all_vel_preds_val) - np.array(all_vel_gts_val))
    rms_omega_val = rms(np.array(all_omega_preds_val) - np.array(all_omega_gts_val))
    avg_val_loss = val_loss / len(val_loader.dataset)

    
    # ---- EARLY STOPPING ----
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0

    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= early_stopping_patience:
        print(f"Early stopping triggered at epoch {epoch+1}!")
        break

    # Evaluación en test
    model.eval()
    test_loss = 0.0
    # Inicializar listas para guardar predicciones y GTs
    all_vel_preds_test = []
    all_vel_gts_test = []
    all_omega_preds_test = []
    all_omega_gts_test = []

    # with torch.no_grad():
    #     for inputs, targets in test_loader:
    #         inputs, targets = inputs.to(device), targets.to(device)
    #         outputs = model(inputs)
    #         loss = criterion(outputs, targets)
    #         test_loss += loss.item() * inputs.size(0)

    #         # Separar velocidad y rotación
    #         outputs_np = outputs.squeeze().cpu().numpy()
    #         targets_np = targets.squeeze().cpu().numpy()

    #         vel_pred = outputs_np[:3]    # x, y, z
    #         omega_pred = outputs_np[3:]  # rot x, y, z

    #         vel_gt = targets_np[:3]
    #         omega_gt = targets_np[3:]

    #         # Guardar para RMS
    #         all_vel_preds_test.append(vel_pred)
    #         all_vel_gts_test.append(vel_gt)
    #         all_omega_preds_test.append(omega_pred)
    #         all_omega_gts_test.append(omega_gt)

    # Calcular RMS
    rms_vel_test = rms(np.array(all_vel_preds_test) - np.array(all_vel_gts_test))
    rms_omega_test = rms(np.array(all_omega_preds_test) - np.array(all_omega_gts_test))
    avg_test_loss = test_loss / len(test_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Test Loss: {avg_test_loss:.4f}, RMS Vel Val: {rms_vel_val:.4f}, RMS Omega Val: {rms_omega_val:.4f}, RMS Vel Test: {rms_vel_test:.4f}, RMS Omega Test: {rms_omega_test:.4f}")


    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "test_loss": avg_test_loss,
        "rms_vel_val": rms_vel_val,
        "rms_omega_val": rms_omega_val,
        "rms_vel_test": rms_vel_test,
        "rms_omega_test": rms_omega_test
    })


wandb.finish()
