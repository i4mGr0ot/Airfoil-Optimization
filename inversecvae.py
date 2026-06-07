import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import h5py
import numpy as np
import os
import sys
import math  

# --- CONFIGURATION ---
DATASET_PATH = "robust_airfoil_dataset_Re600k_fine.h5"
POINTNET_PATH = "pointnet_physics_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 100
LATENT_DIM = 16 

# ==========================================
# 1. DIFFERENTIABLE CST LAYER (TORCH)
# ==========================================
class DifferentiableCST(nn.Module):
    def __init__(self, n_params=8, resolution=200):
        super().__init__()
        self.n_params = n_params
        self.resolution = resolution
        
        # Buffers (Non-learnable constants)
        beta = torch.linspace(0, np.pi, resolution)
        x = 0.5 * (1 - torch.cos(beta))
        self.register_buffer('x', x)
        
        self.C = torch.sqrt(x) * (1 - x)
        
        # Bernstein Matrix B
        B = torch.zeros(resolution, n_params)
        n = n_params - 1
        for k in range(n_params):
            # FIXED: Use math.factorial instead of np.math.factorial
            coeff = math.factorial(n) / (math.factorial(k) * math.factorial(n-k))
            B[:, k] = coeff * (x**k) * ((1 - x)**(n - k))
        self.register_buffer('B', B)

    def forward(self, weights):
        # weights: (Batch, 17) -> [8 upper, 8 lower, 1 dz]
        w_u = weights[:, 0:8]
        w_l = weights[:, 8:16]
        dz  = weights[:, 16].unsqueeze(1)

        # Enforce LE Continuity (Differentiable)
        w_l = torch.cat([w_u[:, 0:1], w_l[:, 1:]], dim=1)

        S_u = torch.matmul(self.B, w_u.T).T 
        S_l = torch.matmul(self.B, w_l.T).T 

        y_u = self.C * S_u + self.x * (dz / 2.0)
        y_l = -self.C * S_l - self.x * (dz / 2.0)

        # Stack into Point Cloud (Batch, 2, Res*2)
        x_cycle = torch.cat([self.x.flip(0), self.x], dim=0)
        y_cycle = torch.cat([y_u.flip(1), y_l], dim=1)
        
        # Create (Batch, 2, N)
        cloud = torch.stack([x_cycle.expand(weights.size(0), -1), y_cycle], dim=1)
        
        # Downsample 400->200
        return cloud[:, :, ::2]

# ==========================================
# 2. CVAE ARCHITECTURE (The Designer)
# ==========================================
class CVAE(nn.Module):
    def __init__(self, input_dim=17, cond_dim=3, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + cond_dim, 128),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 64),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, input_dim) 
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, cond):
        inputs = torch.cat([x, cond], dim=1)
        h = self.encoder(inputs)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        z_cond = torch.cat([z, cond], dim=1)
        recon_w = self.decoder(z_cond)
        return recon_w, mu, logvar

# ==========================================
# 3. PHYSICS CRITIC (Must Match Training Script)
# ==========================================
class PointNetLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, 1)
        self.bn = nn.BatchNorm1d(out_ch)
    def forward(self, x): return F.relu(self.bn(self.conv(x)))

class PointNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp1 = nn.Sequential(PointNetLayer(2, 64), PointNetLayer(64, 128))
        self.mlp2 = nn.Sequential(PointNetLayer(128, 256), PointNetLayer(256, 1024))
    def forward(self, x):
        x = self.mlp1(x)
        x = self.mlp2(x)
        return torch.max(x, 2)[0], x

class AerodynamicDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_scalars = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 3)
        )
        # Field head omitted for memory efficiency (we only need scalars for optimization)
        
    def forward(self, x):
        return self.fc_scalars(x)

class PointNetPhysicsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PointNetEncoder()
        self.decoder = AerodynamicDecoder()
        
    def forward(self, x):
        global_feat, _ = self.encoder(x)
        return self.decoder(global_feat)

# ==========================================
# 4. TRAINING ENGINE
# ==========================================
class H5Dataset(Dataset):
    def __init__(self, path):
        with h5py.File(path, 'r') as f:
            self.weights = torch.tensor(f['weights'][:], dtype=torch.float32)
            self.scalars = torch.tensor(f['scalars'][:], dtype=torch.float32)
            
            # Normalize
            self.s_mean = self.scalars.mean(dim=0)
            self.s_std = self.scalars.std(dim=0)
            self.scalars = (self.scalars - self.s_mean) / (self.s_std + 1e-6)

    def __len__(self): return len(self.weights)
    def __getitem__(self, idx): return self.weights[idx], self.scalars[idx]

def train_inverse():
    print("--- Initializing Inverse Training ---")
    
    dataset = H5Dataset(DATASET_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Init Models
    cvae = CVAE(latent_dim=LATENT_DIM).to(DEVICE)
    diff_cst = DifferentiableCST().to(DEVICE)
    
    # Load Critic
    critic = PointNetPhysicsModel().to(DEVICE)
    try:
        # strict=False allows loading even if the Cp head is missing in this definition
        critic.load_state_dict(torch.load(POINTNET_PATH, map_location=DEVICE), strict=False)
        print("Physics Critic Loaded.")
        for param in critic.parameters(): param.requires_grad = False
        critic.eval()
    except Exception as e:
        print(f"Warning: Critic load failed ({e}). Physics loss disabled.")
        critic = None
    
    optimizer = torch.optim.Adam(cvae.parameters(), lr=LR)
    
    print(f"--- Training on {DEVICE} ---")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        
        for true_w, cond in loader:
            true_w, cond = true_w.to(DEVICE), cond.to(DEVICE)
            
            optimizer.zero_grad()
            
            # 1. CVAE Forward
            recon_w, mu, logvar = cvae(true_w, cond)
            
            # 2. Losses
            loss_recon = F.mse_loss(recon_w, true_w)
            loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / (BATCH_SIZE * 17)
            
            # 3. Physics Loss
            loss_phys = torch.tensor(0.0).to(DEVICE)
            if critic:
                gen_cloud = diff_cst(recon_w) # Generate Shape
                pred_phys = critic(gen_cloud) # Check Physics
                loss_phys = F.mse_loss(pred_phys, cond) # Must match target
            
            # Weighted Sum
            loss = loss_recon + 0.1 * loss_kl + 1.0 * loss_phys
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.4f}")
    
    torch.save(cvae.state_dict(), "inverse_cvae_model.pth")
    np.savez("norm_stats.npz", mean=dataset.s_mean.numpy(), std=dataset.s_std.numpy())
    print("--- Training Complete. Model Saved. ---")

if __name__ == "__main__":
    train_inverse()
