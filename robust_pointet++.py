import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from scipy.special import comb
import os
import sys

DATASET_PATH = "robust_airfoil_dataset_Re600k_fine.h5"
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_POINTS = 200  

class CST_Reconstructor:
    """
    Helper to convert Weights -> Point Cloud inside the Dataloader.
    Keeps the HDF5 file small, generates geometry on CPU.
    """
    def __init__(self, n_params=8, resolution=200):
        self.n_params = n_params
        self.resolution = resolution

        self.beta = np.linspace(0, np.pi, self.resolution)
        self.x = 0.5 * (1 - np.cos(self.beta))
        self.C = np.sqrt(self.x) * (1 - self.x)
        self.B = np.zeros((self.resolution, self.n_params))
        n = self.n_params - 1
        for k in range(self.n_params):
            self.B[:, k] = comb(n, k) * (self.x**k) * ((1 - self.x)**(n - k))

    def get_cloud(self, w_vec):

        w_u = w_vec[0:8]
        w_l = w_vec[8:16]
        dz = w_vec[16]
        
        S_u = self.B @ w_u
        S_l = self.B @ w_l
        y_u = self.C * S_u + self.x * (dz / 2.0)
        y_l = -self.C * S_l - self.x * (dz / 2.0)

        x_all = np.concatenate((self.x[::-1], self.x))
        y_all = np.concatenate((y_u[::-1], y_l))

        points = np.vstack((x_all, y_all)).T
        dist = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
        dist = np.insert(dist, 0, 0)
        cum_dist = np.cumsum(dist)

        new_dist = np.linspace(0, cum_dist[-1], NUM_POINTS)
        x_new = np.interp(new_dist, cum_dist, x_all)
        y_new = np.interp(new_dist, cum_dist, y_all)
        
        return np.vstack((x_new, y_new)).astype(np.float32)

class AirfoilDataset(Dataset):
    def __init__(self, h5_file):
        if not os.path.exists(h5_file):
            raise FileNotFoundError(f"Dataset {h5_file} not found!")
            
        self.h5_file = h5_file
        with h5py.File(h5_file, 'r') as f:
            self.len = f['weights'].shape[0]

            self.weights = f['weights'][:]
            self.scalars = f['scalars'][:] 
            self.cp = f['cp'][:]
            
        self.cst = CST_Reconstructor(n_params=8, resolution=128)

        self.scalar_mean = np.mean(self.scalars, axis=0)
        self.scalar_std = np.std(self.scalars, axis=0) + 1e-6

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        
        w = self.weights[idx]
        point_cloud = self.cst.get_cloud(w) 
   
        s = (self.scalars[idx] - self.scalar_mean) / self.scalar_std
        
        cp = self.cp[idx]
        
        return torch.tensor(point_cloud), torch.tensor(s, dtype=torch.float32), torch.tensor(cp, dtype=torch.float32)

class PointNetLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, 1)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class PointNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp1 = nn.Sequential(
            PointNetLayer(2, 64),
            PointNetLayer(64, 128)
        )
        self.mlp2 = nn.Sequential(
            PointNetLayer(128, 256),
            PointNetLayer(256, 1024)
        )
        
    def forward(self, x):
        # Local Features
        x = self.mlp1(x) # (B, 128, N)
        local_feat = x
        
        # Global Features
        x = self.mlp2(x) # (B, 1024, N)
        global_feat = torch.max(x, 2)[0] # (B, 1024) - The Latent Vector
        
        return global_feat, local_feat

class AerodynamicDecoder(nn.Module):
    """
    Decodes the Latent Vector into Physics.
    """
    def __init__(self):
        super().__init__()
        
        # --- HEAD 1: SCALAR PREDICTION (Cl, Cd, Alpha) ---
        self.fc_scalars = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3) # Output: Cl, Cd, Alpha
        )
        

        self.fc_field = nn.Linear(1024, 256 * 25) # Reshape to (B, 256, 25)
        
        self.cnn_field = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1), # -> 50
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),  # -> 100
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),   # -> 200
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=3, padding=1) # -> (B, 1, 200) Final Cp
        )

    def forward(self, global_feat):
        # Scalars
        scalars = self.fc_scalars(global_feat)
        
        # Field (Cp)
        x = self.fc_field(global_feat)
        x = x.view(-1, 256, 25)
        cp_curve = self.cnn_field(x).squeeze(1) # (B, 200)
        
        return scalars, cp_curve

class PointNetPhysicsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PointNetEncoder()
        self.decoder = AerodynamicDecoder()
        
    def forward(self, x):
        global_feat, _ = self.encoder(x)
        return self.decoder(global_feat)

def train():
    print(f"--- Loading Dataset: {DATASET_PATH} ---")
    try:
        dataset = AirfoilDataset(DATASET_PATH)
    except Exception as e:
        print(e)
        return

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # 0 for Windows safety
    
    model = PointNetPhysicsModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    criterion_scalar = nn.MSELoss()
    criterion_cp = nn.HuberLoss()

    print(f"--- Starting Training on {DEVICE} ---")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        total_s_loss = 0
        total_c_loss = 0
        
        for geo, target_s, target_cp in train_loader:
            geo = geo.to(DEVICE)
            target_s = target_s.to(DEVICE)
            target_cp = target_cp.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward
            pred_s, pred_cp = model(geo)
            
            # Loss Calculation
            loss_s = criterion_scalar(pred_s, target_s)
            loss_c = criterion_cp(pred_cp, target_cp)
            
            # Weighted Sum (Cp is harder, give it weight)
            loss = loss_s + 5.0 * loss_c
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_s_loss += loss_s.item()
            total_c_loss += loss_c.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} (Scalar: {total_s_loss:.4f} | Cp: {total_c_loss:.4f})")

    # Save
    torch.save(model.state_dict(), "pointnet_physics_model.pth")
    print("--- Model Saved: pointnet_physics_model.pth ---")

if __name__ == "__main__":
    train()
