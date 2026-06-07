import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import h5py
import numpy as np
from scipy.special import comb
import os
import sys

# ==========================================
#           INDUSTRIAL CONFIGURATION
# ==========================================
CONFIG = {
    "DATASET_PATH": "airfoil_dataset_Re600k_with_limits.h5",
    "BATCH_SIZE": 128,
    "LEARNING_RATE": 0.001,
    "EPOCHS": 300,
    "NUM_POINTS": 200,
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "TRAIN_SPLIT": 0.9,
    
    # --- ABSOLUTE PHYSICS GATES ---
    # Any data point outside these bounds is flagged as "Numerical Garbage"
    "CONSTRAINTS": {
        # 1. Geometry
        "THICKNESS_MIN": 0.08,     # 8% (Structural limit)
        "THICKNESS_MAX": 0.22,     # 22% (Drag limit)
        "TE_GAP_MAX": 0.015,       # Kutta condition check
        
        # 2. Performance (Scalars)
        "LD_MIN": 10.0,            # If L/D < 10, it's inefficient
        "LD_MAX": 200.0,           # If L/D > 200 @ Re=600k, it's a Solver BUG
        "CL_MAX_LOWER": 1.1,       # Must have decent lift capability
        "CL_MAX_UPPER": 3.0,       # Cl > 3 is impossible without flaps
        
        # 3. Flow Field (Cp)
        "CP_ABS_MAX": 8.0,         # Pressure coefficient > 8 implies singularity/shock
        "CP_TE_DIV": 1.0           # Trailing edge pressure must recover
    }
}

# ==========================================
#           ROBUST GEOMETRY ENGINE
# ==========================================
class CST_Reconstructor:
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

    def generate_surface(self, w_vec):
        w_u, w_l, dz = w_vec[0:8], w_vec[8:16], w_vec[16]
        S_u = self.B @ w_u
        S_l = self.B @ w_l
        y_u = self.C * S_u + self.x * (dz / 2.0)
        y_l = -self.C * S_l - self.x * (dz / 2.0)
        return self.x, y_u, y_l, dz

    def get_cloud(self, w_vec):
        x_raw, y_u, y_l, dz = self.generate_surface(w_vec)
        x_all = np.concatenate((x_raw[::-1], x_raw[1:]))
        y_all = np.concatenate((y_u[::-1], y_l[1:]))

        points = np.vstack((x_all, y_all)).T
        dist = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
        dist = np.insert(dist, 0, 0)
        cum_dist = np.cumsum(dist)

        new_dist = np.linspace(0, cum_dist[-1], CONFIG["NUM_POINTS"])
        x_new = np.interp(new_dist, cum_dist, x_all)
        y_new = np.interp(new_dist, cum_dist, y_all)
        
        return np.vstack((x_new, y_new)).astype(np.float32)

# ==========================================
#           DEEP SANITIZATION DATASET
# ==========================================
class StrictAirfoilDataset(Dataset):
    def __init__(self, h5_file):
        if not os.path.exists(h5_file): raise FileNotFoundError(f"{h5_file} missing")
        self.cst = CST_Reconstructor(n_params=8, resolution=150)
        
        print(f"--- INITIALIZING FORENSIC DATA CLEANING ---")
        
        with h5py.File(h5_file, 'r') as f:
            raw_w = f['weights'][:]
            raw_s = f['scalars'][:] # [L/D, Cl_max, Bucket]
            raw_cp = f['cp'][:]
            
            # --- FILTER 1: NUMERICAL STABILITY ---
            mask = np.isfinite(raw_s).all(axis=1) & np.isfinite(raw_cp).all(axis=1)
            
            # --- FILTER 2: SCALAR BOUNDARIES ---
            # Efficiency Gate
            mask &= (raw_s[:, 0] >= CONFIG["CONSTRAINTS"]["LD_MIN"])
            mask &= (raw_s[:, 0] <= CONFIG["CONSTRAINTS"]["LD_MAX"]) # Catch solver explosions
            
            # Lift Capacity Gate
            mask &= (raw_s[:, 1] >= CONFIG["CONSTRAINTS"]["CL_MAX_LOWER"])
            mask &= (raw_s[:, 1] <= CONFIG["CONSTRAINTS"]["CL_MAX_UPPER"])
            
            # --- FILTER 3: FIELD SINGULARITY CHECK (Cp) ---
            # If Cp spikes to -20, it's a singularity.
            max_cp_val = np.max(np.abs(raw_cp), axis=1)
            mask &= (max_cp_val <= CONFIG["CONSTRAINTS"]["CP_ABS_MAX"])
            
            # --- FILTER 4: GEOMETRIC INTEGRITY ---
            final_indices = []
            
            # Pre-filter for speed
            temp_w = raw_w[mask]
            temp_s = raw_s[mask]
            temp_cp = raw_cp[mask]
            
            print(f"Scanning geometry of {len(temp_w)} candidates...")
            
            for i in range(len(temp_w)):
                w_vec = temp_w[i]
                x, y_u, y_l, dz = self.cst.generate_surface(w_vec)
                
                # Check A: Crossing
                if np.any((y_u - y_l) < -1e-5): continue 
                
                # Check B: Thickness
                t_max = np.max(y_u - y_l)
                if t_max < CONFIG["CONSTRAINTS"]["THICKNESS_MIN"] or \
                   t_max > CONFIG["CONSTRAINTS"]["THICKNESS_MAX"]:
                    continue
                    
                # Check C: Kutta (TE Gap)
                if abs(y_u[-1] - y_l[-1]) > CONFIG["CONSTRAINTS"]["TE_GAP_MAX"]:
                    continue
                
                final_indices.append(i)

            # Compile Final Clean Dataset
            self.weights = torch.tensor(temp_w[final_indices], dtype=torch.float32)
            self.scalars = torch.tensor(temp_s[final_indices], dtype=torch.float32)
            self.cp = torch.tensor(temp_cp[final_indices], dtype=torch.float32)
            
            self.len = len(final_indices)
            print(f"--- DATASET CLEANED AND LOCKED ---")
            print(f"Final Count: {self.len} (Rejected {len(raw_w) - self.len} bogus samples)")

        # Normalize
        self.s_mean = torch.mean(self.scalars, dim=0)
        self.s_std = torch.std(self.scalars, dim=0) + 1e-6

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        w = self.weights[idx].numpy()
        point_cloud = self.cst.get_cloud(w) 
        s = (self.scalars[idx] - self.s_mean) / self.s_std
        cp = self.cp[idx]
        return torch.tensor(point_cloud), s, cp

# ==========================================
#           PHYSICS-AWARE NETWORK
# ==========================================
class TNet(nn.Module):
    def __init__(self, k=2):
        super().__init__()
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        
        self.bn1 = nn.BatchNorm1d(64); self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024); self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.k = k

    def forward(self, x):
        bs = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = torch.eye(self.k, requires_grad=True).repeat(bs, 1, 1).to(x.device)
        return x.view(-1, self.k, self.k) + iden

class PointNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stn = TNet(k=2)
        self.conv1 = nn.Conv1d(2, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return torch.max(x, 2, keepdim=False)[0]

class IndustrialDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_s = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.Mish(), nn.Dropout(0.3),
            nn.Linear(512, 256),  nn.BatchNorm1d(256), nn.Mish(), nn.Dropout(0.2),
            nn.Linear(256, 3) 
        )
        self.fc_f = nn.Linear(1024, 256 * 25) 
        self.cnn_f = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 4, 2, 1), nn.BatchNorm1d(128), nn.Mish(),
            nn.ConvTranspose1d(128, 64, 4, 2, 1),  nn.BatchNorm1d(64),  nn.Mish(),
            nn.ConvTranspose1d(64, 32, 4, 2, 1),   nn.BatchNorm1d(32),  nn.Mish(),
            nn.Conv1d(32, 1, 3, 1, 1)
        )
    def forward(self, x):
        return self.fc_s(x), self.cnn_f(self.fc_f(x).view(-1, 256, 25)).squeeze(1)

class PhysicsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PointNetEncoder()
        self.decoder = IndustrialDecoder()
    def forward(self, x):
        return self.decoder(self.encoder(x))

# ==========================================
#           INDUSTRIAL LOSS FUNCTION
# ==========================================
class IndustrialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.huber = nn.HuberLoss(delta=0.1)

    def forward(self, s_pred, s_true, cp_pred, cp_true):
        # 1. Mission Critical Loss (Efficiency + Safety)
        loss_mission = self.mse(s_pred[:, 0:2], s_true[:, 0:2]) * 20.0
        
        # 2. Versatility Loss (Bucket Width)
        loss_bucket = self.mse(s_pred[:, 2], s_true[:, 2]) * 5.0
        
        # 3. Field Consistency (Cp)
        loss_field = self.huber(cp_pred, cp_true)
        
        return loss_mission + loss_bucket + loss_field

# ==========================================
#           TRAINING ENGINE
# ==========================================
def train():
    try:
        full_dataset = StrictAirfoilDataset(CONFIG["DATASET_PATH"])
        if len(full_dataset) < 100:
            print("CRITICAL: Data rejection too high. Check GA constraints."); return
    except Exception as e:
        print(f"Init Failed: {e}"); return

    train_size = int(CONFIG["TRAIN_SPLIT"] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=CONFIG["BATCH_SIZE"], shuffle=False)

    model = PhysicsNet().to(CONFIG["DEVICE"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["LEARNING_RATE"], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=CONFIG["LEARNING_RATE"], 
        steps_per_epoch=len(train_loader), epochs=CONFIG["EPOCHS"],
        pct_start=0.15, anneal_strategy='cos'
    )
    
    criterion = IndustrialLoss()
    best_loss = float('inf')

    print(f"--- TRAINING START: {CONFIG['EPOCHS']} Epochs ---")

    for epoch in range(CONFIG["EPOCHS"]):
        model.train()
        train_loss = 0.0
        
        for geo, s_target, cp_target in train_loader:
            geo, s_target, cp_target = geo.to(CONFIG["DEVICE"]), s_target.to(CONFIG["DEVICE"]), cp_target.to(CONFIG["DEVICE"])
            
            optimizer.zero_grad()
            s_pred, cp_pred = model(geo)
            loss = criterion(s_pred, s_target, cp_pred, cp_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for geo, s_target, cp_target in val_loader:
                geo, s_target, cp_target = geo.to(CONFIG["DEVICE"]), s_target.to(CONFIG["DEVICE"]), cp_target.to(CONFIG["DEVICE"])
                s_pred, cp_pred = model(geo)
                val_loss += criterion(s_pred, s_target, cp_pred, cp_target).item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)

        if epoch % 5 == 0:
            print(f"Epoch {epoch+1:03d} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), "industrial_physics_model.pth")

    print(f"--- DONE. Best Val Loss: {best_loss:.4f} ---")

if __name__ == "__main__":
    train()
