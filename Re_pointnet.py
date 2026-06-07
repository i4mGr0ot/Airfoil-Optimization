import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import h5py
from scipy.special import comb
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import sys

# --- CONFIGURATION ---
CONFIG = {
    "H5_FILE": "airfoil_dataset_re_sweep.h5",
    "BATCH_SIZE": 64,  # Increased for stability
    "EPOCHS": 150,     # Enough for convergence
    "LR": 0.001,
    "NUM_POINTS": 200,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu"
}

# --- GEOMETRY KERNEL ---
class CST_Kernel:
    def __init__(self, n_params=8, resolution=200):
        self.n_params = n_params
        self.resolution = resolution
        self.beta = np.linspace(0, np.pi, self.resolution)
        self.x = 0.5 * (1 - np.cos(self.beta))
        self.B = np.zeros((self.resolution, self.n_params))
        self.C = np.sqrt(self.x) * (1 - self.x)
        n = self.n_params - 1
        for k in range(self.n_params):
            self.B[:, k] = comb(n, k) * (self.x**k) * ((1 - self.x)**(n - k))
            
    def compute(self, w_u, w_l, dz):
        S_u = self.B @ w_u
        S_l = self.B @ w_l
        y_u = self.C * S_u + self.x * (dz / 2.0)
        y_l = -self.C * S_l - self.x * (dz / 2.0)
        x_coords = np.concatenate((self.x[::-1], self.x[1:]))
        y_coords = np.concatenate((y_u[::-1], y_l[1:]))
        points = np.column_stack((x_coords, y_coords))
        if len(points) != CONFIG["NUM_POINTS"]:
            indices = np.linspace(0, len(points)-1, CONFIG["NUM_POINTS"])
            px = np.interp(indices, np.arange(len(points)), points[:,0])
            py = np.interp(indices, np.arange(len(points)), points[:,1])
            points = np.column_stack((px, py))
        return points.astype(np.float32)

# --- ROBUST DATA PIPELINE ---
def get_dataloaders(h5_path):
    if not os.path.exists(h5_path):
        print(f"CRITICAL: {h5_path} missing.")
        sys.exit(1)

    print(f"--- 1. Loading Data from {h5_path} ---")
    with h5py.File(h5_path, 'r') as f:
        weights = f['weights'][:]
        scalars = f['scalars'][:] # [CL, CD, CM, Alpha, Reynolds]
        cp = f['cp'][:]

    # Filter NaNs/Infs immediately
    valid_mask = np.isfinite(scalars).all(axis=1) & np.isfinite(cp).all(axis=1)
    weights = weights[valid_mask]
    scalars = scalars[valid_mask]
    cp = cp[valid_mask]
    print(f"    Kept {len(weights)} valid samples.")

    # --- 2. Reconstruct Geometries (CPU Intensive) ---
    print("--- 2. Reconstructing Point Clouds ---")
    kernel = CST_Kernel()
    point_clouds = []
    for w in tqdm(weights, desc="Meshing", leave=False):
        pc = kernel.compute(w[0:8], w[8:16], w[16])
        point_clouds.append(pc.T) # [2, 200]
    point_clouds = np.array(point_clouds)

    # --- 3. Split Data ---
    print("--- 3. Splitting & Scaling ---")
    X_pc = point_clouds
    y_scalars = scalars[:, 0:3]     # CL, CD, CM
    y_re = scalars[:, 4].reshape(-1, 1) # Reynolds
    y_cp = cp

    # Split indices first
    pc_train, pc_val, s_train, s_val, re_train, re_val, cp_train, cp_val = train_test_split(
        X_pc, y_scalars, y_re, y_cp, test_size=0.1, random_state=42, shuffle=True
    )

    # --- 4. Shared Scaling (THE FIX) ---
    # Fit scalers ONLY on training data
    scalar_scaler = StandardScaler()
    s_train_norm = scalar_scaler.fit_transform(s_train)
    s_val_norm = scalar_scaler.transform(s_val) # Use TRAIN scaler on VAL

    re_scaler = MinMaxScaler()
    re_train_norm = re_scaler.fit_transform(re_train)
    re_val_norm = re_scaler.transform(re_val)   # Use TRAIN scaler on VAL

    print(f"    Train Samples: {len(pc_train)} | Val Samples: {len(pc_val)}")
    
    # Create TensorDatasets
    train_ds = TensorDataset(
        torch.tensor(pc_train, dtype=torch.float32),
        torch.tensor(re_train_norm, dtype=torch.float32),
        torch.tensor(s_train_norm, dtype=torch.float32),
        torch.tensor(cp_train, dtype=torch.float32)
    )
    val_ds = TensorDataset(
        torch.tensor(pc_val, dtype=torch.float32),
        torch.tensor(re_val_norm, dtype=torch.float32),
        torch.tensor(s_val_norm, dtype=torch.float32),
        torch.tensor(cp_val, dtype=torch.float32)
    )

    return (
        DataLoader(train_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, drop_last=True),
        DataLoader(val_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, drop_last=False)
    )

# --- NETWORK UTILS ---
def index_points(points, idx):
    B, C, N = points.shape
    S, K = idx.shape[1], idx.shape[2]
    points_t = points.transpose(2, 1).contiguous() 
    flat_points = points_t.view(B * N, C)
    offset = torch.arange(B, device=points.device).view(B, 1, 1) * N
    flat_idx = (idx + offset).view(-1)
    res = flat_points[flat_idx] 
    return res.view(B, S, K, C)

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel + 3 
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        B, C, N = xyz.shape
        if self.group_all:
            if points is not None: new_points = torch.cat([xyz, points], dim=1)
            else: new_points = xyz
            new_xyz = None
            new_points = new_points.unsqueeze(2) 
        else:
            stride = N // self.npoint
            idx = torch.arange(0, N, stride, device=xyz.device)[:self.npoint]
            new_xyz = xyz[:, :, idx]
            dist = torch.cdist(new_xyz.transpose(1,2), xyz.transpose(1,2))
            val, group_idx = torch.topk(dist, self.nsample, dim=2, largest=False)
            grouped_xyz = index_points(xyz, group_idx).permute(0, 3, 1, 2)
            grouped_xyz -= new_xyz.transpose(1,2).unsqueeze(2).permute(0, 3, 1, 2)
            if points is not None:
                grouped_points = index_points(points, group_idx).permute(0, 3, 1, 2)
                new_points = torch.cat([grouped_xyz, grouped_points], dim=1)
            else: new_points = grouped_xyz

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, 3)[0]
        return new_xyz, new_points

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        if xyz2 is None: interpolated_points = points2.repeat(1, 1, xyz1.shape[2])
        else:
            dist = torch.cdist(xyz1.transpose(1,2), xyz2.transpose(1,2))
            dists, idx = torch.topk(dist, 3, dim=2, largest=False)
            weight = 1.0 / (dists + 1e-10)
            weight = weight / torch.sum(weight, dim=2, keepdim=True)
            grouped = index_points(points2, idx) 
            interpolated_points = torch.sum(grouped * weight.unsqueeze(-1), dim=2)
            interpolated_points = interpolated_points.transpose(1, 2)

        if points1 is not None: new_points = torch.cat([points1, interpolated_points], dim=1)
        else: new_points = interpolated_points

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

# --- PHYSICS-AWARE MODEL ---
class ReAwareAirfoilPointNet(nn.Module):
    def __init__(self):
        super(ReAwareAirfoilPointNet, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=100, radius=0.1, nsample=32, in_channel=0, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=32, radius=0.2, nsample=32, in_channel=128, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256, mlp=[256, 512, 1024], group_all=True)

        self.fp3 = PointNetFeaturePropagation(in_channel=1025+256, mlp=[256, 256]) # 1024 + 1 (Re) + 256
        self.fp2 = PointNetFeaturePropagation(in_channel=256+128, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128+3, mlp=[128, 128, 128]) 
        
        self.conv_cp = nn.Conv1d(128, 1, 1)

        self.fc1 = nn.Linear(1025, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, 3)

    def forward(self, xyz, re_input):
        B, C, N = xyz.shape
        z = torch.zeros((B, 1, N), device=xyz.device)
        xyz_3d = torch.cat([xyz, z], dim=1)

        l1_xyz, l1_points = self.sa1(xyz_3d, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        global_feat = l3_points.view(B, 1024)
        physics_feat = torch.cat([global_feat, re_input], dim=1) 
        
        x = F.relu(self.bn1(self.fc1(physics_feat)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        scalars = self.fc3(x)

        physics_feat_expanded = physics_feat.unsqueeze(-1) 
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, physics_feat_expanded)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz_3d, l1_xyz, xyz_3d, l1_points)
        
        cp_dist = self.conv_cp(l0_points).squeeze(1)
        return scalars, cp_dist

# --- TRAINING LOOP ---
def train():
    train_loader, val_loader = get_dataloaders(CONFIG["H5_FILE"])
    model = ReAwareAirfoilPointNet().to(CONFIG["DEVICE"])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3)
    # FIX: Removed 'verbose=True' which is deprecated in newer PyTorch versions
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_loss = float('inf')
    
    print(f"\n--- Starting Training on {CONFIG['DEVICE']} ---")
    for epoch in range(CONFIG["EPOCHS"]):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False)
        for pc, re_in, s_target, cp_target in pbar:
            pc, re_in = pc.to(CONFIG["DEVICE"]), re_in.to(CONFIG["DEVICE"])
            s_target, cp_target = s_target.to(CONFIG["DEVICE"]), cp_target.to(CONFIG["DEVICE"])
            
            optimizer.zero_grad()
            s_pred, cp_pred = model(pc, re_in)
            
            loss = F.mse_loss(s_pred, s_target) + 2.0 * F.mse_loss(cp_pred, cp_target)
            
            if torch.isnan(loss):
                print("CRITICAL: NaN Loss detected! Reducing LR or checking data.")
                break
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
        avg_train = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for pc, re_in, s_target, cp_target in val_loader:
                pc, re_in = pc.to(CONFIG["DEVICE"]), re_in.to(CONFIG["DEVICE"])
                s_target, cp_target = s_target.to(CONFIG["DEVICE"]), cp_target.to(CONFIG["DEVICE"])
                s_pred, cp_pred = model(pc, re_in)
                loss = F.mse_loss(s_pred, s_target) + 2.0 * F.mse_loss(cp_pred, cp_target)
                val_loss += loss.item()
                
        avg_val = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        scheduler.step(avg_val)
        
        print(f"Epoch {epoch+1:03d} | Train: {avg_train:.5f} | Val: {avg_val:.5f}")
        
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), "best_physics_pointnet.pth")
            print(f"    >>> Model Saved (Val Loss: {avg_val:.5f})")

if __name__ == "__main__":
    train()
