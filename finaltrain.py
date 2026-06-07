import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from scipy.special import comb
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os
import sys

# --- CONFIGURATION ---
CONFIG = {
    "H5_FILE": "airfoil_dataset_nsga2.h5", # Your file from the previous step
    "BATCH_SIZE": 32,
    "EPOCHS": 200,
    "LR": 0.001,
    "NUM_POINTS": 200,    # Resolution of the airfoil (must match XFOIL used)
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu"
}

# --- GEOMETRY RECONSTRUCTION (Same as before, needed to make point clouds) ---
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
        # Reconstruct coordinates from weights
        S_u = self.B @ w_u
        S_l = self.B @ w_l
        y_u = self.C * S_u + self.x * (dz / 2.0)
        y_l = -self.C * S_l - self.x * (dz / 2.0)
        
        # Combine into (200, 2) array [x, y]
        # Note: XFOIL wraps around, but PointNet likes ordered surface points.
        # We will stack Upper and Lower surfaces.
        # Ideally, PointNet works on sets, but for Cp regression, ordering helps.
        # Let's create a single sequence: Trailing Edge -> Leading Edge -> Trailing Edge
        
        x_coords = np.concatenate((self.x[::-1], self.x[1:]))
        y_coords = np.concatenate((y_u[::-1], y_l[1:]))
        
        # We need exactly CONFIG['NUM_POINTS']. 
        # The CST output size varies slightly based on resolution. 
        # For PointNet, we often just sample the surface.
        # Let's stick to the raw Upper/Lower definition to keep it aligned with Cp.
        
        # Simplified: Just stack [x, y_u] and [x, y_l] for the network
        # Shape: (200, 2) - This is a "surface patch"
        # We will use the Cp indices 0..200 directly from the H5 which are mapped to these.
        
        # Actually, the H5 Cp data corresponds to the XFOIL pane nodes.
        # The easiest robust way is to pass the X coordinate and Y coordinate as features.
        
        points = np.column_stack((x_coords, y_coords))
        
        # Resample to fixed size if necessary (simple linear interp)
        if len(points) != CONFIG["NUM_POINTS"]:
            # Simple fix: We assume the H5 Cp data is already interpolated to 200 
            # (as per the genetic algo script).
            # If coordinates mismatch length, we force resize.
            indices = np.linspace(0, len(points)-1, CONFIG["NUM_POINTS"])
            points_x = np.interp(indices, np.arange(len(points)), points[:,0])
            points_y = np.interp(indices, np.arange(len(points)), points[:,1])
            points = np.column_stack((points_x, points_y))
            
        return points.astype(np.float32)

# --- DATASET LOADER ---
class AirfoilDataset(Dataset):
    def __init__(self, h5_path, train=True):
        self.kernel = CST_Kernel()
        
        with h5py.File(h5_path, 'r') as f:
            # Load raw data
            self.weights = f['weights'][:]  # CST params
            self.scalars = f['scalars'][:]  # [CL, CD, CM, Alpha, ...]
            self.cp = f['cp'][:]            # Cp distribution
            
        # Select Targets: CL, CD, CM (Indices 0, 1, 2)
        self.targets = self.scalars[:, 0:3]
        
        # Normalize Targets (Critical for Regression)
        self.scaler = StandardScaler()
        if train:
            self.targets = self.scaler.fit_transform(self.targets)
        
        # Pre-compute Geometries (Optimization)
        print("--- Pre-generating Point Clouds from Genotypes ---")
        self.point_clouds = []
        for w in tqdm(self.weights):
            w_u = w[0:8]
            w_l = w[8:16]
            dz  = w[16]
            pc = self.kernel.compute(w_u, w_l, dz) # (200, 2)
            self.point_clouds.append(pc)
            
        self.point_clouds = np.array(self.point_clouds)
        
        # Split Train/Val (80/20)
        split = int(0.8 * len(self.point_clouds))
        if train:
            self.point_clouds = self.point_clouds[:split]
            self.targets = self.targets[:split]
            self.cp = self.cp[:split]
        else:
            self.point_clouds = self.point_clouds[split:]
            self.targets = self.targets[split:]
            self.cp = self.cp[split:]

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        # PointNet expects (Channels, Points) -> (2, 200)
        pc = torch.tensor(self.point_clouds[idx]).transpose(0, 1) 
        scalar_y = torch.tensor(self.targets[idx], dtype=torch.float32)
        cp_y = torch.tensor(self.cp[idx], dtype=torch.float32)
        return pc, scalar_y, cp_y

# --- POINTNET++ MODULES ---

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        last_channel = in_channel + 3 # +3 for XYZ coords
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        # xyz: [B, C, N]
        B, C, N = xyz.shape
        
        if self.group_all:
            new_xyz = None
            new_points = torch.cat([xyz, points], dim=1) if points is not None else xyz
            new_points = new_points.unsqueeze(-1) # [B, C+D, N, 1]
        else:
            # Farthest Point Sampling (Simplified: Random/Stride for speed in 1D)
            # For airfoils (1D manifold), striding is often better than FPS
            stride = N // self.npoint
            idx = torch.arange(0, N, stride, device=xyz.device)[:self.npoint]
            new_xyz = xyz[:, :, idx]
            
            # Grouping (kNN or Ball Query)
            # Since data is ordered (airfoil curve), we can just take neighbors in index
            # But true PointNet uses geometric distance.
            dist = torch.cdist(new_xyz.transpose(1,2), xyz.transpose(1,2)) # [B, npoint, N]
            val, idx = torch.topk(dist, self.nsample, dim=2, largest=False) # [B, npoint, nsample]
            
            # Gather
            grouped_xyz = self.index_points(xyz, idx) # [B, npoint, nsample, C]
            grouped_xyz -= new_xyz.transpose(1,2).unsqueeze(2) # Center
            
            if points is not None:
                grouped_points = self.index_points(points, idx)
                new_points = torch.cat([grouped_xyz.permute(0,3,1,2), grouped_points.permute(0,3,1,2)], dim=1)
            else:
                new_points = grouped_xyz.permute(0,3,1,2)

        # MLP
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 3)[0] # Max Pool
        return new_xyz, new_points

    def index_points(self, points, idx):
        # points: [B, C, N]
        # idx: [B, S, K]
        B, C, N = points.shape
        S, K = idx.shape[1], idx.shape[2]
        
        idx_base = torch.arange(0, B, device=points.device).view(-1, 1, 1) * N
        idx = idx + idx_base
        idx = idx.view(-1)
        
        res = points.transpose(2, 1).contiguous().view(B*N, C)[idx, :]
        res = res.view(B, S, K, C)
        return res

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
        # Interpolate features from xyz2 to xyz1
        if xyz2 is None:
            interpolated_points = points2.repeat(1, 1, xyz1.shape[2])
        else:
            dist = torch.cdist(xyz1.transpose(1,2), xyz2.transpose(1,2))
            dists, idx = torch.topk(dist, 3, dim=2, largest=False)
            
            weight = 1.0 / (dists + 1e-10)
            weight = weight / torch.sum(weight, dim=2, keepdim=True)
            
            interpolated_points = torch.sum(self.index_points(points2, idx) * weight.view(xyz1.shape[0], xyz1.shape[2], 3, 1), dim=2)
            interpolated_points = interpolated_points.transpose(1, 2)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=1)
        else:
            new_points = interpolated_points

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

    def index_points(self, points, idx):
        B, C, N = points.shape
        idx = idx.reshape(B, -1)
        res = points.transpose(2, 1).contiguous().view(B*N, C)[idx.view(-1) + torch.arange(0, B, device=points.device).view(-1,1)*N]
        return res.view(B, idx.shape[1]//3, 3, C)

# --- MAIN NETWORK ---

class AirfoilPointNet(nn.Module):
    def __init__(self):
        super(AirfoilPointNet, self).__init__()
        
        # Encoders (Set Abstraction)
        # 200 points -> 100 points
        self.sa1 = PointNetSetAbstraction(npoint=100, radius=0.1, nsample=32, in_channel=0, mlp=[64, 64, 128], group_all=False)
        # 100 points -> 32 points
        self.sa2 = PointNetSetAbstraction(npoint=32, radius=0.2, nsample=32, in_channel=128+3, mlp=[128, 128, 256], group_all=False)
        # 32 points -> Global Feature
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256+3, mlp=[256, 512, 1024], group_all=True)

        # Decoder for Cp (Feature Propagation)
        self.fp3 = PointNetFeaturePropagation(in_channel=1024+256, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256+128, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128+0+3+3, mlp=[128, 128, 128]) # +3+3 for interpolated coords
        
        self.conv_cp = nn.Conv1d(128, 1, 1) # Predicts 1 value (Cp) per point

        # Global Head for Scalars (CL, CD, CM)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, 3) # Predicts CL, CD, CM

    def forward(self, xyz):
        # xyz: [B, 2, 200]
        # We need to add a Z coordinate for standard PointNet math (even if it's 0)
        B, C, N = xyz.shape
        z = torch.zeros((B, 1, N), device=xyz.device)
        xyz_3d = torch.cat([xyz, z], dim=1) # [B, 3, 200]

        # Encoder
        l1_xyz, l1_points = self.sa1(xyz_3d, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Branch 1: Global Scalar Prediction
        global_feat = l3_points.view(B, 1024)
        x = F.relu(self.bn1(self.fc1(global_feat)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        scalars = self.fc3(x)

        # Branch 2: Dense Cp Prediction (Decoder)
        # Propagate features back up to original resolution
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz_3d, l1_xyz, None, l1_points)
        
        cp_dist = self.conv_cp(l0_points).squeeze(1) # [B, 200]

        return scalars, cp_dist

# --- TRAINING LOOP ---

def train():
    if not os.path.exists(CONFIG["H5_FILE"]):
        print(f"Error: {CONFIG['H5_FILE']} not found.")
        return

    print("Initialize Dataset...")
    train_ds = AirfoilDataset(CONFIG["H5_FILE"], train=True)
    val_ds   = AirfoilDataset(CONFIG["H5_FILE"], train=False)
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=0)
    
    model = AirfoilPointNet().to(CONFIG["DEVICE"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["LR"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["EPOCHS"])
    
    # Loss Weights
    w_scalar = 1.0
    w_cp     = 5.0 # Give more weight to Cp curve accuracy
    
    best_val_loss = float('inf')
    
    print("\n--- Starting Training ---")
    for epoch in range(CONFIG["EPOCHS"]):
        model.train()
        train_loss = 0.0
        
        for pc, s_target, cp_target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['EPOCHS']}", leave=False):
            pc, s_target, cp_target = pc.to(CONFIG["DEVICE"]), s_target.to(CONFIG["DEVICE"]), cp_target.to(CONFIG["DEVICE"])
            
            optimizer.zero_grad()
            s_pred, cp_pred = model(pc)
            
            loss_s = F.mse_loss(s_pred, s_target)
            loss_cp = F.mse_loss(cp_pred, cp_target)
            
            loss = (w_scalar * loss_s) + (w_cp * loss_cp)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for pc, s_target, cp_target in val_loader:
                pc, s_target, cp_target = pc.to(CONFIG["DEVICE"]), s_target.to(CONFIG["DEVICE"]), cp_target.to(CONFIG["DEVICE"])
                s_pred, cp_pred = model(pc)
                loss = (w_scalar * F.mse_loss(s_pred, s_target)) + (w_cp * F.mse_loss(cp_pred, cp_target))
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_airfoil_pointnet.pth")
            print("  >>> Model Saved")

    print("Training Complete.")

if __name__ == "__main__":
    train()
