import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import math

# --- CONFIGURATION ---
CVAE_PATH = "inverse_cvae_model.pth"
POINTNET_PATH = "pointnet_physics_model.pth"
STATS_PATH = "norm_stats.npz"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. DIFFERENTIABLE CST KERNEL
# ==========================================
class DifferentiableCST(nn.Module):
    def __init__(self, n_params=8, resolution=200):
        super().__init__()
        self.n_params = n_params
        beta = torch.linspace(0, np.pi, resolution)
        x = 0.5 * (1 - torch.cos(beta))
        self.register_buffer('x', x)
        self.C = torch.sqrt(x) * (1 - x)
        B = torch.zeros(resolution, n_params)
        n = n_params - 1
        for k in range(n_params):
            coeff = math.factorial(n) / (math.factorial(k) * math.factorial(n-k))
            B[:, k] = coeff * (x**k) * ((1 - x)**(n - k))
        self.register_buffer('B', B)

    def forward(self, weights):
        w_u = weights[:, 0:8]
        w_l = weights[:, 8:16]
        dz  = weights[:, 16].unsqueeze(1)
        
        # Enforce LE Continuity
        w0 = (w_u[:, 0] + w_l[:, 0]) / 2.0
        w_u = torch.cat([w0.unsqueeze(1), w_u[:, 1:]], dim=1)
        w_l = torch.cat([w0.unsqueeze(1), w_l[:, 1:]], dim=1)
        
        # Enforce Closed TE (Strict)
        dz = torch.zeros_like(dz)

        S_u = torch.matmul(self.B, w_u.T).T
        S_l = torch.matmul(self.B, w_l.T).T 
        
        y_u = self.C * S_u + self.x * (dz / 2.0)
        y_l = -self.C * S_l - self.x * (dz / 2.0)
        
        # Build Point Cloud (2, 200)
        x_cycle = torch.cat([self.x.flip(0), self.x], dim=0)
        y_cycle = torch.cat([y_u.flip(1), y_l], dim=1)
        cloud = torch.stack([x_cycle.expand(weights.size(0), -1), y_cycle], dim=1)
        
        return cloud[:, :, ::2], y_u, y_l, w_u, w_l

# ==========================================
# 2. SYSTEM ARCHITECTURES
# ==========================================
class CVAE(nn.Module):
    def __init__(self, input_dim=17, cond_dim=3, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim+cond_dim,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128,64), nn.ReLU())
        self.fc_mu = nn.Linear(64, latent_dim); self.fc_logvar = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim+cond_dim,64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64,128), nn.ReLU(), nn.Linear(128,input_dim))
    def reparameterize(self, mu, logvar): return mu + torch.randn_like(mu)*torch.exp(0.5*logvar)
    def forward(self, x, cond): 
        z = self.reparameterize(*self.encode(x, cond))
        return self.decoder(torch.cat([z, cond], dim=1))
    def encode(self, x, cond):
        h = self.encoder(torch.cat([x, cond], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)

class PointNetPhysicsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(2,64,1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64,128,1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128,256,1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256,1024,1), nn.BatchNorm1d(1024), nn.ReLU()
        )
        self.decoder = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 3))
    def forward(self, x):
        return self.decoder(torch.max(self.encoder(x), 2)[0])

def load_system():
    if not os.path.exists(CVAE_PATH): print(f"Missing {CVAE_PATH}"); sys.exit(1)
    stats = np.load(STATS_PATH)
    mean = torch.tensor(stats['mean'], dtype=torch.float32).to(DEVICE)
    std = torch.tensor(stats['std'], dtype=torch.float32).to(DEVICE)
    
    cvae = CVAE(latent_dim=16).to(DEVICE)
    cvae.load_state_dict(torch.load(CVAE_PATH, map_location=DEVICE))
    cvae.eval()
    for p in cvae.parameters(): p.requires_grad = False # Freeze Generator
    
    critic = PointNetPhysicsModel().to(DEVICE)
    critic.load_state_dict(torch.load(POINTNET_PATH, map_location=DEVICE), strict=False)
    critic.eval()
    for p in critic.parameters(): p.requires_grad = False # Freeze Critic
    
    geo = DifferentiableCST().to(DEVICE)
    return cvae, critic, geo, mean, std

# ==========================================
# 3. OPTIMIZATION LOGIC (The "Fixer")
# ==========================================
def optimize_airfoil(target_ld, target_cd, target_alpha):
    cvae, critic, geo, s_mean, s_std = load_system()
    
    target_cl = target_ld * target_cd
    print(f"--- OPTIMIZING: L/D={target_ld} (Cl={target_cl:.3f}), Cd={target_cd} ---")
    
    # 1. SETUP OPTIMIZATION
    # We optimize a batch of 100 latent vectors simultaneously
    batch_size = 100
    z = torch.randn(batch_size, 16, device=DEVICE, requires_grad=True)
    
    target_vec = torch.tensor([[target_cl, target_cd, target_alpha]], device=DEVICE)
    target_norm = (target_vec - s_mean) / s_std
    cond = target_norm.repeat(batch_size, 1)
    
    optimizer = torch.optim.Adam([z], lr=0.1)
    
    print("Step | Phys Loss | Geom Loss | Smooth Loss")
    
    # 2. GRADIENT DESCENT LOOP
    for i in range(200):
        optimizer.zero_grad()
        
        # A. Decode
        z_cond = torch.cat([z, cond], dim=1)
        weights = cvae.decoder(z_cond)
        
        # B. Geometry
        coords, yu, yl, wu, wl = geo(weights)
        
        # C. Physics Prediction
        phys_pred_norm = critic(coords)
        phys_pred = phys_pred_norm * s_std + s_mean
        
        # --- LOSS FUNCTIONS ---
        
        # 1. Physics Loss (Target Match)
        err = torch.abs(phys_pred - target_vec)
        loss_phys = torch.mean(err[:, 0] + 50.0 * err[:, 1]) # Heavy Drag Penalty
        
        # 2. Geometry Validity Loss (No Crossing)
        thickness = yu - yl
        # Penalize negative thickness heavily
        loss_cross = torch.mean(F.relu(-thickness)) * 1000.0
        
        # 3. Thickness Constraints (Target 12% - 15%)
        max_t = torch.max(thickness, dim=1)[0]
        # Penalize if outside [0.10, 0.16]
        loss_thick_low = torch.mean(F.relu(0.10 - max_t)) * 100.0
        loss_thick_high = torch.mean(F.relu(max_t - 0.16)) * 100.0
        loss_thick = loss_thick_low + loss_thick_high
        
        # 4. Smoothness Loss (Anti-Waviness)
        # Minimize second derivative of weights
        d2_wu = torch.mean(torch.abs(wu[:, 2:] - 2*wu[:, 1:-1] + wu[:, :-2]))
        d2_wl = torch.mean(torch.abs(wl[:, 2:] - 2*wl[:, 1:-1] + wl[:, :-2]))
        loss_smooth = (d2_wu + d2_wl) * 10.0
        
        # TOTAL LOSS
        loss = loss_phys + loss_cross + loss_thick + loss_smooth
        
        loss.backward()
        optimizer.step()
        
        if i % 50 == 0:
            print(f"{i:4d} | {loss_phys.item():.4f}    | {loss_cross.item() + loss_thick.item():.4f}    | {loss_smooth.item():.4f}")

    # 3. SELECT BEST RESULT
    with torch.no_grad():
        # Re-evaluate final state
        z_cond = torch.cat([z, cond], dim=1)
        weights = cvae.decoder(z_cond)
        coords, yu, yl, wu, wl = geo(weights)
        phys_pred = critic(coords) * s_std + s_mean
        
        # Final Ranking
        # Strict Boolean Filters
        t = yu - yl
        valid_cross = torch.min(t, dim=1)[0] >= -1e-5
        valid_thick = (torch.max(t, dim=1)[0] > 0.08)
        
        valid_mask = valid_cross & valid_thick
        valid_idx = torch.where(valid_mask)[0]
        
        if len(valid_idx) == 0:
            print("Warning: Optimization struggled. Returning best available structural match.")
            valid_idx = torch.where(valid_cross)[0]
            
        # Pick best physics match from valid ones
        if len(valid_idx) > 0:
            phys_err = torch.abs(phys_pred[valid_idx] - target_vec)
            score = phys_err[:, 0] + 50.0 * phys_err[:, 1]
            best_local = torch.argmin(score)
            best_global = valid_idx[best_local]
        else:
            best_global = 0 # Fallback
            
        final_coords = coords[best_global].cpu().numpy()
        final_phys = phys_pred[best_global].cpu().numpy()
        final_t = torch.max(t[best_global]).item()

    # --- OUTPUT ---
    print(f"\n=== OPTIMIZED AIRFOIL ===")
    print(f"L/D: {final_phys[0]/final_phys[1]:.1f}")
    print(f"Cd:  {final_phys[1]:.5f}")
    print(f"Thickness: {final_t:.1%} (Constraint: 10-16%)")
    
    np.savetxt("optimized_airfoil.dat", final_coords.T, header="Optimized_CST", fmt="%.6f")
    
    # Plot
    x = final_coords[0]
    y = final_coords[1]
    
    
    plt.figure(figsize=(12, 5))
    plt.plot(x, y, 'k-', lw=2, label='Surface')
    plt.fill_between(x, y, color='dodgerblue', alpha=0.1)
    plt.axis('equal')
    plt.grid(True, linestyle='--')
    plt.title(f"Optimized Realistic Airfoil\nTarget L/D: {target_ld} | Result L/D: {final_phys[0]/final_phys[1]:.1f}")
    plt.show()

if __name__ == "__main__":
    optimize_airfoil(target_ld=85.0, target_cd=0.007, target_alpha=2.0)
