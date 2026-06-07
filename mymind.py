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
        
        # Enforce Closed TE
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
# 2. SYSTEM LOADING
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
    cvae.eval(); 
    for p in cvae.parameters(): p.requires_grad = False 
    
    critic = PointNetPhysicsModel().to(DEVICE)
    critic.load_state_dict(torch.load(POINTNET_PATH, map_location=DEVICE), strict=False)
    critic.eval(); 
    for p in critic.parameters(): p.requires_grad = False 
    
    geo = DifferentiableCST().to(DEVICE)
    return cvae, critic, geo, mean, std

# ==========================================
# 3. STRICT TARGET OPTIMIZATION
# ==========================================
def optimize_to_target(target_ld, target_cd, target_alpha):
    cvae, critic, geo, s_mean, s_std = load_system()
    
    # Calculate Target Cl
    target_cl = target_ld * target_cd
    
    print(f"--- OPTIMIZING TO MATCH: L/D={target_ld} (Cl={target_cl:.3f}), Cd={target_cd} ---")
    
    # 1. INITIALIZATION
    batch_size = 50
    # Start with standard normal noise (diverse starting shapes)
    z = torch.randn(batch_size, 16, device=DEVICE, requires_grad=True)
    
    # Target Vector
    target_vec = torch.tensor([[target_cl, target_cd, target_alpha]], device=DEVICE)
    target_norm = (target_vec - s_mean) / s_std
    cond = target_norm.repeat(batch_size, 1)
    
    # Optimizer (Slower learning rate for stability)
    optimizer = torch.optim.Adam([z], lr=0.02)
    
    print(f"{'Step':^5} | {'L/D Error':^10} | {'Shape Loss':^10} | {'Smoothness':^10} | {'Best L/D':^10}")
    print("-" * 55)

    # 2. OPTIMIZATION LOOP
    for i in range(300):
        optimizer.zero_grad()
        
        # Decode
        z_cond = torch.cat([z, cond], dim=1)
        weights = cvae.decoder(z_cond)
        
        # Geometry
        coords, yu, yl, wu, wl = geo(weights)
        
        # Physics Prediction
        phys_pred_norm = critic(coords)
        phys_pred = phys_pred_norm * s_std + s_mean
        
        # --- STRICT LOSS FUNCTION ---
        
        # A. TARGET MATCHING (Not Maximization)
        # We want (Pred - Target)^2. 
        # If Pred L/D is 175 and Target is 85, this error will be huge.
        current_ld = phys_pred[:, 0] / (phys_pred[:, 1] + 1e-6)
        ld_error = torch.mean((current_ld - target_ld)**2)
        
        # Also enforce Cd matching strictly
        cd_error = torch.mean((phys_pred[:, 1] - target_cd)**2) * 10000.0
        
        loss_phys = ld_error + cd_error
        
        # B. REALITY CHECK (Ghost Physics Penalty)
        # If AI predicts Cd < 0.004 (Impossible at Re=600k), punish it.
        # This prevents it from finding "magic" adversarial shapes.
        loss_reality = torch.mean(F.relu(0.005 - phys_pred[:, 1])) * 50000.0
        
        # C. GEOMETRY (No Crossing)
        thickness = yu - yl
        loss_cross = torch.mean(F.relu(-thickness)) * 10000.0
        
        # D. THICKNESS CONSTRAINT (Target 12%)
        max_t = torch.max(thickness, dim=1)[0]
        loss_thick = torch.mean((max_t - 0.12)**2) * 500.0
        
        # E. SMOOTHNESS (Massive Boost)
        # Penalize 2nd derivative of weights heavily
        d2_wu = torch.mean(torch.abs(wu[:, 2:] - 2*wu[:, 1:-1] + wu[:, :-2]))
        d2_wl = torch.mean(torch.abs(wl[:, 2:] - 2*wl[:, 1:-1] + wl[:, :-2]))
        loss_smooth = (d2_wu + d2_wl) * 200.0 # Increased from 10.0 to 200.0
        
        # Total
        loss = loss_phys + loss_reality + loss_cross + loss_thick + loss_smooth
        
        loss.backward()
        optimizer.step()
        
        # Clamp latent z to stay within distribution (avoid outliers)
        with torch.no_grad():
            z.clamp_(-3.0, 3.0)
            
        if i % 50 == 0:
            best_curr_ld = torch.max(current_ld).item()
            print(f"{i:5d} | {loss_phys.item():10.4f} | {loss_cross.item()+loss_thick.item():10.4f} | {loss_smooth.item():10.4f} | {best_curr_ld:10.1f}")

    # 3. SELECT BEST MATCH
    with torch.no_grad():
        z_cond = torch.cat([z, cond], dim=1)
        weights = cvae.decoder(z_cond)
        coords, yu, yl, wu, wl = geo(weights)
        phys_pred = critic(coords) * s_std + s_mean
        
        # Calculate deviation from target
        curr_ld = phys_pred[:, 0] / phys_pred[:, 1]
        dist_to_target = torch.abs(curr_ld - target_ld)
        
        # Filter for validity
        t = yu - yl
        valid = (torch.min(t, dim=1)[0] >= -1e-5) & (torch.max(t, dim=1)[0] > 0.09)
        
        valid_idx = torch.where(valid)[0]
        
        if len(valid_idx) > 0:
            # Pick the one closest to Target L/D (not the highest!)
            best_local = torch.argmin(dist_to_target[valid_idx])
            best_idx = valid_idx[best_local]
        else:
            print("Warning: Optimization compromised. Picking best structural match.")
            best_idx = torch.argmax(torch.max(t, dim=1)[0]) # Safest thick wing

        final_phys = phys_pred[best_idx].cpu().numpy()
        final_coords = coords[best_idx].cpu().numpy()
        final_thick = torch.max(t[best_idx]).item()

    # --- OUTPUT ---
    print(f"\n=== TARGET MATCHED ===")
    print(f"Goal L/D: {target_ld}  |  Achieved L/D: {final_phys[0]/final_phys[1]:.1f}")
    print(f"Goal Cd:  {target_cd}  |  Achieved Cd:  {final_phys[1]:.5f}")
    print(f"Thickness: {final_thick:.1%} (Constraint: ~12%)")
    
    # Save & Plot
    np.savetxt("target_airfoil.dat", final_coords.T, header="Strict_Target_Opt", fmt="%.6f")
    
    x = final_coords[0]
    y = final_coords[1]
    plt.figure(figsize=(12, 5))
    plt.plot(x, y, 'k-', lw=2, label='Surface')
    plt.fill_between(x, y, color='dodgerblue', alpha=0.1)
    plt.axis('equal'); plt.grid(True, linestyle='--')
    plt.title(f"Optimized to Target\nReq L/D: {target_ld} | Result L/D: {final_phys[0]/final_phys[1]:.1f}")
    plt.show()

if __name__ == "__main__":
    optimize_to_target(target_ld=85.0, target_cd=0.007, target_alpha=2.0)
