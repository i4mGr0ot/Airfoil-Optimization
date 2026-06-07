import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import os
import sys
import shutil
import tempfile
import random
import subprocess
import matplotlib.pyplot as plt
from scipy.special import comb
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# --- CONFIGURATION ---
CONFIG = {
    "MODEL_PATH": "best_physics_pointnet.pth",
    "DATASET_PATH": "airfoil_dataset_re_sweep.h5", 
    
    # Target
    "TARGET_REYNOLDS": 1000000.0,
    "TARGET_CL": 0.85,            # Slightly lower target helps smoothness
    "MIN_THICKNESS": 0.11,        # 11% thickness (Structural limit)
    
    # Robustness Constraints (THE FIX)
    "MIN_LE_RADIUS": 0.015,       # Forces a Round Nose (Prevents sharp stall)
    "MAX_CURVATURE_ENERGY": 5.0,  # Prevents wavy surfaces
    
    "POPULATION_SIZE": 1000,      # Large population for better search
    "GENERATIONS": 150,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu"
}

# --- 1. NEURAL ARCHITECTURE (MUST MATCH SAVED MODEL) ---
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

class ReAwareAirfoilPointNet(nn.Module):
    def __init__(self):
        super(ReAwareAirfoilPointNet, self).__init__()
        # Matches 'High-Precision' reduced architecture [64,64,128]
        self.sa1 = PointNetSetAbstraction(npoint=100, radius=0.1, nsample=32, in_channel=0, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=32, radius=0.2, nsample=32, in_channel=128, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256, mlp=[256, 512, 1024], group_all=True)

        self.fp3 = PointNetFeaturePropagation(in_channel=1025+256, mlp=[256, 256]) 
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

# --- 2. ROBUST OPTIMIZER KERNEL ---
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
        
        # 1. Thickness Check
        thickness = y_u - y_l
        if np.any(thickness < -1e-6): return None, 0, 0 
        
        # 2. Leading Edge Radius Approximation (Simple)
        # Radius ~ proportional to sqrt(w_u[0])
        # We ensure w_u[0] (Class shape at LE) is large enough
        le_metric = w_u[0] + w_l[0] 

        # 3. Curvature Energy (Smoothness)
        # Minimize 2nd derivative energy
        k_u = np.gradient(np.gradient(y_u))
        k_l = np.gradient(np.gradient(y_l))
        energy = np.sum(k_u**2) + np.sum(k_l**2)

        x_coords = np.concatenate((self.x[::-1], self.x[1:]))
        y_coords = np.concatenate((y_u[::-1], y_l[1:]))
        points = np.column_stack((x_coords, y_coords))
        
        if len(points) != 200:
             idx = np.linspace(0, len(points)-1, 200)
             px = np.interp(idx, np.arange(len(points)), points[:,0])
             py = np.interp(idx, np.arange(len(points)), points[:,1])
             points = np.column_stack((px, py))
             
        return points.astype(np.float32), le_metric, energy

class NeuralOptimizer:
    def __init__(self):
        self.kernel = CST_Kernel()
        
        print(f"--- Loading Model: {CONFIG['MODEL_PATH']} ---")
        self.model = ReAwareAirfoilPointNet().to(CONFIG['DEVICE'])
        try:
            self.model.load_state_dict(torch.load(CONFIG['MODEL_PATH'], map_location=CONFIG['DEVICE']))
        except Exception as e:
            print(f"Error loading model: {e}"); sys.exit(1)
        self.model.eval()
        
        print("--- Calibrating Scalers ---")
        if not os.path.exists(CONFIG['DATASET_PATH']):
            print("Dataset missing."); sys.exit(1)
            
        with h5py.File(CONFIG['DATASET_PATH'], 'r') as f:
            scalars = f['scalars'][:]
        valid_mask = np.isfinite(scalars).all(axis=1)
        scalars = scalars[valid_mask]
        
        self.target_scaler = StandardScaler()
        self.target_scaler.fit(scalars[:, 0:3]) 
        
        self.re_scaler = MinMaxScaler()
        self.re_scaler.fit(scalars[:, 4].reshape(-1, 1))
        
        re_val = np.array([[CONFIG['TARGET_REYNOLDS']]])
        self.re_norm = torch.tensor(self.re_scaler.transform(re_val), dtype=torch.float32).to(CONFIG['DEVICE'])

    def evaluate(self, population):
        valid_pop, point_clouds, smoothness_penalties = [], [], []
        
        for ind in population:
            # Generate shape and GET GEOMETRIC METRICS
            pc, le_metric, energy = self.kernel.compute(ind['w_u'], ind['w_l'], ind['dz'])
            
            if pc is not None:
                # CONSTRAINT 1: Thickness
                t_max = np.max(pc[:, 1]) - np.min(pc[:, 1])
                
                # CONSTRAINT 2: Round Nose (Prevents Sharp Stall)
                # w[0] roughly correlates to nose radius in CST
                # We enforce a minimum LE width parameter
                if t_max >= CONFIG['MIN_THICKNESS'] and le_metric > CONFIG['MIN_LE_RADIUS']:
                    
                    point_clouds.append(pc.T)
                    valid_pop.append(ind)
                    
                    # Calculate penalty for waviness
                    # If energy > max, apply linear penalty
                    pen = max(0, energy - CONFIG['MAX_CURVATURE_ENERGY']) * 0.1
                    smoothness_penalties.append(pen)
        
        if not point_clouds: return []

        batch_pc = torch.tensor(np.array(point_clouds), dtype=torch.float32).to(CONFIG['DEVICE'])
        batch_re = self.re_norm.repeat(len(batch_pc), 1)
        
        with torch.no_grad():
            scalars_pred, _ = self.model(batch_pc, batch_re)
            
        real_vals = self.target_scaler.inverse_transform(scalars_pred.cpu().numpy())
        
        for i, vals in enumerate(real_vals):
            cl, cd, cm = vals
            
            # --- ROBUST FITNESS FUNCTION ---
            # 1. Minimize Drag (Primary)
            # 2. Minimize Pitching Moment (Secondary - Improves Stability)
            # 3. Penalize Lift Miss (Constraint)
            # 4. Penalize Wavy Surfaces (Geometric Constraint)
            
            score_drag = - (cd * 100.0) 
            score_moment = - (abs(cm) * 10.0) # Penalty for instability
            score_lift = - (max(0, CONFIG['TARGET_CL'] - cl) * 50.0)
            score_smooth = - smoothness_penalties[i]
            
            fitness = score_drag + score_moment + score_lift + score_smooth
            
            valid_pop[i].update({'cl': float(cl), 'cd': float(cd), 'cm': float(cm), 'fitness': fitness})
            
        return valid_pop

    def run(self):
        # Init population with slightly larger LE weights for round noses
        pop = []
        for _ in range(CONFIG['POPULATION_SIZE']):
            w_u = np.random.uniform(-0.1, 0.4, 8)
            w_l = np.random.uniform(-0.2, 0.2, 8)
            w_u[0] = abs(w_u[0]) + 0.1 # Biased towards round nose
            w_l[0] = abs(w_l[0]) + 0.1
            pop.append({'w_u': w_u, 'w_l': w_l, 'dz': 0.005})
        
        print(f"\nROBUST OPTIMIZATION -> Re: {CONFIG['TARGET_REYNOLDS']:.0f} | Round Nose Enforced")
        
        best_overall = None
        for gen in range(CONFIG['GENERATIONS']):
            eval_pop = self.evaluate(pop)
            if not eval_pop: continue
            
            eval_pop.sort(key=lambda x: x['fitness'], reverse=True)
            if best_overall is None or eval_pop[0]['fitness'] > best_overall['fitness']:
                best_overall = eval_pop[0]
            
            if gen % 10 == 0:
                print(f"Gen {gen:03d} | CL:{best_overall['cl']:.2f} CD:{best_overall['cd']:.5f} CM:{best_overall['cm']:.3f}")
            
            next_gen = eval_pop[:50]
            while len(next_gen) < CONFIG['POPULATION_SIZE']:
                p1, p2 = random.choice(eval_pop[:100]), random.choice(eval_pop[:100])
                alpha = random.random()
                
                # Smoother Mutation
                child_wu = p1['w_u']*alpha + p2['w_u']*(1-alpha) + np.random.normal(0, 0.01, 8)
                child_wl = p1['w_l']*alpha + p2['w_l']*(1-alpha) + np.random.normal(0, 0.01, 8)
                
                # Keep Nose Round
                child_wu[0] = max(0.05, child_wu[0]) 
                child_wl[0] = max(0.05, child_wl[0])
                
                next_gen.append({'w_u': child_wu, 'w_l': child_wl, 'dz': (p1['dz'] + p2['dz'])/2.0})
            pop = next_gen
            
        return best_overall

# --- 3. VALIDATION ---
def perform_alpha_sweep(ind):
    xfoil = shutil.which("xfoil.exe") or "xfoil"
    if not xfoil: print("XFOIL not found."); return

    print("\n--- Running XFOIL Robustness Check (0-16 deg) ---")
    kernel = CST_Kernel()
    pts, _, _ = kernel.compute(ind['w_u'], ind['w_l'], ind['dz'])
    
    with open("robust_airfoil.dat", "w") as f:
        f.write("ROBUST_AF\n")
        for p in pts: f.write(f" {p[0]:.6f}  {p[1]:.6f}\n")
    
    polar_file = os.path.abspath("robust_polar.dat")
    if os.path.exists(polar_file): os.remove(polar_file)

    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy("robust_airfoil.dat", os.path.join(tmp, "run.dat"))
        cmds = (
            f"load run.dat\n"
            "ppar\n N 200\n pane\n \n \n"
            "oper\n"
            f"v {CONFIG['TARGET_REYNOLDS']}\n"
            "iter 150\n" # More iterations for convergence
            "pacc\n"
            f"{polar_file}\n \n"
            "aseq 0 16 0.5\n" # Finer resolution
            "pacc\n"
            "quit\n"
        )
        subprocess.run(xfoil, input=cmds, text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=tmp)

    if os.path.exists(polar_file):
        print(f"\n{'Alpha':<8} {'CL':<8} {'CD':<8} {'CM':<8}")
        print("-" * 45)
        cl_vals, cd_vals = [], []
        with open(polar_file, 'r') as f:
            for line in f:
                if any(c.isalpha() for c in line[:5]): continue
                vals = line.split()
                if len(vals) > 4:
                    try:
                        a, cl, cd, cm = float(vals[0]), float(vals[1]), float(vals[2]), float(vals[4])
                        print(f"{a:<8.1f} {cl:<8.4f} {cd:<8.5f} {cm:<8.4f}")
                        cl_vals.append(cl)
                        cd_vals.append(cd)
                    except: pass
        
        # Plot for User Visual Check
        plt.figure(figsize=(6,6))
        plt.plot(cd_vals, cl_vals, 'b-o')
        plt.xlabel('Cd')
        plt.ylabel('Cl')
        plt.title('Drag Polar (Check for Sharp Corners)')
        plt.grid(True)
        plt.savefig("robust_polar_plot.png")
        print("\n

[Image of Polar Curve]
 Saved to 'robust_polar_plot.png'. Check for smoothness.")

if __name__ == "__main__":
    opt = NeuralOptimizer()
    best = opt.run()
    if best: perform_alpha_sweep(best)
