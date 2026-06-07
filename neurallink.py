import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import os
import sys
import time
import random
import subprocess
import shutil
import tempfile
import matplotlib.pyplot as plt
from scipy.special import comb
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# --- CONFIGURATION ---
CONFIG = {
    # File Paths
    "MODEL_PATH": "best_physics_pointnet.pth",
    "DATASET_PATH": "airfoil_dataset_re_sweep.h5", # Needed to calibrate scalers
    
    # Optimization Target
    "TARGET_REYNOLDS": 1000000.0, # The speed you want to fly at
    "TARGET_CL": 0.8,             # The Lift you need
    "MIN_THICKNESS": 0.10,        # Structural limit (10%)
    
    # Optimization Settings
    "POPULATION_SIZE": 500,       # Massive population (Neural Net is fast!)
    "GENERATIONS": 100,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu"
}

# --- 1. ARCHITECTURE (MUST MATCH TRAINING EXACTLY) ---
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

# --- 2. HELPERS ---
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
        
        # Thickness check
        thickness = y_u - y_l
        if np.any(thickness < -1e-6): return None # Crossed
        
        x_coords = np.concatenate((self.x[::-1], self.x[1:]))
        y_coords = np.concatenate((y_u[::-1], y_l[1:]))
        points = np.column_stack((x_coords, y_coords))
        
        if len(points) != 200:
             # Basic interp
             idx = np.linspace(0, len(points)-1, 200)
             px = np.interp(idx, np.arange(len(points)), points[:,0])
             py = np.interp(idx, np.arange(len(points)), points[:,1])
             points = np.column_stack((px, py))
             
        return points.astype(np.float32)

def calibrate_scalers(h5_path):
    print("--- Calibrating Scalers from Dataset ---")
    with h5py.File(h5_path, 'r') as f:
        scalars = f['scalars'][:]
        
    valid_mask = np.isfinite(scalars).all(axis=1)
    scalars = scalars[valid_mask]
    
    # Target Scaler (CL, CD, CM)
    target_scaler = StandardScaler()
    target_scaler.fit(scalars[:, 0:3])
    
    # Reynolds Scaler
    re_scaler = MinMaxScaler()
    re_scaler.fit(scalars[:, 4].reshape(-1, 1))
    
    return target_scaler, re_scaler

# --- 3. OPTIMIZER CLASS ---
class NeuralOptimizer:
    def __init__(self):
        self.kernel = CST_Kernel()
        
        # Load Model
        self.model = ReAwareAirfoilPointNet().to(CONFIG['DEVICE'])
        try:
            self.model.load_state_dict(torch.load(CONFIG['MODEL_PATH'], map_location=CONFIG['DEVICE']))
            print(">>> Neural Network Loaded Successfully.")
        except:
            print("CRITICAL: Model file not found.")
            sys.exit(1)
        self.model.eval()
        
        # Load Scalers
        self.target_scaler, self.re_scaler = calibrate_scalers(CONFIG['DATASET_PATH'])
        
        # Prepare Target Re
        re_val = np.array([[CONFIG['TARGET_REYNOLDS']]])
        self.re_norm = torch.tensor(self.re_scaler.transform(re_val), dtype=torch.float32).to(CONFIG['DEVICE'])

    def evaluate_population(self, population):
        """
        Population: List of dicts {'w_u', 'w_l', 'dz'}
        Returns: List of dicts with 'cl', 'cd', 'fitness' added
        """
        valid_pop = []
        valid_indices = []
        point_clouds = []
        
        # 1. Generate Geometry (CPU)
        for i, ind in enumerate(population):
            pc = self.kernel.compute(ind['w_u'], ind['w_l'], ind['dz'])
            if pc is not None:
                # Thickness Check
                t_max = np.max(pc[:, 1]) - np.min(pc[:, 1]) # Rough approx
                if t_max >= CONFIG['MIN_THICKNESS']:
                    point_clouds.append(pc.T)
                    valid_pop.append(ind)
                    valid_indices.append(i)
        
        if not point_clouds: return population # All invalid

        # 2. Batch Inference (GPU)
        batch_pc = torch.tensor(np.array(point_clouds), dtype=torch.float32).to(CONFIG['DEVICE'])
        # Broadcast Re to batch size
        batch_re = self.re_norm.repeat(len(batch_pc), 1)
        
        with torch.no_grad():
            scalars_pred, _ = self.model(batch_pc, batch_re)
            
        # 3. Inverse Transform
        scalars_np = scalars_pred.cpu().numpy()
        real_vals = self.target_scaler.inverse_transform(scalars_np)
        
        # 4. Assign Fitness
        for i, vals in enumerate(real_vals):
            cl_pred, cd_pred, cm_pred = vals
            ind = valid_pop[i]
            
            ind['cl'] = float(cl_pred)
            ind['cd'] = float(cd_pred)
            
            # FITNESS FUNCTION
            # Goal: Minimize Drag, subject to CL >= Target
            
            cl_penalty = max(0, CONFIG['TARGET_CL'] - cl_pred) * 10.0 # Heavy penalty for low lift
            fitness = - (cd_pred * 100.0) - cl_penalty # Negative because we maximize fitness
            
            ind['fitness'] = fitness
            
        return valid_pop

    def run_evolution(self):
        # Init Random Population
        pop = []
        for _ in range(CONFIG['POPULATION_SIZE']):
            pop.append({
                'w_u': np.random.uniform(-0.1, 0.4, 8),
                'w_l': np.random.uniform(-0.2, 0.2, 8),
                'dz': np.random.uniform(0.002, 0.01)
            })
            
        print(f"\n--- Starting Neural Optimization (Target Re={CONFIG['TARGET_REYNOLDS']:.0f}, CL={CONFIG['TARGET_CL']}) ---")
        
        best_overall = None
        
        for gen in range(CONFIG['GENERATIONS']):
            # Eval
            evaluated_pop = self.evaluate_population(pop)
            if not evaluated_pop: continue
            
            # Sort
            evaluated_pop.sort(key=lambda x: x['fitness'], reverse=True)
            best = evaluated_pop[0]
            
            if best_overall is None or best['fitness'] > best_overall['fitness']:
                best_overall = best
            
            if gen % 10 == 0:
                print(f"Gen {gen:03d} | Best CL: {best['cl']:.3f} | Best CD: {best['cd']:.5f} | Fit: {best['fitness']:.3f}")
            
            # Selection & Crossover
            next_gen = evaluated_pop[:50] # Elitism
            
            while len(next_gen) < CONFIG['POPULATION_SIZE']:
                p1 = random.choice(evaluated_pop[:100])
                p2 = random.choice(evaluated_pop[:100])
                
                # Crossover
                alpha = random.random()
                c_u = p1['w_u']*alpha + p2['w_u']*(1-alpha)
                c_l = p1['w_l']*alpha + p2['w_l']*(1-alpha)
                c_dz = (p1['dz'] + p2['dz']) / 2.0
                
                # Mutation
                if random.random() < 0.1:
                    c_u += np.random.normal(0, 0.02, 8)
                    c_l += np.random.normal(0, 0.02, 8)
                
                next_gen.append({'w_u': c_u, 'w_l': c_l, 'dz': c_dz})
            
            pop = next_gen
            
        print("\n>>> Optimization Complete. Verifying with XFOIL...")
        return best_overall

# --- 4. VERIFICATION ---
def verify_xfoil(ind):
    xfoil_path = shutil.which("xfoil.exe") or "xfoil"
    if not xfoil_path: 
        print("XFOIL not found. Skipping verification.")
        return

    kernel = CST_Kernel()
    pts = kernel.compute(ind['w_u'], ind['w_l'], ind['dz'])
    
    with tempfile.TemporaryDirectory() as tmp:
        # Save Geometry
        with open(f"{tmp}/test.dat", 'w') as f:
            f.write("OPTIMIZED_AF\n")
            for p in pts: f.write(f" {p[0]:.6f}  {p[1]:.6f}\n")
            
        # Run XFOIL
        cmds = (
            f"load {tmp}/test.dat\n"
            "ppar\n N 160\n pane\n \n \n"
            "oper\n"
            f"v {CONFIG['TARGET_REYNOLDS']}\n"
            "iter 100\n"
            "pacc\n"
            f"{tmp}/log.txt\n \n"
            "alfa 0\n" # Just check 0 alpha for now, typically optimizer finds optimum alpha
            "cl 0.8\n" # Solve for Target CL directly
            "quit\n"
        )
        
        subprocess.run(xfoil_path, input=cmds, text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=tmp)
        
        if os.path.exists(f"{tmp}/log.txt"):
            with open(f"{tmp}/log.txt") as f:
                lines = f.readlines()
                for line in lines:
                    if "OPTIMIZED" not in line and len(line.split()) > 5:
                        vals = line.split()
                        try:
                            # Alpha  CL  CD ...
                            print(f"\n--- XFOIL VERIFICATION ---")
                            print(f"Predicted CD: {ind['cd']:.5f}")
                            print(f"Actual    CD: {float(vals[2]):.5f}")
                            print(f"Actual Alpha: {float(vals[0]):.2f}")
                            
                            # Plot
                            plt.figure(figsize=(10,3))
                            plt.plot(pts[:,0], pts[:,1], 'k-')
                            plt.axis('equal')
                            plt.title(f"AI Optimized Airfoil (Re={CONFIG['TARGET_REYNOLDS']:.0e})")
                            plt.grid(True)
                            plt.savefig("optimized_airfoil.png")
                            print("Saved geometry to 'optimized_airfoil.png'")
                            return
                        except: pass
        print("XFOIL failed to converge on the result (Common for high-performance shapes).")

if __name__ == "__main__":
    opt = NeuralOptimizer()
    best_design = opt.run_evolution()
    
    if best_design:
        verify_xfoil(best_design)
