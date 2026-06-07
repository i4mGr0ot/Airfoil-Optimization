import pandas as pd
import numpy as np
import h5py
import subprocess
import multiprocessing
import os
import sys
import time
import random
import shutil
import tempfile
from scipy.special import comb
from scipy.stats import linregress
import traceback

# ==========================================
#           INDUSTRIAL CONFIGURATION
# ==========================================
CONFIG = {
    "INPUT_CSV": "UIUC_CST_Database.csv",
    "OUTPUT_H5": "industrial_airfoil_dataset.h5", # This is the file PointNet needs
    "N_CORES": 6,
    "TARGET_SAMPLES": 7500,
    
    # --- PHYSICS ---
    "REYNOLDS": 600000,
    "MACH": 0.0,
    "ALPHA_SEQ": (-4.0, 18.0, 0.5), 
    "ITERATIONS": 300,
    "XFOIL_TIMEOUT": 45.0, # Increased slightly for stability

    # --- TARGETS ---
    "CL_CRUISE_TARGET": 0.5,
    "CL_MAX_MIN_REQ": 1.3,
    "BUCKET_MARGIN": 0.002,

    # --- GEOMETRIC GATES ---
    "THICKNESS_MIN": 0.05,  
    "THICKNESS_MAX": 0.18,  
    "DZ_TE_MAX": 0.015,     
    "MAX_INFLECTIONS": 1,   

    # --- AERODYNAMIC GATES ---
    "CL_MAX_LIMIT": 2.4,    
    "CD_MIN_LIMIT": 0.002,  
    "MIN_FITNESS": 30.0,    
}

def find_xfoil():
    if os.path.exists("xfoil.exe"): return os.path.abspath("xfoil.exe")
    path = shutil.which("xfoil.exe")
    if path: return path
    common_paths = [r"C:\XFOIL\xfoil.exe", r"C:\Program Files\XFOIL\xfoil.exe"]
    for p in common_paths:
        if os.path.exists(p): return p
    return None

XFOIL_PATH = find_xfoil()

class RobustCST:
    def __init__(self, n_params=8, resolution=200):
        self.n_params = n_params; self.resolution = resolution
        self.beta = np.linspace(0, np.pi, self.resolution)
        self.x = 0.5 * (1 - np.cos(self.beta))
        self.B = np.zeros((self.resolution, self.n_params))
        self.C = np.sqrt(self.x) * (1 - self.x)
        n = self.n_params - 1
        for k in range(self.n_params):
            self.B[:, k] = comb(n, k) * (self.x**k) * ((1 - self.x)**(n - k))

    def check_curvature(self, y_coords):
        curv = np.gradient(np.gradient(y_coords))
        return np.count_nonzero(np.diff(np.sign(curv))) <= CONFIG["MAX_INFLECTIONS"]

    def generate(self, w_u, w_l, dz_te):
        w_l = np.array(w_l, dtype=np.float64); w_u = np.array(w_u, dtype=np.float64)
        w_l[0] = w_u[0] 

        S_u = self.B @ w_u; S_l = self.B @ w_l
        y_u = self.C * S_u + self.x * (dz_te / 2.0)
        y_l = -self.C * S_l - self.x * (dz_te / 2.0)

        # 1. Kutta Condition
        if (-w_u[-1] + dz_te/2) > 0.05 or (w_l[-1] - dz_te/2) < -0.05:
            return None, None, "INVALID: Diverging TE"

        # 2. Intersection
        if np.any((y_u - y_l) < -1e-6): return None, None, "INVALID: Intersection"
        
        # 3. Thickness
        t_max = np.max(y_u - y_l)
        if t_max < CONFIG["THICKNESS_MIN"]: return None, None, "INVALID: Too Thin"
        if t_max > CONFIG["THICKNESS_MAX"]: return None, None, "INVALID: Too Thick"

        # 4. Smoothness
        if not self.check_curvature(y_u) or not self.check_curvature(y_l):
            return None, None, "INVALID: Wavy"

        x_coords = np.concatenate((self.x[::-1], self.x[1:]))
        y_coords = np.concatenate((y_u[::-1], y_l[1:]))
        return x_coords, y_coords, "VALID"

def analyze_polar(df):
    """Calculates Industrial Metrics correctly."""
    stats = {'ld_cruise': 0, 'cl_max': 0, 'bucket': 0, 'linearity': 0, 'valid': False}
    
    try:
        # 1. Cruise Efficiency
        df_sort = df.sort_values('cl')
        # Check targets are within data range
        if CONFIG['CL_CRUISE_TARGET'] > df['cl'].max() or CONFIG['CL_CRUISE_TARGET'] < df['cl'].min():
            return 0.0, stats
            
        cd_cruise = np.interp(CONFIG['CL_CRUISE_TARGET'], df_sort['cl'], df_sort['cd'])
        stats['ld_cruise'] = CONFIG['CL_CRUISE_TARGET'] / (cd_cruise + 1e-9)

        # 2. Bucket Width
        min_cd = df['cd'].min()
        in_bucket = df[df['cd'] <= (min_cd + CONFIG['BUCKET_MARGIN'])]
        if not in_bucket.empty:
            stats['bucket'] = in_bucket['cl'].max() - in_bucket['cl'].min()

        # 3. Max Lift
        stats['cl_max'] = df['cl'].max()
        
        # 4. Linearity
        linear_region = df[(df['alpha'] >= -2) & (df['alpha'] <= 6)]
        if len(linear_region) > 4:
            slope, _, r_val, _, _ = linregress(linear_region['alpha'], linear_region['cl'])
            stats['linearity'] = r_val ** 2
            if slope < 0.09: stats['linearity'] *= 0.5 

        # Fitness Calculation
        lift_penalty = 0.1 if stats['cl_max'] < CONFIG['CL_MAX_MIN_REQ'] else 1.0
        fitness = stats['ld_cruise'] * (1.0 + stats['bucket']) * stats['linearity'] * lift_penalty
        
        if fitness > CONFIG['MIN_FITNESS']: stats['valid'] = True
        return fitness, stats
    except:
        return 0.0, stats

def worker_task(task_data):
    if XFOIL_PATH is None: return None
    job_id, w_u, w_l, dz = task_data
    
    geo = RobustCST()
    x, y, status = geo.generate(w_u, w_l, dz)
    if status != "VALID": return None

    with tempfile.TemporaryDirectory() as tmp_dir:
        f_dat = os.path.join(tmp_dir, "af.dat"); f_log = os.path.join(tmp_dir, "polar.log")
        f_cp = os.path.join(tmp_dir, "cp.txt")

        # Write Geometry
        with open(f_dat, 'w') as f:
            f.write(f"AF_{job_id}\n")
            for i in range(len(x)): f.write(f" {x[i]:.6f}  {y[i]:.6f}\n")

        # Run Sweep (With Timeout Protection)
        cmds = f"load {f_dat}\n ppar\n N 200\n pane\n \n \n oper\n v {CONFIG['REYNOLDS']}\n mach {CONFIG['MACH']}\n iter {CONFIG['ITERATIONS']}\n pacc\n {f_log}\n \n aseq {CONFIG['ALPHA_SEQ'][0]} {CONFIG['ALPHA_SEQ'][1]} {CONFIG['ALPHA_SEQ'][2]}\n pacc\n quit\n"
        
        try:
            subprocess.run(XFOIL_PATH, input=cmds, text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=tmp_dir, timeout=CONFIG['XFOIL_TIMEOUT'])
        except subprocess.TimeoutExpired:
            return None # Skip hung processes

        # Parse Polar
        polar_data = []
        if os.path.exists(f_log):
            with open(f_log, 'r') as f:
                for line in f:
                    v = line.split()
                    if len(v) == 7 and not any(c.isalpha() for c in v[0]) and "#" not in line:
                        try:
                            a, cl, cd = float(v[0]), float(v[1]), float(v[2])
                            if cd > CONFIG['CD_MIN_LIMIT'] and abs(cl) < CONFIG['CL_MAX_LIMIT']:
                                polar_data.append({'alpha': a, 'cl': cl, 'cd': cd})
                        except: continue

        if len(polar_data) > 5:
            fitness, s = analyze_polar(pd.DataFrame(polar_data))
            
            if s['valid']:
                # Run Cp at Cruise Condition
                cmds_cp = f"load {f_dat}\n ppar\n N 200\n pane\n \n \n oper\n v {CONFIG['REYNOLDS']}\n mach {CONFIG['MACH']}\n iter {CONFIG['ITERATIONS']}\n cl {CONFIG['CL_CRUISE_TARGET']}\n cpwr {f_cp}\n quit\n"
                try:
                    subprocess.run(XFOIL_PATH, input=cmds_cp, text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=tmp_dir, timeout=10)
                except: pass
                
                cp_fixed = np.zeros(200)
                if os.path.exists(f_cp):
                    try:
                        raw = np.loadtxt(f_cp, skiprows=3)
                        if raw.ndim == 2 and len(raw) > 10:
                            cp_fixed = np.interp(np.linspace(0, 1, 200), raw[:,0], raw[:,2])
                    except: pass

                # --- FIX: RETURN LIST OF SCALARS ---
                return {
                    'w_u': w_u, 'w_l': w_l, 'dz': dz,
                    'scalars': [s['ld_cruise'], s['cl_max'], s['bucket'], s['linearity']],
                    'cp': cp_fixed, 'fitness': fitness
                }
    return None

class GeneticEngine:
    def __init__(self):
        self.population = []; self.TOURNAMENT_SIZE = 3

    def load_initial_seeds(self):
        print(f"--- Loading Seeds: {CONFIG['INPUT_CSV']} ---")
        if not os.path.exists(CONFIG['INPUT_CSV']): sys.exit("CSV Missing")
        df = pd.read_csv(CONFIG['INPUT_CSV']); df.columns = df.columns.str.strip()
        w_u_cols = [f'wu_{i}' for i in range(8)]; w_l_cols = [f'wl_{i}' for i in range(8)]
        for _, row in df.iterrows():
            self.population.append({
                'w_u': row[w_u_cols].values.astype(float),
                'w_l': row[w_l_cols].values.astype(float),
                'dz': float(row['TE_Thickness_dz']), 'fitness': 0.0
            })
        print(f"Loaded {len(self.population)} seeds")

    def save_batch(self, h5f, batch, datasets):
        if not batch: return
        n = len(batch)
        w = np.array([np.concatenate([r['w_u'], r['w_l'], [r['dz']]]) for r in batch])
        
        # --- FIX: EXTRACT LIST CORRECTLY ---
        s = np.array([r['scalars'] for r in batch]) 
        c = np.array([r['cp'] for r in batch])
        
        idx = datasets['w'].shape[0]
        datasets['w'].resize(idx+n, axis=0); datasets['w'][idx:] = w
        datasets['s'].resize(idx+n, axis=0); datasets['s'][idx:] = s
        datasets['c'].resize(idx+n, axis=0); datasets['c'][idx:] = c
        h5f.flush()

    def tournament_select(self):
        k = min(self.TOURNAMENT_SIZE, len(self.population))
        contenders = random.sample(self.population, k)
        contenders.sort(key=lambda x: x['fitness'], reverse=True)
        return contenders[0]

    def enforce_geometric_constraints(self, u, l, dz):
        """Pre-correction to avoid XFOIL crashes"""
        for _ in range(3):
            geo = RobustCST()
            _, y, status = geo.generate(u, l, dz)
            if status == "VALID": return u, l, dz
            if "Thin" in status: u *= 1.05; l *= 1.05 
            elif "Thick" in status: u *= 0.95; l *= 0.95 
            else: return u, l, dz 
        return u, l, dz

    def run(self):
        if not XFOIL_PATH: sys.exit("XFOIL missing")
        self.load_initial_seeds()
        
        # Clean up old file to prevent Scalar Mismatch
        if os.path.exists(CONFIG['OUTPUT_H5']):
            try: os.remove(CONFIG['OUTPUT_H5'])
            except: pass

        with h5py.File(CONFIG['OUTPUT_H5'], 'w') as h5f:
            datasets = {
                'w': h5f.create_dataset("weights", (0, 17), maxshape=(None, 17)),
                's': h5f.create_dataset("scalars", (0, 4), maxshape=(None, 4)), # 4 Metrics
                'c': h5f.create_dataset("cp", (0, 200), maxshape=(None, 200))
            }
            
            pool = multiprocessing.Pool(CONFIG['N_CORES'])
            print("--- Benchmarking Seeds ---")
            results = pool.map(worker_task, [(i, p['w_u'], p['w_l'], p['dz']) for i, p in enumerate(self.population)])
            valid = [r for r in results if r is not None]
            
            if not valid: print("FATAL: No seeds valid. Check input CSV."); return
            self.population = valid
            self.save_batch(h5f, valid, datasets)
            
            total, gen = len(valid), 0
            print(f"Start Evolution. Target: {CONFIG['TARGET_SAMPLES']}")
            
            while total < CONFIG['TARGET_SAMPLES']:
                gen += 1; tasks = []
                for _ in range(CONFIG['N_CORES'] * 4):
                    if len(self.population) < 2: break
                    p1 = self.tournament_select(); p2 = self.tournament_select()
                    
                    # --- FIX: USE 'IS' TO AVOID CRASH ---
                    while p1 is p2: p2 = self.tournament_select() 
                    
                    alpha = random.random()
                    c_u = p1['w_u']*alpha + p2['w_u']*(1-alpha)
                    c_l = p1['w_l']*alpha + p2['w_l']*(1-alpha)
                    c_dz = (p1['dz'] + p2['dz']) / 2.0
                    
                    if random.random() < 0.35:
                        c_u += np.random.normal(0, 0.05, 8); c_l += np.random.normal(0, 0.05, 8)
                        c_dz += np.random.normal(0, 0.002); c_l[0] = c_u[0]
                    
                    c_dz = max(0.0, min(c_dz, CONFIG['DZ_TE_MAX']))
                    c_u, c_l, c_dz = self.enforce_geometric_constraints(c_u, c_l, c_dz)
                    
                    tasks.append((random.randint(0, 1e9), c_u, c_l, c_dz))
                
                # Use imap_unordered to prevent freezing if one worker hangs
                batch = []
                for res in pool.imap_unordered(worker_task, tasks):
                    if res: batch.append(res)
                
                if batch:
                    self.save_batch(h5f, batch, datasets)
                    total += len(batch)
                    combined = self.population + batch
                    combined.sort(key=lambda x: x['fitness'], reverse=True)
                    self.population = combined[:100] + random.sample(combined[100:], min(len(combined[100:]), 50))
                    
                    best = self.population[0]
                    s = best['scalars']
                    print(f"Gen {gen:03d} | Total: {total:04d} | Fit: {best['fitness']:.1f} [L/D: {s[0]:.1f}, Cl_max: {s[1]:.2f}, Bkt: {s[2]:.2f}]")

            pool.close(); pool.join()
            print("--- DONE ---")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    try: GeneticEngine().run()
    except: traceback.print_exc()
