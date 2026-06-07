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
import atexit
from scipy.special import comb
from scipy.stats import linregress
import traceback

CONFIG = {
    "INPUT_CSV": "UIUC_CST_Database.csv",
    "OUTPUT_H5": "ultimate_mission_dataset.h5",
    "N_CORES": 6,
    "TARGET_SAMPLES": 7500,
    
    "RE_TAKEOFF": 600000,    
    "RE_CRUISE": 1500000,    
    "MACH": 0.1,
    
    "ALPHA_SEQ": (-4.0, 19.0, 0.5), 
    "ITERATIONS": 300,             
    "XFOIL_TIMEOUT": 60.0,      
    
    "CL_CRUISE_TARGET": 0.45,  
    "CL_MAX_MIN_REQ": 1.4,     
    
    "BUCKET_MARGIN": 0.002, 

    "THICKNESS_MIN": 0.09,      
    "THICKNESS_MAX": 0.18,  
    "DZ_TE_MAX": 0.015,     
    "MAX_INFLECTIONS": 1,      
    
    "CD_MIN_LIMIT": 0.003,      
    "CL_MAX_LIMIT": 2.4,    
}


def find_xfoil():
    if os.path.exists("xfoil.exe"): return os.path.abspath("xfoil.exe")
    path_exec = shutil.which("xfoil.exe")
    if path_exec: return path_exec
    common_paths = [r"C:\XFOIL\xfoil.exe", r"C:\Program Files\XFOIL\xfoil.exe", "/usr/bin/xfoil"]
    for p in common_paths:
        if os.path.exists(p): return p
    return None

XFOIL_PATH = find_xfoil()

class RobustCST:
    def __init__(self, n_params=8, resolution=240): 
        self.n_params = n_params
        self.resolution = resolution
        self.beta = np.linspace(0, np.pi, self.resolution)
        self.x = 0.5 * (1 - np.cos(self.beta))
        self.B = np.zeros((self.resolution, self.n_params))
        self.C = np.sqrt(self.x) * (1 - self.x)
        n = self.n_params - 1
        for k in range(self.n_params):
            self.B[:, k] = comb(n, k) * (self.x**k) * ((1 - self.x)**(n - k))

    def check_curvature(self, y_coords):
        curv = np.gradient(np.gradient(y_coords))
        sign_changes = np.diff(np.sign(curv))
        inflections = np.count_nonzero(sign_changes)
        return inflections <= CONFIG["MAX_INFLECTIONS"]

    def generate(self, w_u, w_l, dz_te):
        w_l = np.array(w_l, dtype=np.float64)
        w_u = np.array(w_u, dtype=np.float64)
        
        w_l[0] = w_u[0] 

        S_u = self.B @ w_u
        S_l = self.B @ w_l
        y_u = self.C * S_u + self.x * (dz_te / 2.0)
        y_l = -self.C * S_l - self.x * (dz_te / 2.0)

        slope_u_TE = -w_u[-1] + (dz_te / 2.0)
        slope_l_TE = w_l[-1] - (dz_te / 2.0)
        if slope_u_TE > 0.05 or slope_l_TE < -0.05: return None, None, "INVALID: Diverging TE"

        if np.any((y_u - y_l) < -1e-6): return None, None, "INVALID: Self-Intersect"
        
        t_dist = y_u - y_l
        max_t = np.max(t_dist)
        if max_t < CONFIG["THICKNESS_MIN"]: return None, None, "INVALID: Too Thin"
        if max_t > CONFIG["THICKNESS_MAX"]: return None, None, "INVALID: Too Thick"

        if not self.check_curvature(y_u) or not self.check_curvature(y_l):
            return None, None, "INVALID: Wavy Surface"

        x_coords = np.concatenate((self.x[::-1], self.x[1:]))
        y_coords = np.concatenate((y_u[::-1], y_l[1:]))
        
        return x_coords, y_coords, "VALID"

def run_simulation(job_id, x, y, reynolds, tmp_dir):

    f_dat = os.path.join(tmp_dir, f"af_{job_id}.dat")
    f_log = os.path.join(tmp_dir, f"polar_Re{reynolds}.log")
    
    with open(f_dat, 'w') as f:
        f.write(f"AF_{job_id}\n")
        for i in range(len(x)):
            f.write(f" {x[i]:.6f}  {y[i]:.6f}\n")

    cmds = (
        f"load {f_dat}\n"
        "ppar\n"
        "N 200\n"
        "pane\n"
        "\n"
        "\n"
        "oper\n"
        f"v {reynolds}\n"
        f"mach {CONFIG['MACH']}\n"
        f"iter {CONFIG['ITERATIONS']}\n" 
        "pacc\n"
        f"{f_log}\n"
        "\n"
        f"aseq {CONFIG['ALPHA_SEQ'][0]} {CONFIG['ALPHA_SEQ'][1]} {CONFIG['ALPHA_SEQ'][2]}\n"
        "pacc\n"
        "quit\n"
    )

    try:
        subprocess.run(XFOIL_PATH, input=cmds, text=True, 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, 
                       timeout=CONFIG['XFOIL_TIMEOUT'], cwd=tmp_dir)
    except subprocess.TimeoutExpired:
        return pd.DataFrame() 

    polar_data = []
    if os.path.exists(f_log):
        with open(f_log, 'r') as f:
            for line in f:
                vals = line.split()
                if len(vals) == 7 and not any(c.isalpha() for c in vals[0]) and "#" not in line:
                    try:
                        a, cl, cd = float(vals[0]), float(vals[1]), float(vals[2])
                        if (cd > CONFIG['CD_MIN_LIMIT'] and 
                            abs(cl) < CONFIG['CL_MAX_LIMIT'] and 
                            cd < 1.0):
                            polar_data.append({'alpha': a, 'cl': cl, 'cd': cd})
                    except ValueError: continue
    
    return pd.DataFrame(polar_data)

def worker_task(task_data):
    if XFOIL_PATH is None: return None
    job_id, w_u, w_l, dz = task_data
    
    geo = RobustCST()
    x, y, status = geo.generate(w_u, w_l, dz)
    if status != "VALID": return None

    result = None
    with tempfile.TemporaryDirectory() as tmp_dir:
        

        df_cruise = run_simulation(job_id, x, y, CONFIG['RE_CRUISE'], tmp_dir)
        if len(df_cruise) < 5: return None 

        try:
            df_sort = df_cruise.sort_values('cl')
            
            cd_cruise = np.interp(CONFIG['CL_CRUISE_TARGET'], df_sort['cl'], df_sort['cd'])
            if CONFIG['CL_CRUISE_TARGET'] > df_sort['cl'].max(): return None
            ld_cruise = CONFIG['CL_CRUISE_TARGET'] / cd_cruise

            min_cd_global = df_sort['cd'].min()

            bucket_threshold = min_cd_global + CONFIG['BUCKET_MARGIN']

            in_bucket = df_sort[df_sort['cd'] <= bucket_threshold]
            
            bucket_width = 0.0
            if not in_bucket.empty:
                bucket_width = in_bucket['cl'].max() - in_bucket['cl'].min()
                
        except: return None


        df_takeoff = run_simulation(job_id, x, y, CONFIG['RE_TAKEOFF'], tmp_dir)
        if len(df_takeoff) < 5: return None 

        cl_max = df_takeoff['cl'].max()
        
        linear_region = df_takeoff[(df_takeoff['alpha'] >= -2) & (df_takeoff['alpha'] <= 6)]
        linearity_score = 0.0
        if len(linear_region) > 4:
            slope, intercept, r_value, p_value, std_err = linregress(linear_region['alpha'], linear_region['cl'])
            linearity_score = r_value ** 2 
            
            if slope < 0.085: linearity_score *= 0.5 


        
        takeoff_factor = 1.0
        if cl_max < CONFIG['CL_MAX_MIN_REQ']:
            takeoff_factor = 0.1 
        else:
            takeoff_factor = 1.0 + 0.5 * (cl_max - CONFIG['CL_MAX_MIN_REQ'])

     
        fitness = ld_cruise * (1.0 + bucket_width) * takeoff_factor * linearity_score

        if fitness > 25.0:

            f_cp = os.path.join(tmp_dir, "cp.txt")
            f_dat = os.path.join(tmp_dir, f"af_{job_id}.dat")
            cmds_cp = (
                f"load {f_dat}\n"
                "ppar\n" "N 200\n" "pane\n" "\n" "\n"
                "oper\n"
                f"v {CONFIG['RE_CRUISE']}\n"
                f"mach {CONFIG['MACH']}\n"
                f"iter {CONFIG['ITERATIONS']}\n"
                f"cl {CONFIG['CL_CRUISE_TARGET']}\n"
                f"cpwr {f_cp}\n"
                "quit\n"
            )
            subprocess.run(XFOIL_PATH, input=cmds_cp, text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=tmp_dir)
            
            cp_fixed = np.zeros(200)
            if os.path.exists(f_cp):
                try:
                    raw_cp = np.loadtxt(f_cp, skiprows=3)
                    if raw_cp.ndim == 2 and raw_cp.shape[0] > 10:
                        cp_fixed = np.interp(np.linspace(0, 1, 200), raw_cp[:,0], raw_cp[:,2])
                except: pass

            result = {
                'w_u': w_u, 'w_l': w_l, 'dz': dz,
                'cl_max': cl_max, 
                'ld_cruise': ld_cruise,
                'bucket_width': bucket_width,
                'linearity': linearity_score,
                'cp': cp_fixed,
                'fitness': fitness
            }
            
    return result


class GeneticEngine:
    def __init__(self):
        self.population = []
        self.TOURNAMENT_SIZE = 3 

    def load_initial_seeds(self):
        print(f"--- Loading Seeds: {CONFIG['INPUT_CSV']} ---")
        if not os.path.exists(CONFIG['INPUT_CSV']):
            print(f"CRITICAL: {CONFIG['INPUT_CSV']} not found.")
            sys.exit(1)
        try:
            df = pd.read_csv(CONFIG['INPUT_CSV'])
            df.columns = df.columns.str.strip()
            w_u_cols = [f'wu_{i}' for i in range(8)]
            w_l_cols = [f'wl_{i}' for i in range(8)]
            for _, row in df.iterrows():
                self.population.append({
                    'w_u': row[w_u_cols].values.astype(float),
                    'w_l': row[w_l_cols].values.astype(float),
                    'dz': float(row['TE_Thickness_dz']),
                    'fitness': 0.0 
                })
            print(f"Loaded {len(self.population)} seeds.")
        except Exception:
            traceback.print_exc(); sys.exit(1)

    def save_batch(self, h5f, batch, datasets):
        if not batch: return
        n = len(batch)
        w = np.array([np.concatenate([r['w_u'], r['w_l'], [r['dz']]]) for r in batch])

        s = np.array([[r['ld_cruise'], r['cl_max'], r['bucket_width']] for r in batch])
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

    def run(self):
        if not XFOIL_PATH:
            print("\nCRITICAL: XFOIL toh add karlo path mein."); sys.exit(1)
        
        self.load_initial_seeds()
        
        with h5py.File(CONFIG['OUTPUT_H5'], 'w') as h5f:
            datasets = {
                'w': h5f.create_dataset("weights", (0, 17), maxshape=(None, 17)),
                's': h5f.create_dataset("scalars", (0, 3), maxshape=(None, 3)), 
                'c': h5f.create_dataset("cp", (0, 200), maxshape=(None, 200))
            }
            
            pool = multiprocessing.Pool(CONFIG['N_CORES'])
            
            print("--- Loading new Pokemons ---")
            seed_tasks = [(i, p['w_u'], p['w_l'], p['dz']) for i, p in enumerate(self.population)]
            results = pool.map(worker_task, seed_tasks[:100]) 
            valid_seeds = [r for r in results if r is not None]
            
            if not valid_seeds:
                print("FATAL: No valid Pokemons"); return

            self.population = valid_seeds
            self.save_batch(h5f, valid_seeds, datasets)
            
            total_saved = len(valid_seeds)
            gen = 0
            print(f"\nStarting Evolution. Target: {CONFIG['TARGET_SAMPLES']}")
            print(f"Mission: Takeoff Re={CONFIG['RE_TAKEOFF']} | Cruise Re={CONFIG['RE_CRUISE']}")
            
            while total_saved < CONFIG['TARGET_SAMPLES']:
                gen += 1
                tasks = []
                for _ in range(CONFIG['N_CORES'] * 4):
                    if len(self.population) < 2: break
                    
                    p1 = self.tournament_select()
                    p2 = self.tournament_select()
                    attempts = 0
                    while p1 == p2 and attempts < 3:
                         p2 = self.tournament_select()
                         attempts += 1
                    
                    alpha = random.random()
                    c_u = p1['w_u']*alpha + p2['w_u']*(1-alpha)
                    c_l = p1['w_l']*alpha + p2['w_l']*(1-alpha)
                    c_dz = (p1['dz'] + p2['dz']) / 2.0
                    
                    if random.random() < 0.35: 
                        c_u += np.random.normal(0, 0.05, 8)
                        c_l += np.random.normal(0, 0.05, 8)
                        c_dz += np.random.normal(0, 0.002)
                        c_l[0] = c_u[0] 
                    
                    c_dz = max(0.0, min(c_dz, CONFIG['DZ_TE_MAX']))
                    tasks.append((random.randint(0, 1e9), c_u, c_l, c_dz))
                
                batch_results = pool.map(worker_task, tasks)
                valid_batch = [r for r in batch_results if r is not None]
                
                if valid_batch:
                    self.save_batch(h5f, valid_batch, datasets)
                    total_saved += len(valid_batch)
                    
                    combined = self.population + valid_batch
                    combined.sort(key=lambda x: x['fitness'], reverse=True)
                    self.population = combined[:100] + random.sample(combined[100:], min(len(combined[100:]), 50))
                    
                    best = self.population[0]
                    print(f"Gen {gen:03d} | Total: {total_saved:04d} | "
                          f"Fitness: {best['fitness']:.1f} [L/D: {best['ld_cruise']:.1f}, Bucket: {best['bucket_width']:.2f}, Cl_max: {best['cl_max']:.2f}]")

            pool.close(); pool.join()
            print(f"\n--- SUCCESS. Saved to {CONFIG['OUTPUT_H5']} ---")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        GeneticEngine().run()
    except Exception:
        traceback.print_exc()
