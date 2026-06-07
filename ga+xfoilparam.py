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
from scipy.special import comb
import traceback

INPUT_CSV = "UIUC_CST_Database.csv"
OUTPUT_H5 = "airfoil_dataset_Re600k_mach0_alphasweep_with_limits.h5"

REYNOLDS = 600000         
MACH = 0.0                
ALPHA_START = 0.0        
ALPHA_END = 15.0         
ALPHA_STEP = 0.5         
ITERATIONS = 200         

TARGET_SAMPLES = 10000     
NUM_CORES = 6             
XFOIL_TIMEOUT = 40.0      

CD_MIN_LIMIT = 0.004     
CL_MAX_LIMIT = 1.8        
THICKNESS_MIN = 0.05     
THICKNESS_MAX = 0.19      
DZ_MAX_LIMIT = 0.02       

def find_xfoil():
    if os.path.exists("xfoil.exe"): return os.path.abspath("xfoil.exe")
    path_exec = shutil.which("xfoil.exe")
    if path_exec: return path_exec
    common_paths = [r"C:\XFOIL\xfoil.exe", r"C:\Program Files\XFOIL\xfoil.exe"]
    for p in common_paths:
        if os.path.exists(p): return p
    return None

XFOIL_PATH = find_xfoil()

class RobustCST:
    def __init__(self, n_params=8, resolution=250): 
        self.n_params = n_params
        self.resolution = resolution
        self.beta = np.linspace(0, np.pi, self.resolution)
        self.x = 0.5 * (1 - np.cos(self.beta))
        self.C = np.sqrt(self.x) * (1 - self.x)
        self.B = np.zeros((self.resolution, self.n_params))
        n = self.n_params - 1
        for k in range(self.n_params):
            self.B[:, k] = comb(n, k) * (self.x**k) * ((1 - self.x)**(n - k))

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
        if slope_u_TE > 0.02 or slope_l_TE < -0.02:
             return None, None, "INVALID: Diverging TE"

        if np.any((y_u - y_l) < -1e-6): 
            return None, None, "INVALID: Criss Cross Apple Sauce"
        
        thickness_dist = y_u - y_l
        max_thickness = np.max(thickness_dist)
        
        if max_thickness < THICKNESS_MIN:
            return None, None, "INVALID: Patlu"
        if max_thickness > THICKNESS_MAX:
            return None, None, "INVALID: Motu"
        
        x_coords = np.concatenate((self.x[::-1], self.x[1:]))
        y_coords = np.concatenate((y_u[::-1], y_l[1:]))
        
        return x_coords, y_coords, "VALID"

def worker_task(task_data):
    if XFOIL_PATH is None: return None
    job_id, w_u, w_l, dz = task_data
    
    geo = RobustCST()
    x, y, status = geo.generate(w_u, w_l, dz)
    if status != "VALID": return None

    pid = os.getpid()
    f_root = f"proc_{pid}_{job_id}"
    f_dat = f"{f_root}.dat"
    f_log = f"{f_root}.log"
    f_cp  = f"{f_root}_cp.txt"

    result = None

    try:
        with open(f_dat, 'w') as f:
            f.write(f"AF_{job_id}\n")
            for i in range(len(x)):
                f.write(f" {x[i]:.6f}  {y[i]:.6f}\n")

        cmds_sweep = (
            f"load {f_dat}\n"
            "ppar\n"
            "N 250\n"
            "\n"
            "\n"
            "oper\n"
            f"v {REYNOLDS}\n"
            f"mach {MACH}\n"
            f"iter {ITERATIONS}\n" 
            "pacc\n"
            f"{f_log}\n"
            "\n"
            f"aseq {ALPHA_START} {ALPHA_END} {ALPHA_STEP}\n"
            "pacc\n"
            "quit\n"
        )

        subprocess.run(XFOIL_PATH, input=cmds_sweep, text=True, 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, 
                       timeout=XFOIL_TIMEOUT)

        best_alpha = None
        best_cl = 0; best_cd = 1.0; max_ld = -1.0
        
        if os.path.exists(f_log):
            with open(f_log, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    vals = line.split()
 
                    if len(vals) == 7 and not any(c.isalpha() for c in vals[0]) and not "#" in line:
                        try:
                            a, cl, cd = float(vals[0]), float(vals[1]), float(vals[2])
                            
                            if cd > CD_MIN_LIMIT and cl < CL_MAX_LIMIT: 
                                ld = cl / cd
                                if ld > max_ld:
                                    max_ld = ld; best_alpha = a; best_cl = cl; best_cd = cd
                        except ValueError: continue

        if best_alpha is not None and max_ld > 5.0:
            cmds_cp = (
                f"load {f_dat}\n"
                "ppar\n"
                "N 250\n"
                "\n"
                "\n"
                "oper\n"
                f"v {REYNOLDS}\n"
                f"mach {MACH}\n"
                f"iter {ITERATIONS}\n"
                f"alfa {best_alpha}\n"
                f"cpwr {f_cp}\n"
                "quit\n"
            )
            
            subprocess.run(XFOIL_PATH, input=cmds_cp, text=True, 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, 
                           timeout=10.0) 
            
            if os.path.exists(f_cp):
                raw_cp = np.loadtxt(f_cp, skiprows=3)

                if raw_cp.ndim == 2 and raw_cp.shape[0] > 10:
                    cp_fixed = np.interp(np.linspace(0, 1, 200), raw_cp[:,0], raw_cp[:,2])
                    result = {
                        'w_u': w_u, 'w_l': w_l, 'dz': dz,
                        'cl': best_cl, 'cd': best_cd, 'cp': cp_fixed,
                        'fitness': max_ld, 'alpha_opt': best_alpha
                    }

    except Exception:
        pass 

    for f in [f_dat, f_log, f_cp]:
        if os.path.exists(f):
            for _ in range(3):
                try: os.remove(f); break
                except: time.sleep(0.1)
    
    return result

class GeneticEngine:
    def __init__(self):
        self.population = []

    def load_csv(self):
        print(f"--- Loading Seeds: {INPUT_CSV} ---")
        if not os.path.exists(INPUT_CSV):
            print(f"BROOO: {INPUT_CSV} kaha hai")
            sys.exit(1)
        try:
            df = pd.read_csv(INPUT_CSV)
            df.columns = df.columns.str.strip()
            for _, row in df.iterrows():
                self.population.append({
                    'w_u': row[[f'wu_{i}' for i in range(8)]].values.astype(float),
                    'w_l': row[[f'wl_{i}' for i in range(8)]].values.astype(float),
                    'dz': float(row['TE_Thickness_dz']),
                    'fitness': 0.0 
                })
            print(f"Loaded {len(self.population)} seeds.")
        except Exception:
            traceback.print_exc(); sys.exit(1)

    def evolve_and_run(self):
        if not XFOIL_PATH:
            print("\nCRITICAL: XFOIL lapsing"); sys.exit(1)
        
        self.load_csv()
        
        with h5py.File(OUTPUT_H5, 'w') as h5f:
            d_w = h5f.create_dataset("weights", (0, 17), maxshape=(None, 17))
            d_s = h5f.create_dataset("scalars", (0, 3), maxshape=(None, 3))
            d_c = h5f.create_dataset("cp", (0, 200), maxshape=(None, 200))
            
            pool = multiprocessing.Pool(NUM_CORES)
            
            print("--- Udghaatan of Seeds ---")
            seed_tasks = [(i, p['w_u'], p['w_l'], p['dz']) for i, p in enumerate(self.population[:50])] 
            results = pool.map(worker_task, seed_tasks)
            valid_seeds = [r for r in results if r is not None]
            
            if not valid_seeds:
                print("FATAL: No seeds are good"); return

            self.population = valid_seeds
            self.save_batch(h5f, valid_seeds, d_w, d_s, d_c)
            
            total_saved = len(valid_seeds)
            gen = 0
            print(f"Pokemon Go. Target: {TARGET_SAMPLES}")

            while total_saved < TARGET_SAMPLES:
                gen += 1
                tasks = []
                for _ in range(NUM_CORES * 5):
                    if len(self.population) < 2: break
                    p1 = random.choice(self.population)
                    p2 = random.choice(self.population)
                    
                    alpha = random.random()
                    c_u = p1['w_u']*alpha + p2['w_u']*(1-alpha)
                    c_l = p1['w_l']*alpha + p2['w_l']*(1-alpha)
                    
                    c_dz = (p1['dz'] + p2['dz']) / 2.0
                    
                    if random.random() < 0.25:
                        c_u += np.random.normal(0, 0.05, 8)
                        c_l += np.random.normal(0, 0.05, 8)
                        c_l[0] = c_u[0] 
                        c_dz += np.random.normal(0, 0.001)
                    
                    c_dz = max(0.0, min(c_dz, DZ_MAX_LIMIT))
                    
                    tasks.append((random.randint(0, 1e8), c_u, c_l, c_dz))
                
                results = pool.map(worker_task, tasks)
                valid_batch = [r for r in results if r is not None]
                
                if valid_batch:
                    self.save_batch(h5f, valid_batch, d_w, d_s, d_c)
                    total_saved += len(valid_batch)
                    
                    combined = self.population + valid_batch
                    combined.sort(key=lambda x: x['fitness'], reverse=True)
                    
                    elites = combined[:100]
                    
                    others = combined[100:]
                    if len(others) > 0:
                        k = min(len(others), 50)
                        diversity_picks = random.sample(others, k)
                        self.population = elites + diversity_picks
                    else:
                        self.population = elites
                    
                    best = self.population[0]
                    print(f"Gen {gen} | +{len(valid_batch)} Saved | Total: {total_saved} | Best L/D: {best['fitness']:.1f} (A:{best['alpha_opt']})")
                
            pool.close(); pool.join()
            print(f"\n--- DONE. Saved to {OUTPUT_H5} ---")

    def save_batch(self, h5f, batch, d_w, d_s, d_c):
        if not batch: return
        n = len(batch)
        w = np.array([np.concatenate([r['w_u'], r['w_l'], [r['dz']]]) for r in batch])
        s = np.array([[r['cl'], r['cd'], r['alpha_opt']] for r in batch])
        c = np.array([r['cp'] for r in batch])
        
        idx = d_w.shape[0]
        d_w.resize(idx+n, axis=0); d_w[idx:] = w
        d_s.resize(idx+n, axis=0); d_s[idx:] = s
        d_c.resize(idx+n, axis=0); d_c[idx:] = c
        h5f.flush()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        GeneticEngine().evolve_and_run()
    except Exception:
        traceback.print_exc()
        input("Press Enter to Exit...")
