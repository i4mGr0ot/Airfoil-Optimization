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
import traceback

CONFIG = {
    "INPUT_CSV": "UIUC_CST_Database.csv",
    "OUTPUT_H5": "airfoil_dataset_Re600k_with_limits.h5",
    "N_CORES": 6,
    "TARGET_SAMPLES": 7500,
    
    "REYNOLDS": 600000,
    "MACH": 0.0,
    "ALPHA_SEQ": (-4.0, 18.0, 0.5), 
    "ITERATIONS": 200,
    "XFOIL_TIMEOUT": 40.0,

    "CL_CRUISE_TARGET": 0.5,    
    "CL_MAX_MIN_REQ": 1.3,

    "THICKNESS_MIN": 0.05,  
    "THICKNESS_MAX": 0.18,  
    "DZ_TE_MAX": 0.015,     
    "MAX_INFLECTIONS": 1,   

    "CL_MAX_LIMIT": 1.9,   
    "CD_MIN_LIMIT": 0.004,  
    "MIN_FITNESS": 30.0,  
}

def find_xfoil():
    if os.path.exists("xfoil.exe"): return os.path.abspath("xfoil.exe")
    path_exec = shutil.which("xfoil.exe")
    if path_exec: return path_exec
    
    common_paths = [
        r"C:\XFOIL\xfoil.exe", 
        r"C:\Program Files\XFOIL\xfoil.exe",
        "/usr/bin/xfoil",
        "/usr/local/bin/xfoil"
    ]
    for p in common_paths:
        if os.path.exists(p): return p
    return None

XFOIL_PATH = find_xfoil()

def calculate_curvature(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    d2y = np.gradient(dy, dx)
    return d2y

class RobustCST:
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
        if slope_u_TE > 0.02 or slope_l_TE < -0.02: 
             return None, None, "INVALID: Diverging TE"

        if np.any((y_u - y_l) < -1e-6): 
            return None, None, "INVALID: Criss cross apple sauce"
        
        t_dist = y_u - y_l
        max_t = np.max(t_dist)
        if max_t < CONFIG["THICKNESS_MIN"]: return None, None, "INVALID: Patlu"
        if max_t > CONFIG["THICKNESS_MAX"]: return None, None, "INVALID: Motu"

        if not self.check_curvature(y_u) or not self.check_curvature(y_l):
            return None, None, "INVALID: Wavy Surface"

        x_coords = np.concatenate((self.x[::-1], self.x[1:]))
        y_coords = np.concatenate((y_u[::-1], y_l[1:]))
        
        return x_coords, y_coords, "VALID"


def analyze_flight_profile(df_polar):

    if df_polar.empty: return 0.0, {}

    try:
        # Sort by Cl to interpolate safely
        df_sort = df_polar.sort_values('cl')
        cd_at_cruise = np.interp(CONFIG['CL_CRUISE_TARGET'], df_sort['cl'], df_sort['cd'])
        
        # If the interpolation is outside the data range, penalize heavily
        if CONFIG['CL_CRUISE_TARGET'] > df_sort['cl'].max() or CONFIG['CL_CRUISE_TARGET'] < df_sort['cl'].min():
            return 0.0, {}
            
        ld_cruise = CONFIG['CL_CRUISE_TARGET'] / cd_at_cruise
    except:
        return 0.0, {}

    # 2. TAKEOFF PERFORMANCE (Cl_max)
    cl_max = df_polar['cl'].max()
    alpha_stall = df_polar.loc[df_polar['cl'].idxmax(), 'alpha']
    
    # 3. STALL LINEARITY (The "Stall Effect" function)
    # We analyze the slope dCl/dAlpha from -2 to +6 degrees (Linear Region)
    linear_region = df_polar[(df_polar['alpha'] >= -2) & (df_polar['alpha'] <= 6)]
    
    linearity_score = 0.0
    if len(linear_region) > 4:
        # Perform Linear Regression: Cl = m * alpha + c
        slope, intercept, r_value, p_value, std_err = linregress(linear_region['alpha'], linear_region['cl'])
        
        # A perfect airfoil has R^2 close to 1.0 in the linear region.
        # Deviations indicate separation bubbles or non-linear flow.
        linearity_score = r_value ** 2 
        
        # Theoretical slope check (approx 0.11 per degree is ideal)
        # If slope is too shallow (< 0.08), it's a "lazy" airfoil.
        if slope < 0.08: linearity_score *= 0.5

    # --- FINAL FITNESS FORMULATION ---
    # J = (Efficiency_Cruise) * (Takeoff_Bonus) * (Stability_Factor)
    
    # Takeoff Bonus: Only reward if Cl_max > Req. Otherwise huge penalty.
    takeoff_factor = 1.0
    if cl_max < CONFIG['CL_MAX_MIN_REQ']:
        takeoff_factor = 0.1 # Fail
    else:
        takeoff_factor = 1.0 + (cl_max - CONFIG['CL_MAX_MIN_REQ']) # Bonus for extra lift

    # Final Weighted Calculation
    # We heavily weight Linearity because non-linear lift curves are dangerous/unpredictable
    fitness = ld_cruise * takeoff_factor * linearity_score
    
    stats = {
        'ld_cruise': ld_cruise,
        'cl_max': cl_max,
        'alpha_stall': alpha_stall,
        'linearity': linearity_score
    }
    
    return fitness, stats

def worker_task(task_data):

    if XFOIL_PATH is None: return None
    job_id, w_u, w_l, dz = task_data

    geo = RobustCST()
    x, y, status = geo.generate(w_u, w_l, dz)
    if status != "VALID": return None

    result = None

    with tempfile.TemporaryDirectory() as tmp_dir:
        f_dat = os.path.join(tmp_dir, "airfoil.dat")
        f_log = os.path.join(tmp_dir, "polar.log")
        f_cp  = os.path.join(tmp_dir, "cp_dist.txt")

        try:

            with open(f_dat, 'w') as f:
                f.write(f"AF_{job_id}\n")
                for i in range(len(x)):
                    f.write(f" {x[i]:.6f}  {y[i]:.6f}\n")

            cmds_sweep = (
                f"load {f_dat}\n"
                "ppar\n"
                "N 200\n"        
                "pane\n"
                "\n"
                "\n"
                "oper\n"
                f"v {CONFIG['REYNOLDS']}\n"
                f"mach {CONFIG['MACH']}\n"
                f"iter {CONFIG['ITERATIONS']}\n" 
                "pacc\n"
                f"{f_log}\n"
                "\n"
                f"aseq {CONFIG['ALPHA_SEQ'][0]} {CONFIG['ALPHA_SEQ'][1]} {CONFIG['ALPHA_SEQ'][2]}\n"
                "pacc\n"
                "quit\n"
            )

            subprocess.run(XFOIL_PATH, input=cmds_sweep, text=True, 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, 
                           timeout=CONFIG['XFOIL_TIMEOUT'], cwd=tmp_dir)

            polar_data = []
            best_stats = {'alpha': None, 'cl': 0, 'cd': 1.0, 'ld': -1.0}
            
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
                                    
                                    ld = cl / cd
                                    if ld > best_stats['ld']:
                                        best_stats = {'alpha': a, 'cl': cl, 'cd': cd, 'ld': ld}
                            except ValueError: continue

            bucket_width = 0.0
            
            if len(polar_data) > 5 and best_stats['ld'] > 5.0:
                df = pd.DataFrame(polar_data)
                min_cd = df['cd'].min()
                
                in_bucket = df[df['cd'] <= (min_cd + 0.002)]
                if not in_bucket.empty:
                    bucket_width = in_bucket['cl'].max() - in_bucket['cl'].min()

                fitness = best_stats['ld'] * (1.0 + bucket_width)

                if fitness > CONFIG['MIN_FITNESS']:
 
                    cmds_cp = (
                        f"load {f_dat}\n"
                        "ppar\n"
                        "N 200\n" 
                        "pane\n"
                        "\n"
                        "\n"
                        "oper\n"
                        f"v {CONFIG['REYNOLDS']}\n"
                        f"mach {CONFIG['MACH']}\n"
                        f"iter {CONFIG['ITERATIONS']}\n"
                        f"alfa {best_stats['alpha']}\n"
                        f"cpwr {f_cp}\n"
                        "quit\n"
                    )
                    subprocess.run(XFOIL_PATH, input=cmds_cp, text=True, 
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, 
                                   timeout=10.0, cwd=tmp_dir)
                    
                    cp_fixed = np.zeros(200)
                    if os.path.exists(f_cp):
                        try:
                            raw_cp = np.loadtxt(f_cp, skiprows=3)
                            if raw_cp.ndim == 2 and raw_cp.shape[0] > 10:

                                cp_fixed = np.interp(np.linspace(0, 1, 200), raw_cp[:,0], raw_cp[:,2])
                        except: pass 

                    result = {
                        'w_u': w_u, 'w_l': w_l, 'dz': dz,
                        'cl': best_stats['cl'], 
                        'cd': best_stats['cd'], 
                        'cp': cp_fixed,
                        'fitness': fitness,
                        'alpha_opt': best_stats['alpha'],
                        'bucket_width': bucket_width
                    }
        except Exception:

            pass 
            
    return result

class GeneticEngine:
    def __init__(self):
        self.population = []
        self.TOURNAMENT_SIZE = 3

    def load_initial_seeds(self):

        print(f"--- Loading Pokemon: {CONFIG['INPUT_CSV']} ---")
        if not os.path.exists(CONFIG['INPUT_CSV']):
            print(f"CRITICAL: {CONFIG['INPUT_CSV']} attach karo folder mein.")
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
            print(f"Loaded {len(self.population)} pokemons")
        except Exception:
            traceback.print_exc(); sys.exit(1)

    def save_batch(self, h5f, batch, datasets):

        if not batch: return
        n = len(batch)

        w = np.array([np.concatenate([r['w_u'], r['w_l'], [r['dz']]]) for r in batch])
        s = np.array([[r['cl'], r['cd'], r['alpha_opt']] for r in batch])
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
            print("\nCRITICAL: XFOIL toh add karlo path pe yaar."); sys.exit(1)
        
        self.load_initial_seeds()
        
        with h5py.File(CONFIG['OUTPUT_H5'], 'w') as h5f:
            datasets = {
                'w': h5f.create_dataset("weights", (0, 17), maxshape=(None, 17)),
                's': h5f.create_dataset("scalars", (0, 3), maxshape=(None, 3)),
                'c': h5f.create_dataset("cp", (0, 200), maxshape=(None, 200))
            }
            
            pool = multiprocessing.Pool(CONFIG['N_CORES'])

            print("--- Listing New Pokemons ---")
            seed_tasks = [(i, p['w_u'], p['w_l'], p['dz']) for i, p in enumerate(self.population)]

            results = pool.map(worker_task, seed_tasks[:100]) 
            
            valid_seeds = [r for r in results if r is not None]
            if not valid_seeds:
                print("FATAL: Too many constraints, airfoil nahi banega.")
                return

            self.population = valid_seeds
            self.save_batch(h5f, valid_seeds, datasets)
            
            total_saved = len(valid_seeds)
            gen = 0
            print(f"\nStarting Evolution. Target: {CONFIG['TARGET_SAMPLES']} samples.")
            print(f"Opt Targets: Cruise Cl={CONFIG['CL_CRUISE_TARGET']} | Min Takeoff Cl_max={CONFIG['CL_MAX_MIN_REQ']}")
            
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
                        noise_scale = 0.05
                        c_u += np.random.normal(0, noise_scale, 8)
                        c_l += np.random.normal(0, noise_scale, 8)
                        c_dz += np.random.normal(0, 0.002)
                        
                        c_l[0] = c_u[0] 
                    
                    c_dz = max(0.0, min(c_dz, CONFIG['DZ_TE_MAX']))
                    
                    tasks.append((random.randint(0, 1000000000), c_u, c_l, c_dz))
                
                batch_results = pool.map(worker_task, tasks)
                valid_batch = [r for r in batch_results if r is not None]
                
                if valid_batch:
                    self.save_batch(h5f, valid_batch, datasets)
                    total_saved += len(valid_batch)
                    
                    combined = self.population + valid_batch
                    combined.sort(key=lambda x: x['fitness'], reverse=True)
                    
                    elites = combined[:100]
                    wildcards = random.sample(combined[100:], min(len(combined[100:]), 50))
                    
                    self.population = elites + wildcards
                    
                    best = elites[0]
                    print(f"Gen {gen:03d} | Saved: +{len(valid_batch):02d} | Total: {total_saved:04d} | "
                          f"Fitness: {best['fitness']:.1f} [L/D_cr: {best['ld_cruise']:.1f}, Cl_max: {best['cl_max']:.2f}, Lin: {best['linearity']:.2f}]")

            pool.close(); pool.join()
            print(f"\n--- SUCCESS. Dataset saved to {CONFIG['OUTPUT_H5']} ---")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        GeneticEngine().run()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting.")
    except Exception:
        traceback.print_exc()
