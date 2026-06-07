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
import traceback

# --- CONFIGURATION ---
CONFIG = {
    "INPUT_CSV": "UIUC_CST_Database.csv",
    "OUTPUT_H5": "airfoil_dataset_re_sweep.h5", # New filename
    "N_CORES": 6,
    "TARGET_SAMPLES": 8000,     
    "POPULATION_SIZE": 100,     
    
    # REYNOLDS SWEEP RANGE
    "RE_MIN": 200000.0,
    "RE_MAX": 3000000.0,
    
    "MACH": 0.0,
    "ALPHA_SEQ": (-4.0, 16.0, 1.0), 
    "ITERATIONS": 200,
    "XFOIL_TIMEOUT": 25.0,

    "THICKNESS_MIN": 0.05,  
    "THICKNESS_MAX": 0.22,  
    "DZ_TE_MAX": 0.015,     
    "MAX_INFLECTIONS": 2,   

    "CL_MAX_LIMIT": 2.2,   
    "CD_MIN_LIMIT": 0.002,  
}

def find_xfoil():
    if os.path.exists("xfoil.exe"): return os.path.abspath("xfoil.exe")
    path_exec = shutil.which("xfoil.exe")
    if path_exec: return path_exec
    common_paths = [r"C:\XFOIL\xfoil.exe", r"C:\Program Files\XFOIL\xfoil.exe", "/usr/bin/xfoil", "/usr/local/bin/xfoil"]
    for p in common_paths:
        if os.path.exists(p): return p
    return None

XFOIL_PATH = find_xfoil()

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
        return np.count_nonzero(sign_changes) <= CONFIG["MAX_INFLECTIONS"]

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
        if slope_u_TE > 0.02 or slope_l_TE < -0.02: return None, None, "INVALID: Diverging TE"
        
        thickness = y_u - y_l
        if np.any(thickness < -1e-6): return None, None, "INVALID: Cross"
        
        max_t = np.max(thickness)
        if max_t < CONFIG["THICKNESS_MIN"]: return None, None, "INVALID: Too Thin"
        if max_t > CONFIG["THICKNESS_MAX"]: return None, None, "INVALID: Too Thick"
        
        if not self.check_curvature(y_u) or not self.check_curvature(y_l): return None, None, "INVALID: Wavy"

        x_coords = np.concatenate((self.x[::-1], self.x[1:]))
        y_coords = np.concatenate((y_u[::-1], y_l[1:]))
        return x_coords, y_coords, "VALID"

def worker_task(task_data):
    if XFOIL_PATH is None: return None
    job_id, w_u, w_l, dz = task_data
    
    geo = RobustCST()
    x, y, status = geo.generate(w_u, w_l, dz)
    if status != "VALID": return None

    # --- REYNOLDS SWEEP LOGIC ---
    # Pick a random Re for this specific evaluation
    # This forces the GA to keep shapes that work across various speeds
    current_re = random.uniform(CONFIG['RE_MIN'], CONFIG['RE_MAX'])

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
                "ppar\n" "N 200\n" "pane\n" "\n" "\n"
                "oper\n"
                f"v {current_re}\n" # Use the randomized Re
                f"mach {CONFIG['MACH']}\n"
                f"iter {CONFIG['ITERATIONS']}\n" 
                "pacc\n" f"{f_log}\n" "\n"
                f"aseq {CONFIG['ALPHA_SEQ'][0]} {CONFIG['ALPHA_SEQ'][1]} {CONFIG['ALPHA_SEQ'][2]}\n"
                "pacc\n" "quit\n"
            )
            subprocess.run(XFOIL_PATH, input=cmds_sweep, text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=CONFIG['XFOIL_TIMEOUT'], cwd=tmp_dir)

            polar_data = []
            best_stats = {'alpha': 0, 'cl': 0, 'cd': 1.0, 'ld': -1.0, 'cm': 0.0}
            
            if os.path.exists(f_log):
                with open(f_log, 'r') as f:
                    for line in f:
                        vals = line.split()
                        if len(vals) >= 7 and not any(c.isalpha() for c in vals[0]) and "#" not in line:
                            try:
                                a, cl, cd, cm = float(vals[0]), float(vals[1]), float(vals[2]), float(vals[4])
                                if (cd > CONFIG['CD_MIN_LIMIT'] and abs(cl) < CONFIG['CL_MAX_LIMIT'] and cd < 1.0):
                                    polar_data.append({'alpha': a, 'cl': cl, 'cd': cd, 'cm': cm})
                                    ld = cl / cd
                                    if ld > best_stats['ld']:
                                        best_stats = {'alpha': a, 'cl': cl, 'cd': cd, 'ld': ld, 'cm': cm}
                            except ValueError: continue

            if len(polar_data) > 3 and best_stats['ld'] > 5.0:
                
                # Objectives: Efficiency, Stall Angle, Pitching Moment
                obj_ld = best_stats['ld']
                obj_stall = best_stats['alpha']
                obj_cm = -abs(best_stats['cm'])

                cmds_cp = (
                    f"load {f_dat}\n" "ppar\n" "N 200\n" "pane\n" "\n" "\n"
                    "oper\n" f"v {current_re}\n" f"mach {CONFIG['MACH']}\n" f"iter {CONFIG['ITERATIONS']}\n"
                    f"alfa {best_stats['alpha']}\n" f"cpwr {f_cp}\n" "quit\n"
                )
                subprocess.run(XFOIL_PATH, input=cmds_cp, text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5.0, cwd=tmp_dir)
                
                cp_fixed = np.zeros(200)
                if os.path.exists(f_cp):
                    try:
                        raw_cp = np.loadtxt(f_cp, skiprows=3)
                        if raw_cp.ndim == 2 and raw_cp.shape[0] > 10:
                            cp_fixed = np.interp(np.linspace(0, 1, 200), raw_cp[:,0], raw_cp[:,2])
                    except: pass 

                result = {
                    'w_u': w_u, 'w_l': w_l, 'dz': dz,
                    'cl': best_stats['cl'], 'cd': best_stats['cd'], 'cm': best_stats['cm'],
                    'alpha_opt': best_stats['alpha'],
                    'reynolds': current_re, # Save the Re used!
                    'cp': cp_fixed,
                    'objectives': [obj_ld, obj_stall, obj_cm]
                }
        except Exception: pass    
    return result

# --- NSGA-II CORE (Preserved from previous optimal code) ---
class NSGA2_Core:
    @staticmethod
    def dominates(p_objs, q_objs):
        better_or_equal = all(x >= y for x, y in zip(p_objs, q_objs))
        strictly_better = any(x > y for x, y in zip(p_objs, q_objs))
        return better_or_equal and strictly_better

    @staticmethod
    def fast_non_dominated_sort(population):
        fronts = [[]]
        for p in population:
            p['domination_count'] = 0
            p['dominated_solutions'] = []
            for q in population:
                if NSGA2_Core.dominates(p['objectives'], q['objectives']):
                    p['dominated_solutions'].append(q)
                elif NSGA2_Core.dominates(q['objectives'], p['objectives']):
                    p['domination_count'] += 1
            if p['domination_count'] == 0:
                p['rank'] = 0
                fronts[0].append(p)
        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in p['dominated_solutions']:
                    q['domination_count'] -= 1
                    if q['domination_count'] == 0:
                        q['rank'] = i + 1
                        next_front.append(q)
            i += 1
            if next_front: fronts.append(next_front)
            else: break
        return fronts

    @staticmethod
    def crowding_distance_assignment(front):
        if len(front) == 0: return
        l = len(front)
        num_obj = len(front[0]['objectives'])
        for p in front: p['distance'] = 0.0
        for m in range(num_obj):
            front.sort(key=lambda x: x['objectives'][m])
            front[0]['distance'] = float('inf')
            front[-1]['distance'] = float('inf')
            scale = front[-1]['objectives'][m] - front[0]['objectives'][m]
            if scale == 0: continue
            for i in range(1, l - 1):
                front[i]['distance'] += (front[i+1]['objectives'][m] - front[i-1]['objectives'][m]) / scale

class GeneticEngine:
    def __init__(self):
        self.population = []

    def load_initial_seeds(self):
        if not os.path.exists(CONFIG['INPUT_CSV']):
            print("CRITICAL: Input CSV not found."); sys.exit(1)
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
                    'objectives': [0,0,0], 'rank':0, 'distance':0
                })
            print(f"Loaded {len(self.population)} seeds.")
        except Exception: traceback.print_exc(); sys.exit(1)

    def save_batch(self, h5f, batch, datasets):
        if not batch: return
        n = len(batch)
        w = np.array([np.concatenate([r['w_u'], r['w_l'], [r['dz']]]) for r in batch])
        # Note: Added 'reynolds' to scalars array (5 columns now)
        s = np.array([[r['cl'], r['cd'], r['cm'], r['alpha_opt'], r['reynolds']] for r in batch])
        c = np.array([r['cp'] for r in batch])
        
        idx = datasets['w'].shape[0]
        datasets['w'].resize(idx+n, axis=0); datasets['w'][idx:] = w
        datasets['s'].resize(idx+n, axis=0); datasets['s'][idx:] = s
        datasets['c'].resize(idx+n, axis=0); datasets['c'][idx:] = c
        h5f.flush()

    def crowded_tournament_selection(self):
        p1 = random.choice(self.population)
        p2 = random.choice(self.population)
        if p1['rank'] < p2['rank']: return p1
        elif p2['rank'] < p1['rank']: return p2
        else:
            if p1['distance'] > p2['distance']: return p1
            else: return p2

    def run(self):
        self.load_initial_seeds()
        
        with h5py.File(CONFIG['OUTPUT_H5'], 'w') as h5f:
            datasets = {
                'w': h5f.create_dataset("weights", (0, 17), maxshape=(None, 17)),
                # Increased to 5 columns to store Reynolds Number
                's': h5f.create_dataset("scalars", (0, 5), maxshape=(None, 5)), 
                'c': h5f.create_dataset("cp", (0, 200), maxshape=(None, 200))
            }
            pool = multiprocessing.Pool(CONFIG['N_CORES'])

            print("--- Initializing Population ---")
            tasks = [(i, p['w_u'], p['w_l'], p['dz']) for i, p in enumerate(self.population)]
            # Run init tasks
            results = pool.map(worker_task, tasks[:CONFIG['POPULATION_SIZE']*2])
            
            valid_pop = [r for r in results if r is not None]
            
            # Initial NSGA-II Sort
            fronts = NSGA2_Core.fast_non_dominated_sort(valid_pop)
            for f in fronts: NSGA2_Core.crowding_distance_assignment(f)
            
            # Truncate
            self.population = []
            for f in fronts:
                if len(self.population) + len(f) <= CONFIG['POPULATION_SIZE']:
                    self.population.extend(f)
                else:
                    f.sort(key=lambda x: x['distance'], reverse=True)
                    self.population.extend(f[:CONFIG['POPULATION_SIZE'] - len(self.population)])
                    break
            
            self.save_batch(h5f, valid_pop, datasets)
            total_saved = len(valid_pop)
            gen = 0
            
            print(f"\nSTARTING RE-SWEEP EVOLUTION | Target: {CONFIG['TARGET_SAMPLES']}")
            
            while total_saved < CONFIG['TARGET_SAMPLES']:
                gen += 1
                offspring_tasks = []
                
                while len(offspring_tasks) < CONFIG['POPULATION_SIZE']:
                    p1 = self.crowded_tournament_selection()
                    p2 = self.crowded_tournament_selection()
                    
                    alpha = random.random()
                    c_u = p1['w_u']*alpha + p2['w_u']*(1-alpha)
                    c_l = p1['w_l']*alpha + p2['w_l']*(1-alpha)
                    c_dz = (p1['dz'] + p2['dz']) / 2.0
                    
                    if random.random() < 0.2:
                        noise = np.random.normal(0, 0.05, 8)
                        c_u += noise
                        c_l += noise
                        c_dz += np.random.normal(0, 0.002)
                        c_l[0] = c_u[0]
                    
                    c_dz = max(0.0, min(c_dz, CONFIG['DZ_TE_MAX']))
                    offspring_tasks.append((random.randint(0, 1e9), c_u, c_l, c_dz))

                batch_results = pool.map(worker_task, offspring_tasks)
                valid_offspring = [r for r in batch_results if r is not None]
                
                if valid_offspring:
                    self.save_batch(h5f, valid_offspring, datasets)
                    total_saved += len(valid_offspring)
                    
                    combined_pop = self.population + valid_offspring
                    fronts = NSGA2_Core.fast_non_dominated_sort(combined_pop)
                    for f in fronts: NSGA2_Core.crowding_distance_assignment(f)
                    
                    next_gen = []
                    for f in fronts:
                        if len(next_gen) + len(f) <= CONFIG['POPULATION_SIZE']:
                            next_gen.extend(f)
                        else:
                            f.sort(key=lambda x: x['distance'], reverse=True)
                            next_gen.extend(f[:CONFIG['POPULATION_SIZE'] - len(next_gen)])
                            break
                    self.population = next_gen
                    
                    print(f"Gen {gen:03d} | Saved: +{len(valid_offspring):02d} | Total: {total_saved:04d} | "
                          f"Re Range: {min(r['reynolds'] for r in valid_offspring):.0f}-{max(r['reynolds'] for r in valid_offspring):.0f}")

            pool.close(); pool.join()
            print(f"\n--- SUCCESS. Dataset saved to {CONFIG['OUTPUT_H5']} ---")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        GeneticEngine().run()
    except KeyboardInterrupt:
        print("\nInterrupt.")
    except Exception:
        traceback.print_exc()
