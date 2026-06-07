import pandas as pd
import numpy as np
import h5py
import subprocess
import multiprocessing
import os
import time
import random
import shutil
from scipy.special import comb
from datetime import datetime

# --- CONFIGURATION ---
INPUT_CSV = "UIUC_CST_Database.csv"
OUTPUT_H5 = "robust_airfoil_dataset.h5"
XFOIL_EXEC = "xfoil.exe"          # Ensure in PATH or local folder
NUM_CORES = 6                     # Adjust to your CPU
TARGET_SAMPLES = 5000             # Goal: 5000 Valid Training Samples
XFOIL_TIMEOUT = 4.0               # Seconds max per run
MUTATION_RATE = 0.15              # High mutation to explore new shapes
MUTATION_SCALE = 0.05             # Magnitude of shape changes

# ==========================================
# 1. THE ROBUST GEOMETRY ENGINE
# ==========================================
class RobustCST:
    """
    Production-Grade CST Parameterization.
    Enforces physical constraints to guarantee valid, closed airfoils.
    """
    def __init__(self, n_params=8, resolution=200):
        self.n_params = n_params
        self.resolution = resolution
        
        # Cosine Spacing (Dense at LE and TE for CFD accuracy)
        self.beta = np.linspace(0, np.pi, self.resolution)
        self.x = 0.5 * (1 - np.cos(self.beta))
        
        # Class Function (Round Nose, Sharp TE)
        # C(x) = x^0.5 * (1-x)^1.0
        self.C = np.sqrt(self.x) * (1 - self.x)
        
        # Precompute Bernstein Matrix to speed up generation
        self.B = np.zeros((self.resolution, self.n_params))
        n = self.n_params - 1
        for k in range(self.n_params):
            self.B[:, k] = comb(n, k) * (self.x**k) * ((1 - self.x)**(n - k))

    def generate(self, w_u, w_l, dz_te):
        """
        Input: Weights (Upper/Lower) and TE Thickness.
        Output: X, Y coordinates formatted for XFOIL.
        Status: Returns 'VALID' or error string.
        """
        # --- CONSTRAINT 1: Leading Edge Continuity ---
        # The first weight controls the LE radius. 
        # They MUST be identical for the curve to be smooth at (0,0).
        w_l = np.array(w_l, dtype=np.float64)
        w_u = np.array(w_u, dtype=np.float64)
        w_l[0] = w_u[0]

        # Calculate Shape Functions S(x)
        S_u = self.B @ w_u
        S_l = self.B @ w_l

        # Calculate Y coordinates
        # y = C(x)*S(x) + x*dz
        y_u = self.C * S_u + self.x * (dz_te / 2.0)
        y_l = -self.C * S_l - self.x * (dz_te / 2.0)

        # --- CONSTRAINT 2: Topology Check (Self-Intersection) ---
        # Thickness must be > 0 everywhere (except LE/TE)
        thickness = y_u - y_l
        # Allow tiny numerical error (-1e-7), but reject crossing curves
        if np.any(thickness < -1e-6):
            return None, None, "INVALID: Curves Cross (Negative Thickness)"

        # --- CONSTRAINT 3: Trailing Edge Geometry ---
        # Prevent "fishtail" crossing at the very last point
        if y_u[-1] < y_l[-1]:
             return None, None, "INVALID: TE Cross"

        # Format for XFOIL: TE (top) -> LE -> TE (bottom)
        x_coords = np.concatenate((self.x[::-1], self.x[1:]))
        y_coords = np.concatenate((y_u[::-1], y_l[1:]))
        
        return x_coords, y_coords, "VALID"

# ==========================================
# 2. THE FAILSAFE XFOIL WORKER
# ==========================================
def worker_simulation(task):
    """
    Runs in a separate process. 
    Handles File I/O, Execution, and Parsing safely.
    """
    job_id, w_u, w_l, dz = task
    
    # Instantiate Geometry Engine locally
    geo_engine = RobustCST(n_params=8)
    
    # Generate Coords
    x, y, status = geo_engine.generate(w_u, w_l, dz)
    if status != "VALID":
        return None # Skip invalid shapes immediately

    # Unique Filenames to prevent collision in multiprocessing
    pid = os.getpid()
    f_root = f"temp_worker_{pid}_{job_id}"
    f_dat = f"{f_root}.dat"
    f_log = f"{f_root}.log"
    f_cp  = f"{f_root}_cp.txt"
    
    result = None

    try:
        # Write Geometry File
        with open(f_dat, 'w') as f:
            f.write(f"AF_{job_id}\n")
            for i in range(len(x)):
                f.write(f" {x[i]:.6f}  {y[i]:.6f}\n")
        
        # XFOIL Command Script
        # Optimized for robustness: 'pane' repairs mesh, 'iter' limited to 60
        cmds = (
            f"load {f_dat}\n"
            "pane\n"      
            "oper\n"
            "v 3e6\n"     # Re = 3 Million
            "iter 60\n"
            "pacc\n"
            f"{f_log}\n"
            "\n"
            "alfa 2.0\n"  # Evaluation at Cruise Angle
            "pacc\n"
            f"cpwr {f_cp}\n"
            "\n"
            "quit\n"
        )
        
        # EXECUTE WITH TIMEOUT
        # If XFOIL hangs > 4s, it gets killed.
        subprocess.run(XFOIL_EXEC, input=cmds, text=True, 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, 
                       timeout=XFOIL_TIMEOUT)
        
        # PARSE RESULTS
        if os.path.exists(f_log) and os.path.exists(f_cp):
            # 1. Parse Scalars (Cl, Cd)
            with open(f_log, 'r') as f:
                lines = f.readlines()
                # Check for convergence line
                if len(lines) > 12:
                    vals = lines[-1].split()
                    if len(vals) == 7:
                        cl = float(vals[1])
                        cd = float(vals[2])
                        
                        # 2. Physics Sanity Filter
                        # Reject non-physical results (e.g., drag < 0, lift > max theoretical)
                        if cd > 0.0001 and cd < 0.5 and abs(cl) < 3.0:
                            
                            # 3. Parse Pressure Distribution (Cp)
                            raw_cp = np.loadtxt(f_cp, skiprows=3)
                            # Normalize to fixed 200 points for Neural Network
                            # XFOIL output size varies; we must interpolate.
                            cp_interp = np.interp(np.linspace(0, 1, 200), raw_cp[:,0], raw_cp[:,2])
                            
                            # Calculate Fitness (L/D)
                            fitness = cl / cd
                            
                            result = {
                                'w_u': w_u, 'w_l': w_l, 'dz': dz,
                                'cl': cl, 'cd': cd, 'cp': cp_interp,
                                'fitness': fitness
                            }

    except Exception:
        pass # Fail silently, worker just returns None
    
    # CLEANUP (Crucial for Windows)
    for f in [f_dat, f_log, f_cp]:
        if os.path.exists(f): 
            try: os.remove(f)
            except: pass
            
    return result

# ==========================================
# 3. THE GENETIC ORCHESTRATOR
# ==========================================
class GeneticGenerator:
    def __init__(self):
        self.population = [] # List of dicts (DNA)
        
    def load_seeds(self):
        """Loads the UIUC Database as the 'Adam & Eve' generation."""
        print(f"--- Loading Seeds from {INPUT_CSV} ---")
        try:
            df = pd.read_csv(INPUT_CSV)
            col_wu = [f'wu_{i}' for i in range(8)]
            col_wl = [f'wl_{i}' for i in range(8)]
            
            count = 0
            for _, row in df.iterrows():
                dna = {
                    'w_u': row[col_wu].values.astype(float),
                    'w_l': row[col_wl].values.astype(float),
                    'dz': float(row['TE_Thickness_dz']),
                    'fitness': 0.0 # Will be evaluated
                }
                self.population.append(dna)
                count += 1
            print(f"Loaded {count} airfoils.")
            
            # Shuffle seeds
            random.shuffle(self.population)
            
        except FileNotFoundError:
            print("ERROR: CSV file not found.")
            exit()

    def create_offspring(self, parents, n_needed):
        """
        Creates new generation using Tournament Selection + Crossover + Mutation.
        """
        offspring = []
        while len(offspring) < n_needed:
            # 1. Tournament Selection (Pick 3, take best)
            pool = random.sample(parents, 3)
            p1 = max(pool, key=lambda x: x['fitness'])
            pool = random.sample(parents, 3)
            p2 = max(pool, key=lambda x: x['fitness'])
            
            # 2. Blended Crossover
            alpha = random.random()
            child_u = p1['w_u'] * alpha + p2['w_u'] * (1 - alpha)
            child_l = p1['w_l'] * alpha + p2['w_l'] * (1 - alpha)
            child_dz = (p1['dz'] + p2['dz']) / 2.0
            
            # 3. Gaussian Mutation
            # Apply to 30% of children
            if random.random() < 0.3:
                # Add noise
                child_u += np.random.normal(0, MUTATION_SCALE, 8)
                child_l += np.random.normal(0, MUTATION_SCALE, 8)
                # Enforce LE continuity immediately in DNA (redundant but safe)
                child_l[0] = child_u[0]
            
            offspring.append((random.randint(0, 1e9), child_u, child_l, child_dz))
            
        return offspring

    def run(self):
        self.load_seeds()
        
        # Initialize HDF5 File
        with h5py.File(OUTPUT_H5, 'w') as h5f:
            # Create Resizable Datasets
            d_weights = h5f.create_dataset("weights", (0, 17), maxshape=(None, 17))
            d_scalars = h5f.create_dataset("scalars", (0, 2), maxshape=(None, 2))
            d_cp      = h5f.create_dataset("cp",      (0, 200), maxshape=(None, 200))
            
            # Start Pool
            pool = multiprocessing.Pool(NUM_CORES)
            
            # --- PHASE 1: EVALUATE SEEDS ---
            print("--- Phase 1: Benchmarking Seeds ---")
            # We need to know fitness of seeds to select parents
            seed_tasks = [(i, p['w_u'], p['w_l'], p['dz']) for i, p in enumerate(self.population)]
            # Run only a subset if database is huge (e.g., first 200)
            seed_tasks = seed_tasks[:200]
            
            results = pool.map(worker_simulation, seed_tasks)
            
            # Filter Valid Seeds
            valid_pop = []
            for r in results:
                if r is not None:
                    valid_pop.append(r)
            
            self.population = valid_pop
            print(f"Valid Seeds: {len(self.population)}")
            
            # Save Seeds to Dataset
            self.save_batch(h5f, valid_pop, d_weights, d_scalars, d_cp)
            
            # --- PHASE 2: EVOLUTION LOOP ---
            total_samples = len(valid_pop)
            generation = 0
            
            while total_samples < TARGET_SAMPLES:
                generation += 1
                
                # Determine how many to generate
                # We generate 2x batch size to keep CPU fed
                batch_size = NUM_CORES * 8 
                tasks = self.create_offspring(self.population, batch_size)
                
                # Parallel Execution
                gen_results = pool.map(worker_simulation, tasks)
                
                # Collect Valid Children
                valid_children = [r for r in gen_results if r is not None]
                
                if not valid_children:
                    print(f"Gen {generation}: No valid survivors. Retrying...")
                    continue
                
                # Save to Disk
                self.save_batch(h5f, valid_children, d_weights, d_scalars, d_cp)
                total_samples += len(valid_children)
                
                # Update Population (Survival Strategy)
                # Combine Parents + Children
                combined = self.population + valid_children
                # Sort by Fitness (L/D)
                combined.sort(key=lambda x: x['fitness'], reverse=True)
                
                # ELITISM + DIVERSITY
                # Keep top 100 Best
                elites = combined[:100]
                # Keep 50 Random from the rest (to maintain genetic diversity)
                others = combined[100:]
                if others:
                    diversity = random.sample(others, min(len(others), 50))
                    self.population = elites + diversity
                else:
                    self.population = elites
                
                # Report
                best_ld = self.population[0]['fitness']
                print(f"Gen {generation} | Added: {len(valid_children)} | Total: {total_samples}/{TARGET_SAMPLES} | Best L/D: {best_ld:.2f}")

            pool.close()
            pool.join()
            print(f"--- SUCCESS. Database generated at {OUTPUT_H5} ---")

    def save_batch(self, h5f, batch_data, d_w, d_s, d_c):
        """Writes a batch of results to HDF5 efficiently."""
        n = len(batch_data)
        if n == 0: return
        
        # Prepare Matrices
        # Weights: [wu_0..7, wl_0..7, dz] -> 17 floats
        mat_w = np.array([np.concatenate([r['w_u'], r['w_l'], [r['dz']]]) for r in batch_data])
        mat_s = np.array([[r['cl'], r['cd']] for r in batch_data])
        mat_c = np.array([r['cp'] for r in batch_data])
        
        # Resize Dataset
        idx = d_w.shape[0]
        d_w.resize(idx + n, axis=0)
        d_s.resize(idx + n, axis=0)
        d_c.resize(idx + n, axis=0)
        
        # Write
        d_w[idx:] = mat_w
        d_s[idx:] = mat_s
        d_c[idx:] = mat_c
        
        # Flush to ensure data safety if crash happens
        h5f.flush()

if __name__ == "__main__":
    # Required for Windows Multiprocessing
    multiprocessing.freeze_support()
    
    # Run Engine
    engine = GeneticGenerator()
    engine.run()
