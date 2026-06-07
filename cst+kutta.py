import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.optimize import least_squares
from pathlib import Path
import warnings
import time

INPUT_FOLDER = Path.home() / "Downloads" / "UIUC_Airfoils"
OUTPUT_FOLDER = Path.home() / "Downloads" / "UIUC_CST_Output"
N_PARAMS = 8  

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
(OUTPUT_FOLDER / "plots").mkdir(exist_ok=True)
warnings.filterwarnings("ignore")

class PhysicCSTFitter:
    
    def __init__(self, n_params=8):
        self.n = n_params - 1
        self.n_params = n_params

    def bernstein_basis(self, x):
        B = np.zeros((len(x), self.n_params))
        for k in range(self.n_params):
            B[:, k] = comb(self.n, k) * (x**k) * ((1 - x)**(self.n - k))
        return B

    def cst_curve(self, x, weights, dz_te, is_upper=True):
        C = np.sqrt(x) * (1 - x)
        S = self.bernstein_basis(x) @ weights
        
        if is_upper:
            y = C * S + x * (dz_te / 2.0)
        else:
            y = -C * S - x * (dz_te / 2.0)
        return y

    def objective_function(self, params, x_u, y_u, x_l, y_l):
        w0 = params[0]
        w_u = np.concatenate(([w0], params[1:self.n_params]))
        w_l = np.concatenate(([w0], params[self.n_params : 2*self.n_params - 1]))
        dz = params[-1]

        y_u_fit = self.cst_curve(x_u, w_u, dz, is_upper=True)
        y_l_fit = self.cst_curve(x_l, w_l, dz, is_upper=False)

        res_u = y_u_fit - y_u
        res_l = y_l_fit - y_l
        
        weight_u = 1 + 2.0 * np.exp(-20 * x_u)
        weight_l = 1 + 2.0 * np.exp(-20 * x_l)
        
        return np.concatenate((res_u * weight_u, res_l * weight_l))

    def fit_airfoil(self, upper_pts, lower_pts):
        x_u, y_u = upper_pts[:, 0], upper_pts[:, 1]
        x_l, y_l = lower_pts[:, 0], lower_pts[:, 1]

        dz_est = max(0.0, abs(y_u[-1] - y_l[-1]))

        initial_guess = np.ones(1 + 2*(self.n_params-1) + 1) * 0.15
        initial_guess[-1] = dz_est

        lb = [-0.1] + [-1.0]*14 + [0.0]
        ub = [ 1.0] + [ 1.0]*14 + [0.1] 

        try:
            res = least_squares(
                self.objective_function,
                initial_guess,
                bounds=(lb, ub),
                args=(x_u, y_u, x_l, y_l),
                method='trf',
                loss='soft_l1'
            )
        except ValueError:
            return {"status": "FAILED_OPTIMIZATION"}

        p = res.x
        w0 = p[0]
        w_u = np.concatenate(([w0], p[1:self.n_params]))
        w_l = np.concatenate(([w0], p[self.n_params : 2*self.n_params - 1]))
        dz = p[-1]

        if res.cost > 0.5: 
            status = "POOR_FIT"
        else:
            status = "SUCCESS"

        return {
            "w_u": w_u, "w_l": w_l, "dz": dz,
            "status": status, "cost": res.cost,
            "fit_u": (x_u, self.cst_curve(x_u, w_u, dz, True)),
            "fit_l": (x_l, self.cst_curve(x_l, w_l, dz, False))
        }

class BatchProcessor:
    def parse_file(self, path):
        try:
            with open(path, 'r', errors='ignore') as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            
            coords = []
            for line in lines[1:]: 
                parts = line.split()
                if len(parts) >= 2:
                    try: coords.append([float(parts[0]), float(parts[1])])
                    except: pass
            
            data = np.array(coords)
            if len(data) < 10: return None, None, None

            le_idx = np.argmin(data[:, 0])
            upper = data[:le_idx+1]
            lower = data[le_idx:]

            upper = np.flip(upper, axis=0)
  
            max_c = np.max(data[:, 0])
            upper /= max_c
            lower /= max_c
            
            return upper, lower, path.stem
        except:
            return None, None, None

    def run(self):
        fitter = PhysicCSTFitter(n_params=N_PARAMS)
        csv_data = []
        files = list(INPUT_FOLDER.glob("*.dat"))
        
        print(f"--- UIUC Database Processor ---")
        print(f"Input: {INPUT_FOLDER}")
        print(f"Output: {OUTPUT_FOLDER}")
        print(f"Found {len(files)} files. Processing...")
        
        start_time = time.time()

        for i, fpath in enumerate(files):
            u_pts, l_pts, name = self.parse_file(fpath)
            
            if u_pts is None: continue

            res = fitter.fit_airfoil(u_pts, l_pts)
            
            if res['status'] == "SUCCESS":
                entry = {
                    "Airfoil_Name": name,
                    "Fit_Error_SSE": round(res['cost'], 6),
                    "TE_Thickness_dz": round(res['dz'], 6)
                }
                
                for idx, w in enumerate(res['w_u']):
                    entry[f"wu_{idx}"] = round(w, 6)
                    
                for idx, w in enumerate(res['w_l']):
                    entry[f"wl_{idx}"] = round(w, 6)
                
                csv_data.append(entry)

                if i < 200 or i % 10 == 0:
                    self.plot_verify(name, u_pts, l_pts, res['fit_u'], res['fit_l'])

            if i % 50 == 0:
                print(f"Processed {i}/{len(files)}...")

        df = pd.DataFrame(csv_data)
        
        cols = ['Airfoil_Name', 'Fit_Error_SSE', 'TE_Thickness_dz']
        cols += [f'wu_{j}' for j in range(N_PARAMS)]
        cols += [f'wl_{j}' for j in range(N_PARAMS)]
        df = df[cols]
        
        save_path = OUTPUT_FOLDER / "UIUC_CST_Database.csv"
        df.to_csv(save_path, index=False)
        
        print(f"\nDone! Processed {len(files)} files in {time.time()-start_time:.1f}s.")
        print(f"Successfully parameterized {len(df)} airfoils.")
        print(f"DATABASE SAVED TO: {save_path}")

    def plot_verify(self, name, u_raw, l_raw, u_fit, l_fit):
        plt.figure(figsize=(8, 3))
        plt.plot(u_raw[:,0], u_raw[:,1], 'k.', markersize=2, label='Raw')
        plt.plot(l_raw[:,0], l_raw[:,1], 'k.', markersize=2)
        plt.plot(u_fit[0], u_fit[1], 'r-', linewidth=1, label='CST')
        plt.plot(l_fit[0], l_fit[1], 'r-', linewidth=1)
        plt.legend()
        plt.title(f"Fit Check: {name}")
        plt.axis("equal")
        plt.grid(True, alpha=0.3)
        plt.savefig(OUTPUT_FOLDER / "plots" / f"{name}_fit.png")
        plt.close()

if __name__ == "__main__":
    if not INPUT_FOLDER.exists():
        print("Error: Airfoil folder not found. Run the downloader first.")
    else:
        BatchProcessor().run()
