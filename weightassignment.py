import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.interpolate import interp1d
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

INPUT_FOLDER = Path.home() / "Downloads" / "UIUC_Airfoils"
OUTPUT_FOLDER = Path.home() / "Downloads" / "UIUC_CST_Output"
N_PARAMS = 8  

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
(OUTPUT_FOLDER / "plots").mkdir(exist_ok=True)

class CSTFitter:

    def __init__(self, n_params=8):
        self.n_params = n_params

    def bernstein_poly(self, x, n, k):
        return comb(n, k) * (x**k) * ((1 - x)**(n - k))

    def build_basis_matrix(self, x_coords):
        n = self.n_params - 1
        matrix = np.zeros((len(x_coords), self.n_params))
        for k in range(self.n_params):
            matrix[:, k] = self.bernstein_poly(x_coords, n, k)
        return matrix

    def fit_surface(self, x, y, fixed_w0=None):

        with np.errstate(divide='ignore', invalid='ignore'):
            C_x = np.sqrt(x) * (1 - x)

        mask = (x > 0.005) & (x < 0.995)
        
        if np.sum(mask) < 5: 
            return np.zeros(self.n_params)

        x_fit = x[mask]
        y_fit = y[mask]
        C_fit = C_x[mask]

        b = y_fit / C_fit

        A = self.build_basis_matrix(x_fit)

        if fixed_w0 is None:
            weights, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        else:
            A_reduced = A[:, 1:]
            
            w0_term = fixed_w0 * A[:, 0]
            b_reduced = b - w0_term
            
            w_remaining, _, _, _ = np.linalg.lstsq(A_reduced, b_reduced, rcond=None)
            
            weights = np.concatenate(([fixed_w0], w_remaining))

        return weights

class BatchProcessor:

    def parse_selig_format(self, file_path):

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            data = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        pt = [float(parts[0]), float(parts[1])]
                        data.append(pt)
                    except ValueError:
                        continue
            
            coords = np.array(data)
            if len(coords) < 10: return None, None 

            le_idx = np.argmin(coords[:, 0])
            
            upper_surf = coords[:le_idx+1]
            lower_surf = coords[le_idx:]
            
            upper_surf = np.flip(upper_surf, axis=0)
            
            return upper_surf, lower_surf

        except Exception as e:
            return None, None

    def reconstruct_curve(self, weights, x_pts=None):
        if x_pts is None: x_pts = np.linspace(0, 1, 100)
        
        fitter = CSTFitter(n_params=len(weights))
        A = fitter.build_basis_matrix(x_pts)
        C = np.sqrt(x_pts) * (1 - x_pts)
        S = A @ weights
        y = C * S
        return x_pts, y

    def run(self):
        fitter = CSTFitter(n_params=N_PARAMS)
        results = []
        files = list(INPUT_FOLDER.glob("*.dat"))
        
        print(f"Found {len(files)} airfoils. Starting optimization...")
        
        for i, file_path in enumerate(files):
            airfoil_name = file_path.stem
            upper_pts, lower_pts = self.parse_selig_format(file_path)
            
            if upper_pts is None:
                print(f"[{i+1}/{len(files)}] SKIP: {airfoil_name} (Parse Error)")
                continue

            try:
                w_upper = fitter.fit_surface(upper_pts[:,0], upper_pts[:,1])
                
                w_lower = fitter.fit_surface(lower_pts[:,0], lower_pts[:,1], fixed_w0=w_upper[0])

                rec_x_u, rec_y_u = self.reconstruct_curve(w_upper)
                rec_x_l, rec_y_l = self.reconstruct_curve(w_lower)

                entry = {
                    "Airfoil": airfoil_name,
                    "Upper_Weights": np.round(w_upper, 5).tolist(),
                    "Lower_Weights": np.round(w_lower, 5).tolist(),
                    "LE_Radius_Weight": w_upper[0] 
                }
                results.append(entry)

                if i < 10 or i % 50 == 0:
                    plt.figure(figsize=(10, 6))
                    # Original Data
                    plt.plot(upper_pts[:,0], upper_pts[:,1], 'k.', label='Original Data')
                    plt.plot(lower_pts[:,0], lower_pts[:,1], 'k.')
                    # CST Fit
                    plt.plot(rec_x_u, rec_y_u, 'r-', linewidth=2, label='CST Fit')
                    plt.plot(rec_x_l, rec_y_l, 'r-', linewidth=2)
                    plt.title(f"CST Parameterization: {airfoil_name}")
                    plt.legend()
                    plt.axis('equal')
                    plt.grid(True)
                    plt.savefig(OUTPUT_FOLDER / "plots" / f"{airfoil_name}.png")
                    plt.close()
                
                print(f"[{i+1}/{len(files)}] SUCCESS: {airfoil_name}")

            except Exception as e:
                print(f"[{i+1}/{len(files)}] FAIL: {airfoil_name} - {str(e)}")

        df = pd.DataFrame(results)
        csv_path = OUTPUT_FOLDER / "CST_Optimized_Parameters.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nBatch Processing Complete.")
        print(f"Parameters saved to: {csv_path}")
        print(f"Validation plots saved to: {OUTPUT_FOLDER}/plots")

if __name__ == "__main__":
    if not INPUT_FOLDER.exists():
        print(f"Error: Input folder not found at {INPUT_FOLDER}")
        print("Please run the downloader script first.")
    else:
        processor = BatchProcessor()
        processor.run()
