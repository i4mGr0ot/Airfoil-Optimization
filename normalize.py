import numpy as np
import h5py
import os

DATASET_PATH = "robust_airfoil_dataset_Re600k_fine.h5"
OUTPUT_PATH = "norm_stats.npz"

def generate_stats():
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset {DATASET_PATH} not found.")
        return

    print(f"--- Calculating Statistics from {DATASET_PATH} ---")
    
    with h5py.File(DATASET_PATH, 'r') as f:

        scalars = f['scalars'][:] 

        mean = np.mean(scalars, axis=0)
        std = np.std(scalars, axis=0)
        
        std[std == 0] = 1.0

    print("Stats Calculated:")
    print(f"Mean [Cl, Cd, Alpha]: {mean}")
    print(f"Std  [Cl, Cd, Alpha]: {std}")
    
    np.savez(OUTPUT_PATH, mean=mean, std=std)
    print(f"\nSuccess! Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_stats()
