import numpy as np
from scipy.special import comb
from shapely.geometry import Polygon, LineString
from shapely.validation import make_valid
import matplotlib.pyplot as plt

class RobustCSTAirfoil:

    def __init__(self, n_params=8, resolution=400):
        self.n_params = n_params
        self.resolution = resolution
        self.beta = np.linspace(0, np.pi, self.resolution)
        self.x = 0.5 * (1 - np.cos(self.beta))

        self.C = np.sqrt(self.x) * (1 - self.x)
        
        self.B = np.zeros((self.resolution, self.n_params))
        n = self.n_params - 1
        for k in range(self.n_params):
            self.B[:, k] = comb(n, k) * (self.x**k) * ((1 - self.x)**(n - k))

    def generate(self, w_upper, w_lower, dz_te=0.0):

        w_lower = np.array(w_lower)
        w_upper = np.array(w_upper)
        w_lower[0] = w_upper[0] 

        S_u = self.B @ w_upper
        S_l = self.B @ w_lower

        y_upper = self.C * S_u + self.x * (dz_te / 2.0)
        y_lower = -self.C * S_l - self.x * (dz_te / 2.0)

        thickness = y_upper - y_lower
        if np.any(thickness < -1e-5): 
            return None, None, "FAILURE: Surfaces Intersect"

        if (y_upper[-1] - y_upper[-2]) > (y_lower[-1] - y_lower[-2]) + 0.1:
             return None, None, "WARNING: TE Divergence"

        x_coords = np.concatenate((self.x[::-1], self.x))
        y_coords = np.concatenate((y_upper[::-1], y_lower))
        
        return x_coords, y_coords, "SUCCESS"

class MultiElementManager:

    def __init__(self):
        self.cst_main = RobustCSTAirfoil(n_params=8) 
        self.cst_flap = RobustCSTAirfoil(n_params=6) 
        
    def build_system(self, main_weights, flap_weights, flap_config):
       
        mx, my, status_m = self.cst_main.generate(main_weights['u'], main_weights['l'])
        if status_m != "SUCCESS": return None, None, status_m

        fx, fy, status_f = self.cst_flap.generate(flap_weights['u'], flap_weights['l'])
        if status_f != "SUCCESS": return None, None, status_f

        theta = np.radians(-flap_config['deflection'])
        c_f = flap_config['chord']
      
        R = np.array([[np.cos(theta), -np.sin(theta)], 
                      [np.sin(theta),  np.cos(theta)]])

        flap_coords = np.vstack((fx, fy))
        rotated_flap = R @ flap_coords
        scaled_flap = rotated_flap * c_f
        
        dx = 1.0 + flap_config['x_gap']
        dy = 0.0 + flap_config['y_gap']
        
        final_fx = scaled_flap[0, :] + dx
        final_fy = scaled_flap[1, :] + dy

        poly_main = Polygon(zip(mx, my))
        poly_flap = Polygon(zip(final_fx, final_fy))

        if not poly_main.is_valid: poly_main = make_valid(poly_main)
        if not poly_flap.is_valid: poly_flap = make_valid(poly_flap)

        if poly_main.intersects(poly_flap):
            return None, None, "FAILURE: Flap collision with Main Element"
          
        exact_gap = poly_main.distance(poly_flap)
        if exact_gap < 0.01:
            return None, None, f"FAILURE: Gap too small ({exact_gap:.4f} < 0.01)"

        return (mx, my), (final_fx, final_fy), "SUCCESS"

def test_robustness():
    manager = MultiElementManager()
    
    mw = {'u': [0.15, 0.15, 0.2, 0.2, 0.1, 0.0], 'l': [0.15, 0.1, 0.1, 0.05, 0.02, 0.0]}
    fw = {'u': [0.2, 0.2, 0.2, 0.1, 0.1, 0.0],   'l': [0.2, 0.1, 0.1, 0.05, 0.0, 0.0]}

    config = {
        'chord': 0.35,
        'deflection': 20,
        'x_gap': 0.015,  
        'y_gap': 0.02    
    }
    
    mx, my, fx, fy, status = None, None, None, None, "INIT"
    
    try:
        res_m, res_f, status = manager.build_system(mw, fw, config)
        if status == "SUCCESS":
            mx, my = res_m
            fx, fy = res_f
    except Exception as e:
        status = f"CRASH: {str(e)}"

    print(f"System Status: {status}")
    
    if status == "SUCCESS":
        plt.figure(figsize=(12, 4))
        plt.plot(mx, my, 'k-', linewidth=2, label='Main')
        plt.plot(fx, fy, 'r-', linewidth=2, label='Flap')
        plt.axis('equal')
        plt.title("Robust Multi-Element Generation")
        plt.legend()
        plt.grid(True)
        plt.show()

test_robustness()
