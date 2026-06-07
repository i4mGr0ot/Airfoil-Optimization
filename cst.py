import numpy as np 
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.special.distance import cdist

#def cst_airfoil(coeffs_upper, coeffs_lower, pts = 100):
#  t = np.linspace(0,np.pi,pts)
#  x = 0.5 * (1-np,cos(t))
#  def calculate_surface (x, coeffs, te_gap):
#    n = len(coeffs)-1
#    s = np.zeroes_like(x)
#    for i,w in enumerate(coeffs):
#      k = comb(n,i)
#      bernstein = k * (x**i) * ((1-x)**(n-i))
#      s += w*bernstein 
#    return s
#  c = (x**0.5)*(1-x)
#  y_upper = c*shape_function(x, coeffs_upper)
#  y_lower = c*shape_function(x, coeffs_lower)

class Element:
	def __init__(self,chord,cst_order_u,cst_order_l):
		self.chord = chord
		self.order_u = cst_order_u
		self.order_l = cst_order_l
		self.n_params = cst_order_u + cst_order_l + 2

    def get_coords(self,weights,pts = 120):
		w_u = weights[:self.order_u + 1]
		w_l = weights[self.order_u + 1:]
		x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, pts)))
		
		def cst_surface(x_vec, w, c):
			n = len(w) - 1
			C = x_vec**0.5 * (1 - x_vec)**1.0
			S = sum( w[i] * comb(n,i) * (x_vec**i) * (1 - x_vec)**(n - i) for i in range(len(w)))
			return C * S * c

        yu = cst_surface(x, w_u, self.chord)
        yl = cst_surface(x, w_l, -self.chord)
        return x * self.chord , yu, yl

class Bassi:
	def __init__(self, elements_configs):
		self.elements = [Element(e['chord'], e['order'][0], e['order'][1]) for e in elements_configs]

    def transform(self, x, y, pivot, r, delta):
		px, py = pivot
		angle = np.radians(delta)
		
		tx = px + r * np.sin(angle)
		ty = py - r * (1 - np.cos(angle))
		
		a = np.radians(0.8 * angle)
		cos_a, sin_a = np.cos(a), np.sin(a)
		
		x_rot = x * cos_a + y * sin_a + tx
		y_rot = y * cos_a - x * sin_a + ty

        return x_rot, y_rot

    def penalty(self, coords_list, min_gap):
		total_penalty = 0
		for i in range(len(coords_list)):
			for j in range(i+1, len(coords_list)):
				p1 = np.column_stack((coords_list[i][0], coords_list[i][1]))
				p2 = np.column_stack((coords_list[j][0], coords_list[j][2]))
				
				dist = np.min(cdist(p1, p2))
				
				if dist < min_gap:
					total_penalty += 1000.0 * (min_gap - dist)**2
		return total_penalty

    def Objective (self, params, gap):
		index = []
		all_coords = []
		for i, elem in enumerate(self.elements):
			w = params[i : i + elem.n_params]
			index += elem.n_params
			
			x_loc, yu_loc, yl_loc = elem.get_coords(w)
			
			if i == 0:
				all_coords.append((x_loc, yu_loc, yl_loc))

            else:
                k_params = params[-(4 * (len(self.elements) - 1)):]
				k_index = ( i - 1 ) * 4
                x_g, y_g = self.transform(x_loc, yu_loc, 
                                           pivot = (k_params[k_indxe+2], k_params[k_index+3]),
                                           r = k_params[k_index+1], 
                                           delta = k_params[k_index])
                _, yl_g = self.transform(x_loc, yl_loc, 
                                         pivot = (k_params[k_index+2], k_params[k_index+3]),
                                         r = k_params[k_index+1], 
                                         delta = k_params[k_index])
                all_coords.append((x_g, yu_g, yl_g))

        penalty = self.penalty(all_coords, gap)

        perf_score = self.dummy_physics_eval(all_coords)

        return perf_score + penalty

    def dummy_physics_eval(self, coords):
		return 0.0
		
		
	
					












		
