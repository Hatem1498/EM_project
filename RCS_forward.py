import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import math
from System_matrix import get_symbolic_system

# --- Physical Constants ---
EPS_0 = 8.854e-12
MU_0 = 4 * np.pi * 10**(-7)
ETA_0 = np.sqrt(MU_0 / EPS_0)
C_LIGHT = 3e8

# --- Initialization ---
A_sym, b_sym, sym_vars = get_symbolic_system()
sol_sym = A_sym.LUsolve(b_sym)

# Prepare lambdify for high-speed numerical evaluation
flat_vars = (*sym_vars[0], *sym_vars[1], *sym_vars[2], *sym_vars[3], sym_vars[4])
sol_func = sp.lambdify(flat_vars, sol_sym, modules=['scipy', 'numpy'])

# --- Simulation Parameters ---
eps_list = np.array([1, 4, 1, 50])
freqs = np.linspace(5e9, 15e9, 50)
radius = np.array([12e-3, 4e-3, 3e-3])
max_order = 68

rcs_results = []

# --- Main Simulation Loop ---
# i is the loop index (for progress/storage), f is the frequency value (for calculation)
for i, f in enumerate(freqs):
    print(f"Progress: {i/len(freqs)*100:.1f}%")
    
    lambdas = C_LIGHT / (np.sqrt(eps_list) * f)
    betas = 2 * np.pi / lambdas
    etas = ETA_0 / np.sqrt(eps_list)
    
    # Far-field role of thumb for numbe rof summations needed.
    limit = math.ceil((2 * np.pi / lambdas[0]) * radius[0] + 10)
    sum_a_theta = 0j
    
    #Calculate the x arguments for each frequency.
    x_args = []
    for j in range(len(radius)):
        x_args.append(betas[j] * radius[j])
        x_args.append(betas[j+1] * radius[j])

    for order in range(1, limit):
        # Map physical parameters to the symbolic arguments
        args = (*x_args, *betas, *radius, *etas, order)
        
        sol_n = sol_func(*args)
        
        # Check convergence/numerical stability
        if np.all(np.abs(sol_n) < 1e-10):
            break
            
        # Forward scattering logic (Theta = 0 simplification)
        # sol_n[0] is b_n, sol_n[1] is c_n
        A_theta_term = ((1j)**order) * order * ((order + 1) / 2) * (sol_n[0] + sol_n[1])
        sum_a_theta += A_theta_term
        
    rcs = (lambdas[0]**2 / np.pi) * (np.abs(sum_a_theta)**2) # The A_phi component is reduced since Phi = Pi. See equation 11-243 in balanis
    rcs_results.append(rcs)

# --- Plotting ---
plt.figure(figsize=(10, 6))
normalized_rcs = 10 * np.log10(np.abs(rcs_results) / (np.pi * radius[0]**2))
plt.plot(freqs / 1e9, normalized_rcs)
plt.title("Normalized Forward RCS of Multi-layer Sphere")
plt.xlabel("Frequency (GHz)")
plt.ylabel("RCS/$\pi a^2$ (dB)")
plt.grid(True)
plt.show()