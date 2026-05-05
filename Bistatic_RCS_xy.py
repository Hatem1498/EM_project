import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import math
from helper_functions import leg_diff  # Import specific helpers
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
freq = 1.6e9
radius = np.array([81e-3, 27e-3, 20e-3])
max_order = 68
theta=np.pi/2
phi=np.linspace(0.0001,1.999*np.pi,160)

# Calculating relevant constants
lambdas = C_LIGHT / (np.sqrt(eps_list) * freq)
betas = 2 * np.pi / lambdas
etas = ETA_0 / np.sqrt(eps_list)

rcs_results = []

# --- Main Simulation Loop ---
limit = math.ceil((2 * np.pi / lambdas[0]) * radius[0] + 10)
x_args = []
for j in range(len(radius)):
    x_args.append(betas[j] * radius[j])
    x_args.append(betas[j+1] * radius[j])

# 3. Initialize accumulation arrays for the angular pattern
sumAtheta = np.zeros(len(phi), dtype='complex_')
sumAphi = np.zeros(len(phi), dtype='complex_')

print(f"Calculating Bistatic RCS at {freq/1e9} GHz...")

# --- Main Modal Summation ---
for order in range(1, limit):
    # Evaluate the problem at the current order
    args = (*x_args, *betas, *radius, *etas, order)
    sol_n = sol_func(*args)
    
    # Check for numerical errors
    if np.all(np.abs(sol_n) < 1e-12):
        break
    
    #The bn and cn coefficients for the current order
    #.item() ensures they are scalar, and not wrapped in an array.
    bn = sol_n[0].item()
    cn = sol_n[1].item()

    # Calculate the vector magnetic potentials (A_theta, A_phi) for all theta angles    
    p_terms = leg_diff(order, np.cos(theta))
    pi_n = p_terms[0]/np.sin(theta) #P_n1(cos(theta))/sin(theta)
    tau_n = p_terms[1] # diff(P_n1(cos(theta))) (Differentiation with respect to the argument x = cos(theta))

    # Calculate vector components for this mode
    # A_theta: TM contributes to Tau, TE contributes to Pi
    # A_phi:   TM contributes to Pi,  TE contributes to Tau
    A_theta = (1j**order) * (bn * np.sin(theta) * tau_n - cn * pi_n)
    A_phi   = (1j**order) * (bn * pi_n - cn * np.sin(theta) * tau_n)

    sumAtheta += A_theta
    sumAphi += A_phi

for i in range(len(phi)):
    # 4. Final RCS Calculation (Balanis 11-243)
    # Combines the theta and phi[i] components based on observation angle phi_obs
    term_theta = (np.cos(phi[i])**2) * (np.abs(sumAtheta)**2)
    term_phi = (np.sin(phi[i])**2) * (np.abs(sumAphi)**2)
    RCS = (lambdas[0]**2 / np.pi) * (term_theta + term_phi)

# --- Plotting (Polar Pattern) ---
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
normalized_rcs = 10 * np.log10(np.abs(RCS) / (np.pi * radius[0]**2))

ax.plot(phi, normalized_rcs)
ax.set_title(f"Bistatic RCS Pattern ({freq/1e9} GHz)", va='bottom')
plt.show()
