import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import math
from scipy.special import riccati_jn, riccati_yn, lpmn

# --- Physical Constants ---
EPS_0 = 8.854e-12
MU_0 = 4 * np.pi * 10**(-7)
ETA_0 = np.sqrt(MU_0 / EPS_0)
C_LIGHT = 3e8

# --- Riccati-Bessel Helper Functions (Numerical) ---
def hankel2wow(B, r, n):
    jn, jn_p = riccati_jn(n, B * r)
    yn, yn_p = riccati_yn(n, B * r)
    return (jn - 1j * yn)[1:], (jn_p - 1j * yn_p)[1:]

def besselwow(B, r, n):
    normal, derivative = riccati_jn(n, B * r)
    return normal[1:], derivative[1:]

def Bcal(c, Beta, radius):
    """Reference calculation for validation."""
    g = np.arange(1, c + 1)
    _, d_psi_x = besselwow(Beta[0], radius[0], c)
    _, d_zeta_x = hankel2wow(Beta[0], radius[0], c)
    
    b_n = d_psi_x / d_zeta_x
    a_n_weight = (1j**g) * ((2 * g + 1) / (g * (g + 1)))
    return -b_n * a_n_weight

# --- Riccati-Bessel Helper Functions (Symbolic) ---
def hankel1(x, n): return (sp.jn(n, x) + sp.I * sp.yn(n, x)) * x
def hankel2(x, n): return (sp.jn(n, x) - sp.I * sp.yn(n, x)) * x
def bessel(x, n):  return sp.jn(n, x) * x

# --- Matrix Population ---
def get_symbolic_system():
    """Generates the symbolic A matrix and B vector."""
    B = sp.symbols('B:4')
    eta = sp.symbols('eta:4')
    r = sp.symbols('r:3')
    n = sp.symbols('n')
    x = sp.symbols('x:6') # x0 to x5

    A = sp.zeros(12, 12)
    # 0 = b_n, 1 = c_n, 2 = d_n, 3 = e_n, 4 = f_n, 5 = g_n, 6 = h_n, 7 = i_n, 8 = j_n, 9 = k_n, 10 = l_n, 11 = m_n
    # Each row represents a boundary condition
    # Boundary 1 (Outer)
    A[0, 0], A[0, 2], A[0, 4] = -B[1]*sp.diff(hankel2(x[0],n),x[0]), B[0]*sp.diff(hankel1(x[1],n),x[1]), B[0]*sp.diff(hankel2(x[1],n),x[1]) # This row is the TM condition for E_theta 
    A[1, 1], A[1, 3], A[1, 5] = -B[1]*hankel2(x[0],n), B[0]*hankel1(x[1],n), B[0]*hankel2(x[1],n) # This row is the TE condition for E_theta
    A[2, 0], A[2, 2], A[2, 4] = -B[1]*eta[1]*hankel2(x[0],n), B[0]*eta[0]*hankel1(x[1],n), B[0]*eta[0]*hankel2(x[1],n) # This row is the TM condition for H_theta
    A[3, 1], A[3, 3], A[3, 5] = -B[1]*eta[1]*sp.diff(hankel2(x[0],n),x[0]), B[0]*eta[0]*sp.diff(hankel1(x[1],n),x[1]), B[0]*eta[0]*sp.diff(hankel2(x[1],n),x[1]) # This row is the TE condition for H_theta

    # Boundary 2 
    A[4, 2], A[4, 4], A[4, 6], A[4, 8] = B[2]*sp.diff(hankel1(x[2],n),x[2]), B[2]*sp.diff(hankel2(x[2],n),x[2]), -B[1]*sp.diff(hankel1(x[3],n),x[3]), -B[1]*sp.diff(hankel2(x[3],n),x[3])
    A[5, 3], A[5, 5], A[5, 7], A[5, 9] = B[2]*hankel1(x[2],n), B[2]*hankel2(x[2],n), -B[1]*hankel1(x[3],n), -B[1]*hankel2(x[3],n)
    A[6, 2], A[6, 4], A[6, 6], A[6, 8] = B[2]*eta[2]*hankel1(x[2],n), B[2]*eta[2]*hankel2(x[2],n), -B[1]*eta[1]*hankel1(x[3],n), -B[1]*eta[1]*hankel2(x[3],n)
    A[7, 3], A[7, 5], A[7, 7], A[7, 9] = B[2]*eta[2]*sp.diff(hankel1(x[2],n),x[2]), B[2]*eta[2]*sp.diff(hankel2(x[2],n),x[2]), -B[1]*eta[1]*sp.diff(hankel1(x[3],n),x[3]), -B[1]*eta[1]*sp.diff(hankel2(x[3],n),x[3])

    # Boundary 3 (Core)
    A[8, 6], A[8, 8], A[8, 10]   = B[3]*sp.diff(hankel1(x[4],n),x[4]), B[3]*sp.diff(hankel2(x[4],n),x[4]), -B[2]*sp.diff(bessel(x[5],n),x[5])
    A[9, 7], A[9, 9], A[9, 11]   = B[3]*hankel1(x[4],n), B[3]*hankel2(x[4],n), -B[2]*bessel(x[5],n)
    A[10, 6], A[10, 8], A[10, 10] = B[3]*eta[3]*hankel1(x[4],n), B[3]*eta[3]*hankel2(x[4],n), -B[2]*eta[2]*bessel(x[5],n)
    A[11, 7], A[11, 9], A[11, 11] = B[3]*eta[3]*sp.diff(hankel1(x[4],n),x[4]), B[3]*eta[3]*sp.diff(hankel2(x[4],n),x[4]), -B[2]*eta[2]*sp.diff(bessel(x[5],n),x[5])

    # Right Hand Side Vector
    b = sp.zeros(12, 1)
    a_n = (sp.I**(-n)) * ((2*n + 1) / (n * (n + 1)))
    b[0] = B[1] * sp.diff(bessel(x[0], n), x[0])
    b[1] = B[1] * bessel(x[0], n)
    b[2] = B[1] * eta[1] * bessel(x[0], n)
    b[3] = B[1] * eta[1] * sp.diff(bessel(x[0], n), x[0])
    
    return A, b * a_n, (x, B, r, eta, n)

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