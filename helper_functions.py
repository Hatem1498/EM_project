import numpy as np
import sympy as sp
from scipy.special import riccati_jn, riccati_yn, lpmn

# ---- Numerical Functions ----
def hankel1wow(B, r, n):
    jn, jn_p = riccati_jn(n, B * r)
    yn, yn_p = riccati_yn(n, B * r)
    
    normal = jn + 1j * yn
    derivative = jn_p + 1j * yn_p
    
    return normal[n], derivative[n]

def hankel2wow(B, r, n):
    # Unpack the tuples FIRST
    jn, jn_p = riccati_jn(n, B * r)
    yn, yn_p = riccati_yn(n, B * r)
    
    # Then do the complex math on the arrays
    normal = jn - 1j * yn
    derivative = jn_p - 1j * yn_p
    
    # Slice off the n=0 order
    return normal[n], derivative[n]

def besselwow(B, r, n):
    normal, derivative = riccati_jn(n, B * r)
    return normal[n], derivative[n]

def Bcal(c,Beta,radius):
    # --- RCS Calculation ---
    # np.arange is safer for exact integers than np.linspace
    g = np.arange(1, c + 1) 

    m = np.sqrt(2.56)

    # 1. Calculate the arrays exactly ONCE to save processing time
    psi_x, d_psi_x   = besselwow(Beta[0], radius[0], c)  # Outside boundary (x)
    psi_mx, d_psi_mx = besselwow(Beta[1], radius[0], c)  # Inside boundary (mx)
    zeta_x, d_zeta_x = hankel2wow(Beta[0], radius[0], c) # Hankel outside

    # 2. Correct standard Mie formula for the b_n (TE/Magnetic) coefficient
    num = (m * psi_mx * d_psi_x) - (psi_x * d_psi_mx)
    den = (m * psi_mx * d_zeta_x) - (zeta_x * d_psi_mx)

    b_n = num / den

    # 3. Incident wave weighting (Using standard 1j**g)
    a_n_weight = (1j**g) * ((2 * g + 1) / (g * (g + 1)))

    # Final combined term for the RCS summation
    breal = b_n * a_n_weight
    return breal

def leg_diff(n, x):
    normal, derivative = lpmn(1, n, x)
    return normal[1,n], derivative[1, n]

# --- Symbolic Functions ---
def hankel1(x, n): return (sp.jn(n, x) + sp.I * sp.yn(n, x)) * x
def hankel2(x, n): return (sp.jn(n, x) - sp.I * sp.yn(n, x)) * x
def bessel(x, n):  return sp.jn(n, x) * x