# --- Matrix Population ---
import sympy as sp
from helper_functions import hankel1, hankel2, bessel # Import helpers
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