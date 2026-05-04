import numpy as np
from scipy.special import riccati_jn, riccati_yn, legendre, lpmv, lpmn
import sympy as sp
import matplotlib.pyplot as plt
import math
from helper_functions import hankel1wow, hankel2wow, besselwow, leg_diff, Bcal
from System_matrix import get_symbolic_system

# --- Physical Constants ---
EPS = 8.854e-12
MU_0 = 4 * np.pi * 10**(-7)
ETA_0 = np.sqrt(MU_0 / EPS)
C_LIGHT = 3e8

#B=sp.symbols('B:4')
#eta=sp.symbols('eta:4')
#r=sp.symbols('r:3')
#n=sp.symbols('n')

# --- Initialization ---
A_sym, b_sym, sym_vars = get_symbolic_system()
sol_sym = A_sym.LUsolve(b_sym)

#defining symbolic letters
B=sp.symbols('B:4')
eta=sp.symbols('eta:4')
r=sp.symbols('r:3')
n=sp.symbols('n')
x0,x1,x2,x3,x4,x5=sp.symbols('x:6')
flat_vars = (
        x0, x1, x2, x3, x4, x5, 
        *B,   
        *r,   
        *eta,
        n
    )
#transfering from sympy to numpy
#solnumber=sp.lambdify(flat_vars,sol,modules=['scipy','numpy'])
A_func = sp.lambdify(flat_vars, A_sym, modules=['scipy', 'numpy'])
b_func = sp.lambdify(flat_vars, b_sym, modules=['scipy', 'numpy'])

#defining problem
eps_list = np.array([1,2.56,2.56,2.56])
freq =10e9
layers = 3
radius = np.array([30e-3, 20e-3, 10e-3])
n=101
rad = np.linspace(0.00001, radius[0]*3, 500)
# Force inclusion of boundaries
rad = np.unique(np.concatenate([rad, [radius[0], radius[1], radius[2]]]))
rad.sort()
#-0.99999*np.pi
theta=np.linspace(0.001,1.999*np.pi,500)
phi=np.pi/2
E0=1

#calculating relevant constants
lambda_list = 3e8/np.sqrt(eps_list)/freq
Beta=2*np.pi/lambda_list
etav=ETA_0/np.sqrt(eps_list)

#error check
b=np.array(Bcal(n,Beta,radius),dtype='complex_')

#definign plotting matrixes
E_theta=np.zeros((len(theta),len(rad)),dtype='complex_')
E_phi=np.zeros((len(theta),len(rad)),dtype='complex_')
amount=math.ceil((2*np.pi/lambda_list[0])*radius[0]+10)

#setting up solving lists
sol_list=np.zeros((n,4*layers), dtype='complex_')

#amount=50
constimagtheta=np.zeros((layers+1,len(rad)), dtype='complex_')
constrealtheta=np.zeros((layers+1,len(rad),len(theta)), dtype='complex_')
constimagphi=np.zeros((layers+1,len(rad)), dtype='complex_')
constrealphi=np.zeros((layers+1,len(rad)), dtype='complex_')

#for the electrical fields
for i in range(layers+1):
    #theta
    constimagtheta[i]=-1j*E0*Beta[i]*np.cos(phi)/(((freq*2*np.pi)**2)*eps_list[i]*EPS*rad)
    constrealtheta[i]=-E0*np.cos(phi)/(eps_list[i]*EPS*np.outer(rad,np.sin(theta))*freq*2*np.pi*etav[i])
    #phi
    constimagphi[i]=1j*E0*np.sin(phi)/(Beta[i]*rad)
    constrealphi[i]=E0*np.sin(phi)/(Beta[i]*rad)
    #print(constimagphi)
    #print(constrealphi)
    #rad
    #constimagrad
    #constrealrad


for order in range(1,amount):
    A_num = A_func(Beta[0]*radius[0], Beta[1]*radius[0], Beta[1]*radius[1],  
                   Beta[2]*radius[1], Beta[2]*radius[2], Beta[3]*radius[2], 
                   *Beta, *radius, *etav, order)
    
    b_num = b_func(Beta[0]*radius[0], Beta[1]*radius[0], Beta[1]*radius[1],  
                   Beta[2]*radius[1], Beta[2]*radius[2], Beta[3]*radius[2], 
                   *Beta, *radius, *etav, order)
    
    A_num = np.array(A_num, dtype=np.complex128)
    b_num = np.array(b_num, dtype=np.complex128)
    #sol_list[order-1]=solnumber(Beta[0]*radius[0], Beta[1]*radius[0], Beta[1]*radius[1],  Beta[2]*radius[1],  Beta[2]*radius[2],  Beta[3]*radius[2],*Beta,*radius,*etav,order )
    sol_list[order-1] = np.linalg.solve(A_num, b_num).flatten()
    #print(sol_list[order-1][0])
    a_n = (1j**(-order))*((2*order+1)/(order*(order+1)))
    
    #check for numerical errors
    if any(abs(x)<1e-7 for x in sol_list[order-1]):
        break; 
    
    sol = sol_list[order-1]
    # --- Precompute Legendre ---
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    P_all = np.zeros(len(theta), dtype=np.complex128)
    Pd_all = np.zeros(len(theta), dtype=np.complex128)

    for i in range(len(theta)):
        val, deriv = leg_diff(order, cos_theta[i])
        P_all[i] = val / sin_theta[i]
        Pd_all[i] = -sin_theta[i] * deriv

    print(f"Version 1 - theta[0] = {theta[0]}")
    print(f"P_all[0] = {P_all[0]}")
    print(f"Pd_all[0] = {Pd_all[0]}")
    # --- Precompute radial functions ---
    h1 = [ [hankel1wow(Beta[l], r, order) for r in rad] for l in range(layers+1) ]
    h2 = [ [hankel2wow(Beta[l], r, order) for r in rad] for l in range(layers+1) ]
    b  = [ [besselwow(Beta[l], r, order)  for r in rad] for l in range(layers+1) ]

    

    # --- Main loops (same structure, just using cache) ---
    for i in range(len(theta)):
        P = P_all[i]
        Pd = Pd_all[i]
        for j in range(len(rad)):
            if radius[1] < rad[j] and rad[j] < radius[0]:
                E_phiimag = (sol[2]*h1[1][j][1] + sol[4]*h2[1][j][1]) * P
                E_phireal = (sol[3]*h1[1][j][0] + sol[5]*h2[1][j][0]) * Pd
                E_phi[i][j] += (constimagphi[1][j]*E_phiimag + constrealphi[1][j]*E_phireal)

            elif radius[2] < rad[j] and rad[j] < radius[1]:
                E_phiimag = (sol[6]*h1[2][j][1] + sol[8]*h2[2][j][1]) * P
                E_phireal = (sol[7]*h1[2][j][0] + sol[9]*h2[2][j][0]) * Pd
                E_phi[i][j] += (constimagphi[2][j]*E_phiimag + constrealphi[2][j]*E_phireal)

            elif rad[j] < radius[2]:
                E_phiimag = sol[10]*b[3][j][1] * P
                E_phireal = sol[11]*b[3][j][0] * Pd
                E_phi[i][j] += (constimagphi[3][j]*E_phiimag + constrealphi[3][j]*E_phireal)

            else:
                E_phiimag = (a_n*b[0][j][1] + sol[0]*h2[0][j][1]) * P
                E_phireal = (a_n*b[0][j][0] + sol[1]*h2[0][j][0]) * Pd
                E_phi[i][j] += (constimagphi[0][j]*E_phiimag + constrealphi[0][j]*E_phireal)

            


#print("E_thet",E_theta)
#print("sol",abs(sol_list))
Theta, R = np.meshgrid(theta, rad)

# 3. Create your matrix data (Z-values)
# Here we just generate some sample data using a mathematical function
# In your case, this would be your actual matrix.
# The shape must match the grid: (num_radii, num_angles)
matrix_data = R * np.sin(3 * Theta) 

# 4. Set up the plot with a polar projection
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 6))

# 5. Plot the data using pcolormesh
# shading='auto' (or 'nearest') handles how the colors fill the grid cells
c = ax.pcolormesh(Theta, R, abs(E_phi).T, cmap='viridis', shading='auto',vmin=0,vmax=2)

# 6. Add a colorbar and styling
fig.colorbar(c, ax=ax, label='Matrix Value')
ax.set_title("Polar Matrix Color Plot", va='bottom')
ax.grid(False)

plt.show()
