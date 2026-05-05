import numpy as np
from scipy.special import riccati_jn, riccati_yn, legendre, lpmv
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
#eps_list = np.array([1,1+1j*10e3,1+1j*10e3,1+1j*10e3])
eps_list = np.array([1,2.56,2.56,2.56])
freq =10e9
layers = 3
radius = np.array([30e-3, 6e-3, 3e-3])
n=101
rad = np.linspace(0.00001, radius[0]*3, 200)
# Force inclusion of boundaries
rad = np.unique(np.concatenate([rad, [radius[0], radius[1], radius[2]]]))
rad.sort()
#-0.99999*np.pi
theta=np.linspace(0.001,1.999*np.pi,200)
phi=np.pi
E0=1

#calculating relevant constants
lambda_list = 3e8/np.sqrt(eps_list)/freq
Beta=2*np.pi/lambda_list
etav=ETA_0/np.sqrt(eps_list)



#error check
#b=np.array(Bcal(n,Beta,radius),dtype='complex128')

#definign plotting matrixes
E_theta=np.zeros((len(theta),len(rad)),dtype='complex128')
E_phi=np.zeros((len(theta),len(rad)),dtype='complex128')
E_r = np.zeros((len(theta),len(rad)),dtype='complex128')
amount=math.ceil(np.real((2*np.pi/lambda_list[0])*radius[0]+10))

#setting up solving lists
sol_list=np.zeros((n,4*layers), dtype='complex128')

#amount=50
constimagtheta=np.zeros((layers+1,len(rad)), dtype='complex128')
constrealtheta=np.zeros((layers+1,len(rad)), dtype='complex128')
constimagphi=np.zeros((layers+1,len(rad)), dtype='complex128')
constrealphi=np.zeros((layers+1,len(rad)), dtype='complex128')
constradial = np.zeros((layers+1, len(rad)), dtype='complex128')

#for the electrical fields
for i in range(layers+1):
    #theta
    constimagtheta[i]=-1j*E0*np.cos(phi)/(Beta[i]*rad)
    constrealtheta[i]=-E0*np.cos(phi)/(Beta[i]*rad)
    #phi
    constimagphi[i]=1j*E0*np.sin(phi)/(Beta[i]*rad)
    constrealphi[i]=E0*np.sin(phi)/(Beta[i]*rad)
    constradial[i] = -1j*E0*np.cos(phi)/(Beta[i]*rad)**2
    #print(constimagphi)
    #print(constrealphi)
    #rad
    #constimagrad
    #constrealrad
    #print(constradial[i])


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

    # --- Precompute radial functions ---
    #example h1[0][1] (0 = layer, 1 = rad), so it is indexed h1[l][r]
    h1 = [ [hankel1wow(Beta[l], r, order) for r in rad] for l in range(layers+1) ]
    h2 = [ [hankel2wow(Beta[l], r, order) for r in rad] for l in range(layers+1) ]
    b  = [ [besselwow(Beta[l], r, order)  for r in rad] for l in range(layers+1) ]

    

    # --- Main loops (same structure, just using cache) ---
    for i in range(len(theta)):
        P = P_all[i]
        Pd = Pd_all[i]
        for r in range(len(rad)):
            n_t = order
            if radius[1] < rad[r] and rad[r] < radius[0]: #layer 1 (first layer in sphere)
                E_phiimag = (sol[2]*h1[1][r][1] + sol[4]*h2[1][r][1]) * P
                E_phireal = (sol[3]*h1[1][r][0] + sol[5]*h2[1][r][0]) * Pd
                E_phi[i][r] += (constimagphi[1][r]*E_phiimag + constrealphi[1][r]*E_phireal)
                
                E_r_imag = ( n_t*(n_t+1)*(sol[2]*h1[1][r][0]+sol[4]*h2[1][r][0]) ) * P
                E_r[i][r] += constradial[1][r]*E_r_imag

                E_thetaimag = (sol[2]*h1[1][r][1] + sol[4]*h2[1][r][1])*Pd
                E_thetareal= (sol[3]*h1[1][r][0] + sol[5]*h2[1][r][0])*P
                E_theta[i][r] += (constrealtheta[1][r]*E_thetareal+constimagtheta[1][r]*E_thetaimag)

            elif radius[2] < rad[r] and rad[r] < radius[1]: #layer 2 (second layer inside sphere)
                E_phiimag = (sol[6]*h1[2][r][1] + sol[8]*h2[2][r][1]) * P
                E_phireal = (sol[7]*h1[2][r][0] + sol[9]*h2[2][r][0]) * Pd
                E_phi[i][r] += (constimagphi[2][r]*E_phiimag + constrealphi[2][r]*E_phireal)
                E_r_imag = ( n_t*(n_t+1)*(sol[6]*h1[2][r][0]+sol[8]*h2[2][r][0]) ) * P
                E_r[i][r] += constradial[2][r]*E_r_imag

                E_thetaimag = (sol[6]*h1[2][r][1] + sol[8]*h2[2][r][1])*Pd
                E_thetareal= (sol[7]*h1[2][r][0] + sol[9]*h2[2][r][0])*P
                E_theta[i][r] += (constrealtheta[2][r]*E_thetareal+constimagtheta[2][r]*E_thetaimag)

            elif rad[r] < radius[2]: #layer 3 (third layer inside sphere), this is where origin is.
                E_phiimag = sol[10]*b[3][r][1] * P
                E_phireal = sol[11]*b[3][r][0] * Pd
                E_phi[i][r] += (constimagphi[3][r]*E_phiimag + constrealphi[3][r]*E_phireal)
                E_r_imag =  n_t*(n_t+1)*(sol[10]*b[3][r][0]) * P
                E_r[i][r] += constradial[3][r]*E_r_imag

                E_thetaimag = (sol[10]*b[3][r][1])*Pd
                E_thetareal= (sol[11]*b[3][r][0])*P
                E_theta[i][r] += (constrealtheta[3][r]*E_thetareal + constimagtheta[3][r]*E_thetaimag)

            else: #Outside the sphere (layer 0)
                E_phiimag = (a_n*b[0][r][1] + sol[0]*h2[0][r][1]) * P
                E_phireal = (a_n*b[0][r][0] + sol[1]*h2[0][r][0]) * Pd
                E_phi[i][r] += (constimagphi[0][r]*E_phiimag + constrealphi[0][r]*E_phireal)

                E_thetaimag = (a_n*b[0][r][1]+sol[0]*h2[0][r][1])*Pd
                E_thetareal= (a_n*b[0][r][0]+sol[1]*h2[0][r][0])*P
                E_theta[i][r] += (constrealtheta[0][r]*E_thetareal+constimagtheta[0][r]*E_thetaimag)

                E_r_imag = ( n_t*(n_t+1)*(a_n*b[0][r][0]+sol[0]*h2[0][r][0]) ) * P
                E_r[i][r] += constradial[0][r]*E_r_imag
                
            

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
c = ax.pcolormesh(Theta, R, abs(E_r).T, cmap='viridis', shading='auto',vmin=0,vmax=2)

# 6. Add a colorbar and styling
fig.colorbar(c, ax=ax, label='Matrix Value')
ax.set_title("Polar Matrix Color Plot", va='bottom')
ax.grid(False)

plt.show()
