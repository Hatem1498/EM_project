import numpy as np
from scipy.special import riccati_jn, riccati_yn, legendre, lpmv, lpmn
import sympy as sp
import matplotlib.pyplot as plt
import math
eps_0 = 8.854e-12
mu_0 =  4*np.pi*10**(-7)
eta_0 = np.sqrt(mu_0/eps_0)
def hankel2wow(B, r, n):
    # Unpack the tuples FIRST
    jn, jn_p = riccati_jn(n, B * r)
    yn, yn_p = riccati_yn(n, B * r)
    
    # Then do the complex math on the arrays
    normal = jn - 1j * yn
    derivative = jn_p - 1j * yn_p
    
    # Slice off the n=0 order
    return normal[n], derivative[n]

def hankel1wow(B, r, n):
    jn, jn_p = riccati_jn(n, B * r)
    yn, yn_p = riccati_yn(n, B * r)
    
    normal = jn + 1j * yn
    derivative = jn_p + 1j * yn_p
    
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





def hankel2(x,n):
     
    return (sp.jn(n,x)-sp.I*sp.yn(n,x))*x

def hankel1(x,n):
    return (sp.jn(n,x)+sp.I*sp.yn(n,x))*x



def bessel(x,n):
    return sp.jn(n,x)*x

def leg_diff(n, x):
    normal, derivative = lpmn(1, n, x)
    return normal[1,n],derivative[1, n]


def populate_Amatrix():
    B=sp.symbols('B:4')
    eta=sp.symbols('eta:4')
    r=sp.symbols('r:3')
    n=sp.symbols('n')
    x0,x1,x2,x3,x4,x5=sp.symbols('x:6')


    layers=3
    A=sp.zeros(4*layers,4*layers)


    #x0=B[0]*r[0]
    #x1=B[1]*r[0]
    #Electric boundary imag
    A[0,0] = -B[1]*sp.diff(hankel2(x0,n),x0)
    A[0,2] = B[0]*sp.diff(hankel1(x1,n),x1)
    A[0,4] = B[0]*sp.diff(hankel2(x1,n),x1)
    #Electric boundary real
    A[1,1] = -B[1]*hankel2(x0,n)
    A[1,3] = B[0]*hankel1(x1,n)
    A[1,5] = B[0]*hankel2(x1,n)
    #magnetic boundary real
    A[2,0] = -B[1]*eta[1]*hankel2(x0,n)
    A[2,2] = B[0]*eta[0]*hankel1(x1,n)
    A[2,4] = B[0]*eta[0]*hankel2(x1,n)
    #magnetic boundary imag
    A[3,1] = -B[1]*eta[1]*sp.diff(hankel2(x0,n),x0)
    A[3,3] = B[0]*eta[0]*sp.diff(hankel1(x1,n),x1)
    A[3,5] = B[0]*eta[0]*sp.diff(hankel2(x1,n),x1)


    #x2=B[1]*r[1]
    #x3=B[2]*r[1]
    #Electric boundary imag
    A[4,2] = B[2]*sp.diff(hankel1(x2, n),x2)
    A[4,4] = B[2]*sp.diff(hankel2(x2, n),x2)
    A[4,6] = -B[1]*sp.diff(hankel1(x3, n),x3)
    A[4,8] = -B[1]*sp.diff(hankel2(x3, n),x3)
    #Electric boundary real
    A[5,3] = B[2]*hankel1(x2, n)
    A[5,5] = B[2]*hankel2(x2, n)
    A[5,7] = -B[1]*hankel1(x3, n)
    A[5,9] = -B[1]*hankel2(x3, n)
    #magnetic boundary real
    A[6,2] = B[2]*eta[2]*hankel1(x2, n)
    A[6,4] = B[2]*eta[2]*hankel2(x2, n)
    A[6,6] = -B[1]*eta[1]*hankel1(x3, n)
    A[6,8] = -B[1]*eta[1]*hankel2(x3, n)
    #magnetic boundary imag
    A[7,3] = B[2]*eta[2]*sp.diff(hankel1(x2, n),x2)
    A[7,5] = B[2]*eta[2]*sp.diff(hankel2(x2, n),x2)
    A[7,7] = -B[1]*eta[1]*sp.diff(hankel1(x3, n),x3)
    A[7,9] = -B[1]*eta[1]*sp.diff(hankel2(x3, n),x3)


    #x4=B[2]*r[2]
    #x5=B[3]*r[2]
    #Electric boundary imag
    A[8,6] = B[3]*sp.diff(hankel1(x4, n),x4)
    A[8,8] = B[3]*sp.diff(hankel2(x4, n),x4)
    A[8,10] = -B[2]*sp.diff(bessel(x5, n),x5)
    #Electric boundary real
    A[9,7] =  B[3]*hankel1(x4, n)
    A[9,9] =  B[3]*hankel2(x4, n)
    A[9,11] = -B[2]*bessel(x5, n)
    #magnetic boundary real
    A[10,6] = B[3]*eta[3]*hankel1(x4, n)
    A[10,8] = B[3]*eta[3]*hankel2(x4, n)
    A[10,10] = -B[2]*eta[2]*bessel(x5, n)
    #magnetic boundary imag
    A[11,7] =  B[3]*eta[3]*sp.diff(hankel1(x4, n),x4)
    A[11,9] = B[3]*eta[3]*sp.diff(hankel2(x4, n),x4)
    A[11,11] = -B[2]*eta[2]*sp.diff(bessel(x5, n),x5)
    
    return A

def populate_Bmatrix():
    B=sp.symbols('B:4')
    eta=sp.symbols('eta:4')
    r=sp.symbols('r:3')
    n=sp.symbols('n')
    x0,x1,x2,x3,x4,x5=sp.symbols('x:6')
    layers=3
    b=sp.zeros(4*layers,1)
    
    a_n = (sp.I**(-n))*((2*n+1)/(n*(n+1)))
    b[0]=B[1]*sp.diff(bessel(x0,n),x0)
    b[1]=B[1]*bessel(x0,n)
    b[2]=B[1]*eta[1]*bessel(x0,n)
    b[3]=B[1]*eta[1]*sp.diff(bessel(x0,n),x0)
    b=b*a_n

    return b
#B=sp.symbols('B:4')
#eta=sp.symbols('eta:4')
#r=sp.symbols('r:3')
#n=sp.symbols('n')


#populating matrix and solving for the expasion coefficients
A=populate_Amatrix()
b=populate_Bmatrix()
print("matrix passed")
#sp.pprint(A, num_columns=10_000)
#sol=A.LUsolve(b)


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
A_func = sp.lambdify(flat_vars, A, modules=['scipy', 'numpy'])
b_func = sp.lambdify(flat_vars, b, modules=['scipy', 'numpy'])

#defining problem
eps_list = np.array([1,2.56,2.56,2.56])
freq =10e9
layers = 3
radius = np.array([30e-3, 4e-3, 3e-3])
n=101
rad=np.linspace(0.00001,radius[0]*3,500)
#-0.99999*np.pi
theta=np.linspace(0.001,1.999*np.pi,500)
phi=np.pi/2
E0=1





   

#calculating relevant constants
lambda_list = 3e8/np.sqrt(eps_list)/freq
Beta=2*np.pi/lambda_list
etav=eta_0/np.sqrt(eps_list)

#error check
b=np.array(Bcal(n,Beta,radius),dtype='complex_')

#definign plotting matrixes
E_theta=np.zeros((len(theta),len(rad)),dtype='complex_')
E_phi=np.zeros((len(theta),len(rad)),dtype='complex_')
amount=math.ceil((2*np.pi/lambda_list[0])*radius[0]+10)

#setting up solving lists
sol_list=np.zeros((n,4*layers,1), dtype='complex_')

#amount=50
constimagtheta=np.zeros((layers+1,len(rad)), dtype='complex_')
constrealtheta=np.zeros((layers+1,len(rad),len(theta)), dtype='complex_')
constimagphi=np.zeros((layers+1,len(rad)), dtype='complex_')
constrealphi=np.zeros((layers+1,len(rad)), dtype='complex_')

#for the electrical fields
for i in range(layers+1):
    #theta
    constimagtheta[i]=-1j*E0*Beta[i]*np.cos(phi)/(((freq*2*np.pi)**2)*eps_list[i]*eps_0*rad)
    constrealtheta[i]=-E0*np.cos(phi)/(eps_list[i]*eps_0*np.outer(rad,np.sin(theta))*freq*2*np.pi*etav[i])
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
    sol_list[order-1] = np.linalg.solve(A_num, b_num)
    #print(sol_list[order-1][0])
    a_n = (1j**(-order))*((2*order+1)/(order*(order+1)))
    
    #check for numerical errors
    if any(abs(x)<1e-7 for x in sol_list[order-1]):
        break; 
    for i in range(len(theta)):
        P=leg_diff(order,np.cos(theta[i]))[0]/np.sin(theta[i])
        #print(P)
        Pd=(-np.sin(theta[i])*leg_diff(order,np.cos(theta[i]))[1])
        #print(Pd)
        for j in range(len(rad)):
            if radius[1]<rad[j] and rad[j]<radius[0]:
                E_phiimag=(sol_list[order-1][2]*hankel1wow(Beta[1],rad[j],order)[1]+sol_list[order-1][4]*hankel2wow(Beta[1],rad[j],order)[1])*P
                print(order)
                print(sol_list[order-1][4])
                print(sol_list[order-1][2])
                E_phireal=(sol_list[order-1][3]*hankel1wow(Beta[1],rad[j],order)[0]+sol_list[order-1][5]*hankel2wow(Beta[1],rad[j],order)[0])*Pd
                E_phi[i][j]+=(constimagphi[1][j]*E_phiimag+constrealphi[1][j]*E_phireal)
            elif radius[2]<rad[j] and rad[j]<radius[1]:
                E_phiimag=(sol_list[order-1][6]*hankel1wow(Beta[2],rad[j],order)[1]+sol_list[order-1][8]*hankel2wow(Beta[2],rad[j],order)[1])*P
                E_phireal=(sol_list[order-1][7]*hankel1wow(Beta[2],rad[j],order)[0]+sol_list[order-1][9]*hankel2wow(Beta[2],rad[j],order)[0])*Pd
                E_phi[i][j]+=(constimagphi[2][j]*E_phiimag+constrealphi[2][j]*E_phireal)
            elif rad[j]<radius[2]:
                E_phiimag=(sol_list[order-1][10]*besselwow(Beta[3],rad[j],order)[1])*P
                E_phireal=(sol_list[order-1][11]*besselwow(Beta[3],rad[j],order)[0])*Pd
                E_phi[i][j]+=(constimagphi[3][j]*E_phiimag+constrealphi[3][j]*E_phireal)
            else:
                #theta
                #E_thetaimag=(a_n*besselwow(Beta[0],rad[j],order)[1]+sol_list[order-1][0]*hankel2wow(Beta[0],rad[j],order)[1])*leg_diff(order,np.cos(theta[i]))[1]
                #E_thetareal=(a_n*besselwow(Beta[0],rad[j],order)[0]+sol_list[order-1][1]*hankel2wow(Beta[0],rad[j],order)[0])*leg_diff(order,np.cos(theta[i]))[0]
                #E_theta[i][j]+=(constimagtheta[j]*E_thetaimag+constrealtheta[j][i]*E_thetareal).item()
                #phi
                E_phiimag=(a_n*besselwow(Beta[0],rad[j],order)[1]+sol_list[order-1][0]*hankel2wow(Beta[0],rad[j],order)[1])*P
                E_phireal=(a_n*besselwow(Beta[0],rad[j],order)[0]+sol_list[order-1][1]*hankel2wow(Beta[0],rad[j],order)[0])*Pd
                E_phi[i][j]+=(constimagphi[0][j]*E_phiimag+constrealphi[0][j]*E_phireal)

                #radial
                #E_radimag
                #E_radeal
                #E_rad

            


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
