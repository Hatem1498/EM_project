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
    return normal[1:], derivative[1:]

def hankel1wow(B, r, n):
    jn, jn_p = riccati_jn(n, B * r)
    yn, yn_p = riccati_yn(n, B * r)
    
    normal = jn + 1j * yn
    derivative = jn_p + 1j * yn_p
    
    return normal[1:], derivative[1:]

def besselwow(B, r, n):
    normal, derivative = riccati_jn(n, B * r)
    return normal[1:], derivative[1:]

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
sol=A.LUsolve(b)


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
solnumber=sp.lambdify(flat_vars,sol,modules=['scipy','numpy'])


#defining problem
eps_list = np.array([1,4,1,50])
freq = 10.08e9 
layers = 3
radius = np.array([12e-3, 4e-3, 3e-3])
n=68
#-0.99999*np.pi
theta=np.linspace(0,2*np.pi,160)
phi=np.pi


#setting up lists
sol_list=np.zeros((n,4*layers,1), dtype='complex_')
RCS_list=np.zeros((len(theta),1), dtype='complex_')

#calculating relevant constants
lambda_list = 3e8/np.sqrt(eps_list)/freq
Beta=2*np.pi/lambda_list
etav=eta_0/np.sqrt(eps_list)

#error check
b=np.array(Bcal(n,Beta,radius),dtype='complex_')

sumAtheta=np.zeros(len(theta),dtype='complex_')
sumAphi=np.zeros(len(theta),dtype='complex_')
amount=math.ceil((2*np.pi/lambda_list[0])*radius[0]+10)
for order in range(1,amount):
    #evaluating our expression
    sol_list[order-1]=solnumber(Beta[0]*radius[0], Beta[1]*radius[0], Beta[1]*radius[1],  Beta[2]*radius[1],  Beta[2]*radius[2],  Beta[3]*radius[2],*Beta,*radius,*etav,order )
    print(sol_list[order-1])
    #check for numerical errors
    if all(abs(x)<1e-10 for x in sol_list[order-1]):
        break; 
    
    

    
    for i in range(len(theta)):
        A_theta = (1j**order*( sol_list[order-1][0]*np.sin(theta[i])*leg_diff(order, np.cos(theta[i]))[1]-sol_list[order-1][1]*leg_diff(order, np.cos(theta[i]))[0]/np.sin(theta[i]) ))
        
        A_phi =  1j**order*( sol_list[order-1][0]*leg_diff(order, np.cos(theta[i]))[0]/np.sin(theta[i])-sol_list[order-1][1]*np.sin(theta[i])*leg_diff(order, np.cos(theta[i]))[1] )
        sumAtheta[i]+=A_theta.item()
        
        sumAphi[i]+=A_phi.item()
    #print(sumAphi)
    #print(sumAtheta)
RCS = ((lambda_list[0]**2)/(np.pi))*((np.cos(phi)**2)*(abs(sumAtheta)**2)+(np.sin(phi)**2)*abs(sumAphi)**2)




#print("Beta",Beta)
#print("sol",sol_list[0][0][1])
#print("b_real",breal)
#sol_list=np.zeros((len(freq),n-1,4*layers,1), dtype='complex_')
#print("dif",sol_list[0][:][1][0]-breal)
#print(radius[0])
#print(sol_list)

lam=3e8/freq

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta,10*np.log10(np.absolute(RCS)/(np.pi*radius[0]**2)) )
#plt.plot(theta,10*np.log10(np.absolute(RCS)/(lam**2)))

plt.show()
 
