import numpy as np
from scipy.special import riccati_jn, riccati_yn, legendre, lpmv, lpmn
import sympy as sp
import matplotlib.pyplot as plt
eps_0 = 8.854*10e-12
mu_0 =  4*np.pi*10e-7
eta_0 = np.sqrt(mu_0/eps_0)


def hankel2(x,n):
     
    return (sp.jn(n,x)-sp.I*sp.yn(n,x))*x

def hankel1(x,n):
    return (sp.jn(n,x)+sp.I*sp.yn(n,x))*x



def bessel(x,n):
    return sp.jn(n,x)*x

def leg_diff(n, x):
    normal, derivative = lpmn(1, n, x)
    return derivative[1, n]



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

A=populate_Amatrix()
b=populate_Bmatrix()
print("matrix passed")
#sp.pprint(A, num_columns=10_000)
print(A)
print(b.shape)
sol=A.LUsolve(b)

E_0=1
print(sol[1])



#A_theta = lambda theta: np.abs((1j**n*( sol[0]*np.sin(theta)*leg_diff(n, np.cos(theta))-sol[1]*lpmv(1, n, np.cos(theta))/np.sin(theta) )))**2
#A_phi = lambda theta: np.abs( 1j**n*( sol[0]*lpmv(1, n, np.cos(theta))/np.sin(theta)-sol[1]*np.sin(theta)*leg_diff(n, np.cos(theta)) ) )**2
#RCS = lambda phi, theta: lambda_list[0]**2/np.pi*(np.cos(phi)**2*A_theta(theta)+np.sin(phi)**2*A_phi(theta))
    
#sum=sum+RCS(0,90)
#y_axis[i]=sum    

#lambda_0 = lambda f: 3*10e8/f
#lam=3*10e8/frequencies
#plt.plot(frequencies,10*np.log10(y_axis/(np.pi*r[0]**2)))
#plt.plot(frequencies,10*np.log10(y_axis/(np.pi*r[0]**2)))
#plt.show()
#Etheta = lambda r,theta,phi: (-1j*E_0/(w*eps_list[0]*r))(B[0]/w)*np.cos(phi)*(a_n*bessel(B[0],r,n)+sol[0]*hankel2(B[0],r,n))*legendre(cos())
