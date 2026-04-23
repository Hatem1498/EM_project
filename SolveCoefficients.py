import numpy as np
from scipy.special import riccati_jn, riccati_yn, legendre, lpmv, lpmn,spherical_jn,spherical_yn
import matplotlib.pyplot as plt
eps_0 = 8.854*10e-12
mu_0 =  4*np.pi*10e-7
eta_0 = np.sqrt(mu_0/eps_0)



def hankel2(B,r,n):
    normal ,derivative= riccati_jn(n, B*r) - 1j * np.array(riccati_yn(n, B*r) )
    return normal[n] ,derivative[n]

def hankel1(B,r,n):
    normal ,derivative= riccati_jn(n, B*r) + 1j * np.array(riccati_yn(n, B*r) )
    return normal[n] ,derivative[n]



def bessel(B,r,n):
    normal ,derivative=riccati_jn(n, B*r)
    return normal[n] ,derivative[n]

def leg_diff(n, x):
    normal, derivative = lpmn(1, n, x)
    return derivative[1, n]



def populate_Amatrix(eps_list,frequency,n, r, lambda_list):


    eta=eta_0/np.sqrt(eps_list)
    B=2*np.pi/lambda_list
    layers=3
    A=np.zeros((4*layers,4*layers), dtype='complex_')

    #Electric boundary imag
    A[0][0] = -B[1]*hankel2(B[0],r[0],n)[1]
    A[0][2] = B[0]*bessel(B[0],r[0],n)[1]
    A[0][4] = B[0]*hankel2(B[0],r[0],n)[1]
    
    #Electric boundary real
    A[1][1] = -B[1]*hankel2(B[1],r[0],n)[0]
    A[1][3] = B[0]*hankel1(B[0],r[0],n)[0]
    A[1][5] = B[0]*hankel2(B[0],r[0],n)[0]
    
    #magnetic boundary real
    A[2][0] = -B[1]*eta[1]*hankel2(B[1],r[0],n)[0]
    A[2][2] = B[0]*eta[0]*hankel1(B[0],r[0],n)[0]
    A[2][4] = B[0]*eta[0]*hankel2(B[0],r[0],n)[0]
    #magnetic boundary imag
    A[3][1] = -B[1]*eta[1]*hankel2(B[1],r[0],n)[1]
    A[3][3] = B[0]*eta[0]*hankel1(B[0],r[0],n)[1]
    A[3][5] = B[0]*eta[0]*hankel2(B[0],r[0],n)[1]
    #Electric boundary imag
    A[4][2] = B[2]*hankel1(B[1], r[1], n)[1]
    A[4][4] = B[2]*hankel2(B[1], r[1], n)[1]
    A[4][6] = -B[1]*hankel1(B[2],r[1], n)[1]
    A[4][8] = -B[1]*hankel2(B[2],r[1], n)[1]
    #Electric boundary real
    A[5][3] = B[2]*hankel1(B[1], r[1], n)[0]
    A[5][5] = B[2]*hankel2(B[1], r[1], n)[0]
    A[5][7] = -B[1]*hankel1(B[2],r[1], n)[0]
    A[5][9] = -B[1]*hankel2(B[2],r[1], n)[0]
    #magnetic boundary real
    A[6][2] = B[2]*eta[2]*hankel1(B[1], r[1], n)[0]
    A[6][4] = B[2]*eta[2]*hankel2(B[1], r[1], n)[0]
    A[6][6] = -B[1]*eta[1]*hankel1(B[2],r[1], n)[0]
    A[6][8] = -B[1]*eta[1]*hankel2(B[2],r[1], n)[0]
    #magnetic boundary imag
    A[7][3] = B[2]*eta[2]*hankel1(B[1], r[1], n)[1]
    A[7][5] = B[2]*eta[2]*hankel2(B[1], r[1], n)[1]
    A[7][7] = -B[1]*eta[1]*hankel1(B[2],r[1], n)[1]
    A[7][9] = -B[1]*eta[1]*hankel2(B[2],r[1], n)[1]
    #Electric boundary imag
    A[8][6] = B[3]*hankel1(B[2], r[2], n)[1]
    A[8][8] = B[3]*hankel2(B[2], r[2], n)[1]
    A[8][10] = -B[2]*bessel(B[3], r[2], n)[1]
    #Electric boundary real
    A[9][7] =  B[3]*hankel1(B[2], r[2], n)[0]
    A[9][9] =  B[3]*hankel2(B[2], r[2], n)[0]
    A[9][11] = -B[2]*bessel(B[3], r[2], n)[0]
    #magnetic boundary real
    A[10][6] = B[3]*eta[3]*hankel1(B[2], r[2], n)[0]
    A[10][8] = B[3]*eta[3]*hankel2(B[2], r[2], n)[0]
    A[10][10] = -B[2]*eta[2]*bessel(B[3], r[2], n)[0]
    #magnetic boundary imag
    A[11][7] =  B[3]*eta[3]*hankel1(B[2], r[2], n)[1]
    A[11][9] = B[3]*eta[3]*hankel2(B[2], r[2], n)[1]
    A[11][11] = -B[2]*eta[2]*bessel(B[3], r[2], n)[1]
    
    return A

def populate_Bmatrix(eps_list,frequency, n, r,lambda_list):
    eta=eta_0/np.sqrt(eps_list)
    B=2*np.pi/lambda_list
    layers=3
    b=np.zeros((4*layers), dtype='complex_')
    
    a_n = (1j**(-n))*((2*n+1)/(n*(n+1)))
    b[0]=B[1]*bessel(B[0],r[0],n)[1]
    b[1]=B[1]*bessel(B[0],r[0],n)[0]
    b[2]=B[1]*eta[1]*bessel(B[0],r[0],n)[0]
    b[3]=B[1]*eta[1]*bessel(B[0],r[0],n)[1]
    b=b*a_n

    return b
eps_list = np.array([1,2.56,2.56,2.56])
freq = 10*10e9
layers = 3
r = np.array([12e-3, 4e-3, 3e-3])
#A=populate_Amatrix(eps_list, freq, n, r)
#b=populate_Bmatrix(eps_list,freq, 1, r)
#sol=np.linalg.solve(A,np.transpose(b))
E_0=1


frequencies = np.linspace(1*10e9,300*10e9 , 200)

lambda_list = np.zeros( layers*4)
y_axis= np.zeros(len(frequencies))
for i in range(len(frequencies)):
    sum=0
    for j in range(1,2):
        n=j
        lambda_list = 3*10e8/np.sqrt(eps_list)/frequencies[i]
        #print(lambda_list)
        B=2*np.pi/lambda_list
        w = 2*np.pi*freq
        A=populate_Amatrix(eps_list,frequencies[i],n,r,lambda_list)
        b=populate_Bmatrix(eps_list,frequencies[i],n,r,lambda_list)
        sol=np.linalg.solve(A,b)
        A_theta = lambda theta: np.abs((1j**n*( sol[0]*np.sin(theta)*leg_diff(n, np.cos(theta))-sol[1]*lpmv(1, n, np.cos(theta))/np.sin(theta) )))**2
        A_phi = lambda theta: np.abs( 1j**n*( sol[0]*lpmv(1, n, np.cos(theta))/np.sin(theta)-sol[1]*np.sin(theta)*leg_diff(n, np.cos(theta)) ) )**2
        print("B",B[0])
        print("n",n)
        print("frequency",frequencies[i])
        print("sol",sol)
        RCS = lambda phi, theta: lambda_list[0]**2/np.pi*(np.cos(phi)**2*A_theta(theta)+np.sin(phi)**2*A_phi(theta))
        
        sum=sum+RCS(0,90)
    y_axis[i]=sum    

#lambda_0 = lambda f: 3*10e8/f
lam=3*10e8/frequencies
plt.plot(r[2]/lam,10*np.log10(y_axis/(np.pi*r[2]**2)))
plt.show()
#Etheta = lambda r,theta,phi: (-1j*E_0/(w*eps_list[0]*r))(B[0]/w)*np.cos(phi)*(a_n*bessel(B[0],r,n)+sol[0]*hankel2(B[0],r,n))*legendre(cos())
