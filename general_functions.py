import numpy as np
import math as ma
import matplotlib.pyplot as plt
import matplotlib.colors as cl

au_m=1.496e11 # m
mp_kg=1.67e-27 # kg
Mearth_kg=5.972e24 # kg
year_s= 3.154e7 # s

def number_to_text(number):

    if number<=0.0:
        return str(number)
    exp=ma.floor(np.log10(number))
    factor=number/(10.0**(exp))
    if exp==0.0:
        if factor==1.0:
            return r'1'
        elif factor-int(factor)==0.0:
            return r'%1.0f'%(factor)
        else:
            return r'%1.1f'%(factor)
    else:
        if str(factor)[:3]=='1.0':
            return r'$10^{%1.0f}$'%(exp)
        elif factor-int(factor)==0.0:
            return r'$%1.0f\times10^{%1.0f}$'%(factor,exp)
        else:
            return r'$%1.1f\times10^{%1.0f}$'%(factor,exp)




def fcolor_black_white(i,N):

    l=i*1.0/(N-1)
    rgb=[l, l, l]

    return cl.colorConverter.to_rgb(rgb)
    
def fcolor_blue_red(i,N):

    rgb=[i*1.0/(N-1), 0.0, 1.0-i*1.0/(N-1)]

    return cl.colorConverter.to_rgb(rgb)

def fcolor_green_yellow(i,N):

    cmap=plt.get_cmap('viridis')

    x=i*1./(N-1)

    return cmap(x)

def fcolor_plasma(i,N):

    cmap=plt.get_cmap('plasma')

    x=i*1./(N-1)

    return cmap(x)


def get_last2d(data):
    if data.ndim == 2:
        return data[:]
    if data.ndim == 3:
        return data[0, :]
    if data.ndim == 4:
        return data[0, 0, :]
    
def get_last3d(data):
    if data.ndim == 2:
        return -1.0
    if data.ndim == 3:
        return data[:]
    if data.ndim == 4:
        return data[0, :]


def power_law_dist(xmin, xmax,alpha, N):

    if alpha==-1.0: sys.exit(0)
    u=np.random.uniform(0.0, 1.0,N)
    beta=1.0+alpha
    return ( (xmax**beta-xmin**beta)*u +xmin**beta  )**(1./beta)


def M_to_L(Mstar): # stellar mass to stellar L MS

    if hasattr(Mstar,"__len__"):
        L=np.zeros(Mstar.shape[0])
        L[Mstar<0.43]=0.23*Mstar[Mstar<0.43]**2.3
        mask2= ((Mstar>=0.43))# & (M<2)).
        L[mask2]=Mstar[mask2]**4.
        mask3= (Mstar>=2.) & (Mstar<20.)
        L[mask3]=1.4*Mstar[mask3]**3.5
        L[Mstar>55.]=3.2e4*Mstar[Mstar>55.]
        
        
    else:
        L=0.0
        if Mstar<0.45:
            L=0.23*Mstar**2.3
        elif Mstar<2.:
            L=Mstar**4.
        elif Mstar<20.:
            L=1.4*Mstar**3.5
        else:
            L=3.2e4*Mstar

    return L

def f_H(r, Lstar, Mstar, mu=2.): # disc scale height based on radius, luminosity and stellar mass assuming blackbody temperature

    return 0.05*r**(1.25) * Lstar**(1./8.) * Mstar**(-0.5) * mu**(-0.5) # in au



def mean_free_path(cross_section, Sigma, r, L, M, mu):


    H=f_H(r, L, M, mu=mu) # au
    n_au= Sigma*Mearth_kg/(np.sqrt(2.*np.pi)*H*mu*mp_kg) # au-3
    
    n_mks= n_au/(au_m**3) # in m-3
    
    return 1./(cross_section*n_mks) # in m
