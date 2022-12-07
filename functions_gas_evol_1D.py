import numpy as np
from scipy import interpolate

# import functions_gas_evol_0D as fgas
#from tqdm import tqdm


#### Physical constants

# constants
mp=1.6726219e-27 # kg
m_c1=12.0*mp # kg/molecule
m_co= 28.0*mp # kg/molecule
muc1co=(m_c1/m_co)
sigma_c1=1.6e-17 # cm2
sigma_co=1/(1.15e15) # cm2 # when  shielding is 0.368 = exp(1)
tphCO0=120.0 # CO photodissociation timescale

Mearth=5.9e24 # kg
Msun = 2.e30 # kg
au_cm=1.496e13 # cm
au_m=1.496e11 # m
G = 6.67408e-11 # mks
kb = 1.38064852e-23 #mks
year_s = 3.154e7 # seconds


### CO PHOTODISSOCIATION PHOTON COUNTING

try:
    SCO_grid=np.loadtxt('./Sigma_CO_Mearth_au2.txt')
    SC1_grid=np.loadtxt('./Sigma_C1_Mearth_au2.txt')
    tauCO_grid=np.loadtxt('./tau_CO_yr.txt')
    log10tau_interp=interpolate.RectBivariateSpline( np.log10(SC1_grid),np.log10(SCO_grid), np.log10(tauCO_grid)) # x and y must be swaped, i.e. (y,x) https://github.com/scipy/scipy/issues/3164
    
    # log10tau_interp=interpolate.interp2d(np.log10(SCO_grid), np.log10(SC1_grid), np.log10(tauCO_grid))

    
    # N=200
    # NCOs2=np.logspace(1, 30, N) # cm-2
    # NCs2=np.logspace(5, 30, N)  # cm-2

    # Sigma_CO2=NCOs2*m_co/Mearth*au_cm**2.
    # Sigma_C12=NCs2*m_c1/Mearth*au_cm**2.
    # tau2D2=10**(log10tau_interp(np.log10(Sigma_CO2),np.log10(Sigma_C12)))
    # print tau2D2

except:
    print('Interpolaiton of CO photodissociation from photon counting did not work')

####### CO PHOTODISSOCIATION
# Visser+2009

kCOs=[1.0, 0.9405, 0.7046, 0.4015, 0.09964, 0.01567, 0.003162, 0.0004839]
NCOs=[1.0, 1.e13, 1.e14, 1.e15, 1.e16, 1.e17, 1.e18, 1.e19]
logfkCO = interpolate.interp1d(np.log10(NCOs), np.log10(kCOs))
slope=np.log(kCOs[-1]/kCOs[-2])/np.log(NCOs[-1]/NCOs[-2])

    

#### FUNCTIONS

#### general

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

def power_law_dist(xmin, xmax,alpha, N):

    if alpha==-1.0: sys.exit(0)
    u=np.random.uniform(0.0, 1.0,N)
    beta=1.0+alpha
    return ( (xmax**beta-xmin**beta)*u +xmin**beta  )**(1./beta)
    

## Photodissociation

def selfshielding_CO(NCO):#, NCOs, logfkCO, slope):

    if isinstance(NCO, np.ndarray):
        kco=np.ones(len(NCO))
        # for j in xrange(len(NCO)):
        #     if NCO[j]>=NCOs[0] and NCO[j]<=NCOs[-1]:
        #         kco[j]=10.0**logfkCO(np.log10(NCO[j]))
        #     elif NCO[j]<NCOs[0]:
        #         kco[j]=1.0
        #     else:
        #         kco[j]=kCOs[-1]*(NCO[j]/NCOs[-1])**slope
                
        mask1=(NCO>=NCOs[0]) & (NCO<=NCOs[-1])
        kco[ mask1]=10.0**logfkCO(np.log10(NCO[mask1]))
        mask2=( NCO<NCOs[0])
        kco[mask2]=1.0
        mask3=(NCO>NCOs[-1])
        kco[mask3]=kCOs[-1]*(NCO[mask3]/NCOs[-1])**slope

    else:
        if NCO>=NCOs[0] and NCO<=NCOs[-1]:
            kco=10.0**logfkCO(np.log10(NCO))
        elif NCO<NCOs[0]:
            kco=1.0
        else:
            kco=kCOs[-1]*(NCO/NCOs[-1])**slope
    return kco

def tau_CO2(Sigma_CO, Sigma_C1): # simple approximation
    #area=2*np.pi*r*dr*au_cm**2.0 # cm2
    
    NC1=Sigma_C1/2.*Mearth/m_c1/au_cm**2.0
    NCO=Sigma_CO/2.*Mearth/m_co/au_cm**2.0
    
    return 120.0 * np.exp( sigma_c1*NC1)/ selfshielding_CO(NCO) # yr

def tau_CO3(Sigma_CO, Sigma_C1): # interpolate calculations based on photon counting

    tau=np.ones(Sigma_CO.shape[0])*130. # unshielded
    # to avoid nans we use a floor value for sigmas of 1e-50
    Sigma_COp=Sigma_CO*1. 
    Sigma_C1p=Sigma_C1*1.
    Sigma_COp[Sigma_COp<1.0e-50]=1.0e-50
    Sigma_C1p[Sigma_C1p<1.0e-50]=1.0e-50

    # mask=(Sigma_CO>1.0e-100) & (Sigma_C1>1.0e-100) # if not we get error in interpolation function and we get NaNs
    # if Sigma_CO[mask].shape[0]>0:
        # tau[mask]=10**(log10tau_interp(np.log10(Sigma_C1[mask]),np.log10(Sigma_CO[mask]), grid=False)) # yr, it must be called with C1 first because of column and raws definition. Tested with jupyter notebook and also here https://github.com/scipy/scipy/issues/3164

    tau=10**(log10tau_interp(np.log10(Sigma_C1),np.log10(Sigma_CO), grid=False)) # yr, it must be called with C1 first because of column and raws definition. Tested with jupyter notebook and also here https://github.com/scipy/scipy/issues/3164
    return tau # yr



## viscous evolution
def tau_vis(r, dr, alpha, cs, Mstar):
    
    Omega=np.sqrt(G*Mstar*Msun/((r*au_m)**3.0)) # s
    return (dr*au_m)**2.0*Omega/(alpha*cs**2.)/year_s #/3.0

def tau_vis_local(r, dr, alpha, cs, Mstar):
    
    Omega=np.sqrt(G*Mstar*Msun/((r*au_m)**3.0)) # s
    return (r*au_m)**2.0*Omega/(alpha*cs**2.)/year_s/12. #/3.0


def tau_vis2(r, dr, alpha, cs, Mstar):

    Omega=np.sqrt(G*Mstar*Msun/((r*au_m)**3.0)) # s
    return (r*au_m)**2.0*Omega/(alpha*cs**2.)/year_s #/3.0



def Sigma_dot_vis(Sigmas, Nr, rsi, rhalfsi, hs, nus_au2_yr):
  

    ########## CALCULATE VR*Sigma=F1
    
    Sigma_tot=Sigmas[0,:]+Sigmas[1,:]*(1.+4./3.) # CO+C+O
    eps=np.ones((2,Nr))*0.5
    mask_m=Sigma_tot>0.0
    eps[0,mask_m]=Sigmas[0,mask_m]/Sigma_tot[mask_m]
    eps[1,mask_m]=Sigmas[1,mask_m]/Sigma_tot[mask_m]
    

    G1s=Sigma_tot*nus_au2_yr*np.sqrt(rsi) # Nr
    Sigma_vr_halfs=-3.0*(G1s[1:]-G1s[:-1])/(rsi[1:]-rsi[:-1])/np.sqrt(rhalfsi[1:-1]) # Nr-1
    
    
    ############## CALCULATE dSIGMA/dT
    eps_halfs=np.zeros((2,Nr-1))
    eps_halfs[:,:]=np.where(Sigma_vr_halfs[:]>0.0, eps[:,:-1], eps[:,1:])
    
    G2s=rhalfsi[1:-1]*Sigma_vr_halfs  # Nr-1
    G3s=G2s*eps_halfs    #  2x(Nr-1)
    Sdot=np.zeros((2,Nr))
    Sdot[:,1:-1]=-(G3s[:,1:]-G3s[:,:-1])*2./(rhalfsi[2:-1]**2.-rhalfsi[1:-2]**2.) # Nr-2

    ### inner boundary condition
    #Fph=G3s[:,0] # F_{+1/2} 2 dim
    #Sigma_vr_1=-3.0*(G1s[2]-G1s[0])/(rs[2]-rs[0])/np.sqrt(rs[1]) 
    #F0=Sigma_vr_1*rs[1]/rs[0]*eps[:,0] # 2 dim
    #Fmh=np.where(Sigma_vr_halfs[0]>0.0, np.array([0.0,0.0]),F0) # 2 dim
    #Sdot[:,0]=-(rhalfs[1]*Fph - rhalfs[0]*Fmh)*2.0/(rhalfs[1]**2.0-rhalfs[0]**2.0)
    ### outer boundary condition
    
        
    return Sdot, Sigma_vr_halfs

# def Diffusion(Sigmas, Nr, rs, rhalfs, hs, nus_au2_yr):

#     Sigma_tot=Sigmas[0,:]+Sigmas[1,:]*(28./12.) # CO+C+O
#     eps=np.ones((2,Nr))*0.5
#     eps_dot=np.zeros((2,Nr))
#     mask_m=Sigma_tot>0.
#     eps[0,mask_m]=Sigmas[0,mask_m]/Sigma_tot[mask_m]
#     eps[1,mask_m]=Sigmas[1,mask_m]/Sigma_tot[mask_m]

#     # # geometric average
#     # eps=np.sqrt(eps[1:]*eps[:-1]) # Nr-1
    
#     eps_dot[:,1:-1]=(eps[:,2:]-eps[:,:-2])/(2*hs[1:-1])
#     eps_dot[:,0]=(eps[:,1]-eps[:,0])/hs[0]
#     eps_dot[:,-1]=(eps[:,-1]-eps[:,-2])/hs[-1]

#     F=rs*nus_au2_yr*Sigma_tot*eps_dot # Nr

#     Sdot_diff=np.zeros((2,Nr))    
#     Sdot_diff[:,1:]= (F[:,1:]-F[:,:-1])/(rs[1:]-rs[:-1])/(rs[1:]) # Nr-2
#     Sdot_diff[:,-1]= (F[:,-1]-F[:,-2])/(rs[-1]-rs[-2])/(rs[-1]) # Nr-2

#     # ### OLD IN 2020 PAPER. F was defined with sigmas instead of sigma_tot
#     # F=rs*nus_au2_yr*Sigmas*eps_dot # Nr
#     # Sdot_diff=np.zeros((2,Nr))   
#     # Sdot_diff[:,1:]= (F[:,1:]-F[:,:-1])/(rs[1:]-rs[:-1])/(rs[1:]) # Nr-2
#     # Sdot_diff[:,-1]= (F[:,-1]-F[:,-2])/(rs[-1]-rs[-2])/(rs[-1]) # Nr-2

#     # Sdot_diff[:,-1]=0.

#     return Sdot_diff

def Diffusion(Sigmas, Nr, rs, rhalfs, hs, nus_au2_yr):

    Sigma_tot=Sigmas[0,:]+Sigmas[1,:]*(28./12.) # CO+C+O
    eps=np.ones((2,Nr))*0.5
    mask_m=Sigma_tot>0.
    eps[0,mask_m]=Sigmas[0,mask_m]/Sigma_tot[mask_m]
    eps[1,mask_m]=Sigmas[1,mask_m]/Sigma_tot[mask_m]

    # # geometric average
    eps_half=np.sqrt(eps[:,1:]*eps[:,:-1]) # Nr-1 at cell boundaries
    #Sigma_tot_halfs=np.sqrt(Sigma_tot[1:]*Sigma_tot[:-1]) # Nr-1 at cell boundaries 
    # eps_dot_halfs=np.zeros((2,Nr-1)) # Nr-1
    eps_dot=(eps_half[:,1:]-eps_half[:,:-1])/(hs[1:-1]) # Nr-2 at cell centers

    # eps_dot_halfs[:,0]=(eps_half[:,1]-eps_half[:,0])/hs[0]
    # eps_dot_halfs[:,-1]=(eps[:,-1]-eps[:,-2])/hs[-1]

    
    F=rs[1:-1]*nus_au2_yr[1:-1]*Sigma_tot[1:-1]*eps_dot # Nr-2

    Sdot_diff=np.zeros((2,Nr))    
    Sdot_diff[:,1:-2]= (F[:,1:]-F[:,:-1])/(rs[2:-1]-rs[1:-2])/(rs[1:-2]) # Nr-3
    
    # Sdot_diff[:,-1]= (F[:,-1]-F[:,-2])/(rs[-1]-rs[-2])/(rs[-1]) # Nr-2

    # ### OLD IN 2020 PAPER. F was defined with sigmas instead of sigma_tot
    # F=rs*nus_au2_yr*Sigmas*eps_dot # Nr
    # Sdot_diff=np.zeros((2,Nr))   
    # Sdot_diff[:,1:]= (F[:,1:]-F[:,:-1])/(rs[1:]-rs[:-1])/(rs[1:]) # Nr-2
    # Sdot_diff[:,-1]= (F[:,-1]-F[:,-2])/(rs[-1]-rs[-2])/(rs[-1]) # Nr-2

    # Sdot_diff[:,-1]=0.

    return Sdot_diff
    
def Sig_dot_p_box(rs, r0, width, Mdot, mask_belt):
    
    sigdot_CO=np.ones(Nr)*Mdot/(np.pi*((r0+width/2.)**2.-(r0-width/2.)**2.))
    sigdot_CO[mask_belt]=0.0
    
    return sigdot_CO


def Sig_dot_p_gauss(rs, hs, r0, sig_g, Mdot, mask_belt):
    
    Sdot_comets=np.zeros(len(rs))

    ### no mask
    # Sdot_comets=np.exp( -2* (rs-r0)**2.0 / (2.*sig_g**2.) ) # /(np.sqrt(2.*np.pi)*sig_g)/(2.*np.pi*rs) # factor 2 inside exponential is to make Mdot prop to Sigma**2 
    # Sdot_comets=Mdot*Sdot_comets/(2.*np.pi*np.sum(Sdot_comets*rs*hs))

    ### with mask to avoid CO input beyond or within the belt
    Sdot_comets[mask_belt]=np.exp( -1* (rs[mask_belt]-r0)**2.0 / (2.*sig_g**2.) ) # /(np.sqrt(2.*np.pi)*sig_g)/(2.*np.pi*rs[mask_belt]) # factor 2 inside exponential is to make Mdot prop to Sigma**2 
    Sdot_comets[mask_belt]=Mdot*Sdot_comets[mask_belt]/(2.*np.pi*np.sum(Sdot_comets[mask_belt]*rs[mask_belt]*hs[mask_belt]))
    return Sdot_comets

def Sigma_next(Sigma_prev, Nr, rs, rhalfs, hs, epsilon, r0, width, Mdot, nus_au2_yr, mask_belt, diffusion=0, photodissociation=True):
    
    ###########################################
    ################ viscous evolution
    ###########################################
    Sdot_vis, Sigma_vr_halfs=Sigma_dot_vis(Sigma_prev,  Nr, rs, rhalfs, hs, nus_au2_yr)
    Snext= Sigma_prev + epsilon*Sdot_vis # viscous evolution

    
    ###########################################
    ############### inner boundary condition
    ###########################################

    #if np.all(Snext[:,2])>0.0:
    #    Snext[:,0]=np.minimum(Snext[:,1]*(nus_au2_yr[1]/nus_au2_yr[0]), Snext[:,1]*(rs[0]/rs[1])**(np.log(Snext[:,2]/Snext[:,1])/np.log(rs[2]/rs[1])))
    #else: 
    Snext[:,0]=Snext[:,1]*(nus_au2_yr[1]/nus_au2_yr[0]) # constant Mdot
    
    ###########################################
    ############# Outer boundary condition (power law or constant mass)
    ###########################################

    if np.all(Snext[:,-3])>0.0: # minimum between power law and constant Mdot
        Snext[:,-1]=np.minimum(Snext[:,-2]*(nus_au2_yr[-2]/nus_au2_yr[-1]), Snext[:,-2]*(rs[-1]/rs[-2])**(np.log(Snext[:,-2]/Snext[:,-3])/np.log(rs[-2]/rs[-3])))
    else: 
        Snext[:,-1]=Snext[:,-2]*(nus_au2_yr[-2]/nus_au2_yr[-1])

    ###########################################
    ################ diffusion evolution (this has to come after photodissociation and input rate, otherwise we get weird strong wiggles that diverge resulting in nans)
    ###########################################
    if diffusion:
        Snext=Snext+epsilon* Diffusion(Snext, Nr, rs, rhalfs, hs, nus_au2_yr)

    ###########################################
    ############### CO mas input rate
    ###########################################
    Snext2=np.zeros((2, Nr))
    Snext2[0,:]=Snext[0,:]+epsilon*Sig_dot_p_gauss(rs, hs, r0, width, Mdot, mask_belt)
    Snext2[1,:]=Snext[1,:]*1.
    ###########################################
    ############## photodissociation
    ###########################################
    if photodissociation:
        tphCO=tau_CO2(Sigma_prev[0,:], Sigma_prev[1,:])
        Sdot_ph=Sigma_prev[0,:]/tphCO #(Snext[0,:]/tphCO)
        #Sdot_ph_epsilon=Sigma_prev[0,:]*(1.-np.exp(-epsilon/tphCO))   
        Snext2[0,:]=Snext2[0,:]-epsilon*Sdot_ph
        Snext2[1,:]=Snext2[1,:]+epsilon*Sdot_ph*muc1co
        #Snext2[0,Snext2[0,:]<0.0]=0.0


 
    
    Snext2[Snext2[:,:]<0.0]=0.0

    return Snext2


def Sigma_next_fMdot(Sigma_prev, Nr, rs, rhalfs, hs, epsilon, fMdot, args_fMdot, Mdot, nus_au2_yr,  diffusion=0):
    
    ###########################################
    ################ viscous evolution
    ###########################################
    Sdot_vis, Sigma_vr_halfs=Sigma_dot_vis(Sigma_prev,  Nr, rs, rhalfs, hs, nus_au2_yr)
    Snext= Sigma_prev + epsilon*Sdot_vis # viscous evolution


  
    
    ###########################################
    ############### inner boundary condition
    ###########################################

    #if np.all(Snext[:,2])>0.0:
    #    Snext[:,0]=np.minimum(Snext[:,1]*(nus_au2_yr[1]/nus_au2_yr[0]), Snext[:,1]*(rs[0]/rs[1])**(np.log(Snext[:,2]/Snext[:,1])/np.log(rs[2]/rs[1])))
    #else: 
    Snext[:,0]=Snext[:,1]*(nus_au2_yr[1]/nus_au2_yr[0]) # constant Mdot
    
    ###########################################
    ############# Outer boundary condition (power law or constant mass)
    ###########################################

    if np.all(Snext[:,-3])>0.0: # minimum between power law and constant Mdot
        Snext[:,-1]=np.minimum(Snext[:,-2]*(nus_au2_yr[-2]/nus_au2_yr[-1]), Snext[:,-2]*(rs[-1]/rs[-2])**(np.log(Snext[:,-2]/Snext[:,-3])/np.log(rs[-2]/rs[-3])))
    else: 
        Snext[:,-1]=Snext[:,-2]*(nus_au2_yr[-2]/nus_au2_yr[-1])


    ###########################################
    ################ diffusion evolution
    ###########################################
    if diffusion:
        Snext=Snext+epsilon* Diffusion(Snext, Nr, rs, rhalfs, hs, nus_au2_yr)
        
    ###########################################
    ############### CO mas input rate
    ###########################################
    Snext2=np.zeros((2, Nr))

    Snext2[0,:]=Snext[0,:]+epsilon*fMdot(rs, hs, Mdot, *args_fMdot)
    Snext2[1,:]=Snext[1,:]*1.
    ###########################################
    ############## photodissociation
    ###########################################

    tphCO=tau_CO2(Sigma_prev[0,:], Sigma_prev[1,:])
    Sdot_ph=Sigma_prev[0,:]/tphCO #(Snext[0,:]/tphCO)
    #Sdot_ph_epsilon=Sigma_prev[0,:]*(1.-np.exp(-epsilon/tphCO))   
    Snext2[0,:]=Snext2[0,:]-epsilon*Sdot_ph
    Snext2[1,:]=Snext2[1,:]+epsilon*Sdot_ph*muc1co
    #Snext2[0,Snext2[0,:]<0.0]=0.0
    
    return Snext2



def viscous_evolution(ts, epsilon, rs, rhalfs, hs, rbelt, sig_g, Mdot, alpha, Mstar=1.0, Lstar=1.0, Sigma0=np.array([-1.0]), mu0=12.0, dt_skip=1, diffusion=True, photodissociation=True):
    ### 
    Nt=len(ts)
    if isinstance(dt_skip, int) and dt_skip>0:
        if dt_skip>1:  #  skips dt_skip to make arrays smaller
            if (Nt-1)%dt_skip==0:
                Nt2=int((Nt-1)/dt_skip+1)
            else:
                Nt2=int((Nt-1)/dt_skip+2)
        elif dt_skip==1: Nt2=int(Nt)
    else:
        print('not a valid dt_skip')
        sys.exit(0)


    epsilon=ts[1]-ts[0]
    ts2=np.zeros(Nt2)

    Nr=len(rs)
    Sigma_g=np.zeros((2,Nr,Nt2))
    wbelt=sig_g*2*np.sqrt(2.*np.log(2))
    mask_belt=((rs<rbelt+wbelt*2) & (rs>rbelt-wbelt*2))

    
    ## Temperature and angular velocity
    Ts=278.3*(Lstar**0.25)*rs**(-0.5) # K
    Omegas=2.0*np.pi*np.sqrt(Mstar/(rs**3.0)) # 1/yr
    Omegas_s=Omegas/year_s # Omega in s-1
    ## default viscosity
    mus=np.ones(Nr)*mu0
    
    
    if np.shape(Sigma0)==(2,Nr) and np.all(Sigma0>=0.0):
        Sigma_g[:,:,0]=Sigma0
    else:
        print(np.shape(Sigma0), Sigma0>0.0)
    Sigma_temp=Sigma_g[:,:,0]*1.0
    j=1
    for i in range(1,Nt):
        mask_m=np.sum(Sigma_temp, axis=0)>0.0
        mus[mask_m]=(Sigma_temp[0,mask_m]+Sigma_temp[1,mask_m]*(1.+16./12.))/(Sigma_temp[0,mask_m]/28.+Sigma_temp[1,mask_m]/6.) # Sigma+Oxigen/(N)
 
        nus=alpha*kb*Ts/(mus*mp)/(Omegas_s) # m2/s 1.0e10*np.zeros(Nr) #
        nus_au2_yr=nus*year_s/(au_m**2.0) # au2/yr  

        Sigma_temp=Sigma_next(Sigma_temp, Nr, rs, rhalfs, hs, epsilon, rbelt, sig_g, Mdot, nus_au2_yr, mask_belt, diffusion=diffusion, photodissociation=photodissociation)

        if i%dt_skip==0.0 or i==Nt-1:
            Sigma_g[:,:,j]=Sigma_temp*1.
            ts2[j]=ts[i]
            j+=1
    return Sigma_g, ts2




def viscous_evolution_fMdot(ts, epsilon, rs, rhalfs, hs, fMdot, par_fMdot, Mdot,  alpha, Mstar=1.0, Lstar=1.0, Sigma0=np.array([-1.0]), mu0=12.0, dt_skip=1, diffusion=0 ):
    ### fixed/constant Mdot
    Nt=len(ts)
    if isinstance(dt_skip, int) and dt_skip>0:
        if dt_skip>1:  #  skips dt_skip to make arrays smaller
            if (Nt-1)%dt_skip==0:
                Nt2=(Nt-1)/dt_skip+1
            else:
                Nt2=(Nt-1)/dt_skip+2
        elif dt_skip==1: Nt2=Nt
    else:
        print('not a valid dt_skip')
        sys.exit(0)


    epsilon=ts[1]-ts[0]
    ts2=np.zeros(Nt2)

    Nr=len(rs)
    
    Sigma_g=np.zeros((2,Nr,Nt2))
    # wbelt=sig_g*2*np.sqrt(2.*np.log(2))
    # mask_belt=((rs<rbelt+wbelt) & (rs>rbelt-wbelt))

    
    ## Temperature and angular velocity
    Ts=278.3*(Lstar**0.25)*rs**(-0.5) # K
    Omegas=2.0*np.pi*np.sqrt(Mstar/(rs**3.0)) # 1/yr
    Omegas_s=Omegas/year_s # Omega in s-1
    ## default viscosity
    mus=np.ones(Nr)*mu0
    
    
    if np.shape(Sigma0)==(2,Nr) and np.all(Sigma0>=0.0):
        Sigma_g[:,:,0]=Sigma0
    else:
        print(np.shape(Sigma0), Sigma0>0.0)
    Sigma_temp=Sigma_g[:,:,0]*1.0
    j=1
    for i in range(1,Nt):
        mask_m=np.sum(Sigma_temp, axis=0)>0.0
        mus[mask_m]=(Sigma_temp[0,mask_m]+Sigma_temp[1,mask_m]*(1.+16./12.))/(Sigma_temp[0,mask_m]/28.+Sigma_temp[1,mask_m]/6.) # Sigma+Oxigen/(N)
 
        nus=alpha*kb*Ts/(mus*mp)/(Omegas_s) # m2/s 1.0e10*np.zeros(Nr) #
        nus_au2_yr=nus*year_s/(au_m**2.0) # au2/yr  

        Sigma_temp=Sigma_next_fMdot(Sigma_temp, Nr, rs, rhalfs, hs, epsilon, fMdot, par_fMdot, Mdot, nus_au2_yr, diffusion=diffusion)

        if i%dt_skip==0.0 or i==Nt-1:
            Sigma_g[:,:,j]=Sigma_temp*1.
            ts2[j]=ts[i]
            j+=1
    return Sigma_g, ts2



def viscous_evolution_adt(tf, epsilon, rs, rhalfs, hs, rbelt, sig_g, Mdot, alpha, Mstar=1.0, Lstar=1.0, Sigma0=np.array([-1.0]), mu0=12.0, tol=1.0e-3, diffusion=0 ):
    ### with adaptative timestep (based on total masses) it is not faster than fixed timestep.
    Nr=len(rs)
    Ntmax=int(tf/epsilon+1)+1
    Sigma_g=np.zeros((2,Nr, Ntmax))
    wbelt=sig_g*2*np.sqrt(2.*np.log(2))
    mask_belt=((rs<rbelt+wbelt) & (rs>rbelt-wbelt))

    
    ## Temperature and angular velocity
    Ts=278.3*(Lstar**0.25)*rs**(-0.5) # K
    Omegas=2.0*np.pi*np.sqrt(Mstar/(rs**3.0)) # 1/yr
    Omegas_s=Omegas/year_s # Omega in s-1
    ## default viscosity
    mus=np.ones(Nr)*mu0
    
    
    if np.shape(Sigma0)==(2,Nr) and np.all(Sigma0>=0.0):
        Sigma_g[:,:,0]=Sigma0
    else:
        print(np.shape(Sigma0), Sigma0>0.0)

    i=1
    ti=0.0
    ts=[ti]
    dti=epsilon*1.0
    
    while ti<tf:     # for i in xrange(1,Nt):
        mask_m=np.sum(Sigma_g[:,:,i-1], axis=0)>0.0
        mus[mask_m]=(Sigma_g[0,mask_m,i-1]+Sigma_g[1,mask_m,i-1]*(1.+16./12.))/(Sigma_g[0,mask_m, i-1]/28.+Sigma_g[1,mask_m, i-1]/6.) # Sigma+Oxigen/(N)
        nus=alpha*kb*Ts/(mus*mp)/(Omegas_s) # m2/s 1.0e10*np.zeros(Nr) #
        nus_au2_yr=nus*year_s/(au_m**2.0) # au2/yr          
        Sigma_g[:,:,i]=Sigma_next(Sigma_g[:,:,i-1], Nr, rs, rhalfs, hs, dti, rbelt, sig_g, Mdot, nus_au2_yr, mask_belt, diffusion=diffusion)

        # MCO0=np.sum(Sigma_g[0,:,i]*rs*hs)
        # MC10=np.sum(Sigma_g[1,:,i]*rs*hs)
        ### adapt time step
        Sh=Sigma_next(Sigma_g[:,:,i-1], Nr, rs, rhalfs, hs, dti/2., rbelt, sig_g, Mdot, nus_au2_yr, mask_belt, diffusion=diffusion)
        Sf=Sigma_next(Sh, Nr, rs, rhalfs, hs, dti/2., rbelt, sig_g, Mdot, nus_au2_yr, mask_belt, diffusion=diffusion)

        # MCOh=np.sum(Sh[0,:]*rs*hs)
        # MC1h=np.sum(Sh[1,:]*rs*hs)
        # MCOf=np.sum(Sf[0,:]*rs*hs)
        # MC1f=np.sum(Sf[1,:]*rs*hs)

        # calculate difference in the solutions
        mCO=Sigma_g[0,:,i]>0.
        mC1=Sigma_g[1,:,i]>0.
        
        errorCO=np.max(np.abs((Sf[0,mCO]-Sigma_g[0,mCO,i]))/Sigma_g[0,mCO,i])  #MCOf-MCO0
        errorC1=np.max(np.abs((Sf[1,mC1]-Sigma_g[1,mC1,i]))/Sigma_g[1,mC1,i])         #MC1f-MC10
        
        i+=1
        ti+=dti
        ts.append(ti)
        
        # dti=max(0.9*dti*min( max( min(tol*MCO0/abs(errorCO),tol*MC10/abs(errorC1)),0.3),2.0), epsilon) # it does not go below epsilon
        dti=max(0.9*dti*min( max( min(tol/errorCO,tol/errorC1),0.3),2.0), epsilon) # it does not go below epsilon
        dti=min(dti,  0.02*hs[0]**2./nus_au2_yr[0])
    return Sigma_g[:,:,:i], np.array(ts)



def viscous_evolution_evolcoll(ts, epsilon, rs, rhalfs, hs, rbelt, sig_g, Mdots, alpha, Mstar=1.0, Lstar=1.0, Sigma0=np.array([-1.0]), mu0=12.0, dt_skip=1, diffusion=True, photodissociation=True ):
    
    Nt=len(ts)
    if isinstance(dt_skip, int) and dt_skip>0:
        if dt_skip>1:  #  skips dt_skip to make arrays smaller
            if (Nt-1)%dt_skip==0:
                Nt2=(Nt-1)/dt_skip+1
            else:
                Nt2=(Nt-1)/dt_skip+2
        elif dt_skip==1: Nt2=Nt
    else:
        print('not a valid dt_skip')
        sys.exit(0)

        
    epsilon=ts[1]-ts[0]
    ts2=np.zeros(Nt2)

    Nr=len(rs)
    Sigma_g=np.zeros((2,Nr,Nt2))
    wbelt=sig_g*2*np.sqrt(2.*np.log(2))
    mask_belt=((rs<rbelt+wbelt) & (rs>rbelt-wbelt))

    
    ## Temperature and angular velocity
    Ts=278.3*(Lstar**0.25)*rs**(-0.5) # K
    Omegas=2.0*np.pi*np.sqrt(Mstar/(rs**3.0)) # 1/yr
    Omegas_s=Omegas/year_s # Omega in s-1
    ## default viscosity
    mus=np.ones(Nr)*mu0
    
    
    if np.shape(Sigma0)==(2,Nr) and np.all(Sigma0>=0.0):
        Sigma_g[:,:,0]=Sigma0
    else:
        print(np.shape(Sigma0), Sigma0>0.0)
    Sigma_temp=Sigma_g[:,:,0]*1.0
    j=1
    for i in range(1,Nt):
        mask_m=np.sum(Sigma_temp, axis=0)>0.0
        mus[mask_m]=(Sigma_temp[0,mask_m]+Sigma_temp[1,mask_m]*(1.+16./12.))/(Sigma_temp[0,mask_m]/28.+Sigma_temp[1,mask_m]/6.) # Sigma+Oxigen/(N)
        nus=alpha*kb*Ts/(mus*mp)/(Omegas_s) # m2/s 
        nus_au2_yr=nus*year_s/(au_m**2.0) # au2/yr  
        Sigma_temp=Sigma_next(Sigma_temp, Nr, rs, rhalfs, hs, epsilon, rbelt, sig_g, Mdots[i], nus_au2_yr, mask_belt, diffusion=diffusion, photodissociation=photodissociation)
        if i%dt_skip==0.0 or i==Nt-1:
            Sigma_g[:,:,j]=Sigma_temp*1.
            ts2[j]=ts[i]
            j+=1
    return Sigma_g, ts2



def radial_grid_powerlaw(rmin, rmax, Nr, alpha):

    u=np.linspace(rmin**alpha, rmax**alpha, Nr+1) # Nr+1
    rhalfs=u**(1./alpha) # Nr+1
    hs=rhalfs[1:]-rhalfs[:-1] # Nr
    rs=0.5*(rhalfs[1:] + rhalfs[:-1])
    return rs, rhalfs, hs



def N_optim_radial_grid(rmin, rmax, rb, res):

    Nr=10
    f=0
    while True:
        rs, rhalfs, hs = radial_grid_powerlaw(rmin, rmax, Nr, 0.5)  #0.5)
        for i in range(1,Nr):
            if rs[i]>rb:
                dr=rs[i]-rs[i-1]
                break
        if hs[i-1]/rs[i-1]<res:
            break
        else:
            Nr=int(Nr*1.2)
    return Nr





############## COLLISIONS


def f_tc_simple(Mtot, r, dr, Dc=10.0, e=0.05, Qd=150.0, Mstar=1.0): # collisional timescale of largest planetesimal

    return 1.4e-3 * r**(13.0/3) * (dr/r) * Dc * Qd**(5./6.) *e**(-5.0/3.0) * Mstar**(-4.0/3.0)*Mtot**(-1.0) # in yr

def f_G(q,Xc):

    return (Xc**(5.-3*q)-1. ) + (6.*q-10.)*(3.*q-4.)**(-1.)*(Xc**(4.-3.*q) -1. ) + (3.*q-5.)*(3.*q-3.)**(-1.)*(Xc**(3.-3.*q)-1. )

def f_Xc(Qd, r, Mstar, e, I):

    return 1.3e-3*(Qd * r / (Mstar*(1.25*e**2. + I**2.)))**(1./3.)

def f_tc_Xc(Mtot, r, dr, rho=2700.0, Dc=10.0, e=0.05, I=0.05, Qd=150.0, Mstar=1.0, q=11./6.): # collisional timescale of largest planetesimal
    A=(3.8 * rho * r**2.5 * dr * Dc  )/(Mstar**0.5 * Mtot) # yr (error in Eq 9 Wyatt, Smith, Su, Rieke, Greaves et al. 2007, equation is in years)
    B= ( (12.*q - 20.)*( 1.+1.25*(e/I)**2.0 )**(-0.5) )/((18.-9.*q)*f_G(q, f_Xc(Qd, r, Mstar, e, I)))
    return A*B # yr
    

def Mtot_t(Mtot0, t, r, dr,  rho=2700.0, Dc=10.0, e=0.05, I=0.05, Qd=150.0, Mstar=1.0, q=11./6.):
    # t in years
    tc0=f_tc_Xc(Mtot0, r, dr, rho, Dc, e, I, Qd, Mstar, q=q)
    if hasattr(tc0, "__len__"):
        for i in range(len(tc0)):
            if tc0[i]<0.0:
                tc0[i]=f_tc_simple(Mtot0[i], r[i], dr[i],  Dc, e, Qd, Mstar[i])
    else:
        if tc0<0.0:
            tc0=f_tc_simple(Mtot0, r, dr,  Dc, e, Qd, Mstar)
        
    return Mtot0/(1.0+t/tc0) 

def Mtot_t_simple(Mtot0, t, r, dr,  rho=2700.0, Dc=10.0, e=0.05, I=0.05, Qd=150.0, Mstar=1.0, q=11./6.):
    # t in years
    tc0=f_tc_simple(Mtot0, r, dr,  Dc, e, Qd, Mstar)

    return Mtot0/(1.0+t/tc0) 

def Mtotdot_t(Mtot0, t, r, dr, rho=2700.0,  Dc=10.0, e=0.05, I=0.05, Qd=150.0, Mstar=1.0, q=11./6.):
    # t in years
    tc0=f_tc_Xc(Mtot0, r, dr, rho,  Dc, e, I, Qd, Mstar, q=q)

    if hasattr(tc0, "__len__"):
        for i in range(len(tc0)):
            if tc0[i]<0.0:
                tc0[i]=f_tc_simple(Mtot0[i], r[i], dr[i],  Dc, e, Qd, Mstar[i])
    else:
        if tc0<0.0:
            tc0=f_tc_simple(Mtot0, r, dr,  Dc, e, Qd, Mstar)
   
    return Mtot0/(1.0+t/tc0)**2. / tc0 # Mearth/yr


def f_Gamma(Lstar): # Fig 8.3 Nicole Pawellek's thesis

    return 6.42*Lstar**(-0.37)

def f_Dbl(Mstar=1.0, Lstar=1.0, rho=2700.0):

    return 0.8*(Lstar/Mstar)*(2700.0/rho)
