import numpy as np
import functions_gas_evol_0D as fgas
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



#### FUNCTIONS


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

def Difussion(Sigmas, Nr, rs, rhalfs, hs, nus_au2_yr):

    Sigma_tot=Sigmas[0,:]+Sigmas[1,:]*(1.+4./3.) # CO+C+O
    eps=np.ones((2,Nr))*0.5
    eps_dot=np.zeros((2,Nr))
    mask_m=Sigma_tot>0.0
    eps[0,mask_m]=Sigmas[0,mask_m]/Sigma_tot[mask_m]
    eps[1,mask_m]=Sigmas[1,mask_m]/Sigma_tot[mask_m]

    eps_dot[:,1:-1]=(eps[:,2:]-eps[:,:-2])/(2*hs[1:-1])
    eps_dot[:,0]=(eps[:,1]-eps[:,0])/hs[0]
    eps_dot[:,-1]=(eps[:,-1]-eps[:,-2])/hs[-1]

    F=rs*nus_au2_yr*Sigmas*eps_dot # Nr

    Sdot_diff=np.zeros((2,Nr))    
    Sdot_diff[:,1:]= (F[:,1:]-F[:,:-1])/(rs[1:]-rs[:-1])/(rs[1:]) # Nr-2
    Sdot_diff[:,-1]= (F[:,-1]-F[:,-2])/(rs[-1]-rs[-2])/(rs[-1]) # Nr-2
    # Sdot_diff[:,~mask_m]=0.

    return Sdot_diff
    
def Sig_dot_p_box(rs, r0, width, Mdot, mask_belt):
    
    sigdot_CO=np.ones(Nr)*Mdot/(np.pi*((r0+width/2.)**2.-(r0-width/2.)**2.))
    sigdot_CO[mask_belt]=0.0
    
    return sigdot_CO


def Sig_dot_p_gauss(rs, hs, r0, sig_g, Mdot, mask_belt):
    
    Sdot_comets=np.zeros(len(rs))
    Sdot_comets[mask_belt]=np.exp( -(rs[mask_belt]-r0)**2.0 / (2.*sig_g**2.) ) # /(np.sqrt(2.*np.pi)*sig_g)/(2.*np.pi*rs[mask_belt])
    Sdot_comets[mask_belt]=Mdot*Sdot_comets[mask_belt]/(2.*np.pi*np.sum(Sdot_comets[mask_belt]*rs[mask_belt]*hs[mask_belt]))
    return Sdot_comets

def Sigma_next(Sigma_prev, Nr, rs, rhalfs, hs, epsilon, r0, width, Mdot, nus_au2_yr, mask_belt, diffusion=0):
    
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
        Snext=Snext+epsilon* Difussion(Snext, Nr, rs, rhalfs, hs, nus_au2_yr)
        
    ###########################################
    ############### CO mas input rate
    ###########################################
    Snext2=np.zeros((2, Nr))

    Snext2[0,:]=Snext[0,:]+epsilon*Sig_dot_p_gauss(rs, hs, r0, width, Mdot, mask_belt)
    Snext2[1,:]=Snext[1,:]*1.
    ###########################################
    ############## photodissociation
    ###########################################

    tphCO=fgas.tau_CO2(Sigma_prev[0,:], Sigma_prev[1,:])
    Sdot_ph=Sigma_prev[0,:]/tphCO #(Snext[0,:]/tphCO)
    #Sdot_ph_epsilon=Sigma_prev[0,:]*(1.-np.exp(-epsilon/tphCO))   
    Snext2[0,:]=Snext2[0,:]-epsilon*Sdot_ph
    Snext2[1,:]=Snext2[1,:]+epsilon*Sdot_ph*muc1co
    #Snext2[0,Snext2[0,:]<0.0]=0.0
    
    return Snext2

def viscous_evolution(ts, epsilon, rs, rhalfs, hs, rbelt, sig_g, Mdot, alpha, Mstar=1.0, Lstar=1.0, Sigma0=np.array([-1.0]), mu0=12.0, dt_skip=1, diffusion=0 ):
    ### 
    Nt=len(ts)
    if isinstance(dt_skip, int) and dt_skip>0:
        if dt_skip>1:  #  skips dt_skip to make arrays smaller
            if (Nt-1)%dt_skip==0:
                Nt2=(Nt-1)/dt_skip+1
            else:
                Nt2=(Nt-1)/dt_skip+2
        elif dt_skip==1: Nt2=Nt
    else:
        print 'not a valid dt_skip'
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
        print np.shape(Sigma0), Sigma0>0.0
    Sigma_temp=Sigma_g[:,:,0]*1.0
    j=1
    for i in xrange(1,Nt):
        mask_m=np.sum(Sigma_temp, axis=0)>0.0
        mus[mask_m]=(Sigma_temp[0,mask_m]+Sigma_temp[1,mask_m]*(1.+16./12.))/(Sigma_temp[0,mask_m]/28.+Sigma_temp[1,mask_m]/6.) # Sigma+Oxigen/(N)
 
        nus=alpha*kb*Ts/(mus*mp)/(Omegas_s) # m2/s 1.0e10*np.zeros(Nr) #
        nus_au2_yr=nus*year_s/(au_m**2.0) # au2/yr  

        Sigma_temp=Sigma_next(Sigma_temp, Nr, rs, rhalfs, hs, epsilon, rbelt, sig_g, Mdot, nus_au2_yr, mask_belt, diffusion=diffusion)

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
        print np.shape(Sigma0), Sigma0>0.0

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



def viscous_evolution_evolcoll(ts, epsilon, rs, rhalfs, hs, rbelt, sig_g, Mdots, alpha, Mstar=1.0, Lstar=1.0, Sigma0=np.array([-1.0]), mu0=12.0, dt_skip=1, diffusion=0 ):
    
    Nt=len(ts)
    if isinstance(dt_skip, int) and dt_skip>0:
        if dt_skip>1:  #  skips dt_skip to make arrays smaller
            if (Nt-1)%dt_skip==0:
                Nt2=(Nt-1)/dt_skip+1
            else:
                Nt2=(Nt-1)/dt_skip+2
        elif dt_skip==1: Nt2=Nt
    else:
        print 'not a valid dt_skip'
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
        print np.shape(Sigma0), Sigma0>0.0
    Sigma_temp=Sigma_g[:,:,0]*1.0
    j=1
    for i in xrange(1,Nt):
        mask_m=np.sum(Sigma_temp, axis=0)>0.0
        mus[mask_m]=(Sigma_temp[0,mask_m]+Sigma_temp[1,mask_m]*(1.+16./12.))/(Sigma_temp[0,mask_m]/28.+Sigma_temp[1,mask_m]/6.) # Sigma+Oxigen/(N)
        nus=alpha*kb*Ts/(mus*mp)/(Omegas_s) # m2/s 
        nus_au2_yr=nus*year_s/(au_m**2.0) # au2/yr  
        Sigma_temp=Sigma_next(Sigma_temp, Nr, rs, rhalfs, hs, epsilon, rbelt, sig_g, Mdots[i], nus_au2_yr, mask_belt, diffusion=diffusion)
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
        for i in xrange(1,Nr):
            if rs[i]>rb:
                dr=rs[i]-rs[i-1]
                break
        if hs[i-1]/rs[i-1]<res:
            break
        else:
            Nr=int(Nr*1.2)
    return Nr





