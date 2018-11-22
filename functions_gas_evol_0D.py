import numpy as np
from scipy.interpolate import interp1d


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
year_s = 3.154e7

## functions

####### CO PHOTODISSOCIATION
# Visser+2009

kCOs=[1.0, 0.9495, 0.7046, 0.4015, 0.09964, 0.01567, 0.003162, 0.0004839]
NCOs=[1.0, 1.e13, 1.e14, 1.e15, 1.e16, 1.e17, 1.e18, 1.e19]
logfkCO = interp1d(np.log10(NCOs), np.log10(kCOs))
slope=np.log(kCOs[-1]/kCOs[-2])/np.log(NCOs[-1]/NCOs[-2])


def index_time(ts, ti):

    if ti>ts[-1] or ti<ts[0]:
        return np.nan

    else:
        for it in xrange(len(ts)):
            if ti<ts[it]:
                break
    return it

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

def tau_CO(r,dr,MCO, MC1):
    area=2*np.pi*r*dr*au_cm**2.0 # cm2
    
    NC1=(MC1*Mearth/m_c1)/area
    NCO=(MCO*Mearth/m_co)/area
    
    return 120.0* np.exp( sigma_c1*NC1)/ selfshielding_CO(NCO) # yr

def tau_CO2(Sigma_CO, Sigma_C1):
    #area=2*np.pi*r*dr*au_cm**2.0 # cm2
    
    NC1=Sigma_C1*Mearth/m_c1/au_cm**2.0
    NCO=Sigma_CO*Mearth/m_co/au_cm**2.0
    
    return 120.0 * np.exp( sigma_c1*NC1)/ selfshielding_CO(NCO) # yr

    
def tau_vis(r, alpha, cs, Mstar):
    
    Omega=np.sqrt(G*Mstar*Msun/((r*au_m)**3.0)) # s
    return (r*au_m)**2.0*Omega/(alpha*cs**2.)/year_s/12.0

def MCOdot_n(r,dr,MCO, MC1, alpha, cs, Mstar):
    
    return MCO*(1./tau_CO(r,dr,MCO, MC1)+1./tau_vis(r, alpha, cs, Mstar)) # Mearth/yr

def MC1dot_n(MC1, r, alpha, cs, Mstar):
    
    return MC1/tau_vis(r, alpha, cs, Mstar)

def MC1dot_p(r,dr,MCO, MC1):
    
    return MCO*(m_c1/m_co)/tau_CO(r,dr,MCO, MC1)


def Onestep(MCO, MC1, MCOdot_p, dt, r,dr, alpha, cs , Mstar):

    MCOdot=MCOdot_p - MCOdot_n(r,dr,MCO, MC1, alpha, cs, Mstar)
    MC1dot=MC1dot_p(r,dr,MCO, MC1) - MC1dot_n(MC1, r, alpha, cs, Mstar)

    MCOp=max(MCO+MCOdot*dt,0.0)
    MC1p=max(MC1+MC1dot*dt, 0.0)

    return MCOp, MC1p#, MCOdot, MC1dot

def Onestep_fast(MCO, MC1, MCOdot_p, dt, r,dr, tph_CO, t_vis , Mstar):

    #tph_CO=tau_CO(r,dr,MCO, MC1) 
    MCOdot=MCOdot_p - MCO*(1./tph_CO+1./t_vis) 
    MC1dot= MCO*muc1co/tph_CO - MC1/t_vis  

    MCOp=max(MCO+MCOdot*dt,0.0)
    MC1p=max(MC1+MC1dot*dt, 0.0)

    return MCOp, MC1p#, MCOdot, MC1dot

def integrate(MCO, MC1, MCOdot_p, dt0,tf,tol, r,dr, alphai, cs,  Mstar=1.0 ):
    
    MCOs=[]
    MC1s=[]
    ts=[]
    
    tvis= tau_vis(r, alphai, cs, Mstar)
    
    ti=0.0
    
    MCOs.append(MCO)
    MC1s.append(MC1)
    ts.append(ti)
      
    # MCOpi= MCOdot_p
    # MC1pi= MCOdot_p*(m_c1/m_co)
    dti=dt0

    while ti<tf:
        
        tph_CO=tau_CO(r,dr,MCOs[-1], MC1s[-1]) 
        # calculate step
        MCOi0, MC1i0 = Onestep_fast(MCOs[-1], MC1s[-1],MCOdot_p, dti, r,dr,tph_CO, tvis , Mstar)
        # calculate mid step stopping in the middle
        MCOh, MC1h = Onestep_fast(MCOs[-1], MC1s[-1],MCOdot_p, dti/2, r,dr,tph_CO, tvis , Mstar)
        # calculate 2nd mid step  
        MCOi1, MC1i1 = Onestep_fast(MCOh, MC1h, MCOdot_p, dti/2, r,dr, tph_CO, tvis , Mstar)
        
        # calculate difference in the solutions
        errorCO=MCOi1-MCOi0
        errorC1=MC1i1-MC1i0
        
        ti+=dti
        MCOs.append(MCOi0)
        MC1s.append(MC1i0)
        ts.append(ti)
        
        # define next dt
        dti=0.9*dti*min( max( min(tol*MCOs[-1]/abs(errorCO),tol*MC1s[-1]/abs(errorC1)),0.3),2.0)
        #print dti
        
    return np.array(ts), np.array(MCOs), np.array(MC1s) #, MC1s[-1], MC1pi


def f_tc(Mtot, r, dr, Dc=10.0, e=0.05, Qd=150.0, Mstar=1.0): # collisional timescale of largest planetesimal

    return 1.4e-3 * r**(13.0/3) * (dr/r) * Dc * Qd *e**(-5.0/3.0) * Mstar**(-4.0/3.0)*Mtot**(-1.0) # in yr

    # return 1.0/Mtot

def Mtot_t(Mtot0, t, r, dr,  Dc=10.0, e=0.05, Qd=150.0, Mstar=1.0):
    # t in years
    tc0=f_tc(Mtot0, r, dr, Dc, e, Qd, Mstar)

    return Mtot0/(1.0+t/tc0) 

def Mtotdot_t(Mtot0, t, r, dr,  Dc=10.0, e=0.05, Qd=150.0, Mstar=1.0):
    # t in years
    tc0=f_tc(Mtot0, r, dr, Dc, e, Qd, Mstar)

    return Mtot0/(1.0+t/tc0)**2. / tc0

def integrate_evolcoll(MCO, MC1, dt0,tf,tol, r,dr, alphai, cs,  Mstar=1.0, fCO=0.1, Mtot0=10.0, Dc=10.0, e=0.05, Qd=150.0):
    
    MCOs=[]
    MC1s=[]
    MCOdots=[]
    ts=[]
    
    tvis= tau_vis(r, alphai, cs, Mstar)
    
    ti=0.0
    
    MCOs.append(MCO)
    MC1s.append(MC1)
    ts.append(ti)
    MCOdot_p0=fCO*Mtotdot_t(Mtot0, 0.0, r, dr,  Dc, e, Qd, Mstar)
    MCOdots.append(MCOdot_p0)
    
    dti=dt0

    while ti<tf:
        
        tph_CO=tau_CO(r,dr,MCOs[-1], MC1s[-1])
        MCOdot_p=fCO*Mtotdot_t(Mtot0, ti+dti, r, dr,  Dc, e, Qd, Mstar)

        # calculate step
        MCOi0, MC1i0 = Onestep_fast(MCOs[-1], MC1s[-1],MCOdot_p, dti, r,dr,tph_CO, tvis , Mstar)
        # calculate mid step stopping in the middle
        MCOh, MC1h = Onestep_fast(MCOs[-1], MC1s[-1],MCOdot_p, dti/2, r,dr,tph_CO, tvis , Mstar)
        # calculate 2nd mid step  
        MCOi1, MC1i1 = Onestep_fast(MCOh, MC1h, MCOdot_p, dti/2, r,dr, tph_CO, tvis , Mstar)
        
        # calculate difference in the solutions
        errorCO=MCOi1-MCOi0
        errorC1=MC1i1-MC1i0
        
        ti+=dti
        MCOs.append(MCOi0)
        MC1s.append(MC1i0)
        ts.append(ti)
        MCOdots.append(MCOdot_p)
        
        # define next dt
        dti=0.9*dti*min( max( min(tol*MCOs[-1]/abs(errorCO),tol*MC1s[-1]/abs(errorC1)),0.3),2.0)
        #print dti
        
    return np.array(ts), np.array(MCOs), np.array(MC1s), np.array(MCOdots) #, MC1s[-1], MC1pi




##### 1D evolution

def Sigma_dot_vis(Sigmas, Nr, rs, rhalfs, hs, nus_au2_yr):
  

    ########## CALCULATE VR*Sigma=F1
    
    Sigma_tot=np.sum(Sigmas,axis=0)
    eps=np.ones((2,Nr))*0.5
    mask_m=Sigma_tot>0.0
    eps[0,mask_m]=Sigmas[0,mask_m]/Sigma_tot[mask_m]
    eps[1,mask_m]=Sigmas[1,mask_m]/Sigma_tot[mask_m]
    

    G1s=Sigma_tot*nus_au2_yr*np.sqrt(rs) # Nr
    Sigma_vr_halfs=-3.0*(G1s[1:]-G1s[:-1])/(rs[1:]-rs[:-1])/np.sqrt(rhalfs[1:-1]) # Nr-1
    

    
    ############## CALCULATE dSIGMA/dT
    eps_halfs=np.zeros((2,Nr-1))
    eps_halfs[:,:]=np.where(Sigma_vr_halfs[:]>0.0, eps[:,:-1], eps[:,1:])
    
    G2s=rhalfs[1:-1]*Sigma_vr_halfs  # Nr-1
    G3s=G2s*eps_halfs    #  2x(Nr-1)
    Sdot=np.zeros((2,Nr))
    Sdot[:,1:-1]=-(G3s[:,1:]-G3s[:,:-1])*2./(rhalfs[2:-1]**2.-rhalfs[1:-2]) # Nr-2

    ### inner boundary condition
    #Fph=G3s[:,0] # F_{+1/2} 2 dim
    #Sigma_vr_1=-3.0*(G1s[2]-G1s[0])/(rs[2]-rs[0])/np.sqrt(rs[1]) 
    #F0=Sigma_vr_1*rs[1]/rs[0]*eps[:,0] # 2 dim
    #Fmh=np.where(Sigma_vr_halfs[0]>0.0, np.array([0.0,0.0]),F0) # 2 dim
    #Sdot[:,0]=-(rhalfs[1]*Fph - rhalfs[0]*Fmh)*2.0/(rhalfs[1]**2.0-rhalfs[0]**2.0)
    
    ### outer boundary condition
    
        
    return Sdot, Sigma_vr_halfs

def Sig_dot_p_box(rs, r0, width, Mdot, mask_belt):
    
    sigdot_CO=np.ones(Nr)*Mdot/(np.pi*((r0+width/2.)**2.-(r0-width/2.)**2.))
    sigdot_CO[mask_belt]=0.0
    
    return sigdot_CO


def Sig_dot_p_gauss(rs, r0, sig_g, Mdot, mask_belt):
    
    Sdot_comets=np.zeros(len(rs))
    Sdot_comets[mask_belt]=Mdot*np.exp( -(rs[mask_belt]-r0)**2.0 / (2.*sig_g**2.) )/(np.sqrt(2.*np.pi)*sig_g)/(2.*np.pi*rs[mask_belt])
    return Sdot_comets

def Sigma_next(Sigma_prev, Nr, rs, rhalfs, hs, epsilon, r0, width, Mdot, nus_au2_yr, mask_belt):
    
    ## viscous evolution
    Sdot_vis, Sigma_vr_halfs=Sigma_dot_vis(Sigma_prev,  Nr, rs, rhalfs, hs, nus_au2_yr)
    Snext= Sigma_prev + epsilon*Sdot_vis # viscous evolution
    
    ## Inner boundary condition (constant Mdot)
    Snext[:,0]=Snext[:,1]*(nus_au2_yr[1]/nus_au2_yr[0]) # constant Mdot
    ## Outer boundary condition (power law or constant mass)
    if np.all(Snext[:,-3])>0.0:
        Snext[:,-1]=np.minimum(Snext[:,-2]*(nus_au2_yr[-2]/nus_au2_yr[-1]), Snext[:,-2]*(rs[-1]/rs[-2])**(np.log(Snext[:,-2]/Snext[:,-3])/np.log(rs[-2]/rs[-3])))
    else: 
        Snext[:,-1]=Snext[:,-2]*(nus_au2_yr[-2]/nus_au2_yr[-1])
    
    #if np.all(Snext[:,-2]<Snext[:,-3]) and np.all(Snext[:,-3]>0.0):
    #    Snext[:,-1]=Snext[:,-2]*(rs[-1]/rs[-2])**(np.log(Snext[:,-2]/Snext[:,-3])/np.log(rs[-2]/rs[-3]))  #Snext2[:,-2]*np.sqrt(rs[-2]/rs[-1]) #*vrs[-2]/(rs[-1]*vrs[-1])
    #elif np.all(Snext[:,-2]==Snext[:,-3]):
    #    Snext[:,-1]=Snext[:,-2]
    #else:
    #    print 'Error: positive slope at outer edge'
    #    sys.exit()
    
    ## CO mas input rate
    Snext[0,:]=Snext[0,:]+epsilon*Sig_dot_p_gauss(rs, r0, width, Mdot, mask_belt)
                                   
    ## photodissociation
    Snext2=np.zeros((2, Nr))
    tphCO=tau_CO2(Snext[0,:], Snext[1,:])
    Sdot_ph=(Snext[0,:]/tphCO)
    Snext2[0,:]=Snext[0,:]-epsilon*Sdot_ph
    Snext2[1,:]=Snext[1,:]+epsilon*Sdot_ph*muc1co
    
    return Snext2

def viscous_evolution(ts, epsilon, rs, rhalfs, hs, rbelt, sig_g, Mdot, alpha, Mstar=1.0, Lstar=1.0, Sigma0=np.array([-1.0]) ):
    
    Nt=len(ts)
    epsilon=ts[1]-ts[0]
    Nr=len(rs)
    Sigma_g=np.zeros((2,Nr,Nt))

    mask_belt=((rs<rbelt+sig_g*2) & (rs>rbelt-sig_g*2))

    
    ## Temperature and angular velocity
    Ts=278.3*(Lstar**0.25)*rs**(-0.5) # K
    Omegas=2.0*np.pi*np.sqrt(Mstar/(rs**3.0)) # 1/yr
    Omegas_s=Omegas/year_s # Omega in s-1
    ## default viscosity
    mus=np.ones(Nr)*12.
    
    
    if np.shape(Sigma0)==(2,Nr) and np.all(Sigma0>=0.0):
        Sigma_g[:,:,0]=Sigma0
    else:
        print np.shape(Sigma0), Sigma0>0.0
    for i in xrange(1,Nt):
        mask_m=np.sum(Sigma_g[:,:,i-1], axis=0)>0.0
        mus[mask_m]=(Sigma_g[0,mask_m,i-1]+Sigma_g[1,mask_m,i-1]*(1.+16./12.))/(Sigma_g[0,mask_m, i-1]/28.+Sigma_g[1,mask_m, i-1]/6.) # Sigma+Oxigen/(N)
        nus=alpha*kb*Ts/(mus*mp)/(Omegas_s) # m2/s
        nus_au2_yr=nus*year_s/(au_m**2.0) # au2/yr  
        Sigma_g[:,:,i]=Sigma_next(Sigma_g[:,:,i-1], Nr, rs, rhalfs, hs, epsilon, rbelt, sig_g, Mdot, nus_au2_yr, mask_belt)
    return Sigma_g

def radial_grid_powerlaw(rmin, rmax, Nr, alpha):

    u=np.linspace(rmin**alpha, rmax**alpha, Nr+1)
    rhalfs=u**(1./alpha)
    hs=rhalfs[1:]-rhalfs[:-1]
    rs=0.5*(rhalfs[1:] + rhalfs[:-1])
    return rs, rhalfs, hs
