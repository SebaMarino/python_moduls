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
    r_min=max(r-dr,0.0)
    r_max=r+dr
    area=np.pi*(r_max**2.-r_min**2.)*au_cm**2.0 # cm2
    # area=2*np.pi*r*dr*au_cm**2.0 # cm2
    
    NC1=(MC1*Mearth/m_c1)/area
    NCO=(MCO*Mearth/m_co)/area
    
    return 120.0* np.exp( sigma_c1*NC1)/ selfshielding_CO(NCO) # yr

def tau_CO2(Sigma_CO, Sigma_C1):
    #area=2*np.pi*r*dr*au_cm**2.0 # cm2
    
    NC1=Sigma_C1*Mearth/m_c1/au_cm**2.0
    NCO=Sigma_CO*Mearth/m_co/au_cm**2.0
    
    return 120.0 * np.exp( sigma_c1*NC1)/ selfshielding_CO(NCO) # yr

    
def tau_vis(r, dr, alpha, cs, Mstar):
    
    Omega=np.sqrt(G*Mstar*Msun/((r*au_m)**3.0)) # s
    return (dr*au_m)**2.0*Omega/(alpha*cs**2.)/year_s #/3.0

def tau_vis2(r, dr, alpha, cs, Mstar):

    Omega=np.sqrt(G*Mstar*Msun/((r*au_m)**3.0)) # s
    return (r*au_m)**2.0*Omega/(alpha*cs**2.)/year_s #/3.0


def MCOdot_n(t,r,dr,MCO, MC1, alpha, cs, Mstar):

    tv=tau_vis(r,dr, alpha, cs, Mstar)
    return MCO*(1./tau_CO(r,dr,MCO, MC1) + 1./(tv)) # Mearth/yr



def MC1dot_n(MC1, r, dr, alpha, cs, Mstar):
    
    return MC1/tau_vis(r, dr, alpha, cs, Mstar)

def MC1dot_p(r,dr,MCO, MC1):
    
    return MCO*(m_c1/m_co)/tau_CO(r,dr,MCO, MC1)


def Onestep(MCO, MC1, MCOdot_p,  dt, r,dr, alpha, cs , Mstar):

    MCOdot=MCOdot_p - MCOdot_n(t, r,dr,MCO, MC1, alpha, cs, Mstar)
    MC1dot=MC1dot_p(r,dr,MCO, MC1) - MC1dot_n(MC1, r, dr, alpha, cs, Mstar)

    MCOp=max(MCO+MCOdot*dt,0.0)
    MC1p=max(MC1+MC1dot*dt, 0.0)

    return MCOp, MC1p#, MCOdot, MC1dot

def Onestep_fast(MCO, MC1, MCOdot_p, dt, r,dr, tph_CO, t_vis , Mstar):

    #tph_CO=tau_CO(r,dr,MCO, MC1) 
    MCOdot=MCOdot_p - MCO*(1./tph_CO+ 1./(t_vis)) 
    MC1dot= MCO*muc1co/tph_CO - MC1/(t_vis)  

    MCOp=max(MCO+MCOdot*dt,0.0)
    MC1p=max(MC1+MC1dot*dt, 0.0)

    return MCOp, MC1p#, MCOdot, MC1dot

def integrate(MCO, MC1, MCOdot_p, dt0,tf,tol, r,dr, alphai, cs,  Mstar=1.0 ):
    
    MCOs=[]
    MC1s=[]
    ts=[]
    
    tvis= tau_vis2(r,dr, alphai, cs, Mstar)
    
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


def f_tc_simple(Mtot, r, dr, Dc=10.0, e=0.05, Qd=150.0, Mstar=1.0): # collisional timescale of largest planetesimal

    return 1.4e-3 * r**(13.0/3) * (dr/r) * Dc * Qd *e**(-5.0/3.0) * Mstar**(-4.0/3.0)*Mtot**(-1.0) # in yr

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

    return Mtot0/(1.0+t/tc0) 

def Mtotdot_t(Mtot0, t, r, dr, rho=2700.0,  Dc=10.0, e=0.05, I=0.05, Qd=150.0, Mstar=1.0, q=11./6.):
    # t in years
    tc0=f_tc_Xc(Mtot0, r, dr, rho,  Dc, e, I, Qd, Mstar, q=q)

    return Mtot0/(1.0+t/tc0)**2. / tc0

def integrate_evolcoll(MCO, MC1, dt0,tf, tol, r,dr, alphai, cs,  Mstar=1.0, fCO=0.1, Mtot0=10.0, Dc=10.0, e=0.05, I=0.05, Qd=150.0, q=11./6., rho=2700.0, gamma=2.0  ):
    
    MCOs=[]
    MC1s=[]
    MCOdots=[]
    ts=[]
    
    tvis= tau_vis(r,dr, alphai, cs, Mstar)
    
    ti=0.0
    
    MCOs.append(MCO)
    MC1s.append(MC1)
    ts.append(ti)
    MCOdot_p0=fCO*Mtotdot_t(Mtot0, 0.0, r=r/gamma, dr=dr/gamma,  Dc=Dc, e=e, I=I, Qd=Qd, Mstar=Mstar)
    MCOdots.append(MCOdot_p0)
    
    dti=dt0

    while ti<tf:
        
        tph_CO=tau_CO(r,dr,MCOs[-1], MC1s[-1])
        MCOdot_p=fCO*Mtotdot_t(Mtot0, ti+dti, r=r/gamma, dr=dr/gamma,  Dc=Dc, e=e, I=I, Qd=Qd, Mstar=Mstar)

        # calculate step
        MCOi0, MC1i0 = Onestep_fast(MCOs[-1], MC1s[-1],MCOdot_p, dti, r,dr,tph_CO, tvis , Mstar)
        # calculate mid step stopping in the middle
        MCOh, MC1h = Onestep_fast(MCOs[-1], MC1s[-1],MCOdot_p, dti/2, r,dr,tph_CO, tvis , Mstar)
        # calculate 2nd mid step  
        MCOi1, MC1i1 = Onestep_fast(MCOh, MC1h, MCOdot_p,  dti/2, r,dr, tph_CO, tvis , Mstar)
        
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
    
