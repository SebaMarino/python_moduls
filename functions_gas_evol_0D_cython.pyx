# compile with python setup_mac.py build_ext --inplace
import numpy as np
# from scipy.interpolate import interp1d
# C functions,attributes
cimport numpy as np
cimport cython
# Cython handles the namespace ambiguity internally in this case
from libc.math cimport sqrt, exp, M_PI, log, log10, abs

###  constants 
cdef:
    double mp=1.6726219e-27 # kg
    double m_c1=12.0*mp # kg/molecule
    double m_co= 28.0*mp # kg/molecule
    double muc1co=m_c1/m_co
    double sigma_c1=1.6e-17 # cm2
    double sigma_co=1/(1.15e15) # cm2 # when  shielding is 0.368 = exp(1)
    double tphCO0=120.0 # CO photodissociation timescale

    double Mearth=5.9e24 # kg
    double Msun = 2.e30 # kg
    double au_cm=1.496e13 # cm
    double au_m=1.496e11 # m
    double G = 6.67408e-11 # mks
    double kb = 1.38064852e-23 #mks
    double year_s = 3.154e7 # seconds

 ####### CO PHOTODISSOCIATION
# Visser+2009

cdef: 
    double[:] NCOs=np.array([1.0, 1.e13, 1.e14, 1.e15, 1.e16, 1.e17, 1.e18, 1.e19])
    double[:] kCOs=np.array([1.0, 0.9495, 0.7046, 0.4015, 0.09964, 0.01567, 0.003162, 0.0004839])
    double[:] log10kCOs=np.array([ 0., -0.02250503, -0.15205736, -0.39631445, -1.00156628, -1.804931, -2.50003813, -3.31524438])
    double[:] log10NCOs=np.array([0.0, 13., 14., 15., 16., 17., 18., 19.])
    double[:] logkCOs=np.array([ 0., -0.05181975, -0.35012501, -0.91254775, -2.30619159, -4.15600722, -5.75655054, -7.63363228])
    double[:] logNCOs=np.array([ 0., 29.93360621, 32.2361913 , 34.53877639, 36.84136149, 39.14394658, 41.44653167, 43.74911677])
    # Not used anymore logfkCO = interp1d(np.log10(NCOs), np.log10(kCOs))
    # spectral index for large NCO
    int Ncol=len(logNCOs)
    double slope=(logkCOs[Ncol-1]-logkCOs[Ncol-2])/(logNCOs[Ncol-1]-logNCOs[Ncol-2])
    
## functions
@cython.boundscheck(False)
#@cython.wraparound(False) # False == Deactivate negative indexing 
@cython.nonecheck(False)
@cython.cdivision(True)

def interpol_log_NCO(double NCO):
    cdef:
        size_t i
        double logkCO
        double logNCO=log(NCO)
    for i in range(1,Ncol):
        if logNCO<logNCOs[i]:
            logkCO=logkCOs[i-1]+ (logNCO-logNCOs[i-1])*(logkCOs[i]-logkCOs[i-1])/(logNCOs[i]-logNCOs[i-1])
            return exp(logkCO)
        
        

def selfshielding_CO(double NCO):#, NCOs, logfkCO, slope):

    
    if (NCO>=NCOs[0]) & (NCO<=NCOs[Ncol-1]):
        return interpol_log_NCO(NCO)
    else:
        if NCO<NCOs[0]:
            return 1.0
        else:
            return kCOs[Ncol-1]*(NCO/NCOs[Ncol-1])**slope # (NCO/NCO[-1])**slope

            
def tau_CO(double r,double dr,double MCO,double MC1):
    cdef:
        double r_min=max(r-dr,0.0)
        double r_max=r+dr
        double area=M_PI*(r_max*r_max-r_min*r_min)*au_cm*au_cm # cm2
        double NC1=(MC1*Mearth/m_c1)/area
        double NCO=(MCO*Mearth/m_co)/area
    
    return 120.0* exp( sigma_c1*NC1)/ selfshielding_CO(NCO) # yr


def tau_vis(double r,double dr,double alpha,double cs,double Mstar):
    
    cdef:
        double r_m=r*au_m
        double dr_m=dr*au_m
        Omega=sqrt(G*Mstar*Msun/(r_m*r_m*r_m)) # s
    return (dr_m*dr_m)*Omega/(alpha*cs*cs)/year_s#/12.0

def MCOdot_n(double r,double dr,double MCO,double MC1,double alpha,double cs,double Mstar):
    
    return MCO*(1./tau_CO(r,dr,MCO, MC1)+1./tau_vis(r,dr, alpha, cs, Mstar)) # Mearth/yr

def MC1dot_n(double MC1,double r,double dr,double alpha,double cs,double Mstar):
    
    return MC1/tau_vis(r, dr, alpha, cs, Mstar)

def MC1dot_p(double r,double dr,double MCO,double MC1):
    
    return MCO*(m_c1/m_co)/tau_CO(r,dr,MCO, MC1)


def Onestep_fast(double MCO,double MC1,double MCOdot_p,double dt,double r,double dr,double tph_CO,double t_vis ,double Mstar):

    #tph_CO=tau_CO(r,dr,MCO, MC1) 
    cdef: 
        double MCOdot=MCOdot_p - MCO*(1./tph_CO+1./t_vis) 
        double MC1dot= MCO*muc1co/tph_CO - MC1/t_vis  

        double MCOp=max(MCO+MCOdot*dt,0.0)
        double MC1p=max(MC1+MC1dot*dt, 0.0)

    return MCOp, MC1p#, MCOdot, MC1dot

def Cintegrate(double MCO,double MC1,double MCOdot_p,double dt0,double tf,double tol,double r,double dr,double alphai,double cs,double  Mstar):
    
    cdef:
        
        double tvis= tau_vis(r,dr, alphai, cs, Mstar)
        
        double ti=0.0
        double dti=dt0
        int it=0
        double tph_CO
        double MCOi0
        double MC1i0
        double MCOh
        double MC1h
        double MCOi1
        double MC1i1
        double errorCO
        double errrorC1
        
        double[:] MCOs=np.array([MCO])
        double[:] MC1s=np.array([MC1])
        double[:] ts=np.array([ti])
    
    while ti<tf:  
        tph_CO=tau_CO(r,dr,MCOs[it], MC1s[it]) 
        # calculate step
        MCOi0, MC1i0 = Onestep_fast(MCOs[it], MC1s[it],MCOdot_p, dti, r,dr,tph_CO, tvis , Mstar)
        # calculate mid step stopping in the middle
        MCOh, MC1h = Onestep_fast(MCOs[it], MC1s[it],MCOdot_p, dti/2, r,dr,tph_CO, tvis , Mstar)
        # calculate 2nd mid step  
        MCOi1, MC1i1 = Onestep_fast(MCOh, MC1h, MCOdot_p, dti/2, r,dr, tph_CO, tvis , Mstar)
        
        # calculate difference in the solutions
        errorCO=MCOi1-MCOi0
        errorC1=MC1i1-MC1i0
        
        ti+=dti
        it+=1
        MCOs=np.append(MCOs, MCOi0)
        MC1s=np.append(MC1s, MC1i0)
        ts=np.append(ts, ti)
        
        # define next dt
        dti=0.9*dti*min( max( min(tol*MCOs[-1]/abs(errorCO),tol*MC1s[-1]/abs(errorC1)),0.3),2.0)
        #print dti
        
    return ts, MCOs, MC1s #, MC1s[-1], MC1pi
