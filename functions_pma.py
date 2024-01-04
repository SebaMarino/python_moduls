import numpy as np
import matplotlib.pyplot as plt

### untis
Msun=1.989e+30 # kg
au=1.496e+11   # m
year=3.154e+7 # s
G=6.67408e-11 # mks
Mjup=1.898e27 # kg



###Define the function that corrects for the window smearing effect (eq. 13 in section 3.6.2 of Kervella+2019):
### where Pp=P/deltat with deltat=1227 days for Hipparcos and deltat=668 days for GAIADR2:
def f_gamma(Pp):

    return Pp/(np.sqrt(2.)*np.pi) * np.sqrt(1. - np.cos(2.*np.pi/Pp))

##Now we define the function that corrects for the orbital period of the system (see eq. 14 and section 3.6.3 in Kervella+2019)
# ### with delta_t_HG=24.25 years:
# def f_zeta(P):

#     if isinstance(P, np.ndarray):
#         zeta=np.ones(len(P))
#         zeta[ (P/delta_t_HG)>3.]=3.*delta_t_HG/P[(P/delta_t_HG)>3.]

#     else:
#         if P>3*delta_t_HG:
#             zeta=3.*delta_t_HG/P
#         else:
#             zeta=1.
#     return zeta

def f_zeta_v2(r, m1, Nrandom, epoch='eDR3', gaia='eDR3'): # This does not incorporates uncertainty as it does not depend on cosi. Uncertainty is absorbed by eta

    if gaia=='DR2':
        delta_t_HG=24.25 # years
    elif gaia=='eDR3':
        delta_t_HG=24.75 # years
    period=f_Period(r, m1) # years

    M1=np.random.uniform(0.0, 2.*np.pi, Nrandom) # mean anomaly
    M2=M1 + 2.0*np.pi* (delta_t_HG)/period # mean anomaly 24.25 years after

    # circular orbit
    if epoch=='DR2' or epoch=='eDR3':

        vx=-np.sin(M2)
        vy=np.cos(M2)
    elif epoch=='Hipparcos':
        vx=-np.sin(M1)
        vy=np.cos(M1)

    else:
        print('wrong epoch')
        return -1.

    deltar_x = np.cos(M2)-np.cos(M1)
    deltar_y = np.sin(M2)-np.sin(M1)
    C=period/(2.*np.pi)
    zeta =  np.sqrt( (vx - C*deltar_x/delta_t_HG)**2. +  (vy - C*deltar_y/delta_t_HG)**2. )

    return zeta

def f_zeta_v3(r, m1, Nrandom, cosi, epoch='eDR3', gaia='eDR3'): # this incorporates uncertainty through cosi. 

    # cosi is one or a set of random inclinations distributed as uniform if no information or as something else
    # e.g. cosi=np.random.uniform(0.0, 1.0, Nrandom)

    if gaia=='DR2':
        delta_t_HG=24.25 # years
    elif gaia=='eDR3':
        delta_t_HG=24.75 # years
        
    period=f_Period(r, m1) # years
    
    M1=np.random.uniform(0.0, 2.*np.pi, Nrandom) # mean anomaly
    
    M2=M1 + 2.0*np.pi* (delta_t_HG)/period # mean anomaly 24.25 years after

    # circular orbit
    if epoch=='DR2' or epoch=='eDR3':

        vx=-np.sin(M2)
        vy=np.cos(M2)
    elif epoch=='Hipparcos':
        vx=-np.sin(M1)
        vy=np.cos(M1)

    else:
        print('wrong epoch')
        return -1.

    deltar_x = (np.cos(M2)-np.cos(M1))
    deltar_y = (np.sin(M2)-np.sin(M1))*cosi
    vy=vy*cosi
    C=period/(2.*np.pi)

    # zeta_x=vx - C*deltar_x/delta_t_HG
    # zeta_y=vy - C*deltar_y/delta_t_HG
    
    zeta =  np.sqrt( (vx - C*deltar_x/delta_t_HG)**2. +  (vy - C*deltar_y/delta_t_HG)**2. )/np.sqrt(vx**2+vy**2)


    return zeta

##def of the orbital period as a function of radial separation and mass of the primary (if mass of the secondary<<<mass of the primary):
def f_Period(r, m): # in years
    # r in au
    # m in Msun
    return np.sqrt(r**3. / m)


###function that takes care of the orbital inclination (section 3.6.1).
##it is a function of the 3D orbital proper motion anomaly vector, the inclination and the parallactic angle
def f_eta(PA_pma, inc, PA):

    # projections of pma normalised to unity
    pma_ra = np.sin(PA_pma*np.pi/180.)
    pma_dec = np.cos(PA_pma*np.pi/180.)

    # projections of pma normalised to unity and rotated such that x is in direction of PA
    pma_x = pma_dec * np.cos(PA*np.pi/180.) + pma_ra * np.sin(PA*np.pi/180.)
    pma_y = pma_dec * np.sin(PA*np.pi/180.) - pma_ra * np.cos(PA*np.pi/180.)

    # deprojected pma with x along PA of system and y perpendicular 
    pma_x_dep=pma_x
    pma_y_dep=pma_y/np.cos(inc*np.pi/180.)

    # deprojected pma
    # pma_ra_dep= pma_x_dep*np.sin(PA*np.pi/180.) - pma_y_dep*np.cos(PA*np.pi/180.)
    # pma_dec_dep= pma_x_dep*np.cos(PA*np.pi/180.) + pma_y_dep*np.sin(PA*np.pi/180.)


    
    eta = 1.0 / np.sqrt( pma_x_dep**2. + pma_y_dep**2. )
    return eta

##define the function that links the mass of the secondary, the radial distance, the mass of the primary and the v (the norm of the
## tangential PMa vector converted to linear velocity  using the GDR2, parallax, "dVt" in the Vizier catalogue):
def f_m2(rs, m1, v, PA_pma, epoch='DR3', inc=0.0, PA=0.0, gaia='eDR3'):

    eta=f_eta(PA_pma, inc, PA)
    periods=f_Period(rs, m1) # years
    zeta=np.zeros(len(rs))
    for i in range(len(rs)):
        zeta[i]=f_zeta_v2(rs[i], m1, 1,  epoch=epoch, gaia=gaia)  # when inc and PA is known
        #zeta[i]=f_zeta_v3(rs[i], m1, 1, np.cos(inc), epoch=epoch) # when inc and PA are unknown
    if epoch=='DR2':
        delta_t= 1038./ 365.24 # years, GAIA
    if epoch=='eDR3':
        delta_t= 668./ 365.24 # years, GAIA        
    elif epoch== 'Hipparcos':
        delta_t= 1227./ 365.24 # years, Hipparcos
    else:
        print('wrong epoch')
        delta=0.0
    return np.sqrt(rs*au) / f_gamma(periods/delta_t) * np.sqrt(m1*Msun/G) * v / (eta * zeta)/Mjup


def get_mass_MC(rs, PMa_ra, PMa_ra_err, PMa_dec, PMa_dec_err, parallax, epoch='eDR3', gaia='eDR3', mstar=1.0, inc=0., PA=0., inc_err=0., PA_err=0.,  Nrandom=3000 ):

    Nr=len(rs)

    PMa_ras=np.random.normal(PMa_ra, PMa_ra_err, Nrandom)
    PMa_decs=np.random.normal(PMa_dec, PMa_dec_err, Nrandom)

    if np.isnan(inc_err) or np.isnan(inc):
        print('unknown system inc')
        incs=np.arccos(np.random.uniform(0.0, 1.0, Nrandom))*180/np.pi 
    else:
        incs=np.random.normal(inc, inc_err, Nrandom)
    if np.isnan(PA) or np.isnan(PA_err):
        print('unknown system PA')
        PAs=np.random.uniform(0.0, 180.0, Nrandom)
    else:
        PAs=np.random.normal(PA, PA_err, Nrandom)

     
    msMC=np.zeros((Nr, Nrandom))
    ms=np.zeros((Nr, 7))

    PAs_PMa=np.zeros(Nrandom)

    #### MC
    for irandom in range(Nrandom):
        v_i=np.sqrt(PMa_ras[irandom]**2.+ PMa_decs[irandom]**2.)/ parallax *4740.470 # m/s
        PA_PMa_i=np.arctan2(PMa_ras[irandom], PMa_decs[irandom])*180./np.pi
        msMC[:,irandom]=f_m2(rs, mstar, v_i, PA_PMa_i, epoch=epoch, gaia=gaia, inc=incs[irandom], PA=PAs[irandom])
        PAs_PMa[irandom]=PA_PMa_i

        
    PAs_PMa[PAs_PMa<0.0]=PAs_PMa[PAs_PMa<0.0]+360.
    print('PA mean and std = %1.2f +- %1.2f'%(np.mean(PAs_PMa), np.std(PAs_PMa)))

    ### Retrieve percentiles
    for ir in range(Nr):
        ms[ir,:]=np.percentile(msMC[ir,:], [0.135, 2.28, 15.9, 50., 84.2, 97.7, 99.865 ])

    return ms

