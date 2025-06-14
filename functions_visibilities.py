import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import frank
from frank.constants import deg_to_rad, rad_to_arcsec
from frank.hankel import DiscreteHankelTransform
from frank.constants import rad_to_arcsec
from frank.utilities import get_fit_stat_uncer, generic_dht

from scipy.special import jv
from scipy.optimize import curve_fit


os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

def reduced_chisq(model, uvtable):
    u,v, vis, w = uvtable

    vis_pred = model.predict(u, v)
    
    return np.sum(w*np.abs(vis_pred - vis)**2) / len(u)

def Chisq(model, uvtable):
    u,v, vis, w = uvtable

    vis_pred = model.predict(u, v)
    
    return np.sum(w*np.abs(vis_pred - vis)**2)


def plot_binned_uvdata(uvtable, geom, bin_size=1e3, ax=None, cut_data=None, imag=False, **plot_args):
    u,v, vis, w = uvtable
    
    up, vp, visp = geom.apply_correction(u,v,vis)
    if cut_data is not None:
        up, vp, visp, w = cut_data([up,vp, visp, w])
        
    binned_vis = frank.utilities.UVDataBinner(np.hypot(up,vp), visp, w, bin_size)
    
    if ax is None:
        ax = plt.gca()
    
    ax.errorbar(binned_vis.uv, binned_vis.V.real, binned_vis.error.real, marker='+', ls='', **plot_args)
    ax.set_ylabel('Re[V] [Jy]')
    if imag:
        ax.set_ylabel('Im[V] [Jy]')

    ax.set_xlabel('Baseline [$\lambda$]')
    ax.set_xscale('log')
    
    return binned_vis

def bin_uvdata(uvtable, geom, bin_size=1e3, cut_data=None):
    u,v, vis, w = uvtable
    
    up, vp, visp = geom.apply_correction(u,v,vis)
    if cut_data is not None:
        up, vp, visp, w = cut_data([up,vp, visp, w])
        
    binned_vis = frank.utilities.UVDataBinner(np.hypot(up,vp), visp, w, bin_size)
    
    return binned_vis

def bin_predicted_visibilites(model, uvtable, bin_size=1e3):
    u,v, vis, w = uvtable

    vis = model.predict(u, v)
    up, vp, visp = model.geometry.apply_correction(u,v,vis)
    
    return frank.utilities.UVDataBinner(np.hypot(up,vp), visp, w, bin_size)


def load_vis(path, fsigma=1.):

    try:
        uvtable=np.load(path)
    except:
        uvtable=np.loadtxt(path)
        
    mask=uvtable[:,4]>0.

    u,v=uvtable[mask,0], uvtable[mask,1]
    
    vis=uvtable[mask,2]+uvtable[mask,3]*1j # no imaginary part
    weights= uvtable[mask,4]/(fsigma**2.)

    #     fsigma=np.sqrt(np.sum(vis**2*weights)/len(u))
    #     weights=weights/(fsigma**2.)
    
    print('Chi_red = ',np.sum(np.abs(vis)**2*weights)/len(u)/2.)
    
    return u,v,vis,weights

class system():
    
    def __init__(self, name, dpc, inc, PA, alpha=1.4, wsmooth=1.e-1, Rmax=2., N=300,fstar=0., fsigma=1.0 ):
        self.name=name
        self.dpc=dpc
        self.inc=inc
        self.PA=PA
        self.alpha=alpha
        self.wsmooth=wsmooth
        self.Rmax=Rmax
        self.N=N
        self.fstar=fstar        
        self.fsigma=fsigma

        
### functions to produce model

def sample_delta(us, vs, ws, geom, r0_rad, flux, noise=False):
    
    usp, vsp = geom.deproject(us, vs) 

    rhosp=np.sqrt(usp**2+vsp**2)
    
    Vm=jv(0., rhosp*2*np.pi*r0_rad)*flux

    if noise:
        Vm+=np.random.normal(0., 1./np.sqrt(ws))
    
    return Vm



def gaussian_model(flux, r0_arcsec, sigma_arcsec, Rmax_arcsec=20, N=1000, return_I=False):
    
    rs=np.linspace(0., Rmax_arcsec, N) # arcsec
    dr=rs[1]-rs[0] # arcsec
    fs=np.zeros(N)
    fs[rs>0]=np.exp( - 0.5 * ((rs[rs>0]-r0_arcsec)/sigma_arcsec)**2.)
    fs=flux*fs/np.sum(fs*dr*2*np.pi*rs) # Jy/arcsec2
    fs=fs*(3600.*180./np.pi)**2. # Jy/arcsec2 to Jy/sr

    q, V=generic_dht(rs, fs, Rmax=Rmax_arcsec, N=N, direction='forward', grid=None,inc=0.0)

    if return_I:
        
        return rs, fs,q,V
    else:
        return q,V

def sample_gaussian(us, vs, ws, geom, flux, r0_arcsec, sigma_arcsec=0.01, noise=False):

    q,V= gaussian_model(flux, r0_arcsec, sigma_arcsec)
    
    usp, vsp = geom.deproject(us, vs) 
    rhosp=np.sqrt(usp**2+vsp**2)

    Vp = np.interp(rhosp, q, V)
    if noise:
        Vp+=np.random.normal(0., 1./np.sqrt(ws))
        
    return Vp






def gauss(x, A, x0, w ):
    
    sig=w/2.355
    
    return A*np.exp(-0.5* (x-x0)**2./sig**2 )

def get_psf(r, I, error, p0):
    
    popt, pcov = curve_fit(gauss, r, I, sigma=error, p0=p0)

    return popt[2]

def get_frank_psf_N(u,v, weights, geom, flux, r0_arcsec, alpha, wsmooth, N, Rmax, plot_sol=False, name=''):
    
    r0_rad=r0_arcsec/3600.*np.pi/180.
    
    vm=sample_delta(u, v, weights, geom, r0_rad, flux, noise=True)
    
    
    frankfit=frank.radial_fitters.FrankFitter(Rmax=Rmax,
                                              N=N,
                                              geometry=geom,
                                              alpha=alpha,
                                              weights_smooth=wsmooth,
                                              method='Normal',
                                              max_iter=4000
                                         )

    print('fitting data')
    sol = frankfit.fit(u, v, vm, weights)
    print('fit done')

    # get non-negative solution
    solp=sol.solve_non_negative() 

    # get error as a function of r
    error=get_fit_stat_uncer(sol)

    #get_psf(r, I, error, p0):
    
    popt, pcov = curve_fit(gauss, sol.r, solp, sigma=error, p0=[np.max(solp), r0_arcsec, r0_arcsec/10.])

    psf=popt[2]
    

    if plot_sol:

        fig=plt.figure(figsize=(8,6))
        ax1=fig.add_subplot(111)

        unit_conversion=1.0e3*(np.pi/(180*3600.))**2 # from Jy/sr to mJy/arcsec2

        ax1.plot(sol.r, sol.I*unit_conversion, '-', color='C0', label='Normal sol')
        ax1.fill_between(sol.r, (sol.I-error)*unit_conversion, (sol.I+error)*unit_conversion, color='C0', alpha=0.3)
        ax1.plot(sol.r, solp*unit_conversion, '-', color='C1', label='Normal Non-neg sol')

        ax1.plot(sol.r, gauss(sol.r, *popt)*unit_conversion, color='C2', label='Gaussian fit')

        ax1.plot([r0_arcsec-psf/2., r0_arcsec+psf/2.], np.array([1., 1.])*popt[0]*unit_conversion/2, color='black')

        ax1.set_xlabel('r [arcsec]')
        ax1.set_ylabel('I [mJy/arcsec2]')
        plt.legend(loc=1)
        plt.tight_layout()
        plt.savefig('psf_fit_{}.pdf'.format(name))
        
    return psf

def get_frank_psf_LN(u,v, weights, geom, flux, r0_arcsec, alpha, wsmooth, N, Rmax, plot_sol=False, name='', sigma_arcsec=0.1):
    
    
    vm=sample_gaussian(u, v, weights, geom, flux, r0_arcsec, sigma_arcsec=sigma_arcsec, noise=True)
    
    
    frankfit=frank.radial_fitters.FrankFitter(Rmax=Rmax,
                                              N=N,
                                              geometry=geom,
                                              alpha=alpha,
                                              weights_smooth=wsmooth,
                                              method='LogNormal',
                                              max_iter=10000,
                                              I_scale=1.0e4
                                         )

    print('fitting data')
    sol = frankfit.fit(u, v, vm, weights)
    print('fit done')


    # get error as a function of r
    error=get_fit_stat_uncer(sol)
    
    popt, pcov = curve_fit(gauss, sol.r, sol.I, p0=[np.max(sol.I), r0_arcsec, r0_arcsec/10.]) # remove error ( sigma=error) because of strange frank uncertainties 

    psf=popt[2]
    

    if plot_sol:

        fig=plt.figure(figsize=(8,6))
        ax1=fig.add_subplot(111)

        unit_conversion=1.0e3*(np.pi/(180*3600.))**2 # from Jy/sr to mJy/arcsec2

        ax1.plot(sol.r, sol.I*unit_conversion, '-', color='C0', label='Normal sol')
        ax1.fill_between(sol.r, (sol.I-error)*unit_conversion, (sol.I+error)*unit_conversion, color='C0', alpha=0.3)

        ax1.plot(sol.r, gauss(sol.r, *popt)*unit_conversion, color='C2', label='Gaussian fit')

        ax1.plot([r0_arcsec-psf/2., r0_arcsec+psf/2.], np.array([1., 1.])*popt[0]*unit_conversion/2, color='black')

        ax1.set_xlabel('r [arcsec]')
        ax1.set_ylabel('I [mJy/arcsec2]')
        ax1.set_ylim(0., 1.2*np.nanmax(sol.I*unit_conversion))
        plt.legend(loc=1)
        plt.tight_layout()
        plt.savefig('psf_fit_{}.pdf'.format(name))
        
    return psf


"""
#### old functions previusly in functions_image

########################################
############ VISIBILITIES #############
########################################


def deproj_vis(u,v,Inc,pa):

    # PA=90 means that both reference systems are aligned (x is positive to the right)
    
    inc=Inc*np.pi/180.0

    PArad=pa*np.pi/180.0
    ### up,vp derived from argument of exp(-2pi (ux+vy))=exp(-2pi(up
    ### xp+vpyp)), where xp and yp are deprojected coordinates with xp along pa 

    up = u*np.sin(PArad)+v*np.cos(PArad)
    vp = -u*np.cos(PArad)*np.cos(inc)+v*np.sin(PArad)*np.cos(inc)

    return up,vp


def bin_dep_vis(uvmin, uvmax, Nr, us, vs, reals, imags, Inc, PA, weights=[1.0]):

    # -el minimo uv distance a considerar de las visibilidades deprojectadas (0 por ejemplo)
    # - el maximo uv distance a considerar de las visibilidades deprojectadas (1e6 por ejemplo)
    # - El numero de bins. Este es un parametro que tienes que jugar para sacarle el mayor provecho a tus datos. En general yo ocupo valores de ~50, pero si tu S/N es bueno, puedes ocupar valores mayores
    # - un array de las coordenadas u
    # - un array de las coordenadas v
    # - un array de las componentes reales
    # - un array de las componentes imaginarias
    # - la inclinacion en grados
    # - el PA en grados

    
    amps=np.sqrt(reals**2+imags**2)
    if len(weights)==1:
        weights=np.ones(len(reals))
        
    u_dep,v_dep= deproj_vis(us,vs,Inc,PA)
    ruv=np.sqrt(u_dep**2.0+v_dep**2.0)

    Rs_edge=np.linspace(uvmin, uvmax, Nr+1)
    dR=Rs_edge[1:]-Rs_edge[:-1]
    Rs=Rs_edge[:-1]+dR/2.0

    Amp_mean=np.zeros(Nr)
    Amp_error=np.zeros(Nr)
    Amp_std=np.zeros(Nr)

    Real_mean=np.zeros(Nr)
    Real_error=np.zeros(Nr)
    Real_std=np.zeros(Nr)


    Imag_mean=np.zeros(Nr)
    Imag_error=np.zeros(Nr)
    Imag_std=np.zeros(Nr)


    for ir in range(Nr):
        #print ir, Nr
        n=0

        mask= ((Rs_edge[ir]<ruv) & (ruv<Rs_edge[ir+1])  & (weights!=0.0) & (reals!=0.0))
        n=len(ruv[mask])
    
        if n>10:#150.0:

            Real_mean[ir]=np.mean(reals[mask])
            Real_std[ir]=np.std(reals[mask])
            Real_error[ir]=Real_std[ir]/np.sqrt(n)
            
            Imag_mean[ir]=np.mean(imags[mask])
            Imag_std[ir]=np.std(reals[mask])
            Imag_error[ir]=Imag_std[ir]/np.sqrt(n)

            Amp_mean[ir]=( Real_mean[ir]**2.0 +  Imag_mean[ir]**2.0)**0.5
            # S_amp=S_amp/n
            Amp_std[ir]= ( Real_std[ir]**2.0 +  Imag_std[ir]**2.0)**0.5
            # np.sqrt(S_amp-Amp_mean[ir]**2.0)
            Amp_error[ir]=( Real_error[ir]**2.0 +  Imag_error[ir]**2.0)**0.5
            # Amp_std[ir]/np.sqrt(n)
        else:
             Real_mean[ir]=np.nan
             Imag_mean[ir]=np.nan
             Amp_mean[ir]=np.nan
             
    return Rs_edge, Rs, np.array([Real_mean, Real_std, Real_error]), np.array([Imag_mean, Imag_std, Imag_error]), np.array([Amp_mean, Amp_std, Amp_error])
"""
