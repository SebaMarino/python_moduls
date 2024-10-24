import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import frank
from frank.constants import deg_to_rad, rad_to_arcsec
from frank.hankel import DiscreteHankelTransform
from frank.constants import rad_to_arcsec
from frank.utilities import get_fit_stat_uncer

from scipy.special import jv
from scipy.optimize import curve_fit


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

def model_vis(us, vs, ws, geom, r0_rad, flux, noise=False):
    
    usp, vsp = geom.deproject(us, vs) 

    rhosp=np.sqrt(usp**2+vsp**2)
    
    Vm=jv(0., rhosp*2*np.pi*r0_rad)*flux

    if noise:
        Vm+=np.random.normal(0., 1./np.sqrt(ws))
    
    return Vm

def gauss(x, A, x0, w ):
    
    sig=w/2.355
    
    return A*np.exp(-0.5* (x-x0)**2./sig**2 )

def get_psf(r, I, error, p0):
    
    popt, pcov = curve_fit(gauss, r, I, sigma=error, p0=p0)

    return popt[2]

def get_frank_psf(u,v, weights, geom, flux, r0_arcsec, alpha, wsmooth, N, Rmax):
    
    r0_rad=r0_arcsec/3600.*np.pi/180.
    
    vm=model_vis(u, v, weights, geom, r0_rad, flux, noise=True)
    
    
    frankfit=frank.radial_fitters.FrankFitter(Rmax=Rmax,
                                              N=N,
                                              geometry=geom,
                                              alpha=alpha,
                                              weights_smooth=wsmooth,
                                              method='Normal',
                                         )

    print('fitting data')
    sol = frankfit.fit(u, v, vm, weights)
    print('fit done')

    # get non-negative solution
    solp=sol.solve_non_negative() 

    # get error as a function of r
    error=get_fit_stat_uncer(sol)
    
    psf=get_psf(sol.r, solp, error, [np.max(solp), r0_arcsec, r0_arcsec/10.])
    
    return psf
