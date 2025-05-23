import numpy as np
import math as ma
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import colorsys
from astropy.coordinates import SkyCoord
# from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
# import astropy.units as units



au_m=1.496e11 # m
mp_kg=1.67e-27 # kg
Mearth_kg=5.972e24 # kg
year_s= 3.154e7 # s
c_light=2.99792458e8 # m/s
c_light_cgs=2.99792458e10 # m/s
Kb=1.38064852e-23
h=6.62607004e-34 # mks
h_cgs=6.62607004e-27 # cgs

G=6.67384e-11 # mks
M_sun= 1.9891e30 # kg


def sci_notation(number, sig_fig=2): # stolen from https://stackoverflow.com/questions/53553377/python-scientific-notation-with-superscript-exponent
    ret_string = "{0:.{1:d}e}".format(number, sig_fig)
    a,b = ret_string.split("e")
    b = int(b)         # removed leading "+" and strips leading zeros too.
    
    # check exponent
    if b==0: 
        c=''
    else: c="x$10^{%i}$"%b
    
    # check digit
    if a=='1':
        if c=='':
            return '1'
        else:
            return c[1:]
    else: 
        return a + c


def number_to_text_10(number):

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


def number_to_text_sf(number, s=2):

    text='{{:#.{}g}}'.format(s).format(number)

    # if 'e' in text:
    #     number2=float(text)
    #     text='{}'.format(number2)
    
    if text[-1]=='.':
        text=text[:-1]

    return text


def round_sig(x, sig=1):
    if x == 0:
        return 0
    return round(x, -int(ma.floor(ma.log10(abs(x))) - (sig - 1)))

def format_number(n):
    return int(n) if n == int(n) else n

def round_asymmetric(value, err_plus=0, err_minus=0):

    if err_minus == 0:
        err_minus=err_plus

    if err_plus == 0 and err_minus == 0:
        return f"{value}"

    # Round errors to 1 significant figure
    err_p_rounded = round_sig(err_plus)
    err_m_rounded = round_sig(err_minus)

    # Use smaller of the two for rounding value
    smallest_mag = min(
        int(ma.floor(ma.log10(abs(err_p_rounded or 1)))), 
        int(ma.floor(ma.log10(abs(err_m_rounded or 1))))
    )
    rounded_value = round(value, -smallest_mag)

    return f"${format_number(rounded_value)}^{{+{format_number(err_p_rounded)}}}_{{-{format_number(err_m_rounded)}}}$"



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

def fcolor_x(i,N, colormap='viridis'):

    cmap=plt.get_cmap(colormap)

    x=i*1./(N-1)

    return cmap(x)

def lighten_color(color, amount=0.5):
    # copied from https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
    # values allowed between [0,2), >1 darkens, <1 lightens
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    try:
        c = cl.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*cl.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

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



def mean_free_path(cross_section, Sigma, r, L, M, mu, H=0.0):

    if H<=0.0:
        H=f_H(r, L, M, mu=mu) # au
    n_au= Sigma*Mearth_kg/(np.sqrt(2.*np.pi)*H*mu*mp_kg) # au-3
    n_mks= n_au/(au_m**3) # in m-3
    
    return 1./(cross_section*n_mks) # in m


def Poisson_error(n, S):
    ### Poissonian error approximation based on Gehrels+1986 http://adsabs.harvard.edu/abs/1986ApJ...303..336G
    ## n is the number of events and s is the confidence limit ("+-sigmas")


    ### upper limit
    # lambda_up= n + S*np.sqrt(n+1) + ( S**2 + 2.0) /3.0 # equation 10
    lambda_up= (n+1.0)* (1.0 - 1.0/(9.*(n+1.)) + S/(3.*np.sqrt(n+.1)))**(3.) # equation 9

    ### lower limit
    if S==1:
        beta=0.0
        gamma=1.0
    elif S==2:
        beta =0.062
        gamma=-2.22
    elif S==3:
        beta=0.222
        gamma=-1.88

    else: 
        print( "S different from 1,2 or 3. Error")
        return -1.0

    lambda_low= n*(1.-1./(9.*n) - S/(3.*np.sqrt(n))+beta*n**gamma )**3. # equation 14
    return lambda_up, lambda_low

def Bbody(lam, T): # function returns Planck function Bnu in Janskys
    # lam in m
    # T in K
    nu=c_light/lam # Hz

    return (2.*h*nu**3.)/(c_light**2.)  * 1./(np.exp(h*nu/(Kb*T))-1.)*1.0e26

def Bbody_lam(lam_um, T): # function returns Planck function Blam in cgs
    # lam in um
    # T in K
    lam_cm=lam_um*1.0e-4 # m
    nu=c_light/(lam_cm*1.0e-2) # Hz

    return (2.*h_cgs*c_light_cgs**2.)/(lam_cm**5.)  * 1./(np.exp(h*nu/(Kb*T))-1.) # cgs


def TempBB(r, Lstar=1.0):
    return 278.3 * r**(-0.5) * Lstar**0.25

def rBB(T, Lstar=1.0):
    return (278.3/T)**2.0*Lstar**0.5


def get_simbad(name, par='sptype'):
    customSimbad = Simbad()
    # customSimbad.list_votable_fields()

    # if len(par)>0:
    customSimbad.add_votable_fields(par)
    table = customSimbad.query_object(name)        
    return table

# def Gaia_ids(list_names, radius_arcsec=2.0):

#     customSimbad = Simbad()
#     customSimbad.add_votable_fields( 'pmra', 'pmdec', 'parallax', 'sptype', 'ubv', 'flux(U)')
#     # customSimbad.list_votable_fields()
 
#     list_gaia_ids=[]
#     for namei in list_names:

#         table = customSimbad.query_object(namei)        
#         try:
#             print 'coordinates ok'
#             c = SkyCoord(table['RA'][0],table['DEC'][0], unit=( units.hourangle, units.degree), frame='icrs')
#             print namei
#             print 'RA,DEC = ',c.ra.deg, c.dec.deg
#             # print table['PLX_VALUE'],table['PMRA'], table['PMDEC']
#             job = Gaia.launch_job_async("SELECT * \
#             FROM gaiadr2.gaia_source \
#             WHERE CONTAINS(POINT('ICRS',gaiadr2.gaia_source.ra,gaiadr2.gaia_source.dec), CIRCLE('ICRS',\
#             COORD1(EPOCH_PROP_POS(%1.8f, %1.8f, %1.8f, %1.8f, %1.8f , .6900, 2000,2015.5))," %(c.ra.deg, c.dec.deg,table['PLX_VALUE'], table['PMRA'], table['PMDEC'])+"\
#             COORD2(EPOCH_PROP_POS(%1.8f, %1.8f, %1.8f, %1.8f, %1.8f, .6900,2000,2015.5)), %1.6f))=1;"%(c.ra.deg, c.dec.deg,table['PLX_VALUE'],table['PMRA'], table['PMDEC'], radius_arcsec/3600.0), dump_to_file=False)

#             # 
#             obj=job.get_results()
#             #print obj.keys()
#             print obj['source_id'][0]
#             list_gaia_ids.append(obj['source_id'][0])
#         except:
#             print 'not in gaia, probably'
#             list_gaia_ids.append('')

#     return list_gaia_ids


###########################################################################
###################### Submm galaxies #####################################
###########################################################################
### From Carniani+2015, ALMA constraints on the faint millimetre source number counts and their contribution to the cosmic infrared background

def dNdS(phistar, Sstar, alpha, S):
    return phistar*(S/Sstar)**alpha * np.exp(-S/Sstar)/Sstar

def N_gtr_carniani(S,
                   phistar=2.7e3, # deg-2
                   Sstar=2.6,     # mJy
                   alpha=-1.81):
    # integrate dNdS from S=S to inf (100*Sstar) N=1000
    Nint=100
    Ss=np.logspace(np.log10(S), np.log10(1000*Sstar), Nint)
    f=Ss[1]/Ss[0]
    dSs=Ss*f-Ss
    return np.sum( dNdS(phistar, Sstar, alpha, Ss)*dSs ) # deg-2

### From Simpson+2015 807, 128, THE SCUBA-2 COSMOLOGY LEGACY SURVEY: ALMA RESOLVES THE BRIGHT-END OF THE SUB-MILLIMETER NUMBER COUNTS
def N_gtr_simpson(S, N0=390., S0=8.4, alpha=1.9, beta= 10.5):
    ### the equation does not make much dense dimensionally, but it
    ### agrees with the left panel in Figure 6 of Simpson+2015.
    return N0/S0 * ( (S/S0)**alpha + (S/S0)**beta )**(-1.0) # deg-2



def Integrate_probability_galaxy(rms, fwhm, wav, rmax=0.0, nsigma=3.):
    #wav in mm
    #rms in mJy
    #fwm in arcsec
    #rmax in arcsec
    
    # Simpson+2015
    if wav>0.85 and wav<0.9:
        N0=390.
        S0=8.4
        alpha=1.9
        beta=10.5
        func='Simpson'
    # Carniani+2015
    elif wav==1.1:
        phistar=2.7e3 # deg-2
        Sstar=2.6     # mJy
        alpha=-1.81
        func='Carniani'
    elif wav==1.3:
        phistar=1.8e3 # deg-2
        Sstar=1.7     # mJy
        alpha=-2.08
        func='Carniani'
    else:
        print('not a valid wavelength')
        return -1.0

    
    sig_PB=fwhm/(2.*np.sqrt(2.*np.log(2.)))
    # print 'sig_PB=',sig_PB
    N=10
    if rmax>0.0:
        print( 'integrate over circle of radius %1.2f arcsec'%rmax)
        redge=np.linspace(0.0, rmax, N+1) # bins over which to integrate
    else:
        print( 'integrate over full PB')
        redge=np.linspace(0.0, fwhm, N+1) # bins over which to integrate
   
    rmid=(redge[:-1]+redge[1:])/2.
    P=0.
    for i in xrange(N):

        rmsi=rms/np.exp( -(rmid[i])**2. / (2.*sig_PB**2.))
        dA=np.pi*(redge[i+1]**2 - redge[i]**2.)/(3600.0**2.) # deg2
        if func=='Carniani':
            Pi=N_gtr_carniani( rmsi*nsigma, phistar, Sstar, alpha)*dA
        elif func=='Simpson':
            Pi=N_gtr_simpson( rmsi*nsigma , N0, S0, alpha, beta)*dA
            
            # print rmsi, Pi
        P+=Pi

    return P


# from https://stackoverflow.com/questions/34017866/arrow-on-a-line-plot-with-matplotlib
def add_arrow(line, position=None, direction='right', size=15, color=None, zorder=100, arrowstyle="->"):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle=arrowstyle, color=color),
                       size=size,
                       zorder=zorder,
    )



def get_percentile_2D(logP, xs, ys, levels=[0.997, 0.95, 0.68]):
    """
    function to calculate contour levels that enclose confidence levels
    logP: 2D matrix containing log(P) with arbitrary normalization
    xs: array with x coordinates linearly or log spaced
    ys: array with y coordinates linearly or log spaced
    levels: array with confidence levels (<1)
    """

    levels_sorted=np.sort(levels)[::-1]
    
    # get dx and dy
    dxs=xs[1:]-xs[:-1]
    dys=ys[1:]-ys[:-1]

    if dxs[0]==dxs[1]: # linear spacing
        dxs=dxs[0]*np.ones_like(xs) 
    else:              # log spacing
        dxs=xs*(xs[1]/xs[0]-1.)
    if dys[0]==dys[1]: # linear spacing
        dys=dys[0]*np.ones_like(ys) 
    else:              # log spacing
        dys=ys*(ys[1]/ys[0]-1.)
    # Normalize P
    
    dxm, dym = np.meshgrid(dxs, dys)
    
    # subtract maximum and normalize
    logPmax=np.max(logP)
    P=np.exp(logP-logPmax)
    I=np.sum(P*dxm*dym)
    P=P/I
    
    Pvalues=np.sort(P.flatten())
    
    clevels=[]
    i=0
    for val in Pvalues:
        mask=P>val
        if i>=len(levels_sorted):
            break
        if np.sum(P[mask]*dxm[mask]*dym[mask])<levels_sorted[i]:
            clevels.append(val)
            i+=1
    
    return np.log(np.array(clevels)*I)+logPmax


def get_continuous_cmap(hex_list, float_list=None):                                                                               
    """
    Taken from https://github.com/KerryHalupka/custom_colormap 
 
    Creates and returns a color map that can be used in heat map figures.                                                             
    If float_list is not provided, the color map returned is a homogeneous gradient of the colors in hex_list.
    If float_list is provided, each color in hex_list is set to start at the corresponding location in float_list. 

    Parameters                                                                                        
    ----------                                                                                          
    hex_list: list of hex code strings                                                                
    float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.

    Returns     
    ----------
    matplotlib cmap

    Examples
    ----------
    fig, ax = plt.subplots(1,1)
    hex_list = ['#0091ad', '#fffffc', '#ffd166']
    x, y = np.mgrid[-5:5:0.05, -5:5:0.05]                                
    z = (np.sqrt(x**2 + y**2) + np.sin(x**2 + y**2))
    im = ax.imshow(z, cmap=get_continuous_cmap(hex_list))                                                         
    fig.colorbar(im)  
    ax.yaxis.set_major_locator(plt.NullLocator()) # remove y axis ticks 
    ax.xaxis.set_major_locator(plt.NullLocator()) # remove x axis ticks
    plt.show()
    """

    rgb_list = [matplotlib.colors.to_rgb(i) for i in hex_list]
    if float_list is None: float_list = np.linspace(0,1,len(rgb_list))

    cdict = dict()                                                                                
    for num, col in enumerate(['red', 'green', 'blue']):                                               
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
        cmap_new = matplotlib.colors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmap_new
