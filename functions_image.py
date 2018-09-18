import numpy as np
import astropy.io.fits as pyfits
from astropy.convolution import convolve_fft
import matplotlib.colors as cl
import matplotlib.pyplot as plt
import os,sys
from matplotlib.patches import Ellipse
from matplotlib.colors import LogNorm
from matplotlib import rc
import copy

G=6.67384e-11 # mks
M_sun= 1.9891e30 # kg
au=1.496e11 # m



def dsdx(x,a,b): # ds/dx over an ellipse
    
    return np.sqrt(1.  +  (b**2/a**4) *  x**2./(1.-(x/a)**2))

def Delta_s(xs, a,b): # delta s over an ellipse

    ys=np.zeros(len(xs))
    mask=((xs >= a ) & (xs<=-a))

    if xs[-1]<xs[0]:
        ys[mask]=b*np.sqrt(1.-xs[mask]**2.0/a**2)
    else:
        ys[mask]=-b*np.sqrt(1.-xs[mask]**2.0/a**2)
    return np.sqrt( (ys[:-1]-ys[1:])**2.0 + (xs[:-1]-xs[1:])**2.0 )

def simple_phi(phi):
    if abs(phi)!=2.0*np.pi:
        phir=phi%(2.0*np.pi)
    else: 
        phir=phi
    if phir<0.0:
        phir=2.0*np.pi+phir
    return phir

def x_phi(phi, a, b):
    
    phi=simple_phi(phi)
    
    if phi<=np.pi/2. or phi>=3.*np.pi/2.0:
        sign=1.0
    else:
        sign=-1.0
    return sign/np.sqrt(np.tan(phi)**2.0/b**2.0 + 1./a**2.)

def y_phi(phi, a, b):
    
    phi=simple_phi(phi)
    
    if phi>=0.0 and phi<=np.pi:
        sign=1.0
    else:
        sign=-1.0
    
    return sign/np.sqrt(1./b**2.0 + 1./(a**2.*np.tan(phi)**2.0))


def separate_intervals(phi1, phi2, Nint):
    # Figure out the right ranges of xs' to integrate. Does phi=0.0 pr phi=180 is contained in range?
    phis=[phi1]
    if phi1>phi2: # passes through zero
        phis.append(0.0)
    if phi2>np.pi and phis[-1]<np.pi:
        phis.append(np.pi)
    phis.append(phi2)
    return np.array(phis)
    
def arc_length(a,b,phi1, phi2):
    # works with dy/dx
    Nint=1000000

    phi1=simple_phi(phi1) 
    phi2=simple_phi(phi2)
    # Figures out the right ranges of xs' to integrate. Does phi=0.0 pr phi=180 is contained in range?
    phis=separate_intervals(phi1, phi2, Nint)
    
    Nph=len(phis)
    Arc_length=0.0

    for i in xrange(Nph-1):
        if phis[i]==0.0:
            phi_i=1./Nint
        elif phis[i]==np.pi:
            phi_i=np.pi+1./Nint
        else:
            phi_i=phis[i]
            
        if phis[i+1]==0.0:
            phi_ii=2.0*np.pi-1./Nint
        elif phis[i+1]==np.pi:
            phi_ii=np.pi-1./Nint
        else:
            phi_ii=phis[i+1]   
        x1=x_phi(phi_i,a,b)
        x2=x_phi(phi_ii, a,b)
        dx=(x2-x1)/(Nint-1)
        xs=np.arange(x1,x2,dx)

        Arc_length+=np.sum(dsdx(xs, a,b)*np.abs(dx))
        
    return Arc_length, phis

def arc_length2(a,b,phi1, phi2):
    # works with delta y / delta x
    Nint=1000000

    phi1=simple_phi(phi1) 
    phi2=simple_phi(phi2)
    # Figure out the right ranges of xs' to integrate. Does phi=0.0 pr phi=180 is contained in range?
    phis=separate_intervals(phi1, phi2, Nint)
    
    Nph=len(phis)
    Arc_length=0.0

    for i in xrange(Nph-1):

        # if phis[i]==0.0:
        #     phi_i=1./Nint
        # elif phis[i]==np.pi:
        #     phi_i=np.pi+1./Nint
        # else:
        phi_i=phis[i]
            
        # if phis[i+1]==0.0:
        #     phi_ii=2.0*np.pi-1./Nint
        # elif phis[i+1]==np.pi:
        #     phi_ii=np.pi-1./Nint
        # else:
        phi_ii=phis[i+1]  
 
        x1=x_phi(phi_i,a,b)
        x2=x_phi(phi_ii, a,b)
        dx=(x2-x1)/(Nint-1)
        xs=np.arange(x1,x2,dx)

        Arc_length+=np.sum(Delta_s(xs, a,b))
        
    return Arc_length, phis



def ellipse1(x0,y0,phi,chi,a, PA):

    ## phi is a position angle, but in the plane of the disc (i.e. inclined)
    
    # x0,y0 ellipse center
    # phi pa at which calculate x,y phi=0 is +y axis
    # chi aspect ratio of ellipse with chi>1
    # a semi-major axis
    # a/chi semi-minor axis
    # PA  pa of ellipse 0 is north and pi/2 is east

    phipp= phi-PA
    
    xpp = (a/chi) * np.sin(phipp) 
    ypp =    a    * np.cos(phipp)

    xp =  xpp*np.cos(PA) + ypp*np.sin(PA)
    yp = -xpp*np.sin(PA) + ypp*np.cos(PA)
    
    xc = xp + x0
    yc = yp + y0
    
    return xc , yc


def ellipse2(x0,y0,phi,chi,a, PA):

    ## phi is a position angle in the plane of the sky
    
    # x0,y0 ellipse center
    # phi pa at which calculate x,y phi=0 is +y axis
    # chi aspect ratio of ellipse with chi>1
    # a semi-major axis
    # a/chi semi-minor axis
    # PA  pa of ellipse 0 is north and pi/2 is east

    phipp= phi-PA
    
    
    xpp=x_phi(np.pi/2.0-phipp, a/chi, a)
    ypp=y_phi(np.pi/2.0-phipp, a/chi, a)
    #xpp = (a/chi) * np.sin(phipp) 
    #ypp =    a    * np.cos(phipp)

    xp =  xpp*np.cos(PA) + ypp*np.sin(PA)
    yp = -xpp*np.sin(PA) + ypp*np.cos(PA)
    
    xc = xp + x0
    yc = yp + y0
    
    return xc , yc
 



def xyarray(Np, ps_arcsec):

    xedge=np.zeros(Np+1)
    yedge=np.zeros(Np+1)

    xs=np.zeros(Np)
    ys=np.zeros(Np)

    for i in xrange(Np+1):

        xedge[i]=-(i-Np/2.0)*ps_arcsec#-ps_arcsec/2.0        
        yedge[i]=(i-Np/2.0)*ps_arcsec#+ps_arcsec/2.0

    for i in xrange(Np):
        xs[i]=-(i-Np/2.0)*ps_arcsec-ps_arcsec/2.0           
        ys[i]=(i-Np/2.0)*ps_arcsec+ps_arcsec/2.0

    return xs, ys, xedge, yedge

def radial_profile(image, image_pb, x0, y0, PA, inc, rmax,Nr, phis, rms, BMAJ_arcsec, ps_arcsec, error_std=False, arc='elipse'):#, plot=False):

    # x0, y0 are RA DEC offsets in arcsec
    # PA and inc are PA and inc of the disc in deg
    # rmax [arcsec] is the maximum deprojected radius at which to do the azimuthal averaging
    # Nr is the number of radial points to calculate
    # phis [deg] is an array with uniform spacing that sets the range of PA at which to do the interpolation (0 is north)

    # ################ SPATIAL GRID

    # XY

    Np=len(image[:,0])

    xs, ys, xedge, yedge = xyarray(Np, ps_arcsec)
    
    # R phi

    PA_rad=PA*np.pi/180.0
    phis_rad=phis*np.pi/180.0 
    dphi=abs(phis_rad[1]-phis_rad[0])
    Nphi=len(phis_rad)

    rs=np.linspace(rmax/1.0e3,rmax,Nr)

    ecc= np.sin(inc*np.pi/180.0)
    chi=1.0/(np.sqrt(1.0-ecc**2.0)) # aspect ratio between major and minor axis (>=1)
    
    ##### Calculate averaged profile
    
    Irs1=np.zeros((Nr,Nphi))
    Irs2=np.zeros((Nr,Nphi)) # pb
    for i_r in xrange(Nr):
        ai=rs[i_r]
        for i_p in xrange(Nphi):

            phi1=phis_rad[i_p]  # in the plane of the disc
            XS1,YS1=ellipse2(x0,y0,phi1,chi,ai, PA_rad)
        
            ip1 = -int(XS1/ps_arcsec)+Np/2
            jp1 = int(YS1/ps_arcsec)+Np/2
            
            Irs1[i_r,i_p] = image[jp1,ip1] 
            Irs2[i_r,i_p] = image_pb[jp1,ip1]

    # if plot:
    #     imagep=image*1.0
    #     for i_p in xrange(Nphi):
            
    #         phi1=phis_rad[i_p] 
    #         XS1,YS1=ellipse(x0,y0,phi1,chi,2.8, PA_rad)
        
    #         ip1 = -int(XS1/ps_arcsec)+Np/2
    #         jp1 = int(YS1/ps_arcsec)+Np/2
            
    #         imagep[jp1,ip1]=0.0#imagep[jp1,ip1]
    #     fig=plt.figure()
    #     ax1=fig.add_subplot(111)
    #     pc=ax1.pcolormesh(imagep, vmin=0.0, vmax=2.0e-5)
    #     cb= fig.colorbar(pc)
    #     ax1.set_aspect('equal')
    #     plt.show()
        
    Ir1=np.nanmean(Irs1, axis=1) # mean intensity in Jy/beam
    Ir2=np.zeros(Nr)
    
    for i in xrange(Nphi):
        Ir2=Ir2+(rms/Irs2[:,i])**2.0

    Ir2=np.sqrt(Ir2/(Nphi))

    # Calculate number of independent points 


    
    # arclength=dphi*(Nphi-1) # radians
    if arc=='simple_elipse':
        arclength=(Nphi-1)*dphi* np.sqrt(  (1.0 + (1.0/chi)**2.0 )/2.0 ) # normalice 
    else:
        arclength, phiint= arc_length2(1.0,1.0/chi, phis_rad[0], phis_rad[-1])

    print 'arc length = [deg] ', arclength*180.0/np.pi
    Nindeps_1=rs*arclength/BMAJ_arcsec
    Nindeps_1[Nindeps_1<1.0]=1.0
    

    if error_std:
        Err_1=np.nanstd(Irs1, axis=1)/np.sqrt(Nindeps_1)

    else:
        Err_1=Ir2/np.sqrt(Nindeps_1)

    
    return np.array([rs, Ir1, Err_1]) # rs, I, eI 




def radial_profile_fits_model(fitsfile, x0, y0, PA, inc, rmax,Nr, phis, arc='elipse'):#, plot=False):


    fit1=pyfits.open(fitsfile)
    data1 	= get_last2d(fit1[0].data) #extraer matriz de datos
    
    # print np.shape(data1)
    header1=fit1[0].header
    ps_deg1=float(header1['CDELT2'])
    ps_arcsec1=ps_deg1*3600.0
    ps_rad1=ps_deg1*np.pi/180.0
    Np1=len(data1[:,0])

    # BMAJ1=float(header1['BMAJ']) # deg
    # BMIN1=float(header1['BMIN']) # deg
    # BPA1=float(header1['BPA']) # deg

    # BMAJ_arcsec1=BMAJ1*3600.0
    # BMIN_arcsec1=BMIN1*3600.0

    # change units from Jy/pix to Jy/arcsec

    if header1['BUNIT']=='JY/PIXEL':
        
        try:
            BMAJ=float(header1['BMAJ']) # deg
            BMIN=float(header1['BMIN']) # deg
            BPA=float(header1['BPA']) # deg

            BMAJ_arcsec1=BMAJ*3600.0
            BMIN_arcsec1=BMIN*3600.0
            print BMAJ_arcsec1, BMIN_arcsec1
            beam_area=np.pi*BMAJ_arcsec1*BMIN_arcsec1/(4.*np.log(2.))
            
            data1=data1*beam_area/(ps_arcsec1**2)
        except:
            data1=data1/(ps_arcsec1**2)
    return radial_profile(data1, np.ones((Np1,Np1)), x0, y0, PA, inc, rmax,Nr, phis, rms=0.0, BMAJ_arcsec=1.0, ps_arcsec=ps_arcsec1, arc=arc)


def radial_profile_fits_image(fitsfile_pbcor, fitsfile_pb, x0, y0, PA, inc, rmax,Nr, phis, rms, error_std=False, arcsec2=False , arc='elipse'):


    fit1=pyfits.open(fitsfile_pbcor)
    data1 = get_last2d(fit1[0].data) #extraer matriz de datos


    # print np.shape(data1)
    header1=fit1[0].header
    ps_deg1=float(header1['CDELT2'])
    ps_arcsec1=ps_deg1*3600.0
    ps_rad1=ps_deg1*np.pi/180.0
    Np1=len(data1[:,0])

    if fitsfile_pb!='':
        fit2=pyfits.open(fitsfile_pb)
        data2 = get_last2d(fit2[0].data) #extraer matriz de datos
    else:
        data2=np.ones((Np1,Np1))

    
    BMAJ=float(header1['BMAJ']) # deg
    BMIN=float(header1['BMIN']) # deg
    BPA=float(header1['BPA']) # deg

    BMAJ_arcsec1=BMAJ*3600.0
    BMIN_arcsec1=BMIN*3600.0
    beam_area=np.pi*BMAJ_arcsec1*BMIN_arcsec1/(4.*np.log(2.))
    
    print "beam = %1.2f x %1.2f" %(BMAJ_arcsec1, BMIN_arcsec1)
    if arcsec2:
        data1=data1/beam_area
        rms=rms/beam_area
        print 'hey'
    return radial_profile(data1, data2, x0, y0, PA, inc, rmax,Nr, phis, rms=rms, BMAJ_arcsec=BMAJ_arcsec1, ps_arcsec=ps_arcsec1, error_std=error_std, arc=arc)




def flux_profile(image, image_pb, x0, y0, PA, inc, rmax,Nr, phis, rms, BMAJ_arcsec, BMIN_arcsec, ps_arcsec):

    # x0, y0 are RA DEC offsets in arcsec
    # PA and inc are PA and inc of the disc in deg
    # rmax [arcsec] is the maximum deprojected radius at which to do the azimuthal averaging
    # Nr is the number of radial points to calculate
    # phis [deg] is an array with uniform spacing that sets the range of PA at which to do the interpolation (0 is north)

    # ################ SPATIAL GRID

    # XY

    Np=len(image[:,0])

    xs, ys, xedge, yedge = xyarray(Np, ps_arcsec)

    
    # R phi

    PA_rad=PA*np.pi/180.0
    phis_rad=phis*np.pi/180.0 
    dphi=abs(phis_rad[1]-phis_rad[0])
    Nphi=len(phis_rad)

    rs=np.linspace(rmax/1.0e3,rmax,Nr)
    print rs
    
    ecc= np.sin(inc*np.pi/180.0)
    chi=1.0/(np.sqrt(1.0-ecc**2.0)) # aspect ratio between major and minor axis (>=1)

    Beam_area=np.pi*BMAJ_arcsec*BMIN_arcsec/(4.0*np.log(2.0)) # in arcsec2

    
    ##### Calculate flux within rs
    F=np.zeros((Nr,2))


    xv, yv = np.meshgrid(xs-x0, ys-y0, sparse=False, indexing='xy')

    xpp = xv * np.cos(PA_rad) - yv *np.sin(PA_rad) ### along minor axis
    ypp = xv * np.sin(PA_rad) + yv *np.cos(PA_rad) ###  along major axis

    # PAs=np.arctan2(xpp,ypp)*180.0/np.pi

    rdep=np.sqrt( (xpp*chi)**2.0 + ypp**2.0 ) # real deprojected radius

    rmsmap2=(rms/image_pb)**2.0
    
    print Beam_area
    for i_r in xrange(Nr):

        F[i_r,0]= np.sum(image[rdep<rs[i_r]]*(ps_arcsec**2.0)/Beam_area)
        F[i_r,1]= np.sum( rmsmap2[rdep<rs[i_r]]) # Jy/beam Note: /beam is ok as then it is correct

        # Correct by number of independent points
        
        F[i_r,1]= np.sqrt(F[i_r,1]) * np.sqrt(ps_arcsec**2.0/Beam_area)
        
    # plt.pcolor(xs, ys, image)
    # plt.contour(xs,ys,rdep)
    # plt.show()
    return np.array([rs, F[:,0], F[:,1]]) # rs, I, eI 


def flux_profile_fits_image(fitsfile_pbcor, fitsfile_pb, x0, y0, PA, inc, rmax,Nr, phis, rms):


    fit1=pyfits.open(fitsfile_pbcor)
    data1=get_last2d(fit1[0].data) 

    # print np.shape(data1)
    header1=fit1[0].header
    ps_deg1=float(header1['CDELT2'])
    ps_arcsec1=ps_deg1*3600.0
    ps_rad1=ps_deg1*np.pi/180.0
    Np1=len(data1[:,0])

    if fitsfile_pb!='':
        fit2=pyfits.open(fitsfile_pb)
        data2 = get_last2d(fit2[0].data) #extraer matriz de datos
    else:
        data2=np.ones((Np1,Np1))

    
    BMAJ=float(header1['BMAJ']) # deg
    BMIN=float(header1['BMIN']) # deg
    BPA=float(header1['BPA']) # deg

    BMAJ_arcsec1=BMAJ*3600.0
    BMIN_arcsec1=BMIN*3600.0
    
    return flux_profile(data1, data2, x0, y0, PA, inc, rmax,Nr, phis, rms=rms, BMAJ_arcsec=BMAJ_arcsec1, BMIN_arcsec=BMIN_arcsec1,  ps_arcsec=ps_arcsec1)




def Gauss2d(xi , yi, x0,y0,sigx,sigy,theta):

        xp= (xi-x0)*np.cos(theta) + (yi-y0)*np.sin(theta)
        yp= -(xi-x0)*np.sin(theta) + (yi-y0)*np.cos(theta)

        a=1.0/(2.0*sigx**2.0)
        b=1.0/(2.0*sigy**2.0)

        return np.exp(- ( a*(xp)**2.0 + b*(yp)**2.0 ) )#/(2.0*np.pi*sigx*sigy)


def Convolve_beam(path_image, BMAJ, BMIN, BPA):

    #  -----cargar fit y extraer imagen

    fit1	= pyfits.open(path_image) #abrir objeto cubo de datos

    try: 
        data1 	= fit1[0].data[0,0,:,:] #extraer matriz de datos
    except:
        try:
            data1 	= fit1[0].data[0,:,:] #extraer matriz de datos
        except:
            data1 	= fit1[0].data[:,:] #extraer matriz de datos

    print np.shape(data1)

    header1	= fit1[0].header
    ps_deg=float(header1['CDELT2'])
    ps_mas= ps_deg*3600.0*1000.0 # pixel size input in mas
    dtheta=ps_deg*np.pi/180.0 # dtheta in rad

    M=len(data1[:,0])

    N=M # dim output

    ps1= ps_mas # pixel size input in mas
    ps2= ps1 # pixel size output in mas

    dtheta2=ps2*np.pi/(3600.0*1000.0*180.0)

    d=0

    Fin1=data1[:,:]#*1e23/(dtheta**2.0) #  JY/PIXEL to ergs/s cm2 Hz sr

    x1=np.zeros(N)
    y1=np.zeros(N)
    for i in range(N):
	x1[i]=(i-M/2.0)*ps1
	y1[i]=(i-N/2.0)*ps1


    # BMAJ = float(sys.argv[1])#2.888e-4#*3600.0 #deg
    # BMIN = float(sys.argv[2])#2.240e-4#*3600.0  #deg
    # PA   = float(sys.argv[3])#63.7       # deg


    sigx=BMIN*3600.0*1000.0/(2.0*np.sqrt(2.0*np.log(2.0))) 
    sigy=BMAJ*3600.0*1000.0/(2.0*np.sqrt(2.0*np.log(2.0)))  
    theta=BPA*np.pi/180.0   #np.pi/4.0

    # Fout1=interpol(N,M,ps1,ps2,Fin1,sigx,sigy,theta)

    Gaussimage=np.zeros((N,N))
    for j in xrange(N):
        for i in xrange(N):
            x=(i-N/2.0)*ps2
            y=(j-N/2.0)*ps2
            Gaussimage[j,i]=Gauss2d(x,y,0.0,0.0,sigx,sigy,theta)
    # Gaussimage=Gaussimage/np.max(Gaussimage)


    Fout1=convolve_fft(Fin1,Gaussimage)

    # a=BMAJ*np.pi/180.0
    # b=BMIN*np.pi/180.0
    # C=np.pi*a*b/(4.0*ma.log(2.0)) # in strad
    # Fout1=Fout1*C/(dtheta**2.0) # Jy/pixel to Jy/beam


    # x2=np.zeros(N+1)
    # y2=np.zeros(N+1)
    # for i in range(N+1):
    #     x2[i]=-(i-(N-1)/2.0)*ps2-ps2/2.0
    #     y2[i]=(i-(N-1)/2.0)*ps2-ps2/2.0

    # x2=x2/1000.0
    # y2=y2/1000.0

    header1['BMIN'] = BMIN
    header1['BMAJ'] = BMAJ
    header1['BPA'] = BPA

    header1['BUNIT']='Jy/beam'

    path_fits=path_image[:-5]+'_beamconvolved.fits'
    os.system('rm '+ path_fits)
    pyfits.writeto(path_fits, Fout1, header1, output_verify='fix')


    
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
    

def inter(Nin,Nout,i,j,ps1,ps2,Fin):

	f=0.0
	S=0.0
	a=1.0*ps1
	di=(i-Nout/2.0)*ps2
	dj=(j-Nout/2.0)*ps2
	
	ni=int(di/ps1+Nin/2.0-2.0*a/ps1)
	mi=int(dj/ps1+Nin/2.0-2.0*a/ps1)
	nmax=int(di/ps1+Nin/2.0+2.0*a/ps1)
	mmax=int(dj/ps1+Nin/2.0+2.0*a/ps1)
	if ni<0: ni=0
	if mi<0: mi=0
	if nmax<0: nmax=0
	if mmax<0: mmax=0
	if ni>Nin-1: ni=Nin-1
	if mi>Nin-1: mi=Nin-1
	if nmax>Nin-1: nmax=Nin-1
	if mmax>Nin-1: mmax=Nin-1
	for n in range(ni,nmax):

		dn=(n-Nin/2.0)*ps1
		k=0
		for m in range(mi,mmax):
			
			dm=(m-Nin/2.0)*ps1
			
			r=np.sqrt((dn-di)**2.0+(dm-dj)**2.0)
			if r<2*a: 
				P=np.exp(-r**2.0/(2.0*a**2.0))
				f=f+P*Fin[n,m]
				S=S+P
				k=1
			elif k==1: break

	if S==0.0: return 0.0		
	else: 
		return f/S

def interpol(Nin,Nout,ps1,ps2,Fin):
    print Nin, Nout
    if ps1!=ps2 or Nin!=Nout:
	F=np.zeros((Nout,Nout), dtype=np.float64)
	for i in range(Nout):
                #print i
		for j in range (Nout):	
			F[i,j]=inter(Nin,Nout,i,j,ps1,ps2,Fin)
    else:
        print 'no interpolation'
        F=Fin
    return F

    
def fload_fits_image(path_image, path_pbcor, rms, ps_final, XMAX): # for images from CASA

    ### PS_final in mas

    ##### LOAD IMAGE
    fit1	= pyfits.open(path_image) # open image cube
    data1 	= get_last2d(fit1[0].data) # [0,0,:,:] # extract image matrix

    #### READ HEADER
    header1	= fit1[0].header
    ps_deg1=float(header1['CDELT2'])
    ps_mas1= ps_deg1*3600.0*1000.0 # pixel size input in mas
    ps_arcsec1=ps_deg1*3600.0
    
    N1=len(data1[:,0])


    if path_pbcor!='':
        fit2	= pyfits.open(path_pbcor) #abrir objeto cubo
        data2 	= get_last2d(fit2[0].data) # [0,0,:,:] #extraer matriz de datos

    else:
        data2=np.ones((N1,N1))
        
    rmsmap=rms/data2

    try:
        BMAJ=float(header1['BMAJ'])*3600.0 # arcsec 
        BMIN=float(header1['BMIN'])*3600.0 # arcsec 
        BPA=float(header1['BPA']) # deg 
        print "beam = %1.2f x %1.2f" %(BMAJ, BMIN)
    except:
        BMAJ=0.0
        BMIN=0.0
        BPA=0.0

    if header1['BUNIT']=='JY/PIXEL':
        data1=data1/(ps_arcsec1**2.0) # Jy/pixel to Jy/arcsec2
    # x1, y1, x1edge, y1edge = xyarray(N1, ps_arcsec1)

    Nf=int(XMAX*2.0/(ps_final/1000.0))
    psf_arcsec=ps_final/1000.0
    
    xf=np.zeros(Nf+1)
    yf=np.zeros(Nf+1)
    for i in range(Nf+1):
        xf[i]=-(i-Nf/2.0)*psf_arcsec  
        yf[i]=(i-Nf/2.0)*psf_arcsec 

    image=interpol(N1,Nf,ps_mas1,ps_final, data1)
    if path_pbcor!='':
        rmsmap_out=interpol(N1,Nf,ps_mas1,ps_final,rmsmap)

        return image, rmsmap_out, xf, yf, BMAJ, BMIN, BPA
    else:
        return image, xf, yf, BMAJ, BMIN, BPA


def fload_fits_image_mira(path_image, ps_final, XMAX): # for images from CASA

    ### PS_final in mas

    ##### LOAD IMAGE
    fit1	= pyfits.open(path_image) # open image cube
    data1 	= get_last2d(fit1[0].data) # [0,0,:,:] # extract image matrix

    #### READ HEADER
    header1	= fit1[0].header
    ps_mas1=float(header1['CDELT2'])
    
    N1=len(data1[:,0])

    Nf=int(XMAX*2.0/(ps_final))
    
    xf=np.zeros(Nf+1)
    yf=np.zeros(Nf+1)
    for i in range(Nf+1):
        xf[i]=-(i-Nf/2.0)*ps_final  
        yf[i]=(i-Nf/2.0)*ps_final

    image=interpol(N1,Nf,ps_mas1,ps_final, data1)
    return image, xf, yf
    


def rainbow_cmap():



    cdict3 = {'red': ((0.0, 0.0, 0.0),
                      (0.4, 0.0, 0.0),
                      (0.5, 0.0, 0.0),
                      (0.6, 0.0, 0.0),
                      (0.7, 1.0, 1.0),
                      (0.8, 1.0, 1.0),
                      (0.9, 1.0, 1.0),
                      (1.0, 1.0, 1.0)),
              'green':((0.0, 0.0, 0.0),
                       (0.4, 0.0, 0.0),
                       (0.5, 1.0, 1.0),
                       (0.6, 1.0, 1.0),
                       (0.7, 1.0, 1.0),
                       (0.8, 0.0, 0.0),
                       (0.9, 0.0, 0.0),
                       (1.0, 1.0, 1.0)),
              'blue': ((0.0, 0.0, 0.0),
                       (0.4, 1.0, 1.0),
                       (0.5, 1.0, 1.0),
                       (0.6, 0.0, 0.0),
                       (0.7, 0.0, 0.0),
                       (0.8, 0.0, 0.0),
                       (0.9, 1.0, 1.0),
                       (1.0, 1.0, 1.0))}



    my_cmap = cl.LinearSegmentedColormap('my_colormap',cdict3,256)

    return my_cmap


#########################################
############## DATA CUBE ANALYSIS #######
#########################################

def fload_fits_cube(path_cube, line='CO32'): # for images from CASA

    if line=='CO32':
        f_line=345.79599 # GHz
    if line=='CO21':
        f_line=230.538000 # GHz
    if line=='HCN43':
        f_line=354.50547590 # GHz


    ##### LOAD CUBE
    fit1	= pyfits.open(path_cube) # open image cube
    data1 	= get_last3d(fit1[0].data) # [0,0,:,:] # extract image matrix

    #### READ HEADER
    header1	= fit1[0].header
    ps_deg1=float(header1['CDELT2'])
    ps_mas1= ps_deg1*3600.0*1000.0 # pixel size input in mas
    ps_arcsec1=ps_deg1*3600.0
    
    N1=len(data1[0,0,:])
    Nf=len(data1[:,0,0])

    try:
        BMAJ=float(header1['BMAJ'])*3600.0 # arcsec 
        BMIN=float(header1['BMIN'])*3600.0 # arcsec 
        BPA=float(header1['BPA']) # deg 
        print "beam = %1.2f x %1.2f" %(BMAJ, BMIN)
    except:
        BMAJ=0.0
        BMIN=0.0
        BPA=0.0

    ########### SPATIAL GRID

    x1, y1, x1edge, y1edge = xyarray(N1, ps_arcsec1)

        
    ########## FREQUENCY GRID
        
    df=float(header1['CDELT3'])/1.0e9 # GHz
    f0=float(header1['CRVAL3'])/1.0e9 # GHz
    ckms=299792.458 # km/s

    fs=np.linspace(f0,f0+df*(Nf),Nf) #GHz
    vs=-(fs-f_line)*ckms/f_line  # km/s
    
    dv=vs[1]-vs[0] # km/s
    print "dv [km/s] = ", dv
    print "dnu [GHz] = ", df


    return data1, ps_arcsec1, x1, y1, x1edge, y1edge, BMAJ, BMIN, BPA, fs, vs, dv
        

def moment_0(path_cube, line='CO32', v0=0.0, dvel=10.0,  rmin=0.0, inc=90.0, M_star=1.0, ps_final=0.0, XMAX=0.0):

    data1, ps_arcsec1, x1, y1, x1edge, y1edge, BMAJ, BMIN, BPA, fs, vs, dv = fload_fits_cube(path_cube, line='CO32')


    if rmin!=0.0:
        Dvel=np.sqrt(G*M_star*M_sun/(rmin*au)) *np.sin(inc*np.pi/180.0) /1.0e3
    else:
        Dvel=dvel
    
    mask_v=(vs>=v0-dvel) & (vs<=v0+dvel)
    moment0=np.sum(data1[mask_v,:,:], axis=0)*abs(dv) # Jy km/s
    print 'dvel, dv = ', Dvel, dv
    if ps_final==0.0 or XMAX==0.0:
        
        return moment0, x1edge, y1edge, BMAJ, BMIN, BPA, dv
    else:
        Nf=int(XMAX*2.0/(ps_final/1000.0))
        psf_arcsec=ps_final/1000.0

        xf, yf, xfedge, yfedge = xyarray(Nf, psf_arcsec)

        N1=len(data1[0,0,:])
        ps_mas1=ps_arcsec1*1.0e3

        image=interpol(N1,Nf,ps_mas1,ps_final, moment0)
   
        return image, xfedge, yfedge, BMAJ, BMIN, BPA, dv
        


def Flux_inside_cube(amin, amax, cube , ps_arcsec, vs, Dvel, v0, PArad, incrad, x0, y0, averaged=True):
    
    # returns flux in Jy (cube must be in Jy/arcsec2) 

    dv=vs[1]-vs[0] # km/s
    Nf=len(vs)
    if Dvel>0.0 and dv>0.0:
        k_min=max(0, int((v0-Dvel-vs[0])/dv) )
        k_max=min(Nf, int((v0+Dvel-vs[0])/dv) )
    elif Dvel>0.0 and dv<0.0:
        k_min=max(0, int((v0+Dvel-vs[0])/dv) )
        k_max=min(Nf, int((v0-Dvel-vs[0])/dv) )
        # print k_min,k_max, dv, Dvel, v0

    else:
        print 'error, dv<0 or Dvel<0'
        return -1
    
    F=0.0 # integrated  flux
    # Npix=0  # number of pixels over which we are integrating (not used)
    Nfr=k_max-k_min+1 # number of frequency points over which we will integrate
    Rmss=np.zeros(Nf-Nfr) # array where we will compute the rms of the spectrum
    N1=len(cube[0,0,:]) # number of pixels image
    chi=1.0/np.cos(incrad) # aspect ratio of disc

    
    x1, y1, x1edge, y1edge = xyarray(N1, ps_arcsec)

    for i in xrange(N1):
        for j in xrange(N1):

            xi=x1[i]-x0
            yi=y1[j]-y0      
            xpp = xi * np.cos(PArad) - yi *np.sin(PArad) ### along minor axis
            ypp = xi * np.sin(PArad) + yi *np.cos(PArad) ###  along major axis       
            r=np.sqrt( (xpp*chi)**2.0 + ypp**2.0 )
            
            if r<=amax and r>=amin:
                # Npix+=1
                # plt.plot(xi,yi, 'o',color='blue')                            
                F+=np.sum(cube[k_min:k_max+1,j,i]) # Jy/arcsec
                Rmss[0:k_min]=  Rmss[0:k_min]+  cube[0:k_min,j,i]     
                Rmss[k_min:]=  Rmss[k_min:]+  cube[k_max+1:,j,i]     
    # plt.xlim(10.0,-10.0)
    # plt.ylim(-10.0,10.0)
    # plt.show()
    Delta=(ps_arcsec**2.0)*abs(dv) # constant to obtain total flux in Jy km/s
    Rms=np.std(Rmss)
    if averaged:
        factor=1.15 # typically for 0.8 km/s wide channels that result from averaging 0.4km/2 channels
        # https://safe.nrao.edu/wiki/pub/Main/ALMAWindowFunctions/Note_on_Spectral_Response.pdf
    else:
        factor=2.0 #  for 0.4 km/s wide channels without averaging
    dF=Rms*np.sqrt(Nfr*factor)*Delta
    if dF==0.0:
        return 1.0e-6, 1.0 
    else:
        return F*Delta , dF


def Spectrum(amin,amax, cube,  ps_arcsec, vs, Dvel, v0, PArad, incrad, x0, y0):
    # return spectrum in Jy (cube must be in Jy/arcsec2) 

    dv=vs[1]-vs[0] # km/s
    Nf=len(vs)

    if Dvel>0.0 and dv>0.0:
        k_min=max(0, int((v0-Dvel-vs[0])/dv) )
        k_max=min(Nf, int((v0+Dvel-vs[0])/dv) )
    elif Dvel>0.0 and dv<0.0:
        k_min=max(0, int((v0+Dvel-vs[0])/dv) )
        k_max=min(Nf, int((v0-Dvel-vs[0])/dv) )
    else:
        print 'error, dv<0 or Dvel<0'
        print k_min,k_max, dv, Dvel, v0
        return -1

    
    F=np.zeros(Nf) # spectrum
    F2=np.zeros(Nf-(k_max-k_min+1)) # spectrum without line where we calculate rms
    Npix=0 # number of pixels over which we integrate
    N1=len(cube[0,0,:]) # number of pixels image
    chi=1.0/np.cos(incrad) # aspect ratio of disc

    x1, y1, x1edge, y1edge = xyarray(N1, ps_arcsec)

    
    for i in xrange(N1):
        for j in xrange(N1):
            
            xi=x1[i]-x0
            yi=y1[j]-y0   
            xpp = xi * np.cos(PArad) - yi *np.sin(PArad) ### along minor axis
            ypp = xi * np.sin(PArad) + yi *np.cos(PArad) ###  along major axis
            # PAi=np.arctan2(xi,yi)*180.0/np.pi # North=0                    
            r=np.sqrt( (xpp*chi)**2.0 + ypp**2.0 )
            
            if r<amax and r>amin:
                Npix+=1.0
                # for k in xrange(Nf):           
                F[:]+=cube[:,j,i] # Jy/arcsec
    F2[0:k_min]=F[0:k_min]
    F2[k_min:]=F[k_max+1:]
    rms=np.std(F2)                        
    Delta=(ps_arcsec**2.0)
    
    return F*Delta, rms*Delta, Delta, Npix, k_max, k_min

def sub_baseline_spectrum(I, vs, Nf, k_min, k_max, order=4 ):


    I2=np.zeros(Nf-(k_max-k_min+1))
    I2[:k_min]=I[:k_min]
    I2[k_min:]=I[k_max+1:]

    vs2=np.zeros(Nf-(k_max-k_min+1))
    vs2[:k_min]=vs[:k_min]
    vs2[k_min:]=vs[k_max+1:]

    pfit = np.polyfit(vs2, I2, order)   # Fit a 2nd order polynomial to (x, y) data
    Baseline = np.polyval(pfit, vs)

    return Baseline, I-Baseline


def f_shift(N1, x0, y0, ps_arcsec, PA, inc, M_star, dpc, vlim=10.0):

    # vlim in km/s
    ### return matrix of the size of the image with the shifts on each pixel
    shifts=np.zeros((N1,N1))

    x1, y1, x1edge, y1edge = xyarray(N1, ps_arcsec)
    chi=1.0/np.cos(inc) # aspect ratio of disc

    for i in xrange(N1):
        for j in xrange(N1):

            xi=x1[i]-x0 # x in sky in arcsec
            yi=y1[j]-y0 # y in sky in arcsec
                    
            xpp = xi * np.cos(PA) - yi *np.sin(PA) ###  along minor axis in arcsec
            ypp = xi * np.sin(PA) + yi *np.cos(PA) ###  along major axis in arcsec
                    
            PAi=np.arctan2(xi,yi) # North=0 in rad

            # if PAi<0.0: PAi=2.0*np.pi+PAi
            
            r=np.sqrt( (xpp*chi)**2.0 + ypp**2.0 )*dpc # deprojected radius of that pixel AU
            vk=np.sqrt(G*M_star/ (r*au) )/1.0e3  # km/s

            fsky=PAi-PA # [rad] angle between disc PA and pixel in plane of sky
            fdisc=np.arctan2(xpp*chi,ypp) # [rad] angle between disc PA and pixel in plane of disc
            vr=vk*np.cos(fdisc)*np.sin(inc)
            if np.abs(vr)<vlim:
                shifts[j,i]=vr
            else: shifts[j,i]=vlim*vr/abs(vr)
            

    return shifts # km/s


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


    for ir in xrange(Nr):
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

def save_image(filename, image, xedge, yedge, rms=0.0, rmsmap=0.0, vmin=0.0, vmax=100.0, colormap='inferno', tickcolor='white', XMAX=10.0, major_ticks=np.arange(-15, 20.0, 5.0) , minor_ticks=np.arange(-15.0, 15.0+1.0, 1.0), BMAJ=0.0, BMIN=0.0, BPA=0.0, show=True, clabel=r'Intensity [$\mu$Jy beam$^{-1}$]', formatcb='%1.0f', cbticks=np.arange(-500.0,500.0,50.0), contours=True, star=True, xstar=0.0, ystar=0.0, cbar_log=False, xunit='arcsec'):


    plt.style.use('style1')
    font= {'family':'Times New Roman', 'size': 12}
    rc('font', **font)

    fig = plt.figure(figsize=(4,4.6))#(8,6))

    ax1=fig.add_subplot(111)


    if not cbar_log:
        pc=ax1.pcolormesh(xedge,yedge,image, vmin=vmin, vmax=vmax,  cmap=colormap, rasterized=True)
        #pc.set_edgecolor('face')
        cb= fig.colorbar(pc,orientation='horizontal',label=clabel,format=formatcb, ticks=cbticks, pad=0.12)
        cb.ax.minorticks_on()
    else:
        my_cmap = copy.copy(plt.cm.get_cmap(colormap)) # copy the default cmap
        my_cmap.set_bad((0,0,0))
        pc=ax1.pcolormesh(xedge,yedge,image, norm=LogNorm(vmin=vmin, vmax=vmax),  cmap=my_cmap, rasterized=True)
        #pc.set_edgecolor('face')
        cb= fig.colorbar(pc,orientation='horizontal',label=clabel, pad=0.12)

    c1=fcolor_black_white(0.5,3)
    c2=fcolor_black_white(1.0,3)
    c3=fcolor_black_white(1.5,3)

    if contours:
        PS=abs(yedge[1]-yedge[0])
        c1=plt.contour(xedge[:-1]-PS/2.0,yedge[:-1]+PS/2.0, image/rmsmap, levels=[3.0,5.0, 8.0],colors=[c1,c2, c3], linewidths=0.7)
        # cb.add_lines(con2)

    ax1.set_xticks(major_ticks)                                                       
    ax1.set_xticks(minor_ticks, minor=True)                                           
    ax1.set_yticks(major_ticks)                                                       
    ax1.set_yticks(minor_ticks, minor=True) 

    for tick in ax1.get_xticklines():
        tick.set_color(tickcolor)

    for minortick in ax1.xaxis.get_minorticklines():
        minortick.set_color(tickcolor)

    for tick in ax1.get_yticklines():
        tick.set_color(tickcolor)

    for minortick in ax1.yaxis.get_minorticklines():
        minortick.set_color(tickcolor)

        ax1.spines['bottom'].set_color(tickcolor)
        ax1.spines['top'].set_color(tickcolor)
        ax1.spines['left'].set_color(tickcolor)
        ax1.spines['right'].set_color(tickcolor)

    ax1.set_xlabel('RA offset ['+xunit+']')
    ax1.set_ylabel('DEC offset ['+xunit+']')
    ax1.set_xlim(XMAX,-XMAX)
    ax1.set_ylim(-XMAX,XMAX)
    ax1.set_aspect('equal')

    #---add beam
    if BMAJ!=0.0 and BMIN!=0.0:
        xc=XMAX-2.0*abs(minor_ticks[1]-minor_ticks[0])
        yc=-XMAX+2.0*abs(minor_ticks[1]-minor_ticks[0])
        width= BMAJ
        height= BMIN
        pa=BPA
        elli=Ellipse((xc,yc),width,height,angle=90.-pa,linewidth=3,fill=True,color='white', alpha=1.0)
        #elli.set_clip_box(ax1.bbox)
        #elli.set_alpha(0.7)
        #elli.set_facecolor('black')
        #ax1.add_artist(elli)
        ax1.add_patch(elli)

    if star:
        # add star
        ax1.plot(xstar , ystar, marker='+', color=tickcolor, markersize=3.5, mew=1.0)


    plt.tight_layout()
    plt.subplots_adjust(left=0.17, bottom=0.01, right=0.97, top=1.0)
    print 'saving...'

    plt.savefig(filename, dpi=500)#,format='png', dpi=500)
    print 'saved'

    if show:
        plt.show()
