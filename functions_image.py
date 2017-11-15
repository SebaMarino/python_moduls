import numpy as np
import astropy.io.fits as pyfits
from astropy.convolution import convolve_fft
import matplotlib.colors as cl
import os,sys

def ellipse(x0,y0,phi,chi,a, PA):

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

    
    # phipp= phi - PA
    
    # xpp =  a/chi        * np.cos(phipp) 
    # ypp =   a    * np.sin(phipp)

    # xp=  xpp*np.cos(-PA) + ypp*np.sin(-PA)
    # yp= -xpp*np.sin(-PA) + ypp*np.cos(-PA)
    
    # xc = xp + x0
    # yc = yp + y0
    
    # return xc , yc

#  non parametric fit 





def radial_profile(image, image_pb, x0, y0, PA, inc, rmax,Nr, phis, rms, BMAJ_arcsec, ps_arcsec):

    # x0, y0 are RA DEC offsets in arcsec
    # PA and inc are PA and inc of the disc in deg
    # rmax [arcsec] is the maximum deprojected radius at which to do the azimuthal averaging
    # Nr is the number of radial points to calculate
    # phis [deg] is an array with uniform spacing that sets the range of PA at which to do the interpolation (0 is north)

    # ################ SPATIAL GRID

    # XY

    Np=len(image[:,0])
    
    xedge=np.zeros(Np+1)
    yedge=np.zeros(Np+1)

    xs=np.zeros(Np)
    ys=np.zeros(Np)

    for i in xrange(Np+1):

        xedge[i]=-(i-Np/2.0)*ps_arcsec-ps_arcsec/2.0        
        yedge[i]=(i-Np/2.0)*ps_arcsec+ps_arcsec/2.0

    for i in xrange(Np):
        xs[i]=-(i-Np/2.0)*ps_arcsec              
        ys[i]=(i-Np/2.0)*ps_arcsec
    
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

            phi1=phis_rad[i_p] 
            XS1,YS1=ellipse(x0,y0,phi1,chi,ai, PA_rad)
        
            ip1 = -int(XS1/ps_arcsec)+Np/2
            jp1 = int(YS1/ps_arcsec)+Np/2
        
            Irs1[i_r,i_p] = image[jp1,ip1] 
            Irs2[i_r,i_p] = image_pb[jp1,ip1]

    Ir1=np.mean(Irs1, axis=1) # mean intensity in Jy/beam
    Ir2=np.zeros(Nr)
    
    for i in xrange(Nphi):
        Ir2=Ir2+(rms/Irs2[:,i])**2.0

    Ir2=np.sqrt(Ir2/(Nphi))

    # Calculate number of independent points (this bit can be edited as it might not be true for highly inclined discs)

    Nindeps_1=np.ones(Nr)

    # arclength=dphi*(Nphi-1) # radians
    arclength=(Nphi-1)*dphi* np.sqrt(  (1.0 + (1.0/chi)**2.0 )/2.0 )
    for i in xrange(Nr):

        Nindeps_1i=rs[i]*arclength/BMAJ_arcsec

        if Nindeps_1i>1.0:  Nindeps_1[i]=Nindeps_1i

    print arclength
    print np.max(Nindeps_1), BMAJ_arcsec
    
    Err_1=Ir2/np.sqrt(Nindeps_1)

    
    
    return np.array([rs, Ir1, Err_1]) # rs, I, eI 


def radial_profile_fits_model(fitsfile, x0, y0, PA, inc, rmax,Nr, phis, rms):


    fit1=pyfits.open(fitsfile)
    try: 
        data1 	= fit1[0].data[0,0,:,:] #extraer matriz de datos
    except:
        try:
            data1 	= fit1[0].data[0,:,:] #extraer matriz de datos
        except:
            data1 	= fit1[0].data[:,:] #extraer matriz de datos
    
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
    return radial_profile(data1, np.ones((Np1,Np1)), x0, y0, PA, inc, rmax,Nr, phis, rms=0.0, BMAJ_arcsec=1.0, ps_arcsec=ps_arcsec1)


def radial_profile_fits_image(fitsfile_pbcor, fitsfile_pb, x0, y0, PA, inc, rmax,Nr, phis, rms):


    fit1=pyfits.open(fitsfile_pbcor)
    fit2=pyfits.open(fitsfile_pb)
    try: 
        data1 	= fit1[0].data[0,0,:,:] #extraer matriz de datos
        data2	= fit2[0].data[0,0,:,:] #extraer matriz de datos
    except:
        try:
            data1 	= fit1[0].data[0,:,:] #extraer matriz de datos
            data2	= fit2[0].data[0,:,:] #extraer matriz de datos

        except:
            data1 	= fit1[0].data[:,:] #extraer matriz de datos
            data2	= fit2[0].data[:,:] #extraer matriz de datos

    # print np.shape(data1)
    header1=fit1[0].header
    ps_deg1=float(header1['CDELT2'])
    ps_arcsec1=ps_deg1*3600.0
    ps_rad1=ps_deg1*np.pi/180.0
    Np1=len(data1[:,0])

    BMAJ=float(header1['BMAJ']) # deg
    BMIN=float(header1['BMIN']) # deg
    BPA=float(header1['BPA']) # deg

    BMAJ_arcsec1=BMAJ*3600.0
    BMIN_arcsec1=BMIN*3600.0
    
    return radial_profile(data1, data2, x0, y0, PA, inc, rmax,Nr, phis, rms=rms, BMAJ_arcsec=BMAJ_arcsec1, ps_arcsec=ps_arcsec1)



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


    
def fcolor_blue_red(i,N):

    rgb=[i*1.0/(N-1), 0.0, 1.0-i*1.0/(N-1)]

    return cl.colorConverter.to_rgb(rgb)


def get_last2d(data):
    if data.ndim == 2:
        return data[:]
    if data.ndim == 3:
        return data[0, :]
    if data.ndim == 4:
        return data[0, 0, :]
    

def inter(Nin,Nout,i,j,ps1,ps2,Fin):

	f=0.0
	S=0.0
	a=2.0*ps1
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
			if r<a: 
				P=np.exp(-r**2.0/(2.0*a**2.0))
				f=f+P*Fin[n,m]
				S=S+P
				k=1
			elif k==1: break

	if S==0.0: return 0.0		
	else: 
		return f/S

def interpol(Nin,Nout,ps1,ps2,Fin):
	F=np.zeros((Nout,Nout), dtype=np.float64)
	for i in range(Nout):
                print i
		for j in range (Nout):	
			F[i,j]=inter(Nin,Nout,i,j,ps1,ps2,Fin)
	return F

    
def fload_fits_image(path_image, path_pbcor, rms, ps_final, XMAX): # for images from CASA

    ### PS_final in mas

    ##### LOAD IMAGE
    fit1	= pyfits.open(path_image) # open image cube
    data1 	= get_last2d(fit1[0].data) # [0,0,:,:] # extract image matrix

    fit2	= pyfits.open(path_pbcor) #abrir objeto cubo
    data2 	= get_last2d(fit2[0].data) # [0,0,:,:] #extraer matriz de datos

    rmsmap=rms/data2

    #### READ HEADER
    header1	= fit1[0].header
    ps_deg1=float(header1['CDELT2'])
    ps_mas1= ps_deg1*3600.0*1000.0 # pixel size input in mas
    ps_arcsec1=ps_deg1*3600.0
    
    N1=len(data1[:,0])

    try:
        BMAJ=float(header1['BMIN'])*3600.0 # arcsec 
        BMIN=float(header1['BMAJ'])*3600.0 # arcsec 
        BPA=float(header1['BPA']) # deg 
        print "beam = %1.2f x %1.2f" %(BMAJ, BMIN)
    except:
        BMAJ=0.0
        BMIN=0.0
        BPA=0.0
        
    x1=np.zeros(N1)
    y1=np.zeros(N1)
    for i in range(N1):
        x1[i]=-(i-N1/2.0)*ps_arcsec1-ps_arcsec1/2.0
	y1[i]=(i-N1/2.0)*ps_arcsec1+ps_arcsec1/2.0

    x1edge=np.zeros(N1+1)
    y1edge=np.zeros(N1+1)
    for i in range(N1+1):
	x1edge[i]=-(i-N1/2.0)*ps_arcsec1 
	y1edge[i]=(i-N1/2.0)*ps_arcsec1 


    Nf=int(XMAX*2.0/(ps_final/1000.0))
    psf_arcsec=ps_final/1000.0
    
    xf=np.zeros(Nf+1)
    yf=np.zeros(Nf+1)
    for i in range(Nf+1):
        xf[i]=-(i-Nf/2.0)*psf_arcsec  
        yf[i]=(i-Nf/2.0)*psf_arcsec 

    image_pbcor=interpol(N1,Nf,ps_mas1,ps_final, data1)
    rmsmap_out=interpol(N1,Nf,ps_mas1,ps_final,rmsmap)

    return image_pbcor, rmsmap_out, xf, yf, BMAJ, BMIN, BPA



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
