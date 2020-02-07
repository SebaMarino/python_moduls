# compile with python setup.py build_ext --inplace
import numpy as np
from astropy.io import fits
import sys,os
# C functions,attributes
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
# Cython handles the namespace ambiguity internally in this case

def Chi2_simvisC_bilinear(pathfitsfile, table, offra=0.0, offdec=0.0):

    cdef:
        double chi2
        int Nvis=len(table[:,0])
        int i
        int N
        double ps_arcsec
        double Xmax
        double[:] ips=np.zeros(Nvis)#, np.float32)
        double[:] jps=np.zeros(Nvis)#, np.float32)
        int il, iu, jl, ju
        double complex fyl, dyu
        double complex[:,:] FIs
        double complex[:,:] FIsc
        double complex[:] vmodel=np.zeros(Nvis, dtype=np.complex_)

    # Import fits file
    # Note: model image has to be multiplied by the primary beam
    fit = fits.open(pathfitsfile) 
    data=np.fliplr(fit[0].data[0,0,:,:]) # in Jy/pixel
    header=fit[0].header
    
    N=len(data[:,0])
    ps_arcsec=float(header['CDELT2'])*3600.0
    #print ps_arcsec

    Xmax=ps_arcsec*N/2.0 # arcsec
    Xmaxrad=Xmax*np.pi/(3600.0*180.0)
    x=np.linspace(-Xmax,Xmax,N)  # arcsec
    dx=x[1]-x[0]
    xedge=np.linspace(-1-dx/2.0, 1+dx/2.0, N+1)

    du=1.0/(2.0*Xmaxrad)
    Umax=du*N/2
    u=np.linspace(-Umax, Umax, N ) # 1/rad
    du2=u[1]-u[0]
    uedge=np.linspace(-Umax-du/2.0, Umax+du/2.0, N )
    

    # print "computing fft"        
    FI=np.fft.fft2(np.fft.ifftshift(data))
    # print "shifting"
    FIs=np.fft.fftshift(FI).astype(np.cdouble)
    # print "done"
    
    ## OFFSET IN IMAGE SPACE TRANSLATES TO PHASE SHIFTS IN VISIBILITY SPACE
    us, vs =np.meshgrid(u,u)
    offra_rad=offra*np.pi/180.0/3600.0	
    offdec_rad=offdec*np.pi/180.0/3600.0
    dx_rad=dx*np.pi/(180.*3600.0)
    if N%2==0:  shift_correction=np.exp(2.0*np.pi*1j*( us*(-offra_rad-dx_rad/2.0) + vs*(-offdec_rad-dx_rad/2.0) ) )
    else: shift_correction=np.exp(2.0*np.pi*1j*( us*(-offra_rad) + vs*(-offdec_rad) ) )
    FIsc=FIs*shift_correction

    # FIr=np.real(FIsc)
    # FIi=np.imag(FIsc)

    # print np.shape(table)
    if Umax<np.max(np.abs(table[:,0])) or Umax<np.max(np.abs(table[:,1])):
        print "error, uv point outside grid",
        print "Umax model = %1.1e klambda"%(Umax/1.0e3)
        print "Umax data = %1.1e klambda"%(np.max(np.abs(table[:,0])/1.0e3))
        print "Vmax data = %1.1e klambda"%(np.max(np.abs(table[:,1])/1.0e3))
        sys.exit()

   
    
    # Bilinear interpolation

    ips=(-table[:,0]/du+N/2.0) # indices of u with decimals
    #ips=ips.astype(int)
    jps=(-table[:,1]/du+N/2.0) # indices of v with decimals
    #jps=jps.astype(int)

    for i in xrange(Nvis):
        il=int(ips[i])   # lower u
        iu=int(ips[i]+1) # upper u
            
        jl=int(jps[i])   # lower j
        ju=int(jps[i]+1) # upper j

        # interpol first in the x direction

        fyl=(iu-ips[i])*FIsc[jl, il] + (ips[i]-il)*FIsc[jl, iu]
        fyu=(iu-ips[i])*FIsc[ju, il] + (ips[i]-il)*FIsc[ju, iu]
            
        # interpol in y direction
            
        vmodel[i]=(ju-jps[i])*fyl + (jps[i]-jl)*fyu
    
    #print vmodel[0,0]
    # plt.plot(table[:,0],ips, 'o',color='blue')
    # plt.show()

    # plt.plot(table[:,1],jps, 'o',color='red')
    # plt.show()

    # vmodel=FIs[jps,ips] #interpol(N,us[i],vs[i],du,FIr)

    # plt.subplot(1,2,1)
    # plt.pcolormesh(uedge, uedge, FIr)
    # plt.subplot(1,2,2)
    # plt.pcolormesh(uedge, uedge, FIi)
    # plt.show()

    # np.save('./vmodel', vmodel)
        
    chi2=np.sum(((np.real(vmodel)-table[:,2])**2.0+(np.imag(vmodel)-table[:,3])**2.0)*table[:,4] )

    ais=np.where(table[:,4]>0.0)
    Nvis_noflaggs=len(ais[0])
    # print chi2, Nvis
    return chi2, Nvis_noflaggs


def Model_simvisC_bilinear(pathfitsfile, table, save=False, tag='', offra=0.0, offdec=0.0):
    cdef:
        double chi2
        int Nvis=len(table[:,0])
        int i
        int N
        double ps_arcsec
        double Xmax
        double[:] ips=np.zeros(Nvis)#, np.float32)
        double[:] jps=np.zeros(Nvis)#, np.float32)
        int il, iu, jl, ju
        double complex fyl, dyu
        double complex[:,:] FIs
        double complex[:,:] FIsc
        double complex[:] vmodel=np.zeros(Nvis, dtype=np.complex_)

    # Import fits file
    # Note: model image has to be multiplied by the primary beam
    fit = fits.open(pathfitsfile) 
    data=np.fliplr(fit[0].data[0,0,:,:]) # in Jy/pixel
    header=fit[0].header
    print tag    
    N=len(data[:,0])
    ps_arcsec=float(header['CDELT2'])*3600.0
    #print ps_arcsec

    Xmax=ps_arcsec*N/2.0 # arcsec
    Xmaxrad=Xmax*np.pi/(3600.0*180.0)
    x=np.linspace(-Xmax,Xmax,N)  # arcsec
    dx=x[1]-x[0]
    xedge=np.linspace(-1-dx/2.0, 1+dx/2.0, N+1)

    du=1.0/(2.0*Xmaxrad)
    Umax=du*N/2
    u=np.linspace(-Umax, Umax, N )
    du2=u[1]-u[0]
    uedge=np.linspace(-Umax-du/2.0, Umax+du/2.0, N )



    # print "computing fft"        
    FI=np.fft.fft2(np.fft.ifftshift(data))
    # print "shifting"
    FIs=np.fft.fftshift(FI).astype(np.cdouble)
    # print "done"

    ## OFFSET IN IMAGE SPACE TRANSLATES TO PHASE SHIFTS IN VISIBILITY SPACE
    us, vs =np.meshgrid(u,u)
    offra_rad=offra*np.pi/180.0/3600.0
    offdec_rad=offdec*np.pi/180.0/3600.0
    dx_rad=dx*np.pi/(180.*3600.0)
    if N%2==0:  shift_correction=np.exp(2.0*np.pi*1j*( us*(-offra_rad-dx_rad/2.0) + vs*(-offdec_rad-dx_rad/2.0) ) )
    else: shift_correction=np.exp(2.0*np.pi*1j*( us*(-offra_rad) + vs*(-offdec_rad) ) )
    FIsc=FIs*shift_correction

    # FIr=np.real(FIsc)
    # FIi=np.imag(FIsc)

    # interpolate vis via nearest pixel

    # print np.shape(table)
    if Umax<np.max(np.abs(table[:,0])) or Umax<np.max(np.abs(table[:,1])):
        print "error, uv point outside grid",
        print "Umax model = %1.1e klambda"%(Umax/1.0e3)
        print "Umax data = %1.1e klambda"%(np.max(np.abs(table[:,0])/1.0e3))
        print "Vmax data = %1.1e klambda"%(np.max(np.abs(table[:,1])/1.0e3))
        sys.exit()


    chi2=0
    # Bilinear interpolation

    ips=(-table[:,0]/du+N/2.0) # indices of u with decimals
    #ips=ips.astype(int)
    jps=(-table[:,1]/du+N/2.0) # indices of v with decimals
    #jps=jps.astype(int)

    for i in xrange(Nvis):
        il=int(ips[i])   # lower u
        iu=int(ips[i]+1) # upper u
            
        jl=int(jps[i])   # lower j
        ju=int(jps[i]+1) # upper j

        # interpol first in the x direction

        fyl=(iu-ips[i])*FIsc[jl, il] + (ips[i]-il)*FIsc[jl, iu]
        fyu=(iu-ips[i])*FIsc[ju, il] + (ips[i]-il)*FIsc[ju, iu]
            
        # interpol in y direction
            
        vmodel[i]=(ju-jps[i])*fyl + (jps[i]-jl)*fyu
    
    #print vmodel[0,0]
    # plt.plot(table[:,0],ips, 'o',color='blue')
    # plt.show()

    # plt.plot(table[:,1],jps, 'o',color='red')
    # plt.show()

    # vmodel=FIs[jps,ips] #interpol(N,us[i],vs[i],du,FIr)

    # plt.subplot(1,2,1)
    # plt.pcolormesh(uedge, uedge, FIr)
    # plt.subplot(1,2,2)
    # plt.pcolormesh(uedge, uedge, FIi)
    # plt.show()

    # np.save('./vmodel', vmodel)
    
    #vobs=table[:,2]+table[:,3]*1j

    #chi2=np.sum( (np.abs(vmodel-vobs))**2.0*table[:,4] )
    
    ais=np.where(table[:,4]>0.0)
    
    chi2=np.sum(((np.real(vmodel)-table[:,2])**2.0+(np.imag(vmodel)-table[:,3])**2.0)*table[:,4] )
    if save:
        np.save('tablevismodel_'+tag, np.array([np.real(vmodel), np.imag(vmodel),table[:,4]  ]))
    
    Nvis_noflags=len(ais[0])
    return chi2, Nvis_noflags