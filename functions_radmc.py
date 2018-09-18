import os, sys
import numpy as np
import cmath as cma
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift
from astropy.io import fits

############################################################
# Module with funtions to use radmc and MCMC routines to fit visibilities by S. Marino
###########################################################



############### CONSTANTS
au=1.496e13           # [cm]
pc=3.08567758e18      # [cm]
M_sun=1.9891e33       # [g]
M_earth=5.97219e27    # [g]
R_sun=6.955e+10       # [cm]
G=6.67259e-8          # cgs
sig=5.6704e-5         # cgs

cc  = 2.9979245800000e10      # Light speed             [cm/s]


##########################################################
######################## OPACITIES #######################
##########################################################

def Intextpol(x,y,xi):

    Nx=len(x)
    if xi<=x[0]: return y[0] # extrapol                                                                                                                                                                         
    elif xi<=x[Nx-1]: #interpol                                                                                                                                                                                                              
        for l in xrange(1,Nx):
            if xi<=x[l]:
                return y[l-1]+(xi-x[l-1])*(y[l]-y[l-1])/(x[l]-x[l-1])

    elif xi>x[Nx-1]:    #extrapol                                                                                                                                                                                                            
        alpha=np.log(y[Nx-1]/y[Nx-2])/np.log(x[Nx-1]/x[Nx-2])
        return y[Nx-1]*(xi/x[Nx-1])**alpha


def effnk(n1,k1,n2,k2,n3,k3,f2,f3): 

    # mixing rule Bruggeman http://en.wikipedia.org/wiki/Effective_medium_approximations
    # Sum fi* (epi-ep)/(epi+2ep) = 0, but normilizing by f1

 
    np1=n1+k1*1j # matrix
    np2=n2+k2*1j # inclusion 1
    np3=n3+k3*1j # inclusion 2

    e1=np1**2.0  # n = sqrt(epsilon_r x mu_r) and mu_r is aprox 1
    e2=np2**2.0
    e3=np3**2.0

    # polynomial of third order
    p=np.zeros(4, dtype=complex)

    p[3]=e1*e2*e3*(1.0+f2+f3) # 0 order
    p[2]=-e1*e3*f2 -e1*e2*f3 - e2*e3 + 2*(e1*e2*f2 + e1*e2 +e1*e3*f3+e1*e3+e2*e3*f2+e2*e3*f3) # 1st order
    p[1]= -2.0*(e1*f2+e1*f3+e3*f2+e2*f3+e2+e3)+4.0*(e1+e2*f2+e3*f3)# 2nd order
    p[0]= -4.0*(1.0+f2+f3)

    roots=np.roots(p) 

    # check roots
    for i in xrange(len(roots)):
        effi=roots[i]
        if effi.real>0.0 and effi.imag>0.0:
            return cma.sqrt(effi)
    ### if nothins satisfy the above condition
        
    return -1.0


def mix_opct_bruggeman(pathout, paths, densities, massfractions, Nw=200, wmin=0.1, wmax=10000.0):

    # Mixing rule Bruggeman 

    path1=paths[0]# './astrosilicate_ext.lnk'
    path2=paths[1]# './ac_opct.lnk'
    path3=paths[2]# './ice_opct.lnk'

    d1=densities[0] # 4.0
    d2=densities[1] # 2.5
    d3=densities[2] # 1.0

    m1=massfractions[0] #0.7  # 0.5
    m2=massfractions[1] # 0.2
    m3=massfractions[2] # 0.15 # 0.3

    V1=m1/d1
    V2=m2/d2
    V3=m3/d3

    df=(d1*V1+d2*V2+d3*V3)/(V1+V2+V3)

    v1=V1/(V1+V2+V3) # volume fraction matrix
    v2=V2/(V1+V2+V3) # volume fraction 1st inclusion
    v3=V3/(V1+V2+V3) # volume fraction 2nd inclusion

    f2=v2/v1
    f3=v3/v1

    print v2, v3
    print "final density = ", df


    O1=np.loadtxt(path1)
    O2=np.loadtxt(path2)
    O3=np.loadtxt(path3)

    # logO1n=interpolate.interp1d(np.log10(O1[:,0]), np.log10(O1[:,1]))
    # logO1k=interpolate.interp1d(np.log10(O1[:,0]), np.log10(O1[:,1]))
    # logO2n=interpolate.interp1d(np.log10(O2[:,0]), np.log10(O2[:,1]))
    # logO2k=interpolate.interp1d(np.log10(O2[:,0]), np.log10(O2[:,1]))
    # logO3n=interpolate.interp1d(np.log10(O3[:,0]), np.log10(O3[:,1]))
    # logO3k=interpolate.interp1d(np.log10(O3[:,0]), np.log10(O3[:,1]))


    Opct1=np.zeros((Nw,3))
    Opct1[0,0]=wmin

    i=0

    # n1=10.0**logO1n(np.log10(Opct1[i,0]))
    # n2=10.0**logO2n(np.log10(Opct1[i,0]))
    # n3=10.0**logO3n(np.log10(Opct1[i,0]))

    # k1=10.0**logO1k(np.log10(Opct1[i,0]))
    # k2=10.0**logO2k(np.log10(Opct1[i,0]))
    # k3=10.0**logO3k(np.log10(Opct1[i,0]))


    n1=Intextpol(O1[:,0],O1[:,1],Opct1[i,0])
    n2=Intextpol(O2[:,0],O2[:,1],Opct1[i,0])
    n3=Intextpol(O3[:,0],O3[:,1],Opct1[i,0])

    k1=Intextpol(O1[:,0],O1[:,2],Opct1[i,0])
    k2=Intextpol(O2[:,0],O2[:,2],Opct1[i,0])
    k3=Intextpol(O3[:,0],O3[:,2],Opct1[i,0])


    eff=effnk(n1,k1,n2,k2,n3,k3,f2,f3)

    Opct1[i,1]=eff.real
    Opct1[i,2]=eff.imag

    for i in xrange(1,Nw):
        Opct1[i,0]=wmin*(wmax/wmin)**(i*1.0/(Nw-1))
    
        # n1=10.0**logO1n(np.log10(Opct1[i,0]))
        # n2=10.0**logO2n(np.log10(Opct1[i,0]))
        # n3=10.0**logO3n(np.log10(Opct1[i,0]))
        
        # k1=10.0**logO1k(np.log10(Opct1[i,0]))
        # k2=10.0**logO2k(np.log10(Opct1[i,0]))
        # k3=10.0**logO3k(np.log10(Opct1[i,0]))

        n1=Intextpol(O1[:,0],O1[:,1],Opct1[i,0])
        n2=Intextpol(O2[:,0],O2[:,1],Opct1[i,0])
        n3=Intextpol(O3[:,0],O3[:,1],Opct1[i,0])

        k1=Intextpol(O1[:,0],O1[:,2],Opct1[i,0])
        k2=Intextpol(O2[:,0],O2[:,2],Opct1[i,0])
        k3=Intextpol(O3[:,0],O3[:,2],Opct1[i,0])

        eff=effnk(n1,k1,n2,k2,n3,k3,f2,f3)

        # print eff
        Opct1[i,1]=eff.real
        Opct1[i,2]=eff.imag

    np.savetxt(pathout,Opct1)



def opac_arg(amin, amax, density, N, Inte, lnk_file, Type, exp=3.5):

    # amin      # microns
    # amax       # microns
    # density      # g/cm3
    # N             # bins
    # Inte # if to  average or not
    # lnk_file # file with optical constants (it assumes it ends with .lnk)
    # Type # output tag (dustkappa_'+Type+'.inp')


    path="./"
    pathout='Tempkappa/dustkappa_'+Type+'.inp'


    # ---------------------- MAIN 
    os.system('rm '+path+'Tempkappa/*')

    Pa=(amax/amin)**(1.0/(N-1.0))

    A=np.zeros(N)
    A[0]=amin


    for i in range(N):
        os.system('rm '+path+'param.inp')
        A[i]=amin*(Pa**(i))
        acm=A[i]*10.0**(-4.0)
        # print "a = %1.2e [um]"  %A[i]
        file_inp=open(path+'param.inp','w')
        file_inp.write(lnk_file+'\n')
        e=round(np.log10(acm))
        b=acm/(10.0**e)
        file_inp.write('%1.2fd%i \n' %(b,e))
        file_inp.write('%1.2f \n' %density) 
        file_inp.write('1')
        
        file_inp.close()

        os.system(path+'makeopac')
        os.system('mv '+path+'dustkappa_'+lnk_file+'.inp '+path+'Tempkappa/dustkappa_'+Type+'_'+str(i+1)+'.inp ')
    
   
    # --------- READ OPACITIES AND COMPUTE MEAN OPACITY

    if Inte==1:
        # read number of wavelengths
        opct=np.loadtxt(path+lnk_file+'.lnk')
        Nw=len(opct[:,0])


        Op=np.zeros((Nw,4)) # wl, kappa_abs, kappa_scat, g 
        Op[:,0]=opct[:,0]

        Ws_mass=np.zeros(N) # weigths by mass and abundances
        Ws_number=np.zeros(N) # wights by abundances

        for i in xrange(N):
            Ws_mass[i]=(A[i]**(-exp))*(A[i]**(3.0))*A[i]  # w(a) propto n(a)*m(a)*da and da propto a
            Ws_number[i]=A[i]**(-exp)*A[i]

        W_mass=Ws_mass/np.sum(Ws_mass)
        W_number=Ws_number/np.sum(Ws_number)

        for i in xrange(N):
            file_inp=open(path+'Tempkappa/dustkappa_'+Type+'_'+str(i+1)+'.inp','r')
            file_inp.readline()
            file_inp.readline()


            for j in xrange(Nw):
                line=file_inp.readline()
                dat=line.split()
                kabs=float(dat[1])
                kscat=float(dat[2])
                g=float(dat[3])

                Op[j,1]+=kabs*W_mass[i]
                Op[j,2]+=kscat*W_mass[i]
                Op[j,3]+=g*kscat*W_mass[i]# g*W_mass[i] 
            
            file_inp.close()
            os.system('rm '+path+'Tempkappa/dustkappa_'+Type+'_'+str(i+1)+'.inp')

        ### normalize g
        for j in xrange(Nw):
            Op[j,3]=Op[j,3]/Op[j,2]

        final=open(path+pathout,'w')
    
        final.write('3 \n')
        final.write(str(Nw)+'\n')
        for i in xrange(Nw):
            final.write('%f \t %f \t %f \t %f\n' %(Op[i,0],Op[i,1],Op[i,2],Op[i,3]))
        final.close()

def write_opacities(optc_file, Nspec, Asedge, grain_density, exp=3.5):

    path='dustopac.inp'
    path_opac='./'

    arch=open(path,'w')
    arch.write("2               Format number of this file \n")
    arch.write(str(Nspec)+"              Nr of dust species \n")
    arch.write("============================================================================ \n")

    for i in xrange(Nspec):
        print i
        opac_arg(Asedge[i]*1.0e4, Asedge[i+1]*1.0e4, grain_density, 100, 1, optc_file, 'dust_'+str(i+1), exp=exp )
        os.system('cp '+path_opac+'Tempkappa/* ./')

        arch.write("1               Way in which this dust species is read \n")
        arch.write("0               0=Thermal grain \n")
        arch.write("dust_"+str(i+1)+ " Extension of name of dustkappa_***.inp file \n")
        arch.write("---------------------------------------------------------------------------- \n")
    arch.close()

def change_dustopac(Nspec=1, tag=''):

    path='dustopac.inp'
    path_opac='./'
    arch=open(path,'w')
    arch.write("2               Format number of this file \n")
    arch.write(str(Nspec)+"              Nr of dust species \n")
    arch.write("============================================================================ \n")
    for i in xrange(Nspec):
        arch.write("1               Way in which this dust species is read \n")
        arch.write("0               0=Thermal grain \n")
        arch.write("dust_"+str(i+1)+tag+ " Extension of name of dustkappa_***.inp file \n")
        arch.write("---------------------------------------------------------------------------- \n")
    arch.close()


def corrupt_opacities(path_in, path_out, columns=[], factor=-1.0, value=-1.0 ):

    try:
        ncol=len(columns)
    except:
        print 'columns is not an array'
        return
    if ncol==0:
        print 'no columns were specify'
        return

    for coli in columns:
        if coli not in [1,2,3] or not isinstance(coli, int):
            print 'no valid column or value in columns is not an int' 
            return
    file1=open(path_in,'r')
    ncolsf=int(file1.readline())
    if ncolsf<ncol: print 'too many columns were specify'
    Nw1=int(file1.readline())
    kappa1=np.zeros((Nw1,ncolsf+1))

    for j in xrange(Nw1):
        line=file1.readline()
        dat=line.split()
        for k in xrange(ncolsf+1):
            kappa1[j,k]=float(dat[k])
    file1.close()

    file2=open(path_out, 'w')

    ################ REMOVE COLUMNS
    if factor==-1.0 and value==-1.0:
        file2.write('%1.0i \n'%(ncolsf-ncol-1))
        file2.write('%1.0i \n'%Nw1)

        for j in xrange(Nw1):
            linef='%1.5f\t'%(kappa1[j,0])
            for k in xrange(ncolsf-ncol):
                linef=linef+'%1.5f\t'%(kappa1[j,k+1])
            linef=linef[:-1]+'\n'
            print linef

            file2.write(linef)

    ################ REPLACE OR MULTIPLY COLUMNS
    else:
        file2.write('%1.0i \n'%(ncolsf-1))
        file2.write('%1.0i \n'%Nw1)
    
        if factor>=0.0 and value<0.0:
            for coli in columns:
                kappa1[:,coli]= kappa1[:,coli]*factor
        elif value>=0.0:
            for coli in columns:
                kappa1[:,coli]= value

        for j in xrange(Nw1):
            linef='%1.5f\t'%(kappa1[j,0])
            for k in xrange(ncolsf):
                linef=linef+'%1.5f\t'%(kappa1[j,k+1])
            linef=linef[:-1]+'\n'
            print linef
            file2.write(linef)
    file2.close()
    
###################################################################
######################## stellar spectrum ########################
def write_stellar_spectrum(dir_stellar_templates,  waves, T_star, R_star, M_star=1.0*M_sun, save_npy=False): # if T_star <0 then it assumes black body of T=- T_star (use this mode if you dont have a stellar template)

    # -------- STAR and wavelength
    
    print 'Setting Stellar flux...'
    Nw=len(waves)
        
    if T_star>0.0:
        # load template
        path=dir_stellar_templates+'Kurucz'+str(int(T_star))+'-4.0.fits.gz'
        fit = fits.open(path) #abrir objeto cubo de datos
        data = fit[0].data # wave is in um, flux is in F_lam, units? #ergs cm**-2 s**-1 A**-1
        ltemp=len(data[0,:])

        tempflux=data[1,:]*(R_star/pc)**2.0 # computing flux at 1 pc
        tempwave=data[0,:]

        stellar_flux=np.zeros(Nw)

        I=0.0
        for i in range(Nw-1):
            wi=waves[i]
            if wi<tempwave[0]: # wavelength lower than model
                stellar_flux[i]=0.0#tempflux[0]
            elif wi >=tempwave[0] and wi < tempwave[ltemp-6]: # wavelength in between model

                for j in range(ltemp-1): # search for interval where to interpolate
                    if wi>=tempwave[j] and wi<tempwave[j+1]: # interpolate
                        m=(tempflux[j+1]-tempflux[j])/(tempwave[j+1]-tempwave[j])
                        stellar_flux[i]=m*(wi-tempwave[j])+tempflux[j]
                        break # stop searching
            elif wi >=tempwave[ltemp-6]: # too sparse point to make an interpolation,  propto lam**-4
                stellar_flux[i]=tempflux[ltemp-6]*(wi/tempwave[ltemp-6])**-4.0
            dlam=(waves[i+1]-waves[i])*1.0e4 # from um to A
            I+=stellar_flux[i]*dlam

        I_star=sig*T_star**4.0*(R_star/pc)**2.0 
        stellar_flux[:]=stellar_flux[:]*I_star/I  # normalizing spectrum
                                              # so Lstar=sig*R**2*T**4

       
        # UV excess between 0.091 um and 0.25 um 
        fUV= 0.0 # fraction of uv excess
        slope=2.2  # slope for Fnu
        fluxuv=np.zeros(Nw)

        # Iuv=0.0
        # for i in range(Nw):
        #     if waves[i]>= 0.091 and waves[i]<=0.25:
        
        #         fluxuv[i]=(waves[i])**(slope-2.0)  # C * erg/ s cm**2 A
        #         dlam=(waves[i+1]-waves[i])*1.0e4 # from um to A
        #         Iuv+=fluxuv[i]*dlam

        # fluxuv=fluxuv*I_star*fUV/Iuv # normalize uv excess

        # transform from F_lam to F_nu

        clight=2.99792458e18 # light speed in A/s
        Flux=np.zeros(Nw)
        Fluxwouv=np.zeros(Nw)
        for i in range(Nw):
            Flux[i]=(stellar_flux[i]+fluxuv[i])*(waves[i]*1.0e4)**2.0/clight
            Fluxwouv[i]=(stellar_flux[i])*(waves[i]*1.0e4)**2.0/clight

    path='stars.inp'
    arch_star=open(path,'w')
    arch_star.write('2 \n')
    arch_star.write('1  '+str(Nw)+'\n')
    arch_star.write(str(R_star)+'\t'+str(M_star)+'\t'+'0.0   0.0   0.0 \n')
    for i in xrange(Nw):
        arch_star.write(str(waves[i])+'\n')

    if T_star>0.0:
        for i in xrange(Nw):
            arch_star.write(str(Flux[i])+'\n')
    else:
            arch_star.write(str(T_star)+'\n')
    arch_star.close()

    if save_npy:
        np.savetxt('stellar_spectrum_Jy_1pc.dat', np.transpose(np.array([waves, Flux*1.0e23])))

#########################################################
##### GRID (grid could be a class...future work) #####
#########################################################

def define_grid_sph(Nr, Nth, Nphi, Rmax, Rmin, Thmax, Thmin, logr=False, logtheta=False, save=True):

    Redge=np.zeros(Nr) #from Rmin to Rmax
    R=np.zeros(Nr-1)

    ### R
    if logr: # log sampling
        Redge[0]=Rmin
        Px_r=(Rmax/Rmin)**(1.0/(Nr-1))
        for i in xrange(1,Nr):
            Redge[i]=Rmin*Px_r**i   
            R[i-1]=( Redge[i]+Redge[i-1])/2.0

    else: # ####### linear sampling
        Redge=np.linspace(Rmin,Rmax,Nr)
        dR=Redge[1]-Redge[0]
        R=Redge[1:]-dR/2.0

    ### Theta
    if logtheta: # log sampling
        
        Pth=(Thmax/Thmin)**(1.0/(Nth-2)) 
        Thedge=np.zeros(Nth) #from Rmin to Rmax
        Th=np.zeros(Nth-1)
        Thedge[0]=0.0
        Thedge[1]=Thmin
        Th[0]=Thmin/2.0

        for i in xrange(1,Nth-1):
            Thedge[i+1]=Thedge[i]*Pth
            Th[i]=(Thedge[i+1]+Thedge[i])/2.0
    else:
        
        Thedge=np.linspace(0.0,Thmax,Nth)
        dth=Thedge[1]-Thedge[0]
        Th=Thedge[:-1]+dth/2.0
        Thmin=Th[0]  # second edge

    ### Phi

    # ### linear sampling in phi
    dphi=2*np.pi/Nphi
    Phiedge=np.arange(0.0,2.0*np.pi,dphi) 
    Phi=Phiedge+dphi/2.0

    if save:

        path='amr_grid.inp' #'amr_grid.inp'

        arch=open(path,'w')
        arch.write('1 \n') # iformat: The format number, at present 1
        arch.write('0 \n') # Grid style (regular = 0)
        arch.write('101 \n') # coordsystem: If 100<=coordsystem<200 the coordinate system is spherical
        arch.write('0 \n') # gridinfo
        arch.write('1 \t 1 \t 1 \n') # incl x, incl y, incl z
        arch.write(str(Nr-1)+ '\t'+ str((Nth-1)*2)+'\t'+ str(Nphi)+'\n') 
        # arch.write('0 \t 0 \t 0 \n') # levelmax, nleafsmax, nbranchmax 

        for i in range(Nr):
            arch.write(str(Redge[i]*au)+'\t')
        arch.write('\n')

        for i in range(Nth):
            arch.write(str(np.pi/2.0-Thedge[Nth-1-i])+'\t')  # from northpole to equator
        for i in range(1,Nth):
            arch.write(str(np.pi/2.0+Thedge[i])+'\t')       # from 0 to -pi/2
        arch.write('\n')

        for i in range(Nphi):
            arch.write(str(Phiedge[i])+'\t')
        arch.write(str(2.0*np.pi)+'\t')
        arch.write('\n')
        arch.close()
    
    return Redge, R, Thedge, Th, Phiedge, Phi


def define_grid_cart(Nx, Nz, Xmax, save=True):
    
    Xedge=np.linspace(-Xmax,Xmax,Nx+1)
    dx=Xedge[1]-Xedge[0]
    dz=dx

    Zmax=dz*(Nz)/2.0
    Zedge=np.linspace(-Zmax,Zmax,Nz+1)

    if save:
        path='amr_grid.inp' #'amr_grid.inp'

        arch=open(path,'w')
        arch.write('1 \n') # iformat: The format number, at present 1
        arch.write('0 \n') # Grid style (regular = 0)
        arch.write('1 \n') # coordsystem: If 100<=coordsystem<200 the coordinate system is spherical
        arch.write('0 \n') # gridinfo
        arch.write('1 \t 1 \t 1 \n') # incl x, incl y, incl z

        arch.write(str(Nx)+ '\t'+ str(Nx)+'\t'+ str(Nz)+'\n')

        for i in xrange(Nx+1):
            arch.write(str(Xedge[i]*au)+'\t')
        arch.write('\n')
        for i in xrange(Nx+1):
            arch.write(str(Xedge[i]*au)+'\t')
        arch.write('\n')
        for i in xrange(Nz+1):
            arch.write(str(Zedge[i]*au)+'\t')
        arch.write('\n')
        arch.close()

    return Xedge, Zedge

#### wavelengths of the model
def def_wavelengths(wmin, wmax, Nw, save=True):

    waves=np.logspace(np.log10(wmin), np.log10(wmax), Nw)

    # ----- write wavelength_micron.inp

    path='wavelength_micron.inp'
    arch=open(path,'w')
    arch.write(str(Nw)+'\n')
    for i in xrange(Nw):
        arch.write(str(waves[i])+'\n')
    arch.close()

    return waves

def grains_size_grid(amin, amax, Nspec, Mdust_tot, Sizedist_exp): ## define grain size grid

    Asedge=np.logspace(np.log10(amin), np.log10(amax),Nspec+1) # bin edges
    As=(Asedge[:-1]+Asedge[1:])/2.0

    Ms=np.zeros(Nspec) # dust mass in each bin
    Ms[:] = np.abs(abs(Asedge[1:]**(Sizedist_exp+3.0+1.0)-Asedge[:-1]**(Sizedist_exp+3.0+1.0)))
    
    if Mdust_tot>0.0: 
        Ms=Ms*Mdust_tot/np.sum(Ms) # normalise to set total dust mass = M_dust
    return Asedge, As, Ms

######################################################
#### DESITY DISTRIBUTION FUNCTIONS ###################
######################################################

def rho_3d_dens(r, phi, z, h, sigmaf, *args):
    
    H=h*r 
    return sigmaf(r,phi, *args)*np.exp(-z**2.0/(2.0*H**2.0))/(np.sqrt(2.0*np.pi)*H)


##### create density matrix that is axisymmetric and save it for radmc 
def save_dens_axisym(Nspec, Redge, R, Thedge, Th, Phiedge, Phi, Ms, h, sigmaf, *args):

    # args has the arguments that sigmaf needs in the right order

    Nr=len(R)+1
    Nth=len(Th)+1
    Nphi=len(Phi)

    rho_d=np.zeros((Nspec,(Nth-1)*2,Nphi,Nr-1)) # density field

    for ia in xrange(Nspec):
        M_dust_temp= 0.0 #np.zeros(Nspec) 
        # print ia
        # print "Dust species = ", ia
  
        for k in xrange(Nth-1):
            theta=Th[Nth-2-k]
            for i in xrange(Nr-1):
                rho=R[i]*np.cos(theta)
                z=R[i]*np.sin(theta)
                # for j in xrange(Nphi):

                rho_d[ia,k,:,i]=rho_3d_dens(rho, 0.0, z,h, sigmaf, *args )
                rho_d[ia,2*(Nth-1)-1-k,:,i]=rho_d[ia,k,:,i]
                M_dust_temp+=2.0*rho_d[ia,k,0,i]*2.0*np.pi*rho*(Redge[i+1]-Redge[i])*(Thedge[Nth-2-k+1]-Thedge[Nth-2-k])*R[i]*au**3.0 
        # for ia in xrange(Nspec):
        rho_d[ia,:,:,:]=rho_d[ia,:,:,:]*Ms[ia]/M_dust_temp
        
        
    # Save 
    path='dust_density.inp'
    dust_d=open(path,'w')
    
    dust_d.write('1 \n') # iformat  
    dust_d.write(str((Nr-1)*2*(Nth-1)*(Nphi))+' \n') # iformat n cells 
    dust_d.write(str(Nspec)+' \n') # n species

    for ai in xrange(Nspec):
        for j in range(Nphi):
            for k in range(2*(Nth-1)):
                for i in range(Nr-1):
                    dust_d.write(str(rho_d[ai,k,j,i])+' \n')
                    
    dust_d.close()


##### create density matrix that is non-axisymmetric and save it for radmc 
def save_dens_nonaxisym(Nspec, Redge, R, Thedge, Th, Phiedge, Phi, Ms, h, sigmaf, *args):

    # args has the arguments that sigmaf needs in the right order

    Nr=len(R)+1
    Nth=len(Th)+1
    Nphi=len(Phi)
    dphi=Phiedge[1]-Phiedge[0]
    rho_d=np.zeros((Nspec,(Nth-1)*2,Nphi,Nr-1)) # density field

    for ia in xrange(Nspec):
        M_dust_temp= 0.0 #np.zeros(Nspec) 
        # print ia
        # print "Dust species = ", ia
  
        for k in xrange(Nth-1):
            theta=Th[Nth-2-k]
            for i in xrange(Nr-1):
                rho=R[i]*np.cos(theta)
                z=R[i]*np.sin(theta)
                for j in xrange(Nphi):
                    phi=Phi[j]
                    rho_d[ia,k,j,i]=rho_3d_dens(rho,phi, z,h, sigmaf, *args )
                    rho_d[ia,2*(Nth-1)-1-k,j,i]=rho_d[ia,k,j,i]
                    
                    M_dust_temp+=2.0*rho_d[ia,k,j,i]*(dphi)*rho*(Redge[i+1]-Redge[i])*(Thedge[Nth-2-k+1]-Thedge[Nth-2-k])*R[i]*au**3.0 
        # for ia in xrange(Nspec):
        rho_d[ia,:,:,:]=rho_d[ia,:,:,:]*Ms[ia]/M_dust_temp
        
        
    # Save 
    path='dust_density.inp'
    dust_d=open(path,'w')
    
    dust_d.write('1 \n') # iformat  
    dust_d.write(str((Nr-1)*2*(Nth-1)*(Nphi))+' \n') # iformat n cells 
    dust_d.write(str(Nspec)+' \n') # n species

    for ai in xrange(Nspec):
        for j in range(Nphi):
            for k in range(2*(Nth-1)):
                for i in range(Nr-1):
                    dust_d.write(str(rho_d[ai,k,j,i])+' \n')
                    
    dust_d.close()


##### create density matrix that is non-axisymmetric, composed of two different components and save it for radmc 
def save_dens_nonaxisym_2comp(Nspec, Redge, R, Thedge, Th, Phiedge, Phi,h, Ms1, Ms2, sigmaf1, sigmaf2, args1, args2):

    # args has the arguments that sigmaf. They need to be in the right order

    Nr=len(R)+1
    Nth=len(Th)+1
    Nphi=len(Phi)
    dphi=Phiedge[1]-Phiedge[0]
    rho_d1=np.zeros((Nspec,(Nth-1)*2,Nphi,Nr-1)) # density field
    rho_d2=np.zeros((Nspec,(Nth-1)*2,Nphi,Nr-1)) # density field
    for ia in xrange(Nspec):
        M_dust_temp1= 0.0 #np.zeros(Nspec) 
        M_dust_temp2= 0.0 #np.zeros(Nspec) 
        # print ia
        # print "Dust species = ", ia
  
        for k in xrange(Nth-1):
            theta=Th[Nth-2-k]
            for i in xrange(Nr-1):
                rho=R[i]*np.cos(theta)
                z=R[i]*np.sin(theta)
                for j in xrange(Nphi):
                    phi=Phi[j]
                    rho_d1[ia,k,j,i]=rho_3d_dens(rho,phi, z,h, sigmaf1, *args1 )
                    rho_d1[ia,2*(Nth-1)-1-k,j,i]=rho_d1[ia,k,j,i]

                    rho_d2[ia,k,j,i]=rho_3d_dens(rho,phi, z,h, sigmaf2, *args2 )
                    rho_d2[ia,2*(Nth-1)-1-k,j,i]=rho_d2[ia,k,j,i]
                    
                    M_dust_temp1+=2.0*rho_d1[ia,k,j,i]*(dphi)*rho*(Redge[i+1]-Redge[i])*(Thedge[Nth-2-k+1]-Thedge[Nth-2-k])*R[i]*au**3.0 
                    M_dust_temp2+=2.0*rho_d2[ia,k,j,i]*(dphi)*rho*(Redge[i+1]-Redge[i])*(Thedge[Nth-2-k+1]-Thedge[Nth-2-k])*R[i]*au**3.0 
        # for ia in xrange(Nspec):
        rho_d1[ia,:,:,:]=rho_d1[ia,:,:,:]*Ms1[ia]/M_dust_temp1
        rho_d2[ia,:,:,:]=rho_d2[ia,:,:,:]*Ms2[ia]/M_dust_temp2
        
        
    # Save 
    path='dust_density.inp'
    dust_d=open(path,'w')
    
    dust_d.write('1 \n') # iformat  
    dust_d.write(str((Nr-1)*2*(Nth-1)*(Nphi))+' \n') # iformat n cells 
    dust_d.write(str(Nspec)+' \n') # n species

    for ai in xrange(Nspec):
        for j in range(Nphi):
            for k in range(2*(Nth-1)):
                for i in range(Nr-1):
                    dust_d.write(str(rho_d1[ai,k,j,i]+rho_d2[ai,k,j,i])+' \n')
                    
    dust_d.close()




def save_density_cartesian_flat(Ms, field, Nspec, Nx, Xmax, Nz):
    # field is a matrix with the surface density of Nx x Nx 


    Xedge=np.linspace(-Xmax,Xmax,Nx+1)
    dx=Xedge[1]-Xedge[0]

    dz=dx
    Zmax=dz*(Nz)/2.0
    Zedge=np.linspace(-Zmax,Zmax,Nz+1)



    rho_d=np.zeros((Nspec,Nz, Nx,Nx))

    rho_d[:]=field[:,:]

    M_dust_temp = np.sum(field)*(dx*au)**2 * (2*Zmax*au)

    for ia in xrange(Nspec):

        rho_d[ia,:]=rho_d[ia,:]*Ms[ia]/M_dust_temp

        # for i in xrange(Nx-1):
        #     x=Xedge[i]+dx/2.0
        #     for j in xrange(Nx-1):
        #         y=Xedge[j]+dx/2.0
        #         rho=np.sqrt(x**2.0+y**2.0)
        #         H=H_odisk*(rho/R_odisk)**flare_odisk
        #         if H==0.0: 
        #             # print 'hey'
        #             print rho
        #         Sigma=field[j,i]
        #         for k in xrange(Nz-1):
        #             z=Zedge[k]+dz/2.0
        #             rho_d[ia,j,i,k]=Sigma #rho_p(Sigma,H,z)
        #             M_dust_temp+=rho_d[ia,j,i,k]*dx**2.0*dz*au**3.0
    

    # -------Save it in a file for RADMC
    path='dust_density.inp'
    
    dust_d=open(path,'w')
    dust_d.write('1 \n') # iformat  
    dust_d.write(str((Nx)*(Nx)*(Nz))+' \n') # iformat n cells 
    dust_d.write(str(Nspec)+' \n') # n species

    for ai in xrange(Nspec):
        for k in range(Nz):
            for j in range(Nx):
                for i in range(Nx):
                    dust_d.write(str(rho_d[ai,k,j,i])+' \n')
                    
    dust_d.close()

def save_density_cartesian(Ms, field, Nspec, Nx, Xmax, Nz):
    # field is a matrix with the surface density of Nx x Nx 


    Xedge=np.linspace(-Xmax,Xmax,Nx+1)
    dx=Xedge[1]-Xedge[0]

    dz=dx
    Zmax=dz*(Nz)/2.0
    Zedge=np.linspace(-Zmax,Zmax,Nz+1)



    rho_d=np.zeros((Nspec,Nz,Nx,Nx))

    rho_d[:]=field

    M_dust_temp = np.sum(field)*(dx*au)**3

    for ia in xrange(Nspec):

        rho_d[ia,:]=rho_d[ia,:]*Ms[ia]/M_dust_temp

        # for i in xrange(Nx-1):
        #     x=Xedge[i]+dx/2.0
        #     for j in xrange(Nx-1):
        #         y=Xedge[j]+dx/2.0
        #         rho=np.sqrt(x**2.0+y**2.0)
        #         H=H_odisk*(rho/R_odisk)**flare_odisk
        #         if H==0.0: 
        #             # print 'hey'
        #             print rho
        #         Sigma=field[j,i]
        #         for k in xrange(Nz-1):
        #             z=Zedge[k]+dz/2.0
        #             rho_d[ia,j,i,k]=Sigma #rho_p(Sigma,H,z)
        #             M_dust_temp+=rho_d[ia,j,i,k]*dx**2.0*dz*au**3.0
    

    # -------Save it in a file for RADMC
    path='dust_density.inp'
    
    dust_d=open(path,'w')
    dust_d.write('1 \n') # iformat  
    dust_d.write(str((Nx)*(Nx)*(Nz))+' \n') # iformat n cells 
    dust_d.write(str(Nspec)+' \n') # n species

    for ai in xrange(Nspec):
        for k in range(Nz):
            for j in range(Nx):
                for i in range(Nx):
                    dust_d.write(str(rho_d[ai,k,j,i])+' \n')
                    
    dust_d.close()




def save_dens_gas_axisym( Redge, R, Thedge, Th, Phiedge, Phi, Mh2, Mco, h, sigmaf, *args):

    # args has the arguments that sigmaf needs in the right order

    Nr=len(R)+1
    Nth=len(Th)+1
    Nphi=len(Phi)

    rho_g=np.zeros(((Nth-1)*2,Nphi,Nr-1)) # density field

    M_gas_temp= 0.0 #np.zeros(Nspec) 
    # print ia
    # print "Dust species = ", ia
  
    for k in xrange(Nth-1):
        theta=Th[Nth-2-k]
        for i in xrange(Nr-1):
            rho=R[i]*np.cos(theta)
            z=R[i]*np.sin(theta)
            # for j in xrange(Nphi):

            rho_g[k,:,i]=rho_3d_dens(rho, 0.0, z,h, sigmaf, *args )
            rho_g[2*(Nth-1)-1-k,:,i]=rho_g[k,:,i]
            M_gas_temp+=2.0*rho_g[k,0,i]*2.0*np.pi*rho*(Redge[i+1]-Redge[i])*(Thedge[Nth-2-k+1]-Thedge[Nth-2-k])*R[i]*au**3.0 
        # for ia in xrange(Nspec):
    rho_g=rho_g/M_gas_temp
        
    
    #### Save

    path_h2='numberdens_h2.inp'
    path_12co='numberdens_12c16o.inp'
    path_13c16o='numberdens_13c16o.inp'
    path_12c18o='numberdens_12c18o.inp'
    path_hcop='numberdens_hcop.inp'

    paths=[path_h2, path_12co, path_13c16o, path_12c18o, path_hcop]
    ms=np.array([2.0, 16.0, 17.0, 17.0, 17.0])* 1.67262178e-24 # molecular masses in grams
    abundances=[Mh2, Mco, Mco*1.0e-2, Mco*1.0e-3, Mco*1.0e-3] # mass in grams

    for ip in xrange(len(paths)):
        file_g=open(paths[ip],'w')
        file_g.write('1 \n') # iformat  
        file_g.write(str((Nr-1)*2*(Nth-1)*(Nphi))+' \n') # iformat n cells 

        for j in range(Nphi):
            for k in range(2*(Nth-1)):
                for i in range(Nr-1):
                    file_g.write(str(rho_g[k,j,i]*abundances[ip]/ms[ip])+' \n')
        file_g.close()

    file_lines=open('lines.inp', 'w')
    file_lines.write('2 \n')
    file_lines.write('4 \n')
    file_lines.write('12c16o    leiden    0    0    1 \n')
    file_lines.write('h2 \n')
    file_lines.write('13c16o    leiden    0    0    1 \n')
    file_lines.write('h2 \n')
    file_lines.write('12c18o    leiden    0    0    1 \n')
    file_lines.write('h2 \n')
    file_lines.write('hcop    leiden    0    0    1 \n')
    file_lines.write('h2 \n')

    file_lines.close()

def gas_velocities( Redge, R, Thedge, Th, Phiedge, Phi, M_star=1.0*M_sun, turb=0.1, ecc=-1.0):
    # turb in km/s
    Nr=len(R)+1
    Nth=len(Th)+1
    Nphi=len(Phi)

    path_velocity='gas_velocity.inp'
    arch=open(path_velocity,'w')
    arch.write('1 \n') # iformat  
    arch.write(str((Nr-1)*2*(Nth-1)*(Nphi))+' \n') # iformat n cells   

    for j in xrange(Nphi):
        phi=Phi[j]
        for k in xrange(2*(Nth-1)):
            for i in xrange(Nr-1):
                r=R[i]
            
                if k<Nth-1:
                    Theta = np.pi/2.0-Th[Nth-2-k]
                else:
                    Theta = np.pi/2.0+Th[k-(Nth-1)]
                
                rho=r*np.sin(Theta)
                z=r*np.cos(Theta)

                vphi=np.sqrt(G*M_star/(rho*au)) 
                vr = 0.0
                vtheta = 0.0
                arch.write(str(vr)+'\t'+str(vtheta)+'\t'+str(vphi)+' \n')


    arch.close() 

    arch_v=open('microturbulence.inp','w')
    arch_v.write('1 \n')
    arch_v.write(str((Nr-1)*2*(Nth-1)*(Nphi))+' \n') # n cells
    if ecc<=0.0:
        # ----------------------TURBULENCES 0.1 km/s
        
        for j in xrange(Nphi):
            for k in xrange(2*(Nth-1)):
                for i in xrange(Nr-1):
                    arch_v.write(str(turb*1.0e5)+' \n')

    else:
        for j in xrange(Nphi):
            for k in xrange(2*(Nth-1)):
                for i in xrange(Nr-1):
                    r=R[i]
                    if k<Nth-1:
                        Theta = np.pi/2.0-Th[Nth-2-k]
                    else:
                        Theta = np.pi/2.0+Th[k-(Nth-1)]
            
                    rho=r*np.sin(Theta)
                    z=r*np.cos(Theta)

                    vphi=np.sqrt(G*M_star/(rho*au)) # cm/s
                    # dvel=vphi*2*ecc/(1.-ecc**2.0) # 0.1 km/s to cm/s
                    dvel=vphi*(np.sqrt(1.+ecc)-np.sqrt(1.-ecc)) # 0.1 km/s to cm/s
                    arch_v.write(str(np.sqrt(dvel**2.0+(turb*1.0e5)**2.0))+' \n')

    arch_v.close()


#################################################################
############### CONVERT IMAGES TO FITS FILES ####################
#################################################################

def load_image(path_image, dpc):

    f=open(path_image,'r')
    iformat=int(f.readline())

    if (iformat < 1) or (iformat > 4):
        print "ERROR: File format of image not recognized"
        return

    nx, ny = tuple(np.array(f.readline().split(),dtype=int))
    nf = int(f.readline()) # number of wavelengths
    sizepix_x, sizepix_y = tuple(np.array(f.readline().split(),dtype=float))

    lam = np.empty(nf)
    for i in range(nf):
        lam[i] = float(f.readline())
    
    f.readline()  

    image = np.zeros((1,nf,ny,nx), dtype=float)

    for k in range(nf):
        for j in range(ny):
            for i in range(nx):

                image[0,k,j,i] = float(f.readline())

                if (j == ny-1) and (i == nx-1):
                    f.readline()

    f.close()

    # Compute the flux in this image as seen at dpc (pc)    
    pixdeg_x = 180.0*(sizepix_x/(dpc*pc))/np.pi
    pixdeg_y = 180.0*(sizepix_y/(dpc*pc))/np.pi

    # Compute the conversion factor from erg/cm^2/s/Hz/ster to erg/cm^2/s/Hz/ster at dpc
    pixsurf_ster = pixdeg_x*pixdeg_y * (np.pi/180.)**2
    factor = 1e+23 * pixsurf_ster
    # And scale the image array accordingly:
    image_in_jypix = factor * image

    return image_in_jypix, nx, ny, nf, lam, pixdeg_x, pixdeg_y


def star_pix(nx, omega):

    omega= omega%360.0

    if nx%2==0.0: # even number of pixels
        if omega>=0.0 and omega<=90.0:
            istar=nx/2 
            jstar=nx/2 
        elif omega>90.0 and omega<=180.0:
            istar=nx/2-1 
            jstar=nx/2 
        elif omega>180.0 and omega<=270.0:
            istar=nx/2-1 
            jstar=nx/2-1
        elif omega>270.0 and omega<360.0:
            istar=nx/2 
            jstar=nx/2 -1
    else:
        istar=nx/2
        jstar=nx/2
    return istar, jstar

def shift_image(image, mx, my, pixdeg_x, pixdeg_y, omega=0.0 ):

    if mx ==0.0 and my==0.0: return image

    mvx_pix=(mx/(pixdeg_x*3600.0))
    mvy_pix=(my/(pixdeg_y*3600.0))

    shiftVector=(0.0, 0.0, mvy_pix, -mvx_pix) # minus sign as left is positive 
    # cp star and remove it
    istar, jstar=star_pix(len(image[0,0,0,:]), omega)
    Fstar=image[0,0,jstar,istar]
    # print 'Fstar=', Fstar
    # print  image[0,0,jstar-1,istar-1  ]

    image[0,0,jstar,istar]=0.0
    
    # shift
    image_shifted=shift(image,shift=shiftVector, order=3)#,mode='wrap')
    # add star in new position
    image_shifted[0,0,jstar+int(mvy_pix),istar-int(mvx_pix)]=Fstar

    return image_shifted

def fpad_image(image_in, pad_x, pad_y, nx, ny):

    if image_in.shape[-2:] != (pad_x,pad_y):
        pad_image = np.zeros((1,1,pad_x,pad_y))
        if nx%2==0 and ny%2==0: # even number of pixels
            pad_image[0,0,
                      pad_y/2-ny/2:pad_y/2+ny/2,
                      pad_x/2-nx/2:pad_x/2+nx/2] = image_in[0,0,:,:]
        else:                  # odd number of pixels
            pad_image[0,0,
                      pad_y/2-(ny-1)/2:pad_y/2+(ny+1)/2,
                      pad_x/2-(nx-1)/2:pad_x/2+(nx+1)/2] = image_in[0,0,:,:]
        return pad_image

    else:                      # padding is not necessary as image is already the right size (potential bug if nx>pad_x)
        return image_in
def convert_to_fits(path_image,path_fits, Npixf, dpc , mx=0.0, my=0.0, x0=0.0, y0=0.0, omega=0.0):

    image_in_jypix, nx, ny, nf, lam, pixdeg_x, pixdeg_y = load_image(path_image, dpc)

    image_in_jypix_shifted= shift_image(image_in_jypix, mx, my, pixdeg_x, pixdeg_y, omega=omega)

    # flux = np.sum(image_in_jypix[0,0,:,:])
    flux = np.sum(image_in_jypix_shifted[0,0,:,:])

    # print "flux [Jy] = ", flux
    ### HEADER

    
    lam0=lam[0]
    reffreq=cc/(lam0*1.0e-4)


    # Make FITS header information:
    header = fits.Header()
    
    #header['SIMPLE']='T'
    header['BITPIX']=-32
    # all the NAXIS are created automatically header['NAXIS']=2
    header['OBJECT']='HD109085'
    header['EPOCH']=2000.0
    header['LONPOLE']=180.0
    header['EQUINOX']=2000.0
    header['SPECSYS']='LSRK'
    header['RESTFREQ']=reffreq
    header['VELREF']=0.0
    header['CTYPE3']='FREQ'
    header['CRPIX3'] = 1.0
    header['CDELT3']  = 1.0
    header['CRVAL3']= reffreq


    header['FLUX']=flux

    header['BTYPE'] = 'Intensity'
    header['BSCALE'] = 1
    header['BZERO'] = 0
    header['BUNIT'] = 'JY/PIXEL'#'erg/s/cm^2/Hz/ster'


    #header['EPOCH'] = 2000.
    #header['LONPOLE'] = 180.
    header['CTYPE1'] = 'RA---TAN'
    header['CRVAL1'] = x0
    header['CTYPE2'] = 'DEC--TAN'
    header['CRVAL2'] = y0
    

    unit = 'DEG'
    multiplier = 1
    # RA
    header['CDELT1'] = -multiplier*pixdeg_x
    header['CUNIT1'] = unit
    # ...Zero point of coordinate system
    header['CRPIX1'] = 1.0*((Npixf+1)/2)
    # DEC
    header['CDELT2'] = multiplier*pixdeg_y
    header['CUNIT2'] = unit
    #
    # ...Zero point of coordinate system
    #
    header['CRPIX2'] = 1.0* ((ny+1)/2)

    # FREQ
    if nf > 1:
        # multiple frequencies - set up the header keywords to define the
        #    third axis as frequency
        header['CTYPE3'] = 'VELOCITY'
        header['CUNIT3'] = 'km/s'
        header['CRPIX3'] = 1.0* ((nf+1)/2)
        header['CRVAL3'] = 0.0
        # Calculate the frequency step, assuming equal steps between all:
        delta_velocity = (lam[1] - lam[0])*cc*1e-5/lam0
        header['CDELT3'] = delta_velocity
        header['RESTWAVE']=lam0
    else:                # only one frequency
        header['RESTWAVE'] = lam[0]
        header['CUNIT3'] = 'Hz'
    # Make a FITS file!
    #

    # PAD IMAGE
    image_in_jypix_shifted=fpad_image(image_in_jypix_shifted, Npixf, Npixf, nx, ny)



    image_in_jypix_float=image_in_jypix_shifted.astype(np.float32)
    fits.writeto(path_fits, image_in_jypix_float, header, output_verify='fix')


def convert_to_fits_alpha(path_image,path_fits, Npixf, dpc , lam0, newlam, mx=0.0, my=0.0, x0=0.0, alpha_dust=3.0, y0=0.0, omega=0.0):

    image_in_jypix, nx, ny, nf, lam, pixdeg_x, pixdeg_y = load_image(path_image, dpc)

    istar, jstar=star_pix(nx, omega) 
    Fstar=image_in_jypix[0,0,istar,istar]
    ### dust spectral index
    image_in_jypix[0,0,istar,istar]=0.0
    image_in_jypix=image_in_jypix*(newlam/lam0)**(-alpha_dust)
    ### star spectral index
    Fstar=Fstar*(newlam/lam0)**(-2.0)
    image_in_jypix[0,0,istar,istar]=Fstar


    image_in_jypix_shifted= shift_image(image_in_jypix, mx, my, pixdeg_x, pixdeg_y, omega=omega)

    # flux = np.sum(image_in_jypix[0,0,:,:])
    flux = np.sum(image_in_jypix_shifted[0,0,:,:])

    # print "flux [Jy] = ", flux
    ### HEADER

    
    lam0=newlam
    reffreq=cc/(lam0*1.0e-4)


    # Make FITS header information:
    header = fits.Header()
    
    #header['SIMPLE']='T'
    header['BITPIX']=-32
    # all the NAXIS are created automatically header['NAXIS']=2
    header['OBJECT']='HD109085'
    header['EPOCH']=2000.0
    header['LONPOLE']=180.0
    header['EQUINOX']=2000.0
    header['SPECSYS']='LSRK'
    header['RESTFREQ']=reffreq
    header['VELREF']=0.0
    header['CTYPE3']='FREQ'
    header['CRPIX3'] = 1.0
    header['CDELT3']  = 1.0
    header['CRVAL3']= reffreq


    header['FLUX']=flux

    header['BTYPE'] = 'Intensity'
    header['BSCALE'] = 1
    header['BZERO'] = 0
    header['BUNIT'] = 'JY/PIXEL'#'erg/s/cm^2/Hz/ster'


    #header['EPOCH'] = 2000.
    #header['LONPOLE'] = 180.
    header['CTYPE1'] = 'RA---TAN'
    header['CRVAL1'] = x0
    header['CTYPE2'] = 'DEC--TAN'
    header['CRVAL2'] = y0
    

    unit = 'DEG'
    multiplier = 1
    # RA
    header['CDELT1'] = -multiplier*pixdeg_x
    header['CUNIT1'] = unit
    # ...Zero point of coordinate system
    header['CRPIX1'] = 1.0*((Npixf+1)/2)
    # DEC
    header['CDELT2'] = multiplier*pixdeg_y
    header['CUNIT2'] = unit
    #
    # ...Zero point of coordinate system
    #
    header['CRPIX2'] = 1.0* ((ny+1)/2)

    # FREQ
    if nf > 1:
        # multiple frequencies - set up the header keywords to define the
        #    third axis as frequency
        header['CTYPE3'] = 'VELOCITY'
        header['CUNIT3'] = 'km/s'
        header['CRPIX3'] = 1.0* ((nf+1)/2)
        header['CRVAL3'] = 0.0
        # Calculate the frequency step, assuming equal steps between all:
        delta_velocity = (lam[1] - lam[0])*cc*1e-5/lam0
        header['CDELT3'] = delta_velocity
        header['RESTWAVE']=lam0
    else:                # only one frequency
        header['RESTWAVE'] = lam[0]
        header['CUNIT3'] = 'Hz'
    # Make a FITS file!
    #

    # PAD IMAGE
    image_in_jypix_shifted=fpad_image(image_in_jypix_shifted, Npixf, Npixf, nx, ny)

    image_in_jypix_float=image_in_jypix_shifted.astype(np.float32)
    fits.writeto(path_fits, image_in_jypix_float, header, output_verify='fix')






def convert_to_fits_canvas(path_image,path_fits,path_canvas, dpc, mx=0.0, my=0.0 , pbm=False, omega=0.0, fstar=-1.0):
   

    image_in_jypix, nx, ny, nf, lam, pixdeg_x, pixdeg_y = load_image(path_image, dpc)
    if fstar>=0.0:
        istar, jstar=star_pix(nx, omega)
        image_in_jypix[:, :, jstar,istar]=fstar
    image_in_jypix_shifted= shift_image(image_in_jypix, mx, my, pixdeg_x, pixdeg_y, omega=omega)
    flux = np.sum(image_in_jypix_shifted[0,0,:,:])

    # Make FITS header information from canvas (this is necessary to
    # interact later with CASA). The commented lines were used when
    # not using the canvas
    

    canvas=fits.open(path_canvas)
    header = canvas[0].header # fits.Header()
    del header['Origin'] # necessary due to a comment that CASA adds automatically
    
    # PAD IMAGE
    pad_x = header['NAXIS1']
    pad_y = header['NAXIS2']

    image_in_jypix_shifted=fpad_image(image_in_jypix_shifted, pad_x, pad_y, nx, ny)


    ##### multiply by primary beam or not. 

    if pbm:
        # print "multiply by pb beam!"
        # multiply by primary beam
        pbfits=fits.open(path_canvas[:-4]+'pb.fits')
        pb=pbfits[0].data[0,0,:,:]
        image_in_jypix_shifted=image_in_jypix_shifted*pb

        inans= np.isnan(image_in_jypix_shifted)
        image_in_jypix_shifted[inans]=0.0
    else:
        print "don't multiply by pbm"

    flux = np.sum(image_in_jypix_shifted[0,0,:,:])
    header['BUNIT'] = 'JY/PIXEL' 
    header['FLUX']=flux
    # print "flux [Jy] = ", flux
        
    image_in_jypix_float=image_in_jypix_shifted.astype(np.float32)
    fits.writeto(path_fits, image_in_jypix_float, header, output_verify='fix')


def convert_to_fits_canvas_fields(path_image,path_fits,path_canvas, dpc, mx=0.0, my=0.0 , pbm=False, x0=0.0, y0=0.0, omega=0.0):

    image_in_jypix, nx, ny, nf, lam, pixdeg_x, pixdeg_y = load_image(path_image, dpc)

    # Make FITS header information from canvas 
    
    canvas=fits.open(path_canvas)
    header = canvas[0].header # fits.Header()
    del header['Origin'] # necessary due to a comment that CASA adds automatically

    #### Shift image by mx and my in arcsec
    x1=header['CRVAL1']
    y1=header['CRVAL2']

    if x0==0.0 and y0==0.0:
        x0=x1
        y0=y1

    offx_arcsec=-(x1-x0)*np.cos(y0*np.pi/180.0)*3600.0 # offset projected in sky
    offy_arcsec=-(y1-y0)*3600.0

    mxp=mx+offx_arcsec
    myp=my+offy_arcsec

    image_in_jypix_shifted= shift_image(image_in_jypix, mxp, myp, pixdeg_x, pixdeg_y, omega=omega)
    
    # PAD IMAGE

    pad_x = header['NAXIS1']
    pad_y = header['NAXIS2']

    image_in_jypix_shifted=fpad_image(image_in_jypix_shifted, pad_x, pad_y, nx, ny)

 
    
    ##### multiply by primary beam or not. 

    if pbm:
        # print "multiply by pb beam!"
        pbfits=fits.open(path_canvas[:-4]+'pb.fits')
        pb=pbfits[0].data[0,0,:,:]
        image_in_jypix_shifted=image_in_jypix_shifted*pb

        inans= np.isnan(image_in_jypix_shifted)
        image_in_jypix_shifted[inans]=0.0
    else:
        print "don't multiply by pbm"

    flux = np.sum(image_in_jypix_shifted[0,0,:,:])
    header['BUNIT'] = 'JY/PIXEL' 
    header['FLUX']=flux
    # print "flux [Jy] = ", flux

    image_in_jypix_float=image_in_jypix_shifted.astype(np.float32)
    fits.writeto(path_fits, image_in_jypix_float, header, output_verify='fix')


def convert_to_fits_canvas_fields_alpha(path_image,path_fits,path_canvas, dpc, lam0, newlam, arcsec=False, mas=False, mx=0.0, my=0.0 , pbm=False, x0=0.0, y0=0.0, alpha_dust=3.0, omega=0.0):
    
    # modifies existing image using a specral index alpha and assuming
    # the star has a spectral index of -2.

    image_in_jypix, nx, ny, nf, lam, pixdeg_x, pixdeg_y = load_image(path_image, dpc)
    
    istar, jstar=star_pix(nx, omega) 
    Fstar=image_in_jypix[0,0,istar,istar]
    ### dust spectral index
    image_in_jypix[0,0,istar,istar]=0.0
    image_in_jypix=image_in_jypix*(newlam/lam0)**(-alpha_dust)
    ### star spectral index
    Fstar=Fstar*(newlam/lam0)**(-2.0)
    image_in_jypix[0,0,istar,istar]=Fstar

    
    # Make FITS header information from canvas 
    
    canvas=fits.open(path_canvas)
    header = canvas[0].header # fits.Header()
    del header['Origin'] # necessary due to a comment that CASA adds automatically

    #### Shift image by mx and my in arcsec
    x1=header['CRVAL1']
    y1=header['CRVAL2']

    if x0==0.0 and y0==0.0:
        x0=x1
        y0=y1

    offx_arcsec=-(x1-x0)*np.cos(y0*np.pi/180.0)*3600.0 # offset projected in sky
    offy_arcsec=-(y1-y0)*3600.0

    mxp=mx+offx_arcsec
    myp=my+offy_arcsec

    image_in_jypix_shifted= shift_image(image_in_jypix, mxp, myp, pixdeg_x, pixdeg_y, omega=omega)
    
    # PAD IMAGE
    pad_x = header['NAXIS1']
    pad_y = header['NAXIS2']

    image_in_jypix_shifted=fpad_image(image_in_jypix_shifted, pad_x, pad_y, nx, ny)


    

    ##### multiply by primary beam or not. Only necessary when simulating visibilities in CASA with ft a posteriori

    if pbm:
        # print "multiply by pb beam!"
        # multiply by primary beam
        pbfits=fits.open(path_canvas[:-4]+'pb.fits')
        pb=pbfits[0].data[0,0,:,:]
        image_in_jypix_shifted=image_in_jypix_shifted*pb

        inans= np.isnan(image_in_jypix_shifted)
        image_in_jypix_shifted[inans]=0.0
    else:
        print "don't multiply by pbm"

    image_in_jypix_float=image_in_jypix_shifted.astype(np.float32)

    flux = np.sum(image_in_jypix_float)
    header['BUNIT'] = 'JY/PIXEL' 
    header['FLUX']=flux
    # print "flux [Jy] = ", flux


    fits.writeto(path_fits, image_in_jypix_float, header, output_verify='fix')


def convert_to_fits_canvas_alpha(path_image,path_fits,path_canvas, dpc, lam0, newlam, arcsec=False, mas=False, mx=0.0, my=0.0 , pbm=False, alpha_dust=3.0, omega=0.0, fstar=-1.0):
    
    # modifies existing image using a specral index alpha and assuming
    # the star has a spectral index of -2.

    image_in_jypix, nx, ny, nf, lam, pixdeg_x, pixdeg_y = load_image(path_image, dpc)
    
    istar, jstar=star_pix(nx, omega) 
    Fstar=image_in_jypix[0,0,istar,istar]
    ### dust spectral index
    image_in_jypix[0,0,istar,istar]=0.0
    image_in_jypix=image_in_jypix*(newlam/lam0)**(-alpha_dust)
    ### star spectral index
    if fstar<0.0:
        Fstar=Fstar*(newlam/lam0)**(-2.0)
        image_in_jypix[0,0,istar,istar]=Fstar
    else:
        image_in_jypix[0,0,istar,istar]=fstar

    image_in_jypix_shifted= shift_image(image_in_jypix, mx, my, pixdeg_x, pixdeg_y, omega=omega)

    
    # Make FITS header information from canvas 
    
    canvas=fits.open(path_canvas)
    header = canvas[0].header # fits.Header()
    del header['Origin'] # necessary due to a comment that CASA adds automatically

    
    # PAD IMAGE
    pad_x = header['NAXIS1']
    pad_y = header['NAXIS2']

    image_in_jypix_shifted=fpad_image(image_in_jypix_shifted, pad_x, pad_y, nx, ny)

    ##### multiply by primary beam or not. Only necessary when simulating visibilities in CASA with ft a posteriori

    if pbm:
        # print "multiply by pb beam!"
        # multiply by primary beam
        pbfits=fits.open(path_canvas[:-4]+'pb.fits')
        pb=pbfits[0].data[0,0,:,:]
        image_in_jypix_shifted=image_in_jypix_shifted*pb

        inans= np.isnan(image_in_jypix_shifted)
        image_in_jypix_shifted[inans]=0.0
    else:
        print "don't multiply by pbm"

    image_in_jypix_float=image_in_jypix_shifted.astype(np.float32)

    flux = np.sum(image_in_jypix_float)
    header['BUNIT'] = 'JY/PIXEL' 
    header['FLUX']=flux
    # print "flux [Jy] = ", flux


    fits.writeto(path_fits, image_in_jypix_float, header, output_verify='fix')




def Simimage(dpc, X0, Y0, imagename, wavelength, Npix, dpix, inc, PA, offx=0.0, offy=0.0, tag='', omega=0.0, Npixf=-1):

    # X0, Y0, stellar position (e.g. useful if using a mosaic)

    # images: array of names for images produced at wavelengths
    # wavelgnths: wavelengths at which to produce images
    # fields: fields where to make images (=[0] unless observations are a mosaic)
    if Npixf==-1:
        Npixf=Npix

    sau=Npix*dpix*dpc

    os.system('radmc3d image incl '+str(inc)+' phi '+str(omega)+' posang '+str(PA-90.0)+'  npix '+str(Npix)+' lambda '+str(wavelength)+' sizeau '+str(sau)+' secondorder  > simimgaes.log')
    pathin ='image_'+imagename+'_'+tag+'.out'
    os.system('mv image.out '+pathin)
    pathout='image_'+imagename+'_'+tag+'.fits'
    convert_to_fits(pathin, pathout,Npixf, dpc, mx=offx, my=offy, x0=X0, y0=Y0, omega=omega)
    os.system('mv '+pathout+' ./images')

def Simimage_alpha(dpc, X0, Y0, imagename0, imagename, lam0, newlam, Npix, dpix, offx=0.0, offy=0.0, tag0='',tag='', alpha_d=3.0, omega=0.0, Npixf=-1):

    # images: array of names for images produced at wavelengths
    # wavelgnths: wavelengths at which to produce images

    if Npixf==-1:
        Npixf=Npix
    sau=Npix*dpix*dpc
    
    pathin ='image_'+imagename0+'_'+tag0+'.out'
    pathout='image_'+imagename+'_'+tag+'.fits'
    convert_to_fits_alpha(pathin, pathout, Npixf, dpc, lam0, newlam, mx=offx, my=offy, x0=X0, y0=Y0, alpha_dust=alpha_d, omega=omega)
     #                     path_image,path_fits, Npixf, dpc , lam0, newlam, mx=0.0, my=0.0, x0=0.0, alpha_dust=3.0, y0=0.0, omega=0.0
    os.system('mv '+pathout+' ./images')


def Simimage_canvas(dpc, imagename, wavelength, Npix, dpix, canvas, inc, PA, offx=0.0, offy=0.0, pb=0.0, tag='', omega=0.0, fstar=-1.0):

    # X0, Y0, stellar position (e.g. useful if using a mosaic)

    # images: array of names for images produced at wavelengths
    # wavelgnths: wavelengths at which to produce images
    # fields: fields where to make images (=[0] unless observations are a mosaic)

    sau=Npix*dpix*dpc
    os.system('radmc3d image incl '+str(inc)+' phi '+str(omega)+' posang '+str(PA-90.0)+'  npix '+str(Npix)+' lambda '+str(wavelength)+' sizeau '+str(sau)+' secondorder > simimgaes.log')
    pathin ='image_'+imagename+'_'+tag+'.out'
    os.system('mv image.out '+pathin)
    pathout='image_'+imagename+'_'+tag+'.fits'
    convert_to_fits_canvas(pathin, pathout, canvas+'.fits' ,dpc, mx=offx, my=offy, pbm=pb, omega=omega, fstar=fstar)
    os.system('mv '+pathout+' ./images')




def Simimage_canvas_fields(dpc, X0, Y0, imagename, wavelength, Npix, dpix, canvas, inc, PA, offx=0.0, offy=0.0, pb=0.0, tag='', omega=0.0):

    # X0, Y0, stellar position (e.g. useful if using a mosaic)

    # images: array of names for images produced at wavelengths
    # wavelgnths: wavelengths at which to produce images
    # fields: fields where to make images (=[0] unless observations are a mosaic)

    sau=Npix*dpix*dpc

    os.system('radmc3d image incl '+str(inc)+' phi '+str(omega)+' posang '+str(PA-90.0)+'  npix '+str(Npix)+' lambda '+str(wavelength)+' sizeau '+str(sau)+' secondorder  > simimgaes.log')
    pathin ='image_'+imagename+'_'+tag+'.out'
    os.system('mv image.out '+pathin)
    pathout='image_'+imagename+'_'+tag+'.fits'
    convert_to_fits_canvas_fields(pathin, pathout, canvas+'.fits' ,dpc, mx=offx, my=offy, pbm=pb, x0=X0, y0=Y0, omega=omega)
    os.system('mv '+pathout+' ./images')



def Simimage_canvas_alpha(dpc, imagename0, imagename, lam0, newlam, Npix, dpix, canvas, offx=0.0, offy=0.0, pb=0.0, tag0='',tag='', alpha_d=3.0, omega=0.0, fstar=-1.0):

    # images: array of names for images produced at wavelengths
    # wavelgnths: wavelengths at which to produce images
    # fields: fields where to make images (=[0] unless observations are a mosaic)

    sau=Npix*dpix*dpc
    
    pathin ='image_'+imagename0+'_'+tag0+'.out'
    pathout='image_'+imagename+'_'+tag+'.fits'
    convert_to_fits_canvas_alpha(pathin, pathout, canvas+'.fits' ,dpc, lam0, newlam, mx=offx, my=offy, pbm=pb, alpha_dust=alpha_d, omega=omega, fstar=fstar)
    os.system('mv '+pathout+' ./images')

def Simimage_canvas_fields_alpha(dpc, X0, Y0, imagename0, imagename, lam0, newlam, Npix, dpix, canvas, offx=0.0, offy=0.0, pb=0.0, tag0='',tag='', alpha_d=3.0, omega=0.0):

    # images: array of names for images produced at wavelengths
    # wavelgnths: wavelengths at which to produce images
    # fields: fields where to make images (=[0] unless observations are a mosaic)

    sau=Npix*dpix*dpc
    
    pathin ='image_'+imagename0+'_'+tag0+'.out'
    pathout='image_'+imagename+'_'+tag+'.fits'
    convert_to_fits_canvas_fields_alpha(pathin, pathout, canvas+'.fits' ,dpc, lam0, newlam, mx=offx, my=offy, pbm=pb, x0=X0, y0=Y0, alpha_dust=alpha_d, omega=omega)
    os.system('mv '+pathout+' ./images')




def Simimages_canvas_fields(dpc, X0, Y0, images, wavelengths, fields, Npix, dpix, canvas, inc, PA, offx=0.0, offy=0.0, pb=0.0, tag='', omega=0.0):

    # X0, Y0, stellar position (center of the mosaic)

    # images: array of names for images produced at wavelengths
    # wavelgnths: wavelengths at which to produce images
    # fields: fields where to make images (=[0] unless observations are a mosaic)

    sau=Npix*dpix*dpc

    for im in xrange(len(images)):
        os.system('radmc3d image incl '+str(inc)+' phi '+str(omega)+' posang '+str(PA-90.0)+'  npix '+str(Npix)+' lambda '+str(wavelengths[im])+' sizeau '+str(sau)+' secondorder  > simimgaes.log')

        pathin ='image_'+images[im]+'_'+tag+'.out'
        pathout='image_'+images[im]+'_'+tag+'.fits'

        os.system('mv image.out '+pathin)

        for fi in fields:
            pathout='image_'+images[im]+'_'+tag+'_field'+str(fi)+'.fits'
            convert_to_fits_canvas_fields(pathin, pathout, canvas+str(fi)+'.fits' ,dpc, wavelengths[im], mx=offx, my=offy, pbm=pb, x0=X0, y0=Y0, omega=omega)
            os.system('mv '+pathout+' ./images')






def Simimages_canvas_fields_alpha(dpc, X0, Y0, image0, image_new, lam0, newlam, fields, Npix, dpix, canvas, offx=0.0, offy=0.0, pb=0.0, tag='', alpha_d=3.0, omega=0.0):

    # images: array of names for images produced at wavelengths
    # wavelgnths: wavelengths at which to produce images
    # fields: fields where to make images (=[0] unless observations are a mosaic)

    sau=Npix*dpix*dpc
    
    pathin ='image_'+image0+'_'+tag+'.out'

    for fi in fields:
        pathout='image_'+image_new+'_'+tag+'_field'+str(fi)+'.fits'
        convert_to_fits_canvas_fields_alpha(pathin, pathout, canvas+str(fi)+'.fits' ,dpc, lam0, newlam, mx=offx, my=offy, pbm=pb, x0=X0, y0=Y0, alpha_dust=alpha_d, omega=omega)
        os.system('mv '+pathout+' ./images')




################## GAS FITS


## ...pending





############### MISCELANEOUS




def dered(lam,f,Av,Rv):  # unredenning according to cardelli's law


    wl=lam # microns

    # definir a y b

    npts = len(wl)
    a = np.zeros(npts)                
    b = np.zeros(npts)
    F_a=np.zeros(npts)
    F_b=np.zeros(npts)

    for i in range(npts):
        x=1.0/wl[i]
        
        if x<1.1:#x>=0.3 and x<1.1:
            a[i]=0.574*x**(1.61)
            b[i]=-0.527*x**(1.61)

        if x>=1.1 and x<3.3:
            y=x-1.82
            c1 = [ 1. , 0.104,   -0.609,    0.701,  1.137, -1.718,   -0.827,    1.647, -0.505 ]  
            c2 = [ 0.,  1.952,    2.908,   -3.989, -7.985, 11.102,    5.491,  -10.805,  3.347 ]
            for j in range(len(c1)): #polinomios
                a[i]=a[i]+c1[j]*y**j 
                b[i]=b[i]+c2[j]*y**j 

        if x>=3.3 and x<8:
            y=x
    
            if x>5.9: 
                y1=y-5.9
                F_a[i]=-0.04473 * y1**2 - 0.009779 *y1**3
                F_b[i]=0.2130 * y1**2 + 0.1207 * y1**3
    
            a[i]=1.752 - 0.316*y - (0.104 / ( (y-4.67)**2 + 0.341 )) + F_a[i]
            b[i]=-3.090 + 1.825*y + (1.206 / ( (y-4.62)**2 + 0.263 )) + F_b[i]

        if x>=8 and x<=11:

            y=x-8.0
            c1 = [ -1.073, -0.628,  0.137, -0.070 ]
            c2 = [ 13.670,  4.257, -0.420,  0.374 ]
            for j in range(len(c1)): #polinomios
                a[i]=a[i]+c1[j]*y**j 
                b[i]=b[i]+c2[j]*y**j 

    #print a
    #print b
    A=Av*(a+b/Rv)
    
    #plt.plot(wl,A,color='blue')
    #plt.plot(wl,a,color='red')
    #plt.plot(wl,b,color='green')
    #plt.show()
    Funred=np.zeros(npts)
    for i in xrange(npts):
        Funred[i]=f[i]*10.0**(0.4*A[i])
    

    return Funred














