import numpy as np
import math as ma
import matplotlib.pyplot as plt
import sys, os
###### Functions that could be helpful when working on planetary
###### system dynamics. Written by Sebastian Marino with
###### the collaboration of Tim Pearce. 

def M_to_f(M,e):

    # Converts mean anomaly M in radians into true anomaly f in radians


    if M>=2.0*ma.pi:
        M=M-ma.floor(M/(2.0*ma.pi))*ma.pi

    # Newton's to find solution to E-e*sin(E)=M
    E=M
    for ip in xrange(10):
        E= E - (E-e*ma.sin(E)-M) / (1-e*ma.cos(E))

    # derive f from E
    f = 2.0 * ma.atan2( (1+e)**0.5 * ma.sin(E/2.0), (1-e)**0.5 * ma.cos(E/2.0)) 

    return f

def f_to_M(f,e):

    # converts true anomaly f to mean anomaly M, both in radians.

    E = 2.0 *  ma.atan2( (1-e)**0.5 * ma.sin(f/2.0), (1+e)**0.5 * ma.cos(f/2.0)) 
    M = E-e*ma.sin(E)    
    return M 
    
def M_to_r(M, a, e ):

    # Converts mean anomaly M in radians, semi-major axis a in AU and
    # eccentricity e into radius r in AU
    
    # get true anomaly f
    f = M_to_f(M,e)
    r = a*(1.0-e**2.0)/(1.0+e*ma.cos(f)) 
    return r,f

def draw_random_r(a,e, Nr):

    # Draw a sample of Nr radius based on M uniform random sample of
    # mean anomalies

    Ms=np.random.uniform(0.0,2.0*ma.pi,Nr)
    Rs=np.zeros(Nr)
    fs=np.zeros(Nr)
    for ir in xrange(Nr):
        Rs[ir],fs[ir]=M_to_r(Ms[ir],a,e)
    return Rs

def draw_random_rf(a,e, Nr):

    # Draw a sample of Nr radius based on M uniform random sample of
    # mean anomalies

    Ms=np.random.uniform(0.0,2.0*ma.pi,Nr)
    Rs=np.zeros(Nr)
    fs=np.zeros(Nr)
    for ir in xrange(Nr):
        Rs[ir],fs[ir]=M_to_r(Ms[ir],a,e)
    return Rs, fs


def draw_r(a,e, Nr):

    # Draw a sample of Nr radius based on M uniform sample of
    # mean anomalies

    Ms=np.linspace(0.0,2.0*ma.pi,Nr)

    Rs=np.zeros(Nr)
    fs=np.zeros(Nr)
    for ir in xrange(Nr):
        Rs[ir],fs[ir]=M_to_r(Ms[ir],a,e)
       
    return Rs

def draw_random_projectedrf(a,e,inc,omega, Nr):
    # Draw a random sample of Nr projected radius based on M uniform random sample of
    # mean anomalies and orbital elements
    # ANGLES (i, Omega) MUST BE IN RAD
    # omega  is the argument of periapsis 

    Ms=np.random.uniform(0.0,2.0*ma.pi,Nr)
    Rs=np.zeros(Nr)
    fs=np.zeros(Nr)
    for ir in xrange(Nr):
        R1,fs[ir]=M_to_r(Ms[ir],a,e)
        Rs[ir]=R1*( np.cos(omega+fs[ir])**2.0 + np.sin(omega+fs[ir])**2.0*np.cos(inc)**2.0 )**0.5
    return Rs

def draw_random_projectedrfz(a,e,inc,omega, Nr):
    # Draw a random sample of Nr projected radius based on M uniform random sample of
    # mean anomalies and orbital elements
    # ANGLES (i, Omega) MUST BE IN RAD
    # omega  is the argument of periapsis 

    Ms=np.random.uniform(0.0,2.0*ma.pi,Nr)
    Rs=np.zeros(Nr)
    Zs=np.zeros(Nr)
    fs=np.zeros(Nr)
    for i in xrange(Nr):
        R1,fs[i]=M_to_r(Ms[i],a,e)
        Rs[i]=R1*( np.cos(omega+fs[i])**2.0+np.sin(omega+fs[i])**2.0*np.cos(inc)**2.0 )**0.5
        Zs[i]=R1*(np.sin(omega+fs[i])*np.sin(inc))
    return Rs, Zs

def cartesian_from_orbelement(a,e,inc, Omega, pomega, M):
    
    r, f= M_to_r(M, a, e )
    arg_peri=pomega-Omega
    x=r* ((np.cos(Omega)*np.cos(arg_peri+f)) -  (np.sin(Omega)*np.sin(arg_peri+f)*np.cos(inc))  )
    y=r* ((np.sin(Omega)*np.cos(arg_peri+f)) +  (np.cos(Omega)*np.sin(arg_peri+f)*np.cos(inc))  )
    z=r* np.sin(arg_peri+f)*np.sin(inc)

    return x,y,z
        
def cartesian_from_orbelement_rotating_frame(a,e,inc, Omega, pomega, M, alpha):
    # alpha is the rotating angle which needs to be subtracted from x and y
    r, f= M_to_r(M, a, e )
    arg_peri=pomega-Omega
    x=r* ((np.cos(Omega)*np.cos(arg_peri+f)) -  (np.sin(Omega)*np.sin(arg_peri+f)*np.cos(inc))  )
    y=r* ((np.sin(Omega)*np.cos(arg_peri+f)) +  (np.cos(Omega)*np.sin(arg_peri+f)*np.cos(inc))  )
    z=r* np.sin(arg_peri+f)*np.sin(inc)

    xp =  x*np.cos(alpha) + y*np.sin(alpha)
    yp =  -x*np.sin(alpha) + y*np.cos(alpha)

    return xp,yp,z

def draw_random_xyz_fromorb(a,e,inc, Omega, pomega, M, NM=0):

    if NM==0:
        return cartesian_from_orbelement(a,e,inc, Omega, pomega, M)
    else:

        Ms=np.random.uniform(0.0,2.0*ma.pi,NM)
        xs=np.zeros(NM)
        ys=np.zeros(NM)
        zs=np.zeros(NM)

        for im in xrange(NM):
            xs[im], ys[im], zs[im]=cartesian_from_orbelement(a,e,inc, Omega, pomega, Ms[im])

        return xs, ys, zs

def f_Tiss(aplt, a, e, I):
    return aplt/a + 2.0*( (1.0-e**2.0)*a/aplt )**0.5 * np.cos(I)

def e_T(a, aplt, I, T):
    alpha=aplt/a
    return ( 1.0 - alpha/(2.0*np.cos(I))**2.0 * (T - alpha)**2.0     )**0.5

def f_qmin(aplt, T):
    # assumes Q at aplt from Bonsor+2012
    if T>3.0 or T<2.0: return 0.0

    else:

        return aplt*( -T**2.0 + 2.0*T + 4.0 -4.0*(3.0-T)**0.5)/(T**2.0-8.0)

def f_amin(aplt, T):
    # assumes Q at aplt from Bonsor+2012
    if T>3.0 or T<2.0: return 0.0

    else:
        qmin=f_qmin(aplt, T)# aplt*( -T**2.0 + 2.0*T + 4.0 -4.0*(3.0-T)**0.5)/(T**2.0-8.0)
        emax=T-3.0+2.0*(3.-T)**0.5
        return qmin/(1.0-emax), emax # returns amin, emax

def f_rhill(a,m, M=1.0):

    return a*(m/(3.0*M))**(1.0/3.0)


def f_qminQ(aplt,Q,T):

    A= 2.*aplt**2.0*T - aplt*Q*T**2. + 4*Q**2
    B=4.0*np.sqrt( 2.0*aplt**3.*Q - aplt**2*Q**2*T+ Q**4 )
    C=aplt*T**2-8*Q
    
    return (A-B)/C   #(A-B)/C

    
def Tdomain(path_sim):  # for REBOUND

    path_file=path_sim+'parameters_run.out'
    file_0=open(path_file,'r')
    Nlines=len(file_0.readlines() )
    file_0.seek(0)

    Ti=0.0
    Tf=0.0
    dT=0.0
    Nplt=0
    Nsmall=0
    for i,line in enumerate(file_0):
        dat=line.split()
        if dat[0]=='t0':
            Ti=float(dat[2])
        elif dat[0]=='tmax':
            Tf=float(dat[2])
        elif dat[0]=='interval_output':
            dT=float(dat[2])
        elif dat[0]=='Nplanets':
            Nplt=int(dat[2])
        elif dat[0]=='Nparticles':
            Nsmall=int(dat[2])

    NT=int((Tf-Ti)/dT +1)
    file_0.close()
    return Ti,Tf,NT,dT, Nplt, Nsmall



def load_particles(path_sim, Npart, Tp, dTaverage ):
    print "loading particles from "+path_sim
    # returns numpy array with list of x y de-rotated positions of
    # Npart particles between ti and tf

    # FIRST, LOAD SIMULATION PARAMETERS

    Ti,Tf,Nt,dT,Nplt, Nsmall=Tdomain(path_sim)

    if Npart>Nsmall: 
        print "error, Npart> simulated particles"
        sys.exit()
    # SECOND, LOAD ORB ELEMENTS OF PLANET TO DE-ROTATE WITH RESPECT ITS POSITION

    # check how many epochs to save (Ntaverage)
    if Tp<Tf and dTaverage<=Tp:
        Ntaverage=int(dTaverage*2/dT) +1
    elif Tp==Tf and dTaverage<=Tp:
        Ntaverage=int(dTaverage/dT) +1
    elif Tp>Tf and Tp-dTaverage<Tf and dTaverage<=Tp:
        Ntaverage=int((Tf-(Tp-dTaverage))/dT) +1
    else: 
        print "error, epoch of interest does not overlay with simulation epochs"
        sys.exit()

    Ni=int((Tp-dTaverage)/dT) # line where to start loading

    # Orb_par_planet=np.zeros((Ntaverage, 8))
    Alphas=np.zeros(Ntaverage)
    orbplanet=np.loadtxt(path_sim+'body_1.txt', delimiter=',')

    for i2 in xrange(Ni, Ni+Ntaverage):
        ti =orbplanet[i2,0]
        ai =orbplanet[i2,1]
        ei =orbplanet[i2,2]
        inci =orbplanet[i2,3]*np.pi/180.0
        Omegai =orbplanet[i2,4]*np.pi/180.0
        pomegai =orbplanet[i2,5]*np.pi/180.0
        Mi =orbplanet[i2,6]*np.pi/180.0
        fi= M_to_f(Mi,ei)
        
        alphai=pomegai+fi
        
        Alphas[i2-Ni]=alphai
        # Orb_par_planet[i2-Ni, :]=[ti,ai,ei,inci,Omegai, pomegai, Mi, fi]

    # Third, LOAD ORB ELEMENTS OF PARTICLES AND DE-ROTATE THEIR X AND Y

    Particles=np.zeros((Npart, Ntaverage, 8)) # t, x, y,z, a_0, a, e, i
    for i1 in xrange(Npart):
        #print i1, Npart
        filei=open(path_sim+'body_'+str(i1+2)+'.txt', 'r')

        filei.readline()
        a0i= float(filei.readline().split(',')[1])

        filei.seek(0)
        filei.readline() # header

        for i2 in xrange(Ni):
            filei.readline()
        for i2 in xrange(Ntaverage):
            dat=filei.readline().split(',')
            ti =float(dat[0])
            ai =float(dat[1])
            ei =float(dat[2])
            inci =float(dat[3])*np.pi/180.0
            Omegai =float(dat[4])*np.pi/180.0
            pomegai =float(dat[5])*np.pi/180.0
            Mi =float(dat[6])*np.pi/180.0

            alphai=Alphas[i2] # pomega+f for planet

            x,y,z=cartesian_from_orbelement_rotating_frame(ai,ei,inci, Omegai, pomegai, Mi, alphai)
            
            Particles[i1, i2, 0]= ti
            Particles[i1, i2, 1]= x
            Particles[i1, i2, 2]= y
            Particles[i1, i2, 3]= z
            Particles[i1, i2, 4]= a0i
            Particles[i1, i2, 5]= ai
            Particles[i1, i2, 6]= ei
            Particles[i1, i2, 7]= inci

        filei.close()
    return Particles


def load_particles_spread(path_sim, Npart, Tp, Nspread ):
    print "loading particles from "+path_sim
    # returns numpy array with list of x y de-rotated positions of
    # Npart particles between ti and tf

    # FIRST, LOAD SIMULATION PARAMETERS

    Ti,Tf,Nt,dT,Nplt, Nsmall=Tdomain(path_sim)

    if Npart>Nsmall: 
        print "error, Npart> simulated particles"
        sys.exit()
    # SECOND, LOAD ORB ELEMENTS OF PLANET TO DE-ROTATE WITH RESPECT ITS POSITION

    # check how many epochs to save (Ntaverage)
    if Tp<Ti or Tp>Tf:
        print "error, epoch of interest does not overlay with simulation epochs"
        sys.exit()

    # closest epoch
    itp=int(round((Tp-Ti)/dT))

    orbplanet=np.loadtxt(path_sim+'body_1.txt', delimiter=',')

    # for i2 in xrange(Ni, Ni+Ntaverage):
    ti =orbplanet[itp,0]
    ai =orbplanet[itp,1]
    ei =orbplanet[itp,2]
    inci =orbplanet[itp,3]*np.pi/180.0
    Omegai =orbplanet[itp,4]*np.pi/180.0
    pomegai =orbplanet[itp,5]*np.pi/180.0
    Mi =orbplanet[itp,6]*np.pi/180.0
    fi= M_to_f(Mi,ei)
        
    alphai=pomegai+fi
        

    # Third, LOAD ORB ELEMENTS OF PARTICLES AND DE-ROTATE THEIR X AND Y

    Particles=np.zeros((Npart, Nspread, 8)) # t, x, y,z, a_0, a, e, i
    
    for i1 in xrange(Npart):
        filei=open(path_sim+'body_'+str(i1+2)+'.txt', 'r')

        filei.readline()
        a0i= float(filei.readline().split(',')[1])

        filei.seek(0)
        filei.readline() # header

        for i2 in xrange(itp):
            filei.readline()
        # for i2 in xrange(Ntaverage):

        dat=filei.readline().split(',')
        ti =float(dat[0])
        ai =float(dat[1])
        ei =float(dat[2])
        inci =float(dat[3])*np.pi/180.0
        Omegai =float(dat[4])*np.pi/180.0
        pomegai =float(dat[5])*np.pi/180.0
        Mi =float(dat[6])*np.pi/180.0

        # alphai=Alphas[i2] # pomega+f for planet
        
        # spread them along orbit
        if Nspread==1:
            x,y,z=cartesian_from_orbelement_rotating_frame(ai,ei,inci, Omegai, pomegai, Mi, alphai)

            Particles[i1, 0, 0]= ti
            Particles[i1, 0, 1]= x
            Particles[i1, 0, 2]= y
            Particles[i1, 0, 3]= z
            Particles[i1, 0, 4]= a0i
            Particles[i1, 0, 5]= ai
            Particles[i1, 0, 6]= ei
            Particles[i1, 0, 7]= inci

        elif Nspread>1:
            Mis=np.random.uniform(0.0, 2.0*np.pi, Nspread)
            for i3 in xrange(Nspread):
                x,y,z=cartesian_from_orbelement_rotating_frame(ai,ei,inci, Omegai, pomegai, Mis[i3], alphai)
                Particles[i1, i3, 0]= ti
                Particles[i1, i3, 1]= x
                Particles[i1, i3, 2]= y
                Particles[i1, i3, 3]= z
                Particles[i1, i3, 4]= a0i
                Particles[i1, i3, 5]= ai
                Particles[i1, i3, 6]= ei
                Particles[i1, i3, 7]= inci


        filei.close()
    return Particles


def Surface_density(Particles, amin, amax, gamma, xmax, Nbinsx):
    # returns Surface density

    # xmax=200.0         # maximum x and y in AU
    # # zmax=20.0

    # Nbinsx=80           # number of bins in x and y
    # # Nbinsz=10           # number of bins in x

    Binsx=np.linspace(-xmax,xmax,Nbinsx+1)
    dxs=Binsx[1:]-Binsx[:-1]  # radial width of each bin
    xs=Binsx[:-1]+dxs/2.0    # mean radius at each bin

    mask= (Particles[:,0,4]>amin) & (Particles[:,0,4]<amax)

    xs=Particles[mask,:,1 ].flatten()
    ys=Particles[mask,:,2 ].flatten()
    a0s=Particles[mask,:,4 ].flatten()

    Nxy=np.array( np.histogram2d(ys, xs, bins=[Binsx,Binsx], weights=a0s**(gamma+1.0))[0], dtype=float)

    Nxy[Nbinsx/2,Nbinsx/2]=0.0

    # Nxy=np.array(np.histogram2d(Particles[mask,:,2 ],Particles[mask,:,3], bins=[Binsx,Binsx], weights=Particles[mask,:,4 ]**(gamma+1.0))[0], dtype=float)
    # print 'hey'
    return Nxy, Binsx

def Surface_density_r(Particles, amin, amax, gamma, rmax, Nbins):
    # returns Surface density

    # xmax=200.0         # maximum x and y in AU
    # # zmax=20.0

    # Nbinsx=80           # number of bins in x and y
    # # Nbinsz=10           # number of bins in x

    Binsr=np.linspace(0.0,rmax,Nbins+1)
    drs=Binsr[1:]-Binsr[:-1]  # radial width of each bin
    Rs=Binsr[:-1]+drs/2.0    # mean radius at each bin

    mask= (Particles[:,0,4]>amin) & (Particles[:,0,4]<amax)

    rs=((Particles[mask,:,1 ]**2.0+Particles[mask,:,2 ]**2.0)**0.5).flatten()
    a0s=Particles[mask,:,4 ].flatten()

    Nr=np.array( np.histogram(rs, bins=Binsr ,weights=a0s**(gamma), density=True)[0], dtype=float)

    Nr[0]=0.0

    return Nr, Rs
