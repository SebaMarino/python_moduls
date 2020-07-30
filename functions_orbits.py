import numpy as np
import math as ma
import matplotlib.pyplot as plt
import sys, os
###### Functions that could be helpful when working on planetary
###### system dynamics. Written by Sebastian Marino with
###### initial input from Tim Pearce. 

def M_to_f(M,e):

    # Converts mean anomaly M in radians into true anomaly f in radians

    
    if M>=2.0*np.pi:
        M=M-np.floor(M/(2.0*np.pi))*2.0*np.pi

    # Newton's to find solution to E-e*sin(E)=M
    E=M
    for ip in range(10):
        E= E - (E-e*np.sin(E)-M) / (1-e*np.cos(E))

    # derive f from E
    f = 2.0 * np.arctan2( (1+e)**0.5 * np.sin(E/2.0), (1-e)**0.5 * np.cos(E/2.0)) 

    return f

def M_to_f_array(M,e):

    # Converts mean anomaly M in radians into true anomaly f in radians
    mask=M>=2.0*np.pi
    M[mask]=M[mask]-np.floor(M[mask]/(2.0*np.pi))*2.0*np.pi

    # Newton's to find solution to E-e*sin(E)=M
    E=M
    for ip in range(10):
        E= E - (E-e*np.sin(E)-M) / (1-e*np.cos(E))

    # derive f from E
    f = 2.0 * np.arctan2( (1+e)**0.5 * np.sin(E/2.0), (1-e)**0.5 * np.cos(E/2.0)) 

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
    if hasattr(e,"__len__"): # many orbits
        mask_e=e<1.0
        r=np.zeros(len(e))
        f=np.zeros(len(e))
        if hasattr(M,"__len__"): # many mean anomalies
            f[mask_e] = M_to_f_array(M[mask_e],e[mask_e])
            r[mask_e] = a[mask_e]*(1.0-e[mask_e]**2.0)/(1.0+e[mask_e]*np.cos(f[mask_e])) 
        else: # one mean anomaly
            f[mask_e] = M_to_f(M,e[mask_e])
            r[mask_e] = a[mask_e]*(1.0-e[mask_e]**2.0)/(1.0+e[mask_e]*np.cos(f[mask_e])) 
        return r,f
    else: # one orbit
        if e<1.0:
            if hasattr(M,"__len__"): # many mean anomalies
                f = M_to_f_array(M,e)
                r = a*(1.0-e**2.0)/(1.0+e*np.cos(f)) 
            else: # one mean anomaly
                f = M_to_f(M,e)
                r = a*(1.0-e**2.0)/(1.0+e*np.cos(f)) 

            return r, f
        
        return 0.0, 0.0

    
    if e<1.0:
        if hasattr(M,"__len__"):
            f = M_to_f_array(M,e)
            r = a*(1.0-e**2.0)/(1.0+e*np.cos(f)) 
        else:
            f = M_to_f(M,e)
            r = a*(1.0-e**2.0)/(1.0+e*np.cos(f)) 
        return r,f
    else:
        return 0.0, 0.0
    
def draw_random_r(a,e, Nr):

    # Draw a sample of Nr radius based on M uniform random sample of
    # mean anomalies

    Ms=np.random.uniform(0.0,2.0*ma.pi,Nr)
    Rs=np.zeros(Nr)
    fs=np.zeros(Nr)
    for ir in range(Nr):
        Rs[ir],fs[ir]=M_to_r(Ms[ir],a,e)
    return Rs

def draw_random_rf(a,e, Nr):

    # Draw a sample of Nr radius based on M uniform random sample of
    # mean anomalies

    Ms=np.random.uniform(0.0,2.0*ma.pi,Nr)
    Rs=np.zeros(Nr)
    fs=np.zeros(Nr)
    for ir in range(Nr):
        Rs[ir],fs[ir]=M_to_r(Ms[ir],a,e)
    return Rs, fs


def draw_r(a,e, Nr):

    # Draw a sample of Nr radius based on M uniform sample of
    # mean anomalies

    Ms=np.linspace(0.0,2.0*ma.pi,Nr)

    Rs=np.zeros(Nr)
    fs=np.zeros(Nr)
    for ir in range(Nr):
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
    for ir in range(Nr):
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
    for i in range(Nr):
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

    xp =   x*np.cos(alpha) + y*np.sin(alpha)
    yp =  -x*np.sin(alpha) + y*np.cos(alpha)

    return xp,yp,z

def draw_random_xyz_fromorb(a,e,inc, Omega, pomega, M=0.0, NM=0):

    if NM==0:
        return cartesian_from_orbelement(a,e,inc, Omega, pomega, M)
    else:

        Ms=np.random.uniform(0.0,2.0*ma.pi,NM)
        xs=np.zeros(NM)
        ys=np.zeros(NM)
        zs=np.zeros(NM)

        for im in range(NM):
            xs[im], ys[im], zs[im]=cartesian_from_orbelement(a,e,inc, Omega, pomega, Ms[im])

        return xs, ys, zs

def draw_random_xyz_fromorb_dist_a(aps,e,inc, Omega, pomega, NM=0, random=True):

    Na=aps.size

    
    Nt=Na*NM
    xs=np.zeros(Nt)
    ys=np.zeros(Nt)
    zs=np.zeros(Nt)

    for ia in range(Na):
        if random: Ms=np.random.uniform(0.0,2.0*np.pi,NM)
        else: Ms=np.linspace(0.0,2.0*np.pi,NM+1)[:-1]
        # for im in range(NM):
        #     xs[ia*NM+im], ys[ia*NM+im], zs[ia*NM+im]=cartesian_from_orbelement(aps[ia],e,inc, Omega, pomega, Ms[im])
        xs[ia*NM:(ia+1)*NM], ys[ia*NM:(ia+1)*NM], zs[ia*NM:(ia+1)*NM]=cartesian_from_orbelement(aps[ia],e,inc, Omega, pomega, Ms)

    return xs, ys, zs


def draw_random_xyz_fromorb_dist_orbital_elemts(aps,e,inc, Omega, pomega, NM=1, random=True, Ms=None):

    Na=aps.size

    
    # NM=Na, i.e. 1 mean anomaly per orbit
    if hasattr(Ms,"__len__") and random==False:
        Nt=Na
        xs=np.zeros(Nt)
        ys=np.zeros(Nt)
        zs=np.zeros(Nt)
        xs, ys, zs=cartesian_from_orbelement(aps,e,inc, Omega, pomega, Ms)

    else:
        Nt=Na*NM
        xs=np.zeros(Nt)
        ys=np.zeros(Nt)
        zs=np.zeros(Nt)
        for ia in range(Na):
            if random: Ms=np.random.uniform(0.0,2.0*np.pi,NM)
            else: Ms=np.linspace(0.0,2.0*np.pi,NM+1)[:-1]
            xs[ia*NM:(ia+1)*NM], ys[ia*NM:(ia+1)*NM], zs[ia*NM:(ia+1)*NM]=cartesian_from_orbelement(aps[ia],e[ia],inc[ia], Omega[ia], pomega[ia], Ms)

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



def load_particles(path_sim, Npart, Tp, dTaverage, delimiter=',' ):
    print("loading particles from "+path_sim)
    # returns numpy array with list of x y de-rotated positions of
    # Npart particles between ti and tf

    # FIRST, LOAD SIMULATION PARAMETERS

    Ti,Tf,Nt,dT,Nplt, Nsmall=Tdomain(path_sim)

    if Npart>Nsmall: 
        print("error, Npart> simulated particles")
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
        print("error, epoch of interest does not overlay with simulation epochs")
        sys.exit()

    Ni=int((Tp-dTaverage)/dT) # line where to start loading
    print( Ni, Ntaverage)
    # Orb_par_planet=np.zeros((Ntaverage, 8))
    Alphas=np.zeros(Ntaverage)

    filei=open(path_sim+'body_1.txt', 'r')

    filei.readline() # header

    for i2 in range(Ni):
        filei.readline()
    for i2 in range(Ntaverage):

        dat=filei.readline().split(delimiter)
        ti =float(dat[0])
        ai =float(dat[1])
        ei =float(dat[2])
        inci =float(dat[3])*np.pi/180.0
        Omegai =float(dat[4])*np.pi/180.0
        pomegai =float(dat[5])*np.pi/180.0
        Mi =float(dat[6])*np.pi/180.0
        # ti =orbplanet[i2,0]
        # ai =orbplanet[i2,1]
        # ei =orbplanet[i2,2]
        # inci =orbplanet[i2,3]*np.pi/180.0
        # Omegai =orbplanet[i2,4]*np.pi/180.0
        # pomegai =orbplanet[i2,5]*np.pi/180.0
        # Mi =orbplanet[i2,6]*np.pi/180.0
        fi= M_to_f(Mi,ei)
        
        alphai=pomegai+fi
        
        Alphas[i2-Ni]=alphai
        # Orb_par_planet[i2-Ni, :]=[ti,ai,ei,inci,Omegai, pomegai, Mi, fi]

    # Third, LOAD ORB ELEMENTS OF PARTICLES AND DE-ROTATE THEIR X AND Y

    Particles=np.zeros((Npart, Ntaverage, 9)) # t, x, y,z, a_0, a, e, i
    for i1 in range(Npart):
        #print i1, Npart
        filei=open(path_sim+'body_'+str(i1+Nplt+1)+'.txt', 'r')

        filei.readline()
        a0i= float(filei.readline().split(delimiter)[1])

        filei.seek(0)
        filei.readline() # header
        for i2 in range(Ni):
            filei.readline()
        for i2 in range(Ntaverage):
            dat=filei.readline().split(delimiter)
            if len(dat)>1: # when running REBOUND with massive particles, if they are lost then their orbitals elements are not save anymore and the file is shorter.
                ti =float(dat[0])
                ai =float(dat[1])
                ei =float(dat[2])
                inci =float(dat[3])*np.pi/180.0
                Omegai =float(dat[4])*np.pi/180.0
                pomegai =float(dat[5])*np.pi/180.0
                Mi =float(dat[6])*np.pi/180.0

            else:
                ti = 0.0
                ai =0.0 # float(dat[1])
                ei = 0.0 #float(dat[2])
                inci = 0.0 #float(dat[3])*np.pi/180.0
                Omegai =0.0 #float(dat[4])*np.pi/180.0
                pomegai =0.0 #float(dat[5])*np.pi/180.0
                Mi = 0.0 #float(dat[6])*np.pi/180.0

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
            Particles[i1, i2, 8]= pomegai

        filei.close()
    return Particles


def load_particles_spread(path_sim, Npart, Tp, Nspread,  delimiter=',' ):
    print("loading particles from "+path_sim)
    # returns numpy array with list of x y de-rotated positions of
    # Npart particles between ti and tf

    # this function places the planet at y=0 x>0, but it does not take
    # into account the rotating frame when spreading the position of
    # particles. For example, resonant structure will not be visible.

    
    # FIRST, LOAD SIMULATION PARAMETERS

    Ti,Tf,Nt,dT,Nplt, Nsmall=Tdomain(path_sim)

    if Npart>Nsmall: 
        print("error, Npart> simulated particles")
        sys.exit()
    # SECOND, LOAD ORB ELEMENTS OF PLANET TO DE-ROTATE WITH RESPECT ITS POSITION

    # check how many epochs to save (Ntaverage)
    if Tp<Ti or Tp>Tf+0.5*dT:
        print("error, epoch of interest does not overlay with simulation epochs")
        sys.exit()

    # closest epoch
    itp=int(round((Tp-Ti)/dT))
    print(itp)
    filei=open(path_sim+'body_1.txt', 'r')
    filei.readline() # header 

    for i2 in range(itp):
        filei.readline()
    dat=filei.readline().split(delimiter)
    ti =float(dat[0])
    ai =float(dat[1])
    ei =float(dat[2])
    inci =float(dat[3])*np.pi/180.0
    Omegai =float(dat[4])*np.pi/180.0
    pomegai =float(dat[5])*np.pi/180.0
    Mi =float(dat[6])*np.pi/180.0
    fi= M_to_f(Mi,ei)
        
    alphai=pomegai+fi
    
    filei.close()
    

    # Third, LOAD ORB ELEMENTS OF PARTICLES AND DE-ROTATE THEIR X AND Y

    Particles=np.zeros((Npart, Nspread, 8)) # t, x, y,z, a_0, a, e, i
    
    for i1 in range(Npart):
        filei=open(path_sim+'body_'+str(i1+2)+'.txt', 'r')

        filei.readline()
        a0i= float(filei.readline().split(delimiter)[1])

        filei.seek(0)
        filei.readline() # header

        for i2 in range(itp):
            filei.readline()
        # for i2 in range(Ntaverage):

        dat=filei.readline().split(delimiter)
        # print dat, i1+2
        if len(dat)>1: # when running REBOUND with massive particles, if they are lost then their orbitals elements are not save anymore and the file is shorter.
            ti =float(dat[0])
            ai =float(dat[1])
            ei =float(dat[2])
            inci =float(dat[3])*np.pi/180.0
            Omegai =float(dat[4])*np.pi/180.0
            pomegai =float(dat[5])*np.pi/180.0
            Mi =float(dat[6])*np.pi/180.0
        else:
            print(i1+2)
            ti =Tp
            ai =0.0 # float(dat[1])
            ei = 0.0 #float(dat[2])
            inci = 0.0 #float(dat[3])*np.pi/180.0
            Omegai =0.0 #float(dat[4])*np.pi/180.0
            pomegai =0.0 #float(dat[5])*np.pi/180.0
            Mi = 0.0 #float(dat[6])*np.pi/180.0
      
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
            
            # for i3 in range(Nspread):
            x,y,z=cartesian_from_orbelement_rotating_frame(ai,ei,inci, Omegai, pomegai, Mis, alphai)
            Particles[i1, :, 0]= ti
            Particles[i1, :, 1]= x
            Particles[i1, :, 2]= y
            Particles[i1, :, 3]= z
            Particles[i1, :, 4]= a0i
            Particles[i1, :, 5]= ai
            Particles[i1, :, 6]= ei
            Particles[i1, :, 7]= inci


        filei.close()
    return Particles





def load_particles_spread_rotframe(path_sim, Npart, Tp, Nspread,  delimiter=',', Mstar=1.0 ):
    print("loading particles from "+path_sim)
    # returns numpy array with list of x y de-rotated positions of
    # Npart particles between ti and tf

    # this function places the planet at y=0 x>0, and it does take
    # into account the rotating frame when spreading the position of
    # particles. For example, resonant structure WILL be visible.

    # FIRST, LOAD SIMULATION PARAMETERS

    Ti,Tf,Nt,dT,Nplt, Nsmall=Tdomain(path_sim)

    if Npart>Nsmall: 
        print("error, Npart> simulated particles")
        sys.exit()
    # SECOND, LOAD ORB ELEMENTS OF PLANET TO DE-ROTATE WITH RESPECT ITS POSITION

    # check how many epochs to save (Ntaverage)
    if Tp<Ti or Tp>Tf+0.5*dT:
        print("error, epoch of interest does not overlay with simulation epochs")
        sys.exit()

    # closest epoch
    itp=int(round((Tp-Ti)/dT))

    filei=open(path_sim+'body_1.txt', 'r')
    filei.readline() # header 

    for i2 in range(itp):
        filei.readline()
    dat=filei.readline().split(delimiter)
    tp =float(dat[0])
    ap =float(dat[1])
    ep =float(dat[2])
    incp =float(dat[3])*np.pi/180.0
    Omegap =float(dat[4])*np.pi/180.0
    pomegap =float(dat[5])*np.pi/180.0
    Mp =float(dat[6])*np.pi/180.0
    fp= M_to_f(Mp,ep)
        
    alphap=pomegap+fp
    
    filei.close()
    
    nplt=np.sqrt(Mstar/(ap**3.0)) # angular speed

    # Third, LOAD ORB ELEMENTS OF PARTICLES AND DE-ROTATE THEIR X AND Y

    Particles=np.zeros((Npart, Nspread, 8)) # t, x, y,z, a_0, a, e, i
    
    for i1 in range(Npart):
        filei=open(path_sim+'body_'+str(i1+2)+'.txt', 'r')

        filei.readline()
        a0i= float(filei.readline().split(delimiter)[1])

        filei.seek(0)
        filei.readline() # header

        for i2 in range(itp):
            filei.readline()
        # for i2 in range(Ntaverage):

        dat=filei.readline().split(delimiter)
        # print dat, i1+2
        if len(dat)>1: # when running REBOUND with massive particles, if they are lost then their orbitals elements are not save anymore and the file is shorter.
            ti =float(dat[0])
            ai =float(dat[1])
            ei =float(dat[2])
            inci =float(dat[3])*np.pi/180.0
            Omegai =float(dat[4])*np.pi/180.0
            pomegai =float(dat[5])*np.pi/180.0
            Mi =float(dat[6])*np.pi/180.0
        else:
            print(i1+2)
            ti =Tp
            ai =0.0 # float(dat[1])
            ei = 0.0 #float(dat[2])
            inci = 0.0 #float(dat[3])*np.pi/180.0
            Omegai =0.0 #float(dat[4])*np.pi/180.0
            pomegai =0.0 #float(dat[5])*np.pi/180.0
            Mi = 0.0 #float(dat[6])*np.pi/180.0
      
        # alphai=Alphas[i2] # pomega+f for planet
        
        # spread them along orbit
        if Nspread==1 and ai>0.0 and ei>=0.0 and ei<1.0:
            x,y,z=cartesian_from_orbelement_rotating_frame(ai,ei,inci, Omegai, pomegai, Mi, alphap)

            Particles[i1, 0, 0]= ti
            Particles[i1, 0, 1]= x
            Particles[i1, 0, 2]= y
            Particles[i1, 0, 3]= z
            Particles[i1, 0, 4]= a0i
            Particles[i1, 0, 5]= ai
            Particles[i1, 0, 6]= ei
            Particles[i1, 0, 7]= inci

        elif Nspread>1 and ai>0.0 and ei>=0.0 and ei<1.0:

            ni=np.sqrt(Mstar/(ai**3.0)) # angular speed
            Tc=2.*np.pi/np.abs(ni-nplt) # conjunction period in years
            if ni*Tc>10.0*np.pi and nplt*Tc>10.0*np.pi:
                Tc=2.0*np.pi/np.min([ni, nplt])
            #print Tc
            Mrand=np.random.uniform(0.0, 2.0*ni*Tc, Nspread)#ni*Tc, Nspread) # random.uniform
            Mis=Mrand+Mi
            Mps=Mrand*(nplt/ni)+Mp
            
            #for i3 in range(Nspread):
            alphas=pomegap+M_to_f(Mps,ep)

            x,y,z=cartesian_from_orbelement_rotating_frame(ai,ei,inci, Omegai, pomegai, Mis, alphas)
            Particles[i1, :, 0]= ti
            Particles[i1, :, 1]= x
            Particles[i1, :, 2]= y
            Particles[i1, :, 3]= z
            Particles[i1, :, 4]= a0i
            Particles[i1, :, 5]= ai
            Particles[i1, :, 6]= ei
            Particles[i1, :, 7]= inci


        filei.close()
    return Particles


def load_planets(path_sim,  rot_frame=True, delimiter=','):

    Ti,Tf,NT,dT, Nplt, Nsmall= Tdomain(path_sim)

    Particles=np.zeros((Nplt,NT,  9)) # t, x, y,z, a_0, a, e, i, M
    Alphas=np.zeros(NT)
    for i1 in range(Nplt):

        filei=open(path_sim+'body_'+str(i1+1)+'.txt', 'r')
        filei.readline()
        a0i= float(filei.readline().split(delimiter)[1])

        filei.seek(0)
        filei.readline() # header

        for i2 in range(NT):

            dat=filei.readline().split(delimiter)
            if len(dat)>1:
                ti =float(dat[0])
                ai =float(dat[1])
                ei =float(dat[2])
                inci =float(dat[3])*np.pi/180.0
                Omegai =float(dat[4])*np.pi/180.0
                pomegai =float(dat[5])*np.pi/180.0
                Mi =float(dat[6])*np.pi/180.0
                Massi =float(dat[7]) # earth masses

            else:
                ti = 0.0 
                ai = 0.0 
                ei = 0.0 
                inci = 0.0
                Omegai = 0.0 
                pomegai = 0.0 
                Mi = 0.0
                Massi = 0.0 # earth masses

            if i1==0:
                fi= M_to_f(Mi,ei)
                alphai=pomegai+fi
                Alphas[i2]=alphai

            if rot_frame:
                x,y,z=cartesian_from_orbelement_rotating_frame(ai,ei,inci, Omegai, pomegai, Mi, Alphas[i2])
            else:
                x,y,z=cartesian_from_orbelement(ai,ei,inci, Omegai, pomegai, Mi)

            Particles[i1, i2, 0]= ti
            Particles[i1, i2, 1]= x
            Particles[i1, i2, 2]= y
            Particles[i1, i2, 3]= z
            Particles[i1, i2, 4]= a0i
            Particles[i1, i2, 5]= ai
            Particles[i1, i2, 6]= ei
            Particles[i1, i2, 7]= inci
            Particles[i1, i2, 8]= Massi
    return Particles


def load_particles_spread_x(path_sim, Nspread, delimiter=',' ): # function to load and spread particles from simulation with output: a_initial, a_final, e, pomega, i, Omega, f
    print("loading particles from "+path_sim)
   
    # LOAD ORB ELEMENTS OF PARTICLES AND DE-ROTATE THEIR X AND Y
    particles0=np.loadtxt(path_sim, delimiter=delimiter)
    Npart=np.shape(particles0)[0]
    print('N particles = ',Npart)
    Particles=np.zeros((Npart, Nspread, 8)) # t, x, y,z, a_0, a, e, i
    
    for i1 in range(Npart):
       
        a0i= particles0[i1,0]
        ai = particles0[i1,1]
        ei = particles0[i1,2]
        pomegai = particles0[i1,3]
        inci = particles0[i1,4]
        Omegai = particles0[i1,5]
        Mi =f_to_M(particles0[i1,6], ei)
        
        # spread them along orbit
        if Nspread==1:
            x,y,z=cartesian_from_orbelement(ai,ei,inci, Omegai, pomegai, Mi)

            Particles[i1, 0, 1]= x
            Particles[i1, 0, 2]= y
            Particles[i1, 0, 3]= z
            Particles[i1, 0, 4]= a0i
            Particles[i1, 0, 5]= ai
            Particles[i1, 0, 6]= ei
            Particles[i1, 0, 7]= inci

        elif Nspread>1:
            Mis=np.random.uniform(0.0, 2.0*np.pi, Nspread)
            
            # for i3 in range(Nspread):
            x,y,z=cartesian_from_orbelement(ai,ei,inci, Omegai, pomegai, Mis[:])
            Particles[i1, :, 1]= x
            Particles[i1, :, 2]= y
            Particles[i1, :, 3]= z
            Particles[i1, :, 4]= a0i
            Particles[i1, :, 5]= ai
            Particles[i1, :, 6]= ei
            Particles[i1, :, 7]= inci


    return Particles

def Surface_density(Particles, amin=0.0, amax=1000.0, gamma=-1.0, xmax=200.0, Nbinsx=50, a1=-1.0, a2=-1.0):
    # returns Surface density

    # xmax=200.0         # maximum x and y in AU
    # # zmax=20.0

    # Nbinsx=80           # number of bins in x and y
    # # Nbinsz=10           # number of bins in x

    Binsx=np.linspace(-xmax,xmax,Nbinsx+1)
    dxs=Binsx[1:]-Binsx[:-1]  # radial width of each bin
    xs=Binsx[:-1]+dxs/2.0    # mean radius at each bin

    if a1==-1.0 and a2==-1.0:
        mask= (Particles[:,0,4]>amin) & (Particles[:,0,4]<amax)

    elif a2>a1 and a1>amin and a2<amax: 
        mask= ((Particles[:,0,4]>amin) & (Particles[:,0,4]<a1)) |  ((Particles[:,0,4]<amax) & (Particles[:,0,4]>a2) )
    else:
        #print 'error when masking particles'
        mask= (Particles[:,0,4]>amin) & (Particles[:,0,4]<amax)

    xs=Particles[mask,:,1 ].flatten()
    ys=Particles[mask,:,2 ].flatten()
    a0s=Particles[mask,:,4 ].flatten()

    Nxy=np.array( np.histogram2d(ys, xs, bins=[Binsx,Binsx], weights=a0s**(gamma+1.0))[0], dtype=float)

    Nxy[int(Nbinsx/2),int(Nbinsx/2)]=0.0

    # Nxy=np.array(np.histogram2d(Particles[mask,:,2 ],Particles[mask,:,3], bins=[Binsx,Binsx], weights=Particles[mask,:,4 ]**(gamma+1.0))[0], dtype=float)
    # print 'hey'
    return Nxy, Binsx


def Volume_density(Particles, amin=0.0, amax=1000.0, gamma=-1.0, xmax=200.0, Nbinsz=5, Nbinsx=50, a1=-1.0, a2=-1.0):
    # returns Surface density

    # xmax=200.0         # maximum x and y in AU
    # # zmax=20.0

    # Nbinsx=80           # number of bins in x and y
    # # Nbinsz=10           # number of bins in x

    Binsx=np.linspace(-xmax,xmax,Nbinsx+1)
    dxs=Binsx[1:]-Binsx[:-1]  # radial width of each bin
    xs=Binsx[:-1]+dxs/2.0    # mean radius at each bin
    if a1==-1.0 and a2==-1.0:
        mask= (Particles[:,0,4]>amin) & (Particles[:,0,4]<amax)

    elif a2>a1 and a1>amin and a2<amax: 
        mask= ((Particles[:,0,4]>amin) & (Particles[:,0,4]<a1)) |  ((Particles[:,0,4]<amax) & (Particles[:,0,4]>a2) )
    else:
        print('error when masking particles')

    Zmax=dxs[0]*(Nbinsz/2.0) 
    Binsz=np.linspace(-Zmax, Zmax, Nbinsz+1)
    xs=Particles[mask,:,1 ].flatten()
    ys=Particles[mask,:,2 ].flatten()
    zs=Particles[mask,:,3 ].flatten()
    a0s=Particles[mask,:,4 ].flatten()

    Nxyz=np.array( np.histogramdd( (zs, ys, xs), bins=(Binsz,Binsx,Binsx), weights=a0s**(gamma+1.0))[0], dtype=float)

    Nxyz[:,int(Nbinsx/2),int(Nbinsx/2)]=0.0

    # Nxy=np.array(np.histogram2d(Particles[mask,:,2 ],Particles[mask,:,3], bins=[Binsx,Binsx], weights=Particles[mask,:,4 ]**(gamma+1.0))[0], dtype=float)
    # print 'hey'
    return Nxyz, Binsx, Binsz

def Surface_density_r(Particles, amin, amax, gamma, rmax, Nbins, a1=-1.0, a2=-1.0):
    # returns Surface density

    # xmax=200.0         # maximum x and y in AU
    # # zmax=20.0

    # Nbinsx=80           # number of bins in x and y
    # # Nbinsz=10           # number of bins in x

    Binsr=np.linspace(0.0,rmax,Nbins+1)
    drs=Binsr[1:]-Binsr[:-1]  # radial width of each bin
    Rs=Binsr[:-1]+drs/2.0    # mean radius at each bin

    if a1==-1.0 and a2==-1.0:
        mask= (Particles[:,0,4]>amin) & (Particles[:,0,4]<amax)

    elif a2>a1 and a1>amin and a2<amax: 
        mask= ((Particles[:,0,4]>amin) & (Particles[:,0,4]<a1)) |  ((Particles[:,0,4]<amax) & (Particles[:,0,4]>a2) )
    else:
        print('error when masking particles')

    rs=((Particles[mask,:,1 ]**2.0+Particles[mask,:,2 ]**2.0)**0.5).flatten()
    a0s=Particles[mask,:,4 ].flatten()
    print(len(rs))
    Nr=np.array( np.histogram(rs, bins=Binsr ,weights=a0s**(gamma), density=True)[0], dtype=float)

    Nr[0]=0.0

    return Nr, Rs


def orbit_resonance_rotframe(aplt,  ecc, inc, p, q,
                             eplt=0.0,
                             iplt=0.0,
                             Omegaplt=0.0,
                             pomegaplt=0.0,
                             Omegap=0.0,
                             pomegap=0.0,
                             Mstar=1.0,
                             res='external'):

    Norbits=100.0
    Np=2000
    #### PLANET ORBIT
    
    Tplt=np.sqrt(aplt**3.0/Mstar) # yr
    
    ts=np.linspace(0.0, Norbits*Tplt, Np)
    Maplt=(ts/Tplt)*2.*np.pi

    alphas=np.zeros(Np)

    for i in range(Np):
        #xi,yi,zi=cartesian_from_orbelement(aplt,eplt,iplt, Omegaplt, pomegaplt, Mplt[i])

        fi=M_to_f(Maplt[i], eplt)
        alphas[i]=pomegaplt+fi#np.arctan2(yi,xi)
        
    #### particles (external resonance)
    if res=='external':
        a=aplt*(float(p+q)/float(p))**(2./3.)
    elif res=='internal':
        a=aplt*(float(p)/float(p+q))**(2./3.)
    else:
        print('error, no internal nor exernal resonance')
        sys.exit()
    Tp=np.sqrt(a**3.0/Mstar) # yr
    print( Tp, Tplt)
    Map=(ts/Tp)*2.*np.pi

    xs=np.zeros(Np)
    ys=np.zeros(Np)
    
    for i in range(Np):
        xs[i],ys[i],zi=cartesian_from_orbelement_rotating_frame(a,ecc,0.0, Omegap, pomegap, Map[i], alphas[i])


    return xs, ys, alphas, a
