import numpy as np
import matplotlib.pyplot as plt

#### physical constants

au=1.496e+11    # [m]
G=6.67408e-11   # [m3 kg-1 s-2]
Msun=1.989e30   # [kg]
Mearth=5.972e24 # [kg]
year= 3.154e+7  # [s]


def f_Qd(Di, vimp=3.0e3, Qs=608.0, bs=0.38, Qg=0.011, bg=1.36 ): ### DISPERSAL THRESHOLD
    # Di in meters
    # # Basalt at 3 km s^-1 (Benz and Asphaug 1999) and assuming rho=2.7 g/cm3
    # Qa=608.0 # J kg-1
    # a=0.38  
    # Qb=0.011 # J kg-1
    # b=1.36
    
    v0=3.0e3 # m/s

    Qd=(Qs*Di**(-bs)+Qg*Di**bg)*(vimp/v0)**0.5

    # if Di<0.001:
    #     return Qd*100 #(Qs*Di**(-bs)+Qg*Di**bg)*(vimp/v0)**0.5
    # else:
    return Qd
    
def f_dmax( vimp=3.0e3, Qg=0.03, bg=1.38 ):
    v0=3.0e3 # m/s
    Xcc=0.95
    return (Xcc**3.0 * vimp**1.5 * v0 **0.5 / Qg / 2.0 )**(1.0/bg)

           
def f_Rcoll(Ds, k, Msi, V, vrel, rho, Qs=500.0, bs=0.37, Qg=0.03, bg=1.38):  #### DESTRUCTIVE COLLISION RATE

    delta=Ds[0]/Ds[1] # -1.0
    ND=len(Ds)
    Xc=(2.0*f_Qd(Ds[k], vrel, Qs, bs, Qg, bg)/vrel**2.0)**(1.0/3.0)
    if Xc*Ds[k]>Ds[0]:
        return 0.0

    kmaxi=min(k-int(np.log(Xc)/np.log(delta)),ND-1) # min size for cat collision
    Ri=3.0*np.sum(Msi[:kmaxi]*(Ds[k]+Ds[:kmaxi])**2.0/(Ds[:kmaxi]**3.0))*np.pi*vrel/(2.0*rho*np.pi*V)
    return Ri


def SizeDist(Mstar=1.0, 
             e=0.05,
             I=0.05/2.0,    
             r0=1.0,          
             dr=0.05*2.0*1.0, 
             Mbeltem=1.0,          
             rho=2700.0,
             dmin=0.8e-6,
             dmax=1.0e5,
             ND=500,
             alphap=-3.7,
             ts=[100.0],
             Qs=608.0,
             bs=0.38,
             Qg=0.011,
             bg=1.36): 
    # ##################################################
    # returns size distribution: Ds, M(Ds)
    # Mstar=1.0, stellar mass in solar masses 
    # e=0.05, mean eccentricities
    # I=0.05/2.0,  mean inclinations in radians
    # r0=1.0,      mean radius in au
    # dr=0.05*2.0*1.0,  width of annulus in au
    # Mbeltem=1.0,     total mass in solids in earth Masses
    # rho=2700.0, bulk density of the solids in kg/m3
    # dmin=1.0e-6, minimum size of solids in m
    # dmax=1.0e5, maximum size of solids in m
    # ND=500,     Number of size log bins 
    # alphap=-3.7, primordial size distribution
    # ts=100.0 age of the system in Myr (it must be an array)
    
    # ##############################################


    Mstar=Mstar*Msun # kg
    rmid=r0*au # m
    Mbelt=Mbeltem*Mearth # kg
    print rmid/au, dr, I
    Vol=2.0*np.pi*rmid*dr*au*I*rmid # [m3]
    print "Vol [au3] = ", Vol/au**3.0
    vk=(G*Mstar/rmid)**0.5 # m/s
    vrel=vk*(1.25*e**2.0+I**2.0)**(0.5) # m/s
    print "vrel [km/s] = ",vrel/1.0e3
    # ######## BINS

    
    Ds=np.logspace(np.log10(dmax), np.log10(dmin), ND) # from big to small [m]
    delta=Ds[0]/Ds[1] # -1.0

    # ########## redistribution function
    alphar=-3.5
    alphare=20
    etarmax=2.0**(-1.0/3.0) # the largest objects has half the mass of the original
    keta=-int(np.log(etarmax)/np.log(delta)) # number of bins
    Fredist=np.zeros(ND)

    for i in xrange(0,ND):
        if (delta)**(-i)<=etarmax:
            Fredist[i]=(delta**(-i))**(4.0+alphar)
        else:
            Fredist[i]=1.0*(delta**(-i))**(4.0+alphare)
    Fredist=Fredist/np.sum(Fredist)

    
    Nt=len(ts)

    Msfinal=np.zeros((Nt, ND))

    # ######################################
    # ########### initial conditions
    # ######################################

    Ms0=np.zeros(ND)
    for i in xrange(ND):
        Ms0[i]=(Ds[i]/dmax)**(alphap+4.0)
    Ms0=Ms0*Mbelt/np.sum(Ms0)

    
    # compute collisional rates to guess maximum size in steady state
    Rs0=np.zeros(ND)
    for i in xrange(ND):
        Rs0[i]=f_Rcoll(Ds, i,Ms0, Vol, vrel, rho, Qs, bs, Qg, bg)

    # plt.plot(Ds,Rs0)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()

    # #############################################
    # Calculate size distribution at different epochs
    # #############################################
    for it in xrange(Nt):

        if ts[it]==0.0:
            Msfinal[it,:]=Ms0
        
        elif ts[it]>0.0:
            

            # Compute initial collisional timescales for each size
            tcolls=1.0/Rs0 /(1.0e6*year) # Myr # this can give inf as Rs0 can be 0 if e<emax to break up planetesimals
            print tcolls[:30]
            # initial guess for Dc
            kmin=1
            kmax=kmin
            for k in xrange(kmin,ND):
                if tcolls[k]<ts[it]:
                    kmax=k
                    break
            kmaxu=kmax*1 # upper limit
            kmaxd=0 # lower limit

            Nit=20
            ncheck=18

            Mp=np.zeros(ND)
            Ms=Ms0*1.0
            Mb=Ms0*2.0 # Ms used to save the one from a previous iteration
            tcollib=0.0
            Rs=np.zeros(ND)
            Rcoll1=np.zeros(ND)
            Cs=np.zeros(ND)

            # #############################################
            # ######## ITERATE TO REACH STEADY STATE SIZE DISTRIBUTION
            # #############################################
       
            ni=0
            while ni<Nit:
                print "it , ni ", it, ni

                Mp=Ms # it should be Mp=Ms*1.0, but without the *1.0 it converges faster
                # Method from Wyatt+2011
                for k in xrange(ND):

                    Rcoll1[k]=f_Rcoll(Ds,k,Ms, Vol, vrel, rho, Qs, bs, Qg, bg)
                for k in xrange(ND):
        
                    C1k=np.sum(Rcoll1[:k]*Ms[:k]*Fredist[k:0:-1])
            
                    if k>=kmax and Rcoll1[k]>0.0:
                        Mp[k]=(C1k/Rcoll1[k]+Ms[k])/2.0
            
                    Rcoll2=f_Rcoll(Ds,k,Mp, Vol, vrel, rho, Qs, bs, Qg, bg)

                    if k>=kmax and Rcoll2>0.0:
                        Ms[k]=(C1k/Rcoll2+Ms[k])/2.0

                    Rs[k]=Rcoll2
                    Cs[k]=C1k




                # compare age and coll timescale
                tcolli=1.0/Rs /(1.0e6*year)
                print ts[it],  kmax, kmin,  tcolli[kmax+1], tcolli[kmax], tcolli[kmax-1],  tcolli[0], kmaxu, kmaxd
                ni+=1

                # correct maximum size if age<tcoll

                if ts[it]*1.1< tcolli[kmax] and kmax<ND-1 and ni>ncheck and kmaxd==0:
                    for ik in xrange(kmax+1,ND):
                        if tcolli[ik]<ts[it]:
                            kmax=int(round( (ik+ kmax)/2.0 ))
                            break
                    ni=0
                    tcollib=tcolli[kmax-1]
                    Mb=Ms*1.0
                    Ms= Ms0*1.0 #/(1.0+ts[it]/tcolli[0])
        

                if ts[it]*1.1< tcolli[kmax] and kmax<ND-1 and ni>ncheck and kmaxd!=0 and (kmaxd-kmax>0 or kmaxd==0):
                    kmaxu=kmax*1
                    kmax=int(round((kmaxd+ kmax)/2.0))
                    ni=0
                    tcollib=tcolli[kmax-1]
                    Mb=Ms*1.0
                    Ms= Ms0*1.0 

                elif ts[it]> 1.1*tcolli[kmax-1] and kmax>kmaxu and ni>ncheck and (kmaxd-kmaxu>1 or kmaxd==0):
                    kmaxd=kmax*1
                    kmax=int(round( (kmaxu + kmax)/2))
                    ni=0
                    Mb=Ms*1.0
                    tcollib=tcolli[kmax-1]
                    Ms= Ms0*1.0 #/(1.0+ts[ti]/tcolli[0])
                
                elif tcolli[kmax-1]==np.inf and ni>ncheck: # tcolli[kmax-1]==np.inf 
                    kmax+=1
                    kmaxu=kmax*1
                    ni=0
                    tcollib=tcolli[kmax-1]
                    Mb=Ms*1.0
                    Ms= Ms0*1.0 #/(1.0+ts[ti]/tcolli[0])

                if kmax==ND-1 or tcolli[-1]>ts[it]: 
                    Mb=Ms*1.0
                    tcollib=tcolli[kmax-1]                
                    Ms= Ms0*1.0 #/(1.0+ts[ti]/tcolli[0])
                    break
        
            if kmaxd==kmaxu+1 and abs(ts[it]-tcolli[kmax-1])/ts[it]>0.1:
                # make weighted average using both size distibution
                print "average"
                w1=min(ts[it], tcolli[kmax-1])/max(ts[it], tcolli[kmax-1])
                w2=min(ts[it], tcollib)/max(ts[it], tcollib)
                Ms=(Ms*w1 + Mb*w2)/(w1+w2)


            # collisionally evolve the whole distribution
            for k in xrange(ND):
                Rs[k]=f_Rcoll(Ds, k,Ms, Vol, vrel, rho, Qs, bs, Qg, bg)
            tcolli=1.0/Rs /(1.0e6*year)
            if tcolli[0]<ts[it] and kmax==kmin:
                Ms = Ms/(ts[it]/tcolli[0])

            Msfinal[it,:]=Ms


        else: 
            print "the epoch must be positive"
            return -1.0

    return Ds, Msfinal





def SizeDist_dmax(Mstar=1.0, 
                  e=0.05,
                  I=0.05/2.0,    
                  r0=1.0,          
                  dr=0.05*2.0*1.0, 
                  Mbeltem=1.0,          
                  rho=2700.0,
                  dmin=0.8e-6,
                  dmax=1.0e5,
                  ND=500,
                  alphap=-3.7,
                  ts=[100.0],
                  Qs=608.0,
                  bs=0.38,
                  Qg=0.011,
                  bg=1.36): 
    # ##################################################
    # returns size distribution: Ds, M(Ds)
    # Mstar=1.0, stellar mass in solar masses 
    # e=0.05, mean eccentricities
    # I=0.05/2.0,  mean inclinations in radians
    # r0=1.0,      mean radius in au
    # dr=0.05*2.0*1.0,  width of annulus in au
    # Mbeltem=1.0,     total mass in solids in earth Masses
    # rho=2700.0, bulk density of the solids in kg/m3
    # dmin=1.0e-6, minimum size of solids in m
    # dmax=1.0e5, maximum size of solids in m
    # ND=500,     Number of size log bins 
    # alphap=-3.7, primordial size distribution
    # ts=100.0 age of the system in Myr (it must be an array)

    # The only difference with the original function is that it
    # recalculates dmax based on Xc=1, and scales the mass of the belt
    # to be consistent.
    # ##############################################


    Mstar=Mstar*Msun # kg
    rmid=r0*au # m
    Mbelt=Mbeltem*Mearth # kg
    print rmid/au, dr, I
    Vol=2.0*np.pi*rmid*dr*au*I*rmid # [m3]
    print "Vol [au3] = ", Vol/au**3.0
    vk=(G*Mstar/rmid)**0.5 # m/s
    vrel=vk*(1.25*e**2.0+I**2.0)**(0.5) # m/s
    print "vrel [km/s] = ",vrel/1.0e3
    # ######## BINS

    print "dmax = ", f_dmax( vimp=vrel, Qg=Qg, bg=bg )
    dmaxn=f_dmax( vimp=vrel, Qg=Qg, bg=bg )

    if dmaxn<dmax:
        Mbeltn=Mbelt*(dmaxn/dmax)**(alphap+4.0)
    else:
        dmaxn=dmax
        Mbeltn=Mbelt
        
    Ds=np.logspace(np.log10(dmaxn), np.log10(dmin), ND) # from big to small [m]
    delta=Ds[0]/Ds[1] # -1.0

    # ########## redistribution function
    alphar=-3.5
    alphare=20
    etarmax=2.0**(-1.0/3.0) # the largest objects has the same mass as the original
    keta=-int(np.log(etarmax)/np.log(delta)) # number of bins
    Fredist=np.zeros(ND)

    for i in xrange(0,ND):
        if (delta)**(-i)<=etarmax:
            Fredist[i]=(delta**(-i))**(4.0+alphar)
        else:
            Fredist[i]=1.0*(delta**(-i))**(4.0+alphare)
    Fredist=Fredist/np.sum(Fredist)

    
    Nt=len(ts)

    Msfinal=np.zeros((Nt, ND))

    # ######################################
    # ########### initial conditions
    # ######################################

    Ms0=np.zeros(ND)
    for i in xrange(ND):
        Ms0[i]=(Ds[i]/dmax)**(alphap+4.0)
    Ms0=Ms0*Mbeltn/np.sum(Ms0)

    
    # compute collisional rates to guess maximum size in steady state
    Rs0=np.zeros(ND)
    for i in xrange(ND):
        Rs0[i]=f_Rcoll(Ds, i,Ms0, Vol, vrel, rho, Qs, bs, Qg, bg)

    # plt.plot(Ds,Rs0)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()

    # #############################################
    # Calculate size distribution at different epochs
    # #############################################
    for it in xrange(Nt):

        if ts[it]==0.0:
            Msfinal[it,:]=Ms0
        
        elif ts[it]>0.0:
            

            # Compute initial collisional timescales for each size
            tcolls=1.0/Rs0 /(1.0e6*year) # Myr # this can give inf as Rs0 can be 0 if e<emax to break up planetesimals
            print tcolls[:30]
            # initial guess for Dc
            kmin=1
            kmax=kmin
            for k in xrange(kmin,ND):
                if tcolls[k]<ts[it]:
                    kmax=k
                    break
            kmaxu=kmax*1 # upper limit
            kmaxd=0 # lower limit

            Nit=20
            ncheck=18

            Mp=np.zeros(ND)
            Ms=Ms0*1.0
            Mb=Ms0*2.0 # Ms used to save the one from a previous iteration
            tcollib=0.0
            Rs=np.zeros(ND)
            Rcoll1=np.zeros(ND)
            Cs=np.zeros(ND)

            # #############################################
            # ######## ITERATE TO REACH STEADY STATE SIZE DISTRIBUTION
            # #############################################
       
            ni=0
            while ni<Nit:
                print "it , ni ", it, ni

                Mp=Ms # it should be Mp=Ms*1.0, but without the *1.0 it converges faster
                # Method from Wyatt+2011
                for k in xrange(ND):

                    Rcoll1[k]=f_Rcoll(Ds,k,Ms, Vol, vrel, rho, Qs, bs, Qg, bg)
                for k in xrange(ND):
        
                    C1k=np.sum(Rcoll1[:k]*Ms[:k]*Fredist[k:0:-1])
            
                    if k>=kmax and Rcoll1[k]>0.0:
                        Mp[k]=(C1k/Rcoll1[k]+Ms[k])/2.0
            
                    Rcoll2=f_Rcoll(Ds,k,Mp, Vol, vrel, rho, Qs, bs, Qg, bg)

                    if k>=kmax and Rcoll2>0.0:
                        Ms[k]=(C1k/Rcoll2+Ms[k])/2.0

                    Rs[k]=Rcoll2
                    Cs[k]=C1k




                # compare age and coll timescale
                tcolli=1.0/Rs /(1.0e6*year)
                print ts[it], r0,  kmax, kmin,  tcolli[kmax+1], tcolli[kmax], tcolli[kmax-1],  tcolli[0], kmaxu, kmaxd
                ni+=1

                # correct maximum size if age<tcoll

                if ts[it]*1.1< tcolli[kmax] and kmax<ND-1 and ni>ncheck and kmaxd==0:
                    for ik in xrange(kmax+1,ND):
                        if tcolli[ik]<ts[it]:
                            kmax=int(round( (ik+ kmax)/2.0 ))
                            break
                    ni=0
                    tcollib=tcolli[kmax-1]
                    Mb=Ms*1.0
                    Ms= Ms0*1.0 #/(1.0+ts[it]/tcolli[0])
        

                if ts[it]*1.1< tcolli[kmax] and kmax<ND-1 and ni>ncheck and kmaxd!=0 and (kmaxd-kmax>0 or kmaxd==0):
                    kmaxu=kmax*1
                    kmax=int(round((kmaxd+ kmax)/2.0))
                    ni=0
                    tcollib=tcolli[kmax-1]
                    Mb=Ms*1.0
                    Ms= Ms0*1.0 

                elif ts[it]> 1.1*tcolli[kmax-1] and kmax>kmaxu and ni>ncheck and (kmaxd-kmaxu>1 or kmaxd==0):
                    kmaxd=kmax*1
                    kmax=int(round( (kmaxu + kmax)/2))
                    ni=0
                    Mb=Ms*1.0
                    tcollib=tcolli[kmax-1]
                    Ms= Ms0*1.0 #/(1.0+ts[ti]/tcolli[0])
                
                elif tcolli[kmax-1]==np.inf and ni>ncheck: # tcolli[kmax-1]==np.inf 
                    kmax+=1
                    kmaxu=kmax*1
                    ni=0
                    tcollib=tcolli[kmax-1]
                    Mb=Ms*1.0
                    Ms= Ms0*1.0 #/(1.0+ts[ti]/tcolli[0])

                if kmax==ND-1 or tcolli[-1]>ts[it]: 
                    Mb=Ms*1.0
                    tcollib=tcolli[kmax-1]                
                    Ms= Ms0*1.0 #/(1.0+ts[ti]/tcolli[0])
                    break
        
            if kmaxd==kmaxu+1 and abs(ts[it]-tcolli[kmax-1])/ts[it]>0.1:
                # make weighted average using both size distibution
                print "average"
                w1=min(ts[it], tcolli[kmax-1])/max(ts[it], tcolli[kmax-1])
                w2=min(ts[it], tcollib)/max(ts[it], tcollib)
                Ms=(Ms*w1 + Mb*w2)/(w1+w2)


            # collisionally evolve the whole distribution
            for k in xrange(ND):
                Rs[k]=f_Rcoll(Ds, k,Ms, Vol, vrel, rho, Qs, bs, Qg, bg)
            tcolli=1.0/Rs /(1.0e6*year)
            if tcolli[0]<ts[it] and kmax==kmin:
                Ms = Ms/(ts[it]/tcolli[0])

            Msfinal[it,:]=Ms


        else: 
            print "the epoch must be positive"
            return -1.0

    ### need to extrapolate Ds to original dmax and Msfinal 

    NDe=int(round(np.log(dmax/dmaxn)/np.log(delta)))
    print NDe
    NDf=ND+NDe
    Msfinalf=np.zeros((Nt,NDf))
    # Rsfinalf=np.zeros((NDf))

    Msfinalf[:,NDe:]=Msfinal
    # Rsfinalf[NDe:]=Rs

    Dsf=np.logspace(np.log10(dmax), np.log10(dmin), NDf)

    for i in xrange(NDe):
        Msfinalf[:,i]=(Dsf[i]/dmax)**(alphap+4.0)


    Msfinalf[:,:NDe]=Msfinalf[:,:NDe]*(Mbelt-Mbeltn)/np.sum(Msfinalf[0,:NDe])

    return Dsf, Msfinalf#, Rs





def SizeDist_dmax_gas(Mstar=1.0, 
                      e=0.05,
                      I=0.05/2.0,    
                      r0=1.0,          
                      dr=0.05*2.0*1.0, 
                      Mbeltem=1.0,          
                      rho=2700.0,
                      dmin=0.8e-6,
                      dmax=1.0e5,
                      ND=500,
                      alphap=-3.7,
                      ts=[100.0],
                      Qs=500.0,
                      bs=0.37,
                      Qg=0.03,
                      bg=1.36): 
    # ##################################################
    # returns size distribution: Ds, M(Ds)
    # Mstar=1.0, stellar mass in solar masses 
    # e=0.05, mean eccentricities
    # I=0.05/2.0,  mean inclinations in radians
    # r0=1.0,      mean radius in au
    # dr=0.05*2.0*1.0,  width of annulus in au
    # Mbeltem=1.0,     total mass in solids in earth Masses
    # rho=2700.0, bulk density of the solids in kg/m3
    # dmin=1.0e-6, minimum size of solids in m
    # dmax=1.0e5, maximum size of solids in m
    # ND=500,     Number of size log bins 
    # alphap=-3.7, primordial size distribution
    # ts=100.0 age of the system in Myr (it must be an array)

    # The only difference with the original function is that it
    # recalculates dmax based on Xc=1, and scales the mass of the belt
    # to be consistent.
    # ##############################################


    Mstar=Mstar*Msun # kg
    rmid=r0*au # m
    Mbelt=Mbeltem*Mearth # kg
    print rmid/au, dr, I
    Vol=2.0*np.pi*rmid*dr*au*I*rmid # [m3]
    print "Vol [au3] = ", Vol/au**3.0
    vk=(G*Mstar/rmid)**0.5 # m/s
    vrel=vk*(1.25*e**2.0+I**2.0)**(0.5) # m/s
    print "vrel [km/s] = ",vrel/1.0e3
    # ######## BINS

    print "dmax = ", f_dmax( vimp=vrel, Qg=Qg, bg=bg )
    dmaxn=f_dmax( vimp=vrel, Qg=Qg, bg=bg )

    if dmaxn<dmax:
        Mbeltn=Mbelt*(dmaxn/dmax)**(alphap+4.0)
    else:
        dmaxn=dmax
        Mbeltn=Mbelt
        
    Ds=np.logspace(np.log10(dmaxn), np.log10(dmin), ND) # from big to small [m]
    delta=Ds[0]/Ds[1] # -1.0

    # ########## redistribution function
    alphar=-3.5
    alphare=20
    etarmax=2.0**(-1.0/3.0) # the largest objects has the same mass as the original
    keta=-int(np.log(etarmax)/np.log(delta)) # number of bins
    Fredist=np.zeros(ND)

    for i in xrange(0,ND):
        if (delta)**(-i)<=etarmax:
            Fredist[i]=(delta**(-i))**(4.0+alphar)
        else:
            Fredist[i]=1.0*(delta**(-i))**(4.0+alphare)
    Fredist=Fredist*0.7/np.sum(Fredist)

    
    Nt=len(ts)

    Msfinal=np.zeros((Nt, ND))

    # ######################################
    # ########### initial conditions
    # ######################################

    Ms0=np.zeros(ND)
    for i in xrange(ND):
        Ms0[i]=(Ds[i]/dmax)**(alphap+4.0)
    Ms0=Ms0*Mbeltn/np.sum(Ms0)

    
    # compute collisional rates to guess maximum size in steady state
    Rs0=np.zeros(ND)
    for i in xrange(ND):
        Rs0[i]=f_Rcoll(Ds, i,Ms0, Vol, vrel, rho, Qs, bs, Qg, bg)

    # plt.plot(Ds,Rs0)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()

    # #############################################
    # Calculate size distribution at different epochs
    # #############################################
    for it in xrange(Nt):

        if ts[it]==0.0:
            Msfinal[it,:]=Ms0
        
        elif ts[it]>0.0:
            

            # Compute initial collisional timescales for each size
            tcolls=1.0/Rs0 /(1.0e6*year) # Myr # this can give inf as Rs0 can be 0 if e<emax to break up planetesimals
            print tcolls[:30]
            # initial guess for Dc
            kmin=1
            kmax=kmin
            for k in xrange(kmin,ND):
                if tcolls[k]<ts[it]:
                    kmax=k
                    break
            kmaxu=kmax*1 # upper limit
            kmaxd=0 # lower limit

            Nit=20
            ncheck=18

            Mp=np.zeros(ND)
            Ms=Ms0*1.0
            Mb=Ms0*2.0 # Ms used to save the one from a previous iteration
            tcollib=0.0
            Rs=np.zeros(ND)
            Rcoll1=np.zeros(ND)
            Cs=np.zeros(ND)

            # #############################################
            # ######## ITERATE TO REACH STEADY STATE SIZE DISTRIBUTION
            # #############################################
       
            ni=0
            while ni<Nit:
                print "it , ni ", it, ni

                Mp=Ms # it should be Mp=Ms*1.0, but without the *1.0 it converges faster
                # Method from Wyatt+2011
                for k in xrange(ND):

                    Rcoll1[k]=f_Rcoll(Ds,k,Ms, Vol, vrel, rho, Qs, bs, Qg, bg)
                for k in xrange(ND):
        
                    C1k=np.sum(Rcoll1[:k]*Ms[:k]*Fredist[k:0:-1])
            
                    if k>=kmax and Rcoll1[k]>0.0:
                        Mp[k]=(C1k/Rcoll1[k]+Ms[k])/2.0
            
                    Rcoll2=f_Rcoll(Ds,k,Mp, Vol, vrel, rho, Qs, bs, Qg, bg)

                    if k>=kmax and Rcoll2>0.0:
                        Ms[k]=(C1k/Rcoll2+Ms[k])/2.0

                    Rs[k]=Rcoll2
                    Cs[k]=C1k




                # compare age and coll timescale
                tcolli=1.0/Rs /(1.0e6*year)
                print ts[it], r0,  kmax, kmin,  tcolli[kmax+1], tcolli[kmax], tcolli[kmax-1],  tcolli[0], kmaxu, kmaxd
                ni+=1

                # correct maximum size if age<tcoll

                if ts[it]*1.1< tcolli[kmax] and kmax<ND-1 and ni>ncheck and kmaxd==0:
                    for ik in xrange(kmax+1,ND):
                        if tcolli[ik]<ts[it]:
                            kmax=int(round( (ik+ kmax)/2.0 ))
                            break
                    ni=0
                    tcollib=tcolli[kmax-1]
                    Mb=Ms*1.0
                    Ms= Ms0*1.0 #/(1.0+ts[it]/tcolli[0])
        

                if ts[it]*1.1< tcolli[kmax] and kmax<ND-1 and ni>ncheck and kmaxd!=0 and (kmaxd-kmax>0 or kmaxd==0):
                    kmaxu=kmax*1
                    kmax=int(round((kmaxd+ kmax)/2.0))
                    ni=0
                    tcollib=tcolli[kmax-1]
                    Mb=Ms*1.0
                    Ms= Ms0*1.0 

                elif ts[it]> 1.1*tcolli[kmax-1] and kmax>kmaxu and ni>ncheck and (kmaxd-kmaxu>1 or kmaxd==0):
                    kmaxd=kmax*1
                    kmax=int(round( (kmaxu + kmax)/2))
                    ni=0
                    Mb=Ms*1.0
                    tcollib=tcolli[kmax-1]
                    Ms= Ms0*1.0 #/(1.0+ts[ti]/tcolli[0])
                
                elif tcolli[kmax-1]==np.inf and ni>ncheck: # tcolli[kmax-1]==np.inf 
                    kmax+=1
                    kmaxu=kmax*1
                    ni=0
                    tcollib=tcolli[kmax-1]
                    Mb=Ms*1.0
                    Ms= Ms0*1.0 #/(1.0+ts[ti]/tcolli[0])

                if kmax==ND-1 or tcolli[-1]>ts[it]: 
                    Mb=Ms*1.0
                    tcollib=tcolli[kmax-1]                
                    Ms= Ms0*1.0 #/(1.0+ts[ti]/tcolli[0])
                    break
        
            if kmaxd==kmaxu+1 and abs(ts[it]-tcolli[kmax-1])/ts[it]>0.1:
                # make weighted average using both size distibution
                print "average"
                w1=min(ts[it], tcolli[kmax-1])/max(ts[it], tcolli[kmax-1])
                w2=min(ts[it], tcollib)/max(ts[it], tcollib)
                Ms=(Ms*w1 + Mb*w2)/(w1+w2)


            # collisionally evolve the whole distribution
            for k in xrange(ND):
                Rs[k]=f_Rcoll(Ds, k,Ms, Vol, vrel, rho, Qs, bs, Qg, bg)
            tcolli=1.0/Rs /(1.0e6*year)
            if tcolli[0]<ts[it] and kmax==kmin:
                Ms = Ms/(ts[it]/tcolli[0])

            Msfinal[it,:]=Ms


        else: 
            print "the epoch must be positive"
            return -1.0

    ### need to extrapolate Ds to original dmax and Msfinal 

    NDe=int(round(np.log(dmax/dmaxn)/np.log(delta)))
    print NDe
    NDf=ND+NDe
    Msfinalf=np.zeros((Nt,NDf))
    Rsfinalf=np.zeros((NDf))

    Msfinalf[:,NDe:]=Msfinal
    Rsfinalf[NDe:]=Rs

    Dsf=np.logspace(np.log10(dmax), np.log10(dmin), NDf)

    for i in xrange(NDe):
        Msfinalf[:,i]=(Dsf[i]/dmax)**(alphap+4.0)


    Msfinalf[:,:NDe]=Msfinalf[:,:NDe]*(Mbelt-Mbeltn)/np.sum(Msfinalf[0,:NDe])

    return Dsf, Msfinalf #, Rs

#### MAIN


# tsystem=[ 10.0] # Myr
# Ds, Msf2= SizeDist(ts=tsystem, r0=1.0, ND=500, Mbeltem=1.0)


# # Qds=np.zeros(len(Ds))
# # for i in xrange(len(Ds)):
# #     Qds[i]=f_Qd(Ds[i])
# # plt.plot(Ds,Qds)
    
# plt.xscale('log')
# plt.yscale('log')
# plt.show()

# for i in xrange(len(tsystem)):
#     plt.plot(Ds, Msf[i,:], color='black')

# plt.xscale('log')
# plt.yscale('log')
# plt.show()
