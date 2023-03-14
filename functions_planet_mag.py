import glob, os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker
from matplotlib import rc
import general_functions as gf
from astropy.io import ascii
import scipy.stats
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter


Msun=1.989e+30 # kg
au=1.496e+11   # m
year=3.154e+7 # s
G=6.67408e-11 # mks
Mjup=1.898e27 # kg
Mearth=5.972e24 #kg




def get_mag_from_mass_age(input_age, input_mass, input_distance=10., obs_filter = 'f444w'):

    #### get curve mass vs mag for lower and upper ages in grid
    
    ## BEX (check if both age curves cover the right mass )
    # get closest two curves

    try:
        bex=get_mass_vs_mag_bex(input_age, input_distance, obs_filter = obs_filter, diff='sandwich' )
    except:
        # filter not in BEX models, use ATMO only (twice as a simple not definitive solution)
        bex=get_mass_vs_mag_atmo(input_age, input_distance, obs_filter = obs_filter, diff='sandwich' )
        
    # try:
    atmo=get_mass_vs_mag_atmo(input_age, input_distance, obs_filter = obs_filter, diff='sandwich' )
    # except:
    #     # filter not in ATMO, use BEX only
    #     atmo=get_mass_vs_mag_bex(input_age, input_distance, obs_filter = obs_filter, diff='sandwich' )

    fill_value=25

    if input_mass>atmo[1,-1]: return atmo[0,-1] # if mass is above mass grid
    
    if bex.size==0: bex_max_m=0.
    else: bex_max_m=bex[1,-1]
    
    if atmo.size==0: atmo_min_m=1.0e3
    else: atmo_min_m=atmo[1,0]
        
    if bex_max_m>atmo_min_m and len(bex[0,:])>1 and len(atmo[0,:])>1: # join two functions
        #print(1)
        fbex=interp1d(np.log10(bex[1,:]), bex[0,:], kind='linear', bounds_error=False, fill_value=fill_value)
        magbex=fbex(np.log10(input_mass))

        fatmo=interp1d(np.log10(atmo[1,:]), atmo[0,:], kind='linear', bounds_error=False, fill_value=fill_value)
        magatmo=fatmo(np.log10(input_mass))

        if input_mass<bex_max_m:
            if input_mass>atmo_min_m: # overlay region
                watmo=np.log(input_mass/atmo_min_m) / np.log(bex_max_m/atmo_min_m) # from 0 to 1 
                wbex=1-watmo
            else:
                watmo=0.
                wbex=1.
        else:
            watmo=1.
            wbex=0.

        magfinal=(magatmo*watmo+magbex*wbex)/(watmo+wbex)

    elif 0.<bex_max_m<input_mass<atmo_min_m: # interpolate between bex and atmo 
        #print(2)

        watmo=np.log(input_mass/bex_max_m) / np.log(atmo_min_m/bex_max_m) # from 0 to 1 
        wbex=1-watmo
        magfinal=(atmo[0,0]*watmo+bex[0,-1]*wbex)/(watmo+wbex)

    elif bex.size>2 and input_mass<bex_max_m: # use only Bex
        #print(3)

        fbex=interp1d(np.log10(bex[1,:]), bex[0,:], kind='linear', bounds_error=False, fill_value=fill_value)
        magfinal=fbex(np.log10(input_mass))

    elif atmo.size>2 and input_mass>atmo_min_m: # use only atmo
        #print(4)
        fatmo=interp1d(np.log10(atmo[1,:]), atmo[0,:], kind='linear', bounds_error=False, fill_value=fill_value)
        magfinal=fatmo(np.log10(input_mass))
 
    else: # none cover the required mass and age
        #print(5)
        magfinal=fill_value
    
    return magfinal
    
    

def get_mass_vs_mag_atmo(input_age, input_distance=10., obs_filter = 'f444w', diff='sandwich' ):


    atmo_grid_dir = '/Users/Sebamarino/Astronomy/JWST/planet_models/ATMO_CEQ/JWST_vega/'
    atmo_grid_files = sorted(glob.glob(atmo_grid_dir+'*.txt'), key=lambda x:float(x.split("/")[-1].split('_')[0].replace('m', '')))

    if 'c' in obs_filter:
        instrument = 'miri'
    else:
        instrument = 'nircam'

        
    atmo_masses_2d, atmo_mags_2d, atmo_ages_2d = [], [], []
    
    for atmo_grid_file in atmo_grid_files: # file by file
        atmo_data = np.genfromtxt(atmo_grid_file).transpose() # get table
        with open(atmo_grid_file, 'r') as f:
            header = f.readlines()[0].replace('#','').lower().split() # get header

        table_ages = atmo_data[header.index('age')]*1e3  #Convert to Myr
        table_mags = atmo_data[header.index(instrument+'-'+obs_filter.lower())] + 5*np.log10(input_distance/10)
        table_mass = atmo_data[header.index('mass')][0]/0.0009543 #Convert to Jupiter masses
        table_masses = atmo_data[header.index('mass')]/0.0009543 #Convert to Jupiter masses

    
    
        if max(table_ages) < input_age or min(table_ages) > input_age:
            #These files will be useless, exclude them  
            continue

        
        atmo_ages_2d.append(table_ages)
        atmo_mags_2d.append(table_mags)
        atmo_masses_2d.append(table_masses)

    atmo_ages=np.concatenate(atmo_ages_2d)
    atmo_mags=np.concatenate(atmo_mags_2d)
    atmo_masses=np.concatenate(atmo_masses_2d)
    
    atmo_ages_nodups=np.unique(atmo_ages)
    atmo_ages_diff = [abs(x-input_age) for x in atmo_ages_nodups]
    
    atmo_masses_nodups =  np.unique(atmo_masses)
    
    if input_age<np.min(atmo_ages_nodups) or input_age>np.max(atmo_ages_nodups):
        return np.array([])
    
    if diff=='sandwich':
        if min(atmo_ages_diff) == 0:
            #Exact age in grid, less interpolation needed.
            closest_age = atmo_ages_nodups[atmo_ages_diff.index(np.min(atmo_ages_diff))]
            closest_ages = [closest_age, closest_age] 
        else:
            #Need two points for interpolation
            ageip=atmo_ages_nodups[0]
            for agei in atmo_ages_nodups[1:]:  # it is sorted already
                if agei>input_age:
                    closest_ages=[ageip, agei]
                    break
                ageip=agei*1
    else:
        if min(atmo_ages_diff) == 0:
            #Exact age in grid, less interpolation needed.
            closest_age = atmo_ages_nodups[atmo_ages_diff.index(np.min(atmo_ages_diff))]
            closest_ages = [closest_age, closest_age] 
        else:
            #Need two points of interpolation
            min_diffs_index = [atmo_ages_diff.index(x) for x in np.partition(atmo_ages_diff, 1)[0:2]] # this selects ages with the two smallest differences
            closest_ages = [atmo_ages_nodups[x] for x in min_diffs_index]

        
        
    # now extract mag and masses for those closest ages
    #print(closest_ages)
    age_1=atmo_ages==closest_ages[0]
    age_2=atmo_ages==closest_ages[1]
    
    common_masses, mask1, mask2=np.intersect1d(atmo_masses[age_1], atmo_masses[age_2], return_indices=True, assume_unique=True)
        
     # make average between the two using weights as ratio of ages
    w1=max(input_age/closest_ages[0], closest_ages[0]/input_age)
    w2=max(input_age/closest_ages[1], closest_ages[1]/input_age)
    
    mag_final=(atmo_mags[age_1][mask1]*w1+atmo_mags[age_2][mask2]*w2)/(w1+w2)
    
    return np.array([mag_final, common_masses])

#     return np.array([atmo_mags[age_1][mask1], atmo_masses[age_1][mask1], [closest_ages[0]]*len(common_masses)]),  np.array([atmo_mags[age_2][mask2], atmo_masses[age_2][mask2], [closest_ages[1]]*len(common_masses)])

def get_mass_vs_mag_bex(input_age, input_distance=10., obs_filter = 'f444w', diff='sandwich' ):
    # age in Myr
    # distance in pc

    obs_filter=obs_filter.lower()
    
    bex_grid_file='/Users/Sebamarino/Astronomy/JWST/planet_models/BEX_evol_mags_-2_MH_0.00_UPDATEDV2.dat'

    with open(bex_grid_file, 'r') as f:
        bex_data = json.load(f)
    
    bex_ages = 10**np.array(bex_data['log(age/yr)']) / 1e6     # In Myr
    bex_masses = np.array(bex_data['mass/mearth']) / 317.8       #In MJup
    if obs_filter in ['f356w', 'f444w', 'f410m', 'f430m','f460m', 'f480m']:
        bex_filter = obs_filter+'_mask335r'
    else:
        bex_filter = obs_filter

    try:
        bex_abs_mags = np.array(bex_data[bex_filter])
    except:
        raise Exception('Could not locate {} filter in BEX grid, it must be computed...'.format(obs_filer))

    bex_mags = bex_abs_mags + 5*np.log10(input_distance/10)      #BEX GRID IS ONLY IN VEGA
    
    
    bex_ages_nodups = list(dict.fromkeys(bex_ages))
    bex_ages_diff = [abs(x-input_age) for x in bex_ages_nodups]
    #bex_ages_ratio_up = [x/input_age for x in bex_ages_nodups] # ratios (>1)
    #bex_ages_ratio_down = [input_age/x for x in bex_ages_nodups] # ratios (>1)

    bex_masses_nodups =  list(dict.fromkeys(bex_masses))
    
    if input_age<np.min(bex_ages_nodups) or input_age>np.max(bex_ages_nodups):
        return np.array([])
    
    if diff=='sandwich':
        if min(bex_ages_diff) == 0:
            #Exact age in grid, less interpolation needed.
            closest_age = bex_ages_nodups[bex_ages_diff.index(np.min(bex_ages_diff))]
            closest_ages = [closest_age, closest_age] 
        else:
            #Need two points for interpolation
            ageip=bex_ages_nodups[0]
            for agei in bex_ages_nodups[1:]:  # it is sorted already
                if agei>input_age:
                    closest_ages=[ageip, agei]
                    break
                ageip=agei*1
    else:
        if min(bex_ages_diff) == 0:
            #Exact age in grid, less interpolation needed.
            closest_age = bex_ages_nodups[bex_ages_diff.index(np.min(bex_ages_diff))]
            closest_ages = [closest_age, closest_age] 
        else:
            #Need two points of interpolation
            min_diffs_index = [bex_ages_diff.index(x) for x in np.partition(bex_ages_diff, 1)[0:2]] # this selects ages with the two smallest differences
            closest_ages = [bex_ages_nodups[x] for x in min_diffs_index]

        
        
    # now extract mag and masses for those closest ages only considering the mass range they share
    age_1=bex_ages==closest_ages[0]
    age_2=bex_ages==closest_ages[1]

    common_masses, mask1, mask2=np.intersect1d(bex_masses[age_1], bex_masses[age_2], return_indices=True, assume_unique=True)
    
    # make average between the two using weights as ratio of ages
    w1=max(input_age/closest_ages[0], closest_ages[0]/input_age)
    w2=max(input_age/closest_ages[1], closest_ages[1]/input_age)
    
    mag_final=(bex_mags[age_1][mask1]*w1+bex_mags[age_2][mask2]*w2)/(w1+w2)
    
    return np.array([mag_final, common_masses])



def detectability_map(separation, contrast,  agemin, agemax, inc=0., dpc=10., NM=100, Na=100,amin=0., Nphi=30, amax=200., Mpmin=0.1, Mpmax=100., Nage=10, absolute_mag=True, simple=False, obs_filter='f1550c' , contrast_type='magnitude', a_log=False):
    # inc in radians    
    
    # add point at separation 0 and at amax (repeat first and last values)
    sep=np.array(separation.copy())
    con=np.array(contrast.copy())
    
    sep=np.insert(sep, 0, 0.)
    sep=np.insert(sep, len(sep), amax/dpc)

    if contrast_type=='magnitude':
        con=np.insert(con, 0, 0.)
    elif contrast_type=='mass':
        con=np.insert(con, 0, 1.0e3)
    else:
        con=np.insert(con, 0, contrast[0])
    con=np.insert(con, len(con), contrast[-1])

    
    
    # interpolate
    fcontrast= interp1d(sep, con, kind='linear')

    if absolute_mag:
        dpc_contrast=10.
    else:
        dpc_contrast=dpc
    ###########################################
    ########## construct detectability map
    ###########################################

    

    ## CONSTRUCT GRID

    Mpgrid=np.logspace(np.log10(Mpmin), np.log10(Mpmax), NM+1) # Jupiter masses
    Mpdiff=Mpgrid[1:]-Mpgrid[:-1]
    Mps=Mpgrid[:-1]+Mpdiff/2.

    amax=min(amax, sep[-1]*dpc)
    if a_log:
        apgrid=np.logspace(np.log10(amin), np.log10(amax), Na+1) # au
    else:
        apgrid=np.linspace(amin, amax, Na+1) # au
    apdiff=apgrid[1:]-apgrid[:-1]
    aps=apgrid[:-1]+apdiff/2.

    ages=np.linspace(agemin, agemax, Nage)

    Detectability=np.zeros((NM,Na))

    phis=np.linspace(0.0, np.pi/2., Nphi+1)[:-1] # only necessary to do one quarter

    ## ITERATE AND CALCULATE PROBABILITY FOR EACH BIN
    for j in range(NM):
        #print(j)

        if contrast_type=='magnitude':
            for k in range(Nage):
            
                magi=get_mag_from_mass_age(ages[k], Mps[j], input_distance=dpc_contrast,obs_filter=obs_filter )
                for i in range(Na):
                    xs=aps[i]*np.cos(phis)
                    ys=aps[i]*np.sin(phis)*np.cos(inc)
                    separations=np.sqrt(xs**2+ys**2)/dpc

                    if simple: # consider detected if above 5sigma, undetected if not
                        mask_detections=fcontrast(separations)>magi
                        Detectability[j, i]=Detectability[j, i]+len(separations[mask_detections])/(Nphi*Nage)
                    else: # consider the probability of detection i.e. given its flux check what fraction would lie above 5sigma with errors
                        Xsigma=5*10**(-0.4*(magi-fcontrast(separations))) # array of sigmas
                        # calculate probability for each of them to produce a 5sigma detection
                        Probs=1.-scipy.stats.norm(Xsigma, 1.).cdf(5.) # probability of being higher than 5 sigma by chance
                        Detectability[j, i]=Detectability[j, i]+np.sum(Probs)/(Nage*Nphi)
                        
        elif contrast_type=='mass': # ages are not used
            for i in range(Na):
                xs=aps[i]*np.cos(phis)
                ys=aps[i]*np.sin(phis)*np.cos(inc)
                separations=np.sqrt(xs**2+ys**2)/dpc
                
                mask_detections=fcontrast(separations)<Mps[j]
                Detectability[j, i]=Detectability[j, i]+len(separations[mask_detections])/(Nphi)

        else:
            print('invalid contrast type. It must be either magnitude or mass')
            sys.exit()
    return apgrid, aps, Mpgrid, Mps, Detectability

def upper_limit(separation, mass_limit, dpc, amin, amax, Mpmin, Mpmax, inc, a_log=True, Na=100, NM=100, Nphi=30, threshold=0.997):

    # extend to amax

    sep=np.array(separation.copy())
    mlim=np.array(mass_limit.copy())

    sep=np.insert(sep, 0, 0.)
    mlim=np.insert(mlim, 0, 1.0e3)
    
    if amax>separation[-1]*dpc:
        sep=np.insert(sep,len(sep), separation[-1]*1.01) # one more point to make abrupt transition
        sep=np.insert(sep,len(sep), amax/dpc)

        mlim=np.insert(mlim,len(mlim), 1.0e3)
        mlim=np.insert(mlim,len(mlim), 1.0e3)

    
    fcontrast= interp1d(sep, mlim, kind='linear')

    
    Mpgrid=np.logspace(np.log10(Mpmin), np.log10(Mpmax), NM+1) # Jupiter masses
    Mpdiff=Mpgrid[1:]-Mpgrid[:-1]
    Mps=Mpgrid[:-1]+Mpdiff/2.
    
    if a_log:
        apgrid=np.logspace(np.log10(amin), np.log10(amax), Na+1) # au
    else:
        apgrid=np.linspace(amin, amax, Na+1) # au
    apdiff=apgrid[1:]-apgrid[:-1]
    aps=apgrid[:-1]+apdiff/2.
    

    print(np.array(sep[-3:])*dpc, aps[-1], amax)

    
    Detectability=np.zeros((NM,Na))

    phis=np.linspace(0.0, np.pi/2., Nphi+1)[:-1] # only necessary to do one quarter

    for j in range(NM):
        for i in range(Na):

            xs=aps[i]*np.cos(phis)
            ys=aps[i]*np.sin(phis)*np.cos(inc)

            separations=np.sqrt(xs**2+ys**2)/dpc
            try:
                mask_detections=fcontrast(separations)<Mps[j]
            except:
                print(separations*dpc)
            Detectability[j, i]=len(separations[mask_detections])/(Nphi)


    ### get right contour level
    
    figtemp=plt.figure()
    axtemp=figtemp.add_subplot(111)

    # axtemp.pcolormesh(aps, Mps, Detectability)
    cs=axtemp.contour(aps, Mps, Detectability, levels=[threshold])
    
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()
    plt.close()
    
    p = cs.collections[0].get_paths()[0]
    v = p.vertices
    x = v[:,0]
    y = v[:,1]

    # limit=interp1d(x, y, kind='linear')
    
    return x, y









####################
###### old #########






# def get_mass_vs_mag_bex(input_age, input_distance=10., obs_filter = 'f1550c', diff='sandwich' ):
#     # age in Myr
#     # distance in pc

#     obs_filter=obs_filter.lower()
    
#     bex_grid_file='/Users/Sebamarino/Astronomy/PlanetModels/BEX/BEX_evol_mags_-2_MH_0.00_UPDATEDV2.dat'

#     with open(bex_grid_file, 'r') as f:
#         bex_data = json.load(f)
    
#     bex_ages = 10**np.array(bex_data['log(age/yr)']) / 1e6     # In Myr
#     bex_masses = np.array(bex_data['mass/mearth']) / 317.8       #In MJup
#     if obs_filter == 'f356w':
#         bex_filter = 'f356w_mask335r'
#     elif obs_filter == 'f444w':
#         bex_filter = 'f444w_mask335r'
#     else:
#         bex_filter = obs_filter

#     try:
#         bex_abs_mags = np.array(bex_data[bex_filter])
#     except:
#         print('Could not locate filter in BEX grid, it must be computed...')

#     bex_mags = bex_abs_mags + 5*np.log10(input_distance/10)      #BEX GRID IS ONLY IN VEGA
    
    
#     bex_ages_nodups = list(dict.fromkeys(bex_ages))
#     bex_ages_diff = [abs(x-input_age) for x in bex_ages_nodups]
#     #bex_ages_ratio_up = [x/input_age for x in bex_ages_nodups] # ratios (>1)
#     #bex_ages_ratio_down = [input_age/x for x in bex_ages_nodups] # ratios (>1)

#     bex_masses_nodups =  list(dict.fromkeys(bex_masses))
   
#     if diff=='sandwich':
#         if min(bex_ages_diff) == 0:
#             #Exact age in grid, less interpolation needed.
#             closest_age = bex_ages_nodups[bex_ages_diff.index(np.min(bex_ages_diff))]
#             closest_ages = [closest_age, closest_age] 
#         else:
#             #Need two points for interpolation
#             ageip=bex_ages_nodups[0]
#             for agei in bex_ages_nodups[1:]:  # it is sorted already
#                 if agei>input_age:
#                     closest_ages=[ageip, agei]
#                     break
#                 ageip=agei*1
#     else:
#         if min(bex_ages_diff) == 0:
#             #Exact age in grid, less interpolation needed.
#             closest_age = bex_ages_nodups[bex_ages_diff.index(np.min(bex_ages_diff))]
#             closest_ages = [closest_age, closest_age] 
#         else:
#             #Need two points of interpolation
#             min_diffs_index = [bex_ages_diff.index(x) for x in np.partition(bex_ages_diff, 1)[0:2]] # this selects ages with the two smallest differences
#             closest_ages = [bex_ages_nodups[x] for x in min_diffs_index]

        
        
#     # now extract mag and masses for those closest ages
#     #print(closest_ages)
#     age_1=bex_ages==closest_ages[0]
#     age_2=bex_ages==closest_ages[1]
#     #print(closest_ages[0], closest_ages[1])
#     return np.array([bex_mags[age_1], bex_masses[age_1], [closest_ages[0]]*len(bex_mags[age_1])]),  np.array([bex_mags[age_2], bex_masses[age_2], [closest_ages[1]]*len(bex_mags[age_2])])

# # def mass_mag(mag, A, B):
# #     return np.exp(A*mag+B)

# def get_mass_from_mag_age(input_age, input_mag, input_distance=10., obs_filter = 'f1550c'):
#     # this doesn't work for ages beyond ~1Gyr since there is only one planet mass for those ages
    
#     # get closest two curves
#     data1, data2=get_mass_vs_mag_bex(input_age, input_distance=input_distance, obs_filter=obs_filter,diff='sandwich')
#     # data1  # mag, mass, age1(fixed)
#     # data2  # mag, mass, age1(fixed)

#     age1=data1[2,0] 
#     age2=data2[2,0] 
    
#     ## fit and extrapolate using last three points
#     #popt1, pcov1 = curve_fit(mass_mag, data1[0,:3], data1[1,:3]) # use first/largest magnitudes
#     #popt2, pcov2 = curve_fit(mass_mag, data2[0,:3], data2[1,:3])
#     #mass1=mass_mag(input_mag, popt1[0], popt1[1])
#     #mass2=mass_mag(input_mag, popt2[0], popt2[1])
    
#     # extrapolate value assuming exponential of mass vs mag or straight line for logM vs mag.
    
#     ## find nearest indices
#     if input_mag>data1[0,0]: # beyond range, hence extrapolate
#         indices1=[1,0] # mag increasing
#     elif input_mag<data1[0,-1]: # beyond range, hence extrapolate
#         indices1=[-1,-2] # mag increasing

#     else: # interpolate
#         for i, magi in enumerate(data1[0,:]): # decreasing order
#             if magi<=input_mag:
#                 indices1=[i,i-1] # mag increasing
#                 break

                
#     if input_mag>data2[0,0]: # beyond range, hence extrapolate
#         indices2=[1,0] # mag increasing
#     elif input_mag<data2[0,-1]: # beyond range, hence extrapolate
#         indices2=[-2,-1] # mag increasing

#     else: # interpolate
#         for i, magi in enumerate(data2[0,:]): # decreasing order
#             if magi<=input_mag:
#                 indices2=[i,i-1] # mag increasing
#                 break
#     # make projection in logspace
#     slope1=(np.log(data1[1,indices1[1]])-np.log(data1[1,indices1[0]]) )/ (data1[0,indices1[1]]-data1[0,indices1[0]])
#     M1=data1[1,indices1[0]]*np.exp(  slope1*(input_mag-data1[0,indices1[0]])    )
    
#     slope2=(np.log(data2[1,indices2[1]])-np.log(data2[1,indices2[0]]) )/ (data2[0,indices2[1]]-data2[0,indices2[0]])
#     M2=data2[1,indices2[0]]*np.exp(  slope2*(input_mag-data2[0,indices2[0]])    )
    
#     # make average between the two using weights as ratio of ages
#     w1=max(input_age/age1, age1/input_age)
#     w2=max(input_age/age2, age2/input_age)
    
#     Mfinal=(M1*w1+M2*w2)/(w1+w2)
#     return Mfinal

# def get_mag_from_mass_age(input_age, input_mass, input_distance=10., obs_filter = 'f1550c'):
#     # this doesn't work for ages beyond ~1Gyr since there is only one planet mass for those ages
    
#     # get closest two curves
#     data1, data2=get_mass_vs_mag_bex(input_age, input_distance=input_distance, obs_filter=obs_filter,diff='sandwich')
#     # data1  # mag, mass, age1(fixed)
#     # data2  # mag, mass, age1(fixed)

#     # mag decreases with index. This deterimines how to select indices
#     age1=data1[2,0] 
#     age2=data2[2,0] 
    
#     # extrapolate value assuming exponential of mass vs mag or straight line for logM vs mag.
    
#     ## find nearest indices
#     if input_mass<data1[1,0]: # beyond range, hence extrapolate
#         indices1=[0,1] # mass increasing
#     elif input_mass>data1[1,-1]: # beyond range, hence extrapolate
#         indices1=[-1,-2] # mass increasing

#     else: # interpolate
#         for i, massi in enumerate(data1[1,:]): # decreasing order
#             if massi>=input_mass:
#                 indices1=[i,i-1] # mass increasing
#                 break

                
#     if input_mass<data2[1,0]: # beyond range, hence extrapolate
#         indices2=[0,1] # mass increasing
#     elif input_mass>data2[1,-1]: # beyond range, hence extrapolate
#         indices2=[-1,-2] # mass increasing

#     else: # interpolate
#         for i, massi in enumerate(data2[1,:]): # decreasing order
#             if massi>=input_mass:
#                 indices2=[i,i-1] # mass increasing
#                 break
#     # make projection in logspace
#     slope1= (data1[0,indices1[1]]-data1[0,indices1[0]])/ (np.log(data1[1,indices1[1]])-np.log(data1[1,indices1[0]]) )
#     mag1=data1[0,indices1[0]]+ slope1*(np.log(input_mass)-np.log(data1[1,indices1[0]]))
    
#     slope2=(data2[0,indices2[1]]-data2[0,indices2[0]])/ (np.log(data2[1,indices2[1]])-np.log(data2[1,indices2[0]]) )
#     mag2=data2[0,indices2[0]]+ slope2*(np.log(input_mass)-np.log(data2[1,indices2[0]])) 
    
#     # make average between the two using weights as ratio of ages
#     w1=max(input_age/age1, age1/input_age)
#     w2=max(input_age/age2, age2/input_age)
    
#     mag_final=(mag1*w1+mag2*w2)/(w1+w2)
#     return mag_final
