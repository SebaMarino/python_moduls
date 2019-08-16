import numpy as np

def extractvis(vis, tablename, ms): # by S. Marino

    cc=2.99792458e8 # m/s


    mse=vis # it will extract the visibilities from the 4 spws

    print "openning ms"
    ms.open(mse, nomodify=True)
    print "ms opened"

    visd0=ms.getdata(['data', 'data_desc_id','axis_info'],ifraxis=False)
    visd=np.array(visd0['data']) # complex visibilities
    spwids=np.array(visd0['data_desc_id']) # spws ids 

    # viscd0=ms.getdata(['corrected_data'])
    # viscd=np.array(viscd0['corrected_data'])

    nchan=np.shape(visd)[1] # number of channels
    nrows=np.shape(visd)[2] # numer of rows (approx nscans x nspws)

    print np.shape(visd)
    print np.shape(spwids)

    # weights (two weights for all channels in one spw that correspond to two polarization modes XX YY)
    wts0=ms.getdata(['weight'])
    wts=np.array(wts0['weight']) 

    
    # baselines
    uvs=ms.getdata(['u', 'v']) #meters
    
    # get flags
    flag_row0=ms.getdata(['flag_row'])
    flag_row=np.array(flag_row0['flag_row'])

    flags0=ms.getdata(['flag'])
    flags=np.array(flags0['flag'])

    print "flag dimension: "+str(flags.shape)
    print "n flags in flagrow: "+str(np.sum(flag_row))
    print "n flags in flags: "+str(np.sum(flags))

    # frequencies
    freqs=visd0['axis_info']['freq_axis']['chan_freq']/1.0e9 # freq in GHz (channel,spw) (e.g, freqs[:,0] are freqs of channels in the 1st spw) 
    
    nspw=len(freqs[0,:])
    table=np.zeros((nrows*nchan,6)) # u, v, Vreal, Vimag, weight lam
    print np.shape(uvs['u'])

    for i in xrange(nrows):

        # print i, nrows
        # if flag_row[i]==1: 
        #     continue
        # if flags[0,0,i]==1 or flags[1,0,i]==1:
        #     continue

        ispw=spwids[i]

        wts1i = wts[0,i]
        wts2i = wts[1,i]

        # 2 polarization (average the two polarization modes) if not flagged
        wtsc=0.0

        if flag_row[i]==1: 
            continue
        if wts1i*wts2i!=0.0:
            wtsc=4.0*wts1i*wts2i/(wts1i+wts2i)
            # sigs=np.sqrt(1.0/wts1) # uncertainties or errors
        elif  wts1i!=0.0:
            wtsc=wtsc1i
        elif  wts2i!=0.0:
            wtsc=wtsc2i
        else: 
            continue

        for j in xrange(nchan):
            u=np.array(uvs['u'][i])*freqs[j,ispw]*1.0e9/cc # us
            v=np.array(uvs['v'][i])*freqs[j,ispw]*1.0e9/cc # vs
            
            table[i*nchan+j,0]=u # lambda
            table[i*nchan+j,1]=v # lambda

            if flags[0,j,i]==0 and flags[1,j,i]==0: # no polarizations flags
                Vi=(visd[0,j,i]+visd[1,j,i])/2 # average two polarizations [Jy]
                table[i*nchan+j,4]=wtsc
            elif flags[0,j,i]==0 or flags[1,j,i]==0: # flag in only one polarization
                Vi=(visd[0,j,i]*(1.0-flags[0,j,i]) + visd[1,j,i]*(1.0-flags[1,j,i]) )/2. # average two polarizations [Jy]
                table[i*nchan+j,4]=wts1i*(1.0-flags[0,j,i]) + wts2i*(1.0-flags[1,j,i]) 
            else: # both are falgged
                Vi=0.0
                table[i*nchan+j,4]=0.0

            # check if visibility is !=0
            if Vi!=0.0:
                table[i*nchan+j,2]=Vi.real
                table[i*nchan+j,3]=Vi.imag
            else: # sometimes ms have v=0 and extreamely high weights that should not be included
                table[i*nchan+j,4]=0.0

            # lambda
            table[i*nchan+j,5]=cc/(freqs[j, ispw]*1.0e9) # [m]

    ms.close()

    print 'Chi red = ', np.sum( (table[:,2]**2.+table[:,3]**2)*table[:,4] )/table[:,0].size

    np.savetxt(tablename+'.dat', table)
    np.save(tablename, table)

