import numpy as np


h = 6.62607e-34 # mks
c = 2.99792e+8  # mks
ckms = 2.99792e+5  # km/s
k = 1.380648e-23 # mks
pc=3.086e16 # m

# http://www.astro.uni-koeln.de/site/vorhersagen/catalog/partition_function.html

Z_CO=np.flipud(np.array([[2000,726.7430],
                      [1000.,	362.6910],
                      [500.0,	181.3025],
                      [300.0,	108.8651],
                      [225.0,	81.7184],
                      [150.0,	54.5814],
                      [75.00,	27.4545],
                      [37.50,	13.8965],
                      [18.75,	7.1223],
                      [9.375,	3.7435],
                      [5.000,	2.1824],
                      [2.725,	1.4053]])) # partition function CO v=0 at diff temperatures

Z_HCN=np.flipud(np.array( [ [1000.0, 3309.78665073],
                            [500.0, 930.67918316],
                            [300.0, 453.52371345],
                            [225, 325.2370401] ,
                            [150.0, 213.10812106],
                            [75.0, 106.80706966],
                            [37.5, 53.91380705],
                            [18.75, 27.472615],
                            [9.375, 14.2724983],
                            [5.0, 8.14516713],
                            [2.725, 5.03037081]]))  # partition function CO v=0 at diff temperatures


def Z_mol(file_trans, T):
    # method only valid until ~1000 K
    trans=np.loadtxt(file_trans)
    Ejs=trans[:,5]
    gjs=trans[:,0]*2+1 # valid for CO!!! check for other molecules

    return np.sum(gjs*np.exp(-Ejs/T))

def x(j, Ej,  T, molecule='CO'):
    # Ej, Energy in K
    # T, Temperature in K
    # j, upper level
    if molecule=='CO':
        Z=Z_CO
        Zt=np.interp(T, Z[:,0], Z[:,1])

    elif molecule=='13CO':
        Zt=Z_mol('/Users/Sebamarino/Astronomy/DebrisDiscs/NLTE/rot_transitions_13CO.txt', T) # cut from leiden database
    elif molecule=='C18O':
        Zt=Z_mol('/Users/Sebamarino/Astronomy/DebrisDiscs/NLTE/rot_transitions_C18O.txt', T) # cut from leiden database
    elif molecule=='HCN':
        Z=Z_HCN
        Zt=np.interp(T, Z[:,0], Z[:,1])

    k=1.38064852e-23 # mks blotzmann constant
    gj=2*j+1
    xj=gj*np.exp(-Ej/T)/Zt
    return xj

def Flux_line_jykms(M_molecule, dpc, nu, j, Ej, A, T, molecule='CO'):

    # A, Einstein coefficient,  s-1 from leiden database
    # Mco in Kg
    if molecule=='CO':
        mol_mass=28.0
    elif molecule=='HCN':
        mol_mass=27.0
    d=dpc*pc # m
    m_molecule=1.672e-27*(mol_mass) # kg
    return M_molecule*h*nu*A*x(j, Ej, T, molecule=molecule)/(4.0*np.pi*m_molecule*d**2)*1.0e26/(nu/3.0e5)

def M_molecule(dpc, nu, j, Ej,  Fjykms, T, A, molecule='CO'):

    # A, Einstein coefficient,  s-1 from leiden database
    if molecule=='CO':
        mol_mass=28.0
    if molecule=='13CO':
        mol_mass=29.0
    if molecule=='C18O':
        mol_mass=30.0
    elif molecule=='HCN':
        mol_mass=27.0

    Fmks=Fjykms*(nu/ckms)*1.0e-26
    d=dpc*pc # m
    m_molecule=1.672e-27*(mol_mass) # kg

    return 4.0*np.pi*m_molecule*d**2.0*Fmks/(h*nu*A*x(j, Ej, T, molecule=molecule))

def planck(wav, T):  # Inu
    nu=c/wav
    a = 2.0*h*nu**3/c**2.0
    b = h*nu/(k*T)
    intensity = a/ (np.exp(b) - 1.0)
    return intensity





def frac_CO_CO2_expected(RCO, RCO2, f_ice=0.5):
    # R are the abundances of x molecule vs water
    # returns fractional mass of CO+CO2
    mco=28.
    mco2=44.
    mh20=18.
    return f_ice*(RCO*mco + RCO2*mco)/(mh20 + RCO*mco + RCO2*mco2) 



def frac_CO_CO2_mdot(R, width, fir, Lstar, Mstar,  MCO, tph=120.):
    # MCO in Mearth

    return 1./ ( 1.+  0.0012*R**1.5 * width**(-1.) *fir**2 *Lstar* Mstar**(-0.5)*tph * MCO**(-1.)  )
