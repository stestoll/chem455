import numpy as np
import math as m
import scipy.special as sp

# Fundamental constants
#-------------------------------------------------------------------
h = 6.62607015e-34      # Planck constant, in J/Hz
hbar = h/2/m.pi         # reduced Planck constant, in J/(rad/s)
c = 299792458           # speed of light in vacuum, in m/s
NA = 6.02214076e23      # Avogadro number, in mol^(-1)
kB =  1.380649e-23      # Boltzmann constant, in J/K
a0 = 5.29177210903e-11  # Bohr radius, in m
e = 1.602176634e-19     # elementary charge, in C
eps0 = 8.8541878128e-12 # vacuum electric permittivity, C^2/J/m
u = 1.66053906660e-27   # atomic unit mass, in kg
me = 9.1093837015e-31   # electron mass, in kg
mp = 1.67262192369e-27  # proton mass, in kg
eV = e


# Black-body radiation
#-------------------------------------------------------------------
def rayleigh_jeans_law_freq(nu,T):
    # (4*pi/c) converts from spectral radiance to spectral energy density
    return (4*m.pi/c)*2*kB/c**2 * nu**2 * T
     
def rayleigh_jeans_law_wavelength(lam,T):
    return (c/lam**2)*rayleigh_jeans_law_freq(c/lam,T)
    
def planck_law_freq(nu,T):
    # (4*pi/c) converts from spectral radiance to spectral energy density
    return (4*m.pi/c)*2*h/c**2*nu**3/(np.exp(h*nu/kB/T)-1)
    
def planck_law_wavelength(lam,T):
    return (c/lam**2)*planck_law_freq(c/lam,T)

# Wien law for maxima as a function of temperature
def wien_law_wavelength(T):
    x5 = 4.965114231744276 # solution of (x-5)*exp(x) + 5 = 0
    return (h*c/x5)/(kB*T)
    
def wien_law_freq(T):
    x3 = 2.8214393721220787 # solution of (x-3)*exp(x) + 3 = 0
    return (x3*kB/h)*T

# Particle in a 1D box
#-------------------------------------------------------------------
def box_wavefunction(n,x,L):
    """
    Calculates wavefunctions of particle-in-the-box
    """
    psi = np.sqrt(2/L)*np.sin(n*m.pi*x/L)
    psi[x<0] = 0
    psi[x>L] = 0
    return psi
    

# Harmonic oscillator
#-------------------------------------------------------------------
def oscillator_wavefunction(n,x,alpha):
    """
    Calculates wavefunctions of harmonic oscillator
    """
    z = x*m.sqrt(alpha)
    N = (alpha/m.pi)**(1/4)/m.sqrt(2**n*sp.factorial(n))
    h = sp.eval_hermite(n,z)
    psi = N*h*np.exp(-z**2/2)
    return psi

"""
def radialfunction(n,l,r,Z=1,a0=a0):
    ""
    Calculates radial wavefunction for one-electron atom
    ""
    rho = 2*Z/n*r/a0
    N = sp.factorial(n-l-1)/2/n/sp.factorial(n+l)**3
    Za = (2*Z/n/a0)**3
    R = m.sqrt(Za*N)*sp.laguerre([2*l+1,n-l-1],rho).*rho.**l.*np.exp(-rho/2)
    return R
"""

