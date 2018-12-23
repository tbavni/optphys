import os
import numpy as np
from . import SI,periodic_table,eV_lambda
from periodictable import xsf

def refractive_index(element,eV,mass_density=None):
    """Refractive index using periodictable package.
    
    Imaginary part is positive i.e. optics rather than x-ray convention.
    
    Args:
        mass_density (float or None): in kg/m^3. If not given, then :func:`physdata.periodic_table.mass_density`
        is used.
    """
    if mass_density is None:
        mass_density=periodic_table.mass_density(element)
    # xsf function accepts keV and g/cm^3
    n=xsf.index_of_refraction(compound=element,energy=eV/1e3,density=mass_density/1e3).conj()
    return n
