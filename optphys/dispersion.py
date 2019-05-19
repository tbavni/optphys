import numpy as np
from scipy.misc import derivative
from scipy.optimize import minimize
from math import pi
from .physdata import SI,eV_lambda
from mathx import sft as fourier
refindinf=None
xray=None

refindinf_entries={'BBOo':('BaB2O4','Eimerl-o'),'BBOe':('BaB2O4','Eimerl-e'),'KDPo':('KH2PO4','Zernike-o'),'KDPe':('KH2PO4','Zernike-e'),
                    'Si':('Si','Jellison'),'Ge':('Ge','Jellison')}

def refractive_index_fun(name=None,book=None,page=None,source='refindinf',**kwargs):
    """
    Refractive index function of wavelength (SI).
    
    Interfaces both refindinf and xray modules (lazy import), chosen by source parameter.
    
    TODO: standardize conditions
    
    Args:
        name (str): refindinf: a key in refindinf_entries which defines the book and page
        book (str): redindinf only: book in refindinf
        page (str): redindinf only: page in refindinf
        source (str): 'refindinf' or 'xray'
        **kwargs: passed on to the underlying function e.g. :func:`physdata.refindinf.lookup_fun` or 
            :func:`physdata.xray.refractive_index`
    Only one of name, or book/name should be given. If book is given, page is
    optional.
    """
    if source=='refindinf':
        global refindinf
        if refindinf is None:
            from .physdata import refindinf
        if name is not None:
            if name in refindinf_entries:
                book,page=refindinf_entries[name]
            else:
                book=name
        n_0=refindinf.lookup_fun(book,page,**kwargs)
    elif source=='xray':
        global xray
        if xray is None:
            from .physdata import xray
        n_0=lambda lamb:xray.refractive_index(name,eV_lambda/lamb)
    # Insert gas correction
    return n_0
        
def refractive_index(lamb,**kwargs):
    return refractive_index_fun(**kwargs)(lamb)
    
def material_dispersion(lamb,order=2,**kwargs):
    """N-th (default 2) derivative of wavenumber of a material.
    Args:
        lamb (numeric): wavelength (SI)
        order (int): dispersion order >=0, 2=GVD
        kwargs: passed to refractive_index_fun.
    Returns:
        numeric: d^n k/domega^n where n is the order and k=n*omega/c
    """
    nlamb=refractive_index_fun(**kwargs)
    return dispersion(lamb,nlamb,order)
        
def dispersion(lamb,nlamb,order=2):
    """N-th (default 2) derivative of wavenumber given refractive index.
    Args:
        lamb (numeric): wavelength (SI)
        order (int): dispersion order >=0, 2=GVD
        kwargs: passed to refractive_index_fun.
    Returns:
        numeric: d^n k/domega^n where n is the order and k=n*omega/c
    """
    def k(omega):
        return nlamb(2*pi*SI['c']/omega)*omega/SI['c']
    omega=2*pi*SI['c']/lamb
    if order>0:
        # num. points used to calculate the derivative must be at least the order+1
        # and also must be odd. 
        return derivative(k,omega,1e11,n=order,order=round((order+1)/2)*2+1)
    else:
        return k(omega)
        
def uniaxial_e_axis_index(n_o,n_e,theta):
    """Refractive index of extraordinary axis of a uniaxial crystal
    Args:
        n_o: refractive index of ordinary axis
        n_e: refractive index of extraordinary axis
        theta: propagation angle w.r.t. crystal axis
    """
    n=(1/((np.cos(theta)/n_o)**2+(np.sin(theta)/n_e)**2))**0.5
    return n
    
def refractive_index_uniaxial(symb,lamb,theta):
    n_o=refractive_index(lamb,name=symb+'o')
    n_e=refractive_index(lamb,name=symb+'e')
    n=uniaxial_e_axis_index(n_o,n_e,theta)
    return n
    
def optimal_propagation(omega,Ef,betaf,l_min,l_max):
    omega=omega.squeeze()
    Ef=Ef.squeeze()
    assert omega.ndim==1
    assert Ef.shape==omega.shape
    assert betaf.shape==Ef.shape
    
    ftd=fourier.FTD(k=omega,sign=1,x_m=0)
    # Normalize peak in time domain for optimizer
    Ef=Ef/max(abs(ftd.inv_trans(Ef)))
    # if l_bounds is not None:
    #     scale_length=l_bounds[1]-l_bounds[0]
    #     l_bounds_sc=[l/scale_length for l in l_bounds]
    # else:
    #     scale_length=1e-3
    def max_It(l):
        Et=ftd.inv_trans(Ef*np.exp(1j*betaf*l))
        It=(Et*Et.conj()).real
        return It.max(axis=-1)
    ls=np.linspace(l_min,l_max,100)[:,None]
    Itl=max_It(ls)
    lsi=Itl.argmax()
    l_guess=ls[lsi]
    def score(l):
        return -max_It(l)+1
    result=minimize(score,l_guess,options=dict(disp=False))
    #print(result.message)
    return result.x
