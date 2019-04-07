from .physdata import SI
import numpy as np
import mathx


def grating_pair_dispersion(Lamb, lamb, theta_inc, order):
    """Calculate dispersion (phase per unit length) of a grating pair.
    
    Dispersion includes both passes i.e. four grating reflections/transmissions,
    *per unit grating separation* along the normal. To
    convert to per unit grating separation along incident beam, multiply by
    cos(theta_inc). All values in SI units. Taken from Walmsley et al. 2001 [1]_, 
    except for inclusion of factor of m in the additive term inside the
    brackets in the expression for phi_3, which I believe is a typo.
    
    Sign convention: incident and diffracted angles are as normally defined for a
    mirror i.e. for the zeroth order, theta_diff=theta_inc. Increasing m increases
    theta_diff.
        
    Args:
        Lamb: line spacing
        lamb: wavelength
        theta_inc: grating incident angle
        order: diffraction order
        
    Returns:
        tuple of second-order dispersion beta_2, third-order dispersion beta_3, 
        diffracted angle theta_diff, and phase beta.
    
    .. [1] Walmsley, I., Waxer, L. & Dorrer, C., The role of dispersion in ultrafast
        optics, Review of Scientific Instruments, 2001, 72, 1-29
    """
    theta_diff = np.arcsin(np.sin(theta_inc) + order*lamb/Lamb)
    beta = 2*pi/lamb*2*np.cos(theta_inc - theta_diff)/np.cos(theta_inc)
    beta_2 = -lamb**3*2/(2*pi*SI['c']**2*Lamb**2*np.cos(theta_diff)**3)
    beta_3 = 2*3*lamb**4/(4*pi**2*SI['c']**3*Lamb**2*np.cos(theta_diff)**3)*(
                1 + order*lamb*np.sin(theta_diff)/(Lamb*np.cos(theta_diff)**2))
    return (beta_2, beta_3, theta_diff, beta)


def focal_len_lens(n, r1, r2, d):
    """Focal length of a lens with lensmaker's equation
    
    Args:
        n: refractive index
        r1: radius of curvature on incident side
        r2: radius of curvature on exit side
        d: minimum thickness between two surfaces
    """
    return 1/((n - 1)*(1/r1 - 1/r2 + (n - 1)*d/(n*r1*r2)))


def propagate_Sziklas_Siegman_cylindrical_symmetry(ht, R, E, k, L, M, axis=None):
    """Paraxial propagate using Sziklas-Siegman transform with cylindrical symmetry.
    
    Sziklas-Siegman transform described in Sziklas & Siegman Appl. Opt.
    14, 1975, p1874. This function largely uses the notation in Hello & Vinet, J.
    Optics (Paris) 27, 1996, p265.

    Args:
        ht (:class:`.QDHT`): Hankel transform object
        R (numeric): aperture radius
        E  (numeric): electric field sampled at Hankel transform radial points
        k  (numeric): wavenumber
        L  (numeric): propagation distance
        M (numeric): aperture magnification factor
        axis (int): radial axis
    
    Returns:
        numeric: propagated electric field
    """
    if axis is None:
        axis = mathx.vector_dim(E)
    r = ht.points(R, axis)
    p = ht.conj_points(R, axis)
    # Eq. (12)       
    z0 = L/(M - 1)
    # Eq. (13)
    E = np.exp(-1j*k*r**2/(2*z0))*E
    # Propagate a distance L/M
    Et = ht.transform(E, R, axis)
    Et = np.exp(-1j*p**2/(2*k)*L/M)*Et
    E = ht.inv_transform(Et, R, axis)
    # Eq. (16)
    Rp = R*M
    rp = r*M
    # Eq. (15)
    E = np.exp(1j*k*rp**2/(2*(z0 + L)))*E/M
    return E


def propagate_Sziklas_Siegman(ftds, E, k, L, M):
    # Eq. (12)
    z0 = L/(M - 1)
    # Eq. (13)
    rsqd = sum(ftd.x**2 for ftd in ftds)
    E = np.exp(-1j*k*rsqd/(2*z0))*E
    # Transform
    Et = E
    for ftd in ftds:
        Et = ftd.trans(Et)
    # Propagate a distance L/M
    Et *= np.exp(-1j*sum(ftd.k**2 for ftd in ftds)/(2*k)*L/M)
    # Inverse transform
    E = Et
    for ftd in ftds:
        E = ftd.inv_trans(E)
    # Eq. (16)
    rpsqd = rsqd*M**2
    # Eq. (15)
    E *= np.exp(1j*k*rpsqd/(2*(z0 + L)))/M
    return E


# def fresnel_reflection(n1, n2, incidence_angle=0):
#     sr = np.sqrt(n2**2 - (n1*np.sin(incidence_angle))**2);
#     r_s = (n1*np.cos(incidence_angle) - sr)/(n1*np.cos(incidence_angle) + sr)
#     r_p = (n2**2*np.cos(incidence_angle) - n1*sr)/(n2**2.*np.cos(incidence_angle) + n1*sr)
#     return r_s, r_p
