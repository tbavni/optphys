from math import pi
import mathx
from mathx import matseq

# def propagate_Gaussian(d,q1,E1):
#     q2=apply_to_Gaussian(propagation(d),q)
#     return q2,q1/q2*E1

def propagation(d):
    return [[1,d],[0,1]]

def thin_lens(f):
    return [[1,0],[-1/f,1]]
    
def curved_interface(n1,n2,R):
    """
    Args:
        n1: incident refractive index
        n2: final refractive index
        R: radius of curvature, >0 for convex
    """
    return [[1,0],[(n1-n2)/(R*n2),n1/n2]]

def thick_lens(n2,r1,r2,t,n1=1):
    #return mathx.mult_mat_mat([[1,0],[(n2-n1)/(r2*n1),n2/n1]],propagation(t),[[1,0],[(n1-n2)/(r1*n2),n1/n2]])
    return matseq.mult_mat_mat(curved_interface(n2,n1,-r2),propagation(t),curved_interface(n1,n2,r1))
    
def transform_Gaussian(m,q):
    return (m[0][0]*q+m[0][1])/(m[1][0]*q+m[1][1])
    
def Gaussian_wR_to_q(w,R,lamb,m=1):
    return 1/(1/R-1j*lamb/(pi*(w/m)**2))
    
def Gaussian_q_to_wR(q,lamb,m=1):
    iq=1/q
    w=(-lamb/(pi*iq.imag))**0.5*m
    try:
        R=1/iq.real
    except ZeroDivisionError:
        R=float('inf')
    return w,R

