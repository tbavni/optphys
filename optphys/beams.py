import math
import numpy as np
import mathx

class GaussianBeam:
    """lamb is in medium"""
    def __init__(self,lamb,**kwargs):
        """Create Gaussian beam
        
        Can specify one of w_0 or z_R.
        Args:
            lamb: wavelength (in the medium)
            w_0: waist
            z_R: Rayleigh range
        """
        self.lamb=np.asarray(lamb)
        self.k=2*math.pi/lamb
        if 'w_0' in kwargs:
            self.w_0=np.asarray(kwargs.pop('w_0'))
            self.z_R=math.pi*self.w_0**2/self.lamb
        elif 'z_R' in kwargs:
            self.z_R=np.asarray(kwargs.pop('z_R'))
            self.w_0=(self.z_R*self.lamb/math.pi)**0.5
        self.area=math.pi*self.w_0**2/2
        if 'P' in kwargs:
            self.P=np.asarray(kwargs.pop('P'))
            self.absE_w=(P/area*2*SI['Z_vac'])**0.5
        elif 'absE_w' in kwargs:
            self.absE_w=np.asarray(kwargs.pop('absE_w'))
        else:
            self.absE_w=1
        if len(kwargs)!=0:
            raise ValueError('Unknown keyword arguments %s'%list(kwargs.keys()))

    def __eq__(self,other):
        return self.lamb==other.lamb and self.w_0==other.w_0 and self.absE_w==other.absE_w
            
    def roc(self,z):
        z=np.asarray(z)
        return z*(1+mathx.divide0(self.z_R,z)**2)
        # try:
        #     return z*(1+(self.z_R/z)**2)
        # except ZeroDivisionError:
        #     return 0
            
    def Gouy(self,z):
        z=np.asarray(z)
        return np.arctan(z/self.z_R)
        
    def waist(self,z):
        z=np.asarray(z)
        return self.w_0*(1+(z/self.z_R)**2)**0.5
        
    def calc_all(self,z,rho):
        z=np.asarray(z)
        rho=np.asarray(rho)
        w=self.waist(z)
        psi=self.Gouy(z)
        R=self.roc(z)
        absE=self.w_0/w*np.exp(-(rho/w)**2)*self.absE_w
        z_c=mathx.divide0(rho**2,2*R)
        # try:
        #     z_c=rho**2/(2*R)
        # except ZeroDivisionError:
        #     z_c=0
        phi=self.k*z_c-psi # leave k out, retarded reference frame
        return {'w':w,'psi':psi,'absE':absE,'phi':phi,'R':R}
        
    def absEphi(self,z,rho=0):
        s=self.calc_all(z,rho)
        return s['absE'],s['phi']
        
    def E(self,z,rho=0):
        absE,phi=self.absEphi(z,rho)
        return absE*np.exp(1j*phi)
        
    @classmethod
    def q_amplitude(cls,q,lamb):
        return cls(lamb,q.imag).absEphi(q.real)