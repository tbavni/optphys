import numpy as np
from optphys.physdata.xray import *

def test_refractive_index():
    # I took these values from http://henke.lbl.gov/optical_constants/getdb2.html
    # on 23/9/2016
    assert np.isclose(refractive_index('Ne',300,0.8999)-1,-1.81337089E-06+3.28078897E-07j)
    assert np.isclose(refractive_index('Ar',200,1.784)-1,-1.82901158E-06+6.52095139E-07j)
    assert np.isclose(refractive_index('Al',400,2.699e3)-1,-0.00317348377+0.000863814494j,rtol=1e-3)
