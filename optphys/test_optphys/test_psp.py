import numpy as np
from mathx import sft
import pyqtgraph_recipes as pgr
import optphys
import pyqtgraph_extended as pg

t = np.linspace(-150, 150, 100)
omega_0 = 2.35
T0 = 50
alpha = 0.004
Et = 2*np.exp(-0.5*(t/T0)**2 - 1j*(omega_0*t - alpha*t**2))
omega = sft.conj_axis(t, omega_0)


def test_PhaseSpacePlot(qtbot):
    ipp = pgr.ImageWithProjsAlignedPlot()
    psp = optphys.PhaseSpacePlot(ipp.plots, ftd=sft.FTD(x=t, k=omega, sign=1), Ex=Et, cbar=ipp.cbar,
                             lut=pg.get_colormap_lut())
    return psp


def test_PhaseSpacePlotAligned(qtbot):
    psp = optphys.PhaseSpacePlotAligned(ftd=sft.FTD(x=t, k=omega, sign=1), Ex=Et, bound_cond='cyclic')
    return psp
