import numpy as np
from pyqtgraph_recipes import ImageWithProjsAlignedPlot
from mathx import sft as fourier
import pyqtgraph_extended as pg
from optphys.physdata import SI

class PhaseSpacePlot:
    """A collection of Pyqtgraph GraphicsObjects representing a phase space distribution and its
    projections.
    """

    def __init__(self, plots, cbar=None, pen='b', ftd=None, Ex=None, Ek=None, lut=None, S_transform=None, log10_range=3,
            bound_cond='pad', scale=None):
        """
        Create image and projection items and set their data.

        Args:
            plots (dict): values are pyqtgraph PlotItems or similar with keys
                horz, vert and image
            cbar: a colorbar, if required
            pen: passed on to plot method for Ix and Ik
            ftd (:class:`fourier.FTD'): object for transforming from x to k domains
            scale (dict or None): dict of scale factors, with keys (all optional)
                x, k, Ix, Ik, S

        Other parameters are passed to :meth:`set_E'
        """
        self.plots = plots
        self.cbar = cbar
        self.lines = {}
        self.lines['Ix'] = self.plots['horz'].plot(pen=pen)
        self.lines['Ik'] = self.plots['vert'].plot(pen=pen)
        self.image = self.plots['image'].image(lut=lut)
        if self.cbar is not None:
            self.cbar.setImage(self.image)
        # scale factors for axes
        if scale is None:
            scale = {}
        scale.setdefault('x', 1)
        scale.setdefault('k', 1)
        scale.setdefault('Ix', 1)
        scale.setdefault('Ik', 1)
        scale.setdefault('S', 1)
        assert scale.keys() == set(('x', 'k', 'Ix', 'Ik', 'S'))
        self.scale = scale
        self.set_E(ftd=ftd, Ex=Ex, Ek=Ek, S_transform=S_transform, log10_range=log10_range, bound_cond=bound_cond)

    def set_E(self, **kwargs):
        """Set signal, calculate phase space distribution and update plots.

        Args:
            Ex: signal in x domain
            Ek: signal in k domain
            lut: Pyqtgraph lookup table
            S_transform (str): None or 'log10'
            log10_range (int): decades for log color scale
            bound_cond (str): passed on to :func:`fourier.spectrogram', one of
                'pad','odd-cyclic','cyclic'
        """
        try:
            self.log10_range = kwargs['log10_range']
        except KeyError:
            pass
        try:
            self.S_transform = kwargs['S_transform']
        except KeyError:
            pass
        assert self.S_transform in (None, 'log10')
        try:
            self.ftd = kwargs['ftd']
        except KeyError:
            pass
        try:
            self.Ek = kwargs['Ek']
            if self.Ek is not None:
                self.Ex = None
        except KeyError:
            pass
        try:
            self.Ex = kwargs['Ex']
            if self.Ex is not None:
                self.Ek = None
        except KeyError:
            pass
        try:
            self.bound_cond = kwargs['bound_cond']
        except KeyError:
            pass
        if self.Ex is None and self.Ek is None:
            return
        if self.Ex is None:
            self.Ex = self.ftd.inv_trans(self.Ek)
        else:
            self.Ek = self.ftd.trans(self.Ex)
        Ix = abs(self.Ex.squeeze())**2
        if self.scale['Ix'] == 'max':
            Ix /= Ix.max()
        else:
            Ix /= self.scale['Ix']
        Ik = abs(self.Ek.squeeze())**2
        if self.scale['Ik'] == 'max':
            Ik /= Ik.max()
        else:
            Ik /= self.scale['Ik']
        self.lines['Ix'].setData(self.ftd.x.squeeze()/self.scale['x'], Ix)
        self.lines['Ik'].setData(Ik, self.ftd.k.squeeze()/self.scale['k'])
        xs, ks, S = fourier.spectrogram(self.ftd.x.squeeze(), 1, self.Ex.squeeze(), 32, self.ftd.k.squeeze(),
                                        self.bound_cond)
        if self.scale['S'] == 'max':
            S /= S.max()
        else:
            S /= self.scale['S']
        if self.S_transform == 'log10':
            S[S == 0] = S[S != 0].min()
            S = np.log10(S)
        self.image.setImage(S.T)
        self.image.setRect(pg.axes_to_rect(xs/self.scale['x'], ks/self.scale['k']))
        if self.S_transform == 'log10':
            mx = S.max()
            self.image.setLevels((mx - self.log10_range, mx))


class PhaseSpacePlotAligned(ImageWithProjsAlignedPlot, PhaseSpacePlot):
    def __init__(self, gl=None, pen='b', ftd=None, Ex=None, Ek=None, lut=None, S_transform=None, cornertexts=None,
            log10_range=3, bound_cond='pad', scale=None):
        ImageWithProjsAlignedPlot.__init__(self, gl, cornertexts)
        PhaseSpacePlot.__init__(self, self.plots, self.cbar, pen, ftd, Ex, Ek, lut, S_transform, log10_range,
                                bound_cond, scale)

    @classmethod
    def setup_ultrashort_pulse_plot(cls, ftd, log10_range=3, S_transform=None, gl=None, lut=None, t_unit='fs',
            omega_unit='rad/fs', It_scale=1, If_scale=1, S_scale=1):
        if gl is None:
            glw = pg.GraphicsLayoutWidget()
            gl = glw.ci
        if lut is None:
            lut = pg.get_colormap_lut()
        scale = dict(x={'fs': 1e-15}[t_unit], k={'rad/fs': 1e15, 'eV': SI['e']/SI['hbar']}[omega_unit], Ix=It_scale,
                     Ik=If_scale, S=S_scale)
        It_str = '|E(t)|<sup>2</sup>'
        If_str = '|E(&omega;)|<sup>2</sup>'
        S_str = 'Spectrogram'
        if It_scale == 'max':
            It_str += ' (norm.)'
        if If_scale == 'max':
            If_str += ' (norm.)'
        if S_scale == 'max':
            S_str += ' (norm.)'
        psp = PhaseSpacePlotAligned(ftd=ftd, lut=lut, log10_range=log10_range, S_transform=S_transform, scale=scale,
                                    gl=gl)
        vp = psp.plots['vert']
        vp.setLabel('left', '&omega; (%s)'%omega_unit)
        vp.setLabel('top', If_str)
        hp = psp.plots['horz']
        hp.setLabel('bottom', 't (%s)'%t_unit)
        hp.setLabel('left', It_str)
        psp.cbar.setLabel(S_str)
        if 'glw' in locals():
            glw.show()
            psp.widget = glw
        return psp