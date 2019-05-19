"""Storage and manipulation of ultrashort pulse reconstructions.

Used by SPIDER programs and others.

The various classes describe different types of pulses and allow convenient
conversion e.g. oversampling or picking a given row. They can also plot themselves
using usp.plot.
"""
import logging, math
from scipy.interpolate import interp1d
from scipy.special import factorial
import numpy as np
from PyQt5 import QtGui, QtCore
import pyqtgraph_extended as pg
import mathx
from mathx import usv
from mathx import sft as fourier

logger = logging.getLogger(__name__)

# These are implicitly hard-coded in some indexing expressions and shouldn't be
# changed.
AXIS_TEMP = -1
AXIS_Y = -2
AXIS_FRAME = -3


class Result:
    axis_attrs = ('omega',)
    array_attrs = ()

    def __init__(self, omega):
        self.ftd = ft.FTD(k=omega, sign=1, x_m=0)
        self.omega = omega
        self.t = self.ftd.x

    def same_axes_as(self, other):
        """Test if all axis attributes are the same as another result."""
        return all(np.array_equal(getattr(self, attr), getattr(other, attr)) for attr in self.axis_attrs)

        # @property

    # def array_shape(self):
    #     return np.broadcast(*[getattr(self,attr) for attr in self.axis_attrs])

    @classmethod
    def kwargs(cls, obj):
        return {key: getattr(obj, key) for key in cls.axis_attrs + cls.array_attrs}


class Temporal(Result):
    axis_attrs = Result.axis_attrs + ('omega_0',)
    array_attrs = Result.array_attrs + ('Ef', 'Et', 'It', 'phit', 'fwhm')

    def __init__(self, omega, Ef, omega_0, Et=None, It=None, phit=None, fwhm=None, *args, **kwargs):
        super().__init__(omega=omega, *args, **kwargs)
        if Et is None:
            Et = ft.trans(self.omega, -1, Ef, self.t, AXIS_TEMP) * np.exp(1j * omega_0 * self.t)
        if It is None:
            It = mathx.abs_sqd(Et)
        if phit is None:
            phit = mathx.unwrap(np.angle(Et), abs(self.t).argmin(), axis=AXIS_TEMP)
        self.Ef = Ef
        self.omega_0 = omega_0
        self.Et = Et
        self.It = It
        self.phit = phit
        if fwhm is None:
            t_half_max = mathx.peak_crossings(self.t, self.It, 0.5, AXIS_TEMP)
            fwhm = t_half_max[1] - t_half_max[0]
        self.fwhm = fwhm

    def transform_limited(self):
        kwargs = self.__class__.kwargs(self)
        kwargs['Ef'] = abs(self.Ef)
        for key in ('Et', 'It', 'phit', 'fwhm'):
            kwargs[key] = None
        return self.__class__(**kwargs)


class Temporal1D(Temporal):
    # def __init__(self,*args,**kwargs):
    #     super().__init__(*args,**kwargs)
    #     for attr in self.array_attrs:
    #         squeezed=getattr(self,attr).squeeze()
    #         assert squeezed.ndim==1
    #         setattr(self,attr,squeezed)
    # self.t_half_max=mathx.peak_crossings_1D(self.t,self.It,0.5)
    # self.fwhm=self.t_half_max[1]-self.t_half_max[0]
    def set_usp_plot(self, plot, key, **kwargs):
        plot.set_pulse(key, t=self.t, It=self.It, phit=self.phit, **kwargs)


class YDependent(Result):
    axis_attrs = Result.axis_attrs + ('yp',)

    def __init__(self, yp, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.yp = yp

    def pick_y(self, yi=None):
        if yi is None:
            yi = abs(self.yp).argmin()
        kwargs = self.y_picked_class.kwargs(self)  # ,self.y_picked_class)
        for key in self.y_dependent_kwargs:
            kwargs[key] = mathx.squeeze_leading(kwargs[key].take([yi], AXIS_Y))
        return self.y_picked_class(**kwargs)

    def oversample_y(self, factor):
        kwargs = self.__class__.kwargs(self)
        for key in self.y_dependent_kwargs:
            kwargs['yp'], kwargs[key] = ft.oversample(self.yp, kwargs[key], factor)
        return self.__class__(**kwargs)

    # def kwargs(self,mro=None):
    #     if mro is None:
    #         mro=self.__class__
    #     kwargs=super(YDependent,mro).kwargs(self,mro)
    #     kwargs.update(yp=self.yp)
    #     return kwargs

    # @property
    # def array_shape(self):
    #     return mathx.set_shape_element(super().array_shape,AXIS_Y,len(self.yp))
    #     
    # def same_axes(self,other):
    #     return super().same_axes(other) and np.array_equal(self.yp,other.yp)
    #     
    # def set_axes(self,other):
    #     super().set_axes(other)
    #     self.yp=other.yp


class TemporalYDep(Temporal, YDependent):
    axis_attrs = tuple(set(Temporal.axis_attrs + YDependent.axis_attrs))
    array_attrs = tuple(set(Temporal.array_attrs + YDependent.array_attrs))

    y_picked_class = Temporal1D
    y_dependent_kwargs = ('Ef', 'Et', 'It', 'phit', 'fwhm')


class Temporal2D(TemporalYDep):
    """Single frame y-dependent temporal reconstruction."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.It.shape == (len(self.yp), len(self.t)) and self.phit.shape == self.It.shape
        # Peaks and pulse front
        self.t_peak = self.t[self.It.argmax(axis=1)][:, None]
        self.cross_correlate_pulse_front()

    def set_usp_plot(self, plot, **kwargs):
        plot.set_pulse_2D(t=self.t, y=self.yp, It=self.It, phit=self.phit, **kwargs)
        yp = self.yp.squeeze()
        xc = self.pulse_front_cross_correlation
        if 'pulse_front_fit' not in plot.image_annotations.intensity:
            line = QtGui.QGraphicsLineItem()
            line.setPen(QtGui.QPen(QtGui.QColor('red'), 0, style=pg.DashLine))
            plot.add_image_annotation('intensity', 'pulse_front_fit', line)
        else:
            line = plot.image_annotations.intensity['pulse_front_fit']
        fit = xc['linear_fit_coeffs']
        line.setLine(fit[1] + fit[0] * yp[0], yp[0], fit[1] + fit[0] * yp[-1], yp[-1])

        if 'pulse_front' not in plot.image_annotations.intensity:
            line = pg.PlotDataItem(pen='r')
            plot.add_image_annotation('intensity', 'pulse_front', line)
        else:
            line = plot.image_annotations.intensity['pulse_front']
        line.setData(xc['t'].squeeze(), yp)

    def cross_correlate_pulse_front(self):
        Ef0c = self.Ef[abs(self.yp).argmin(), :].conj()
        t = np.array([mathx.phase.optimal_linear_phase(self.omega, Ef * Ef0c)[0] for Ef in self.Ef])[:, None]
        # Fit cross correlation
        maxt_It = self.It.max(axis=1)
        fit_weight = mathx.thresholded(maxt_It, 0.5 * maxt_It.max())
        linear_fit_coeffs = mathx.fit_sparse_1d_poly(self.yp.squeeze(), t.squeeze(), fit_weight, [1, 0])
        linear_fit = np.polyval(linear_fit_coeffs, self.yp)
        self.pulse_front_cross_correlation = dict(t=t, fit_weight=fit_weight, linear_fit=linear_fit,
                                                  linear_fit_coeffs=linear_fit_coeffs)


class PolarSpectral(Result):
    axis_attrs = Result.axis_attrs + ('omega_0_ind',)
    array_attrs = Result.array_attrs + (
        'If', 'phif', 'Ef', 'Et', 'omega_centroid', 'omega_hm0', 'omega_hm1', 'fwhm', 'phif_taylor')

    def __init__(self, omega, If, phif, omega_0_ind, Ef=None, Et=None, omega_centroid=None,
                 omega_hm0=None, omega_hm1=None, fwhm=None, phif_taylor=None, *args, **kwargs):
        super().__init__(omega=omega, *args, **kwargs)
        if Ef is None:
            Ef = If ** 0.5 * np.exp(1j * phif)
        if Et is None:
            Et = ft.trans(self.omega, -1, Ef, self.t, AXIS_TEMP)
        self.If = If
        self.phif = phif
        self.Ef = Ef
        self.Et = Et
        self.omega_0_ind = omega_0_ind
        self.omega_0 = omega[omega_0_ind]
        self.omega_centroid = mathx.moment(self.omega, self.If, 1, axis=AXIS_TEMP)
        if omega_hm0 is None or omega_hm1 is None or fwhm is None:
            omega_hm0, omega_hm1 = mathx.peak_crossings(self.omega, self.If, 0.5, AXIS_TEMP)
            fwhm = omega_hm1 - omega_hm0
        self.omega_hm0 = omega_hm0
        self.omega_hm1 = omega_hm1
        self.fwhm = fwhm
        if phif_taylor is None:
            omegap = omega - self.omega_0
            deg = 3
            w = np.nanmean(If.reshape(-1, len(omega)), axis=0)
            taylor_shape = mathx.set_shape_element(self.phif.shape, -1, deg + 1)
            if np.isnan(w).any():
                # Can't do fit, so create NaN array of correct shape
                phif_taylor = np.empty(taylor_shape)
                phif_taylor[:] = np.nan
            else:
                fit = np.polyfit(omegap, phif.reshape(-1, len(omega)).T, deg, w=w)
                phif_taylor = fit[::-1].T.reshape(taylor_shape) * factorial(range(deg + 1))
        self.phif_taylor = phif_taylor

    # def kwargs(self,mro=None):
    #     if mro is None:
    #         mro=self.__class__
    #     kwargs=super(PolarSpectral,mro).kwargs(self,mro)
    #     kwargs.update(If=self.If,phif=self.phif,omega_0_ind=self.omega_0_ind)
    #     return kwargs

    def oversample_omega(self, factor):
        if factor == 0:
            return self
        kwargs = self.__class__.kwargs(self)
        omegao, Efo, _ = ft.trans_oversample(self.t, 1, self.Et, self.omega, factor, AXIS_TEMP)
        phifo_orig = interp1d(self.omega, self.phif, copy=False, axis=AXIS_TEMP, fill_value='extrapolate')(omegao)
        phifo = np.angle(Efo)
        phifo += 2 * math.pi * ((phifo_orig - phifo) / (2 * math.pi)).round()
        kwargs.update(omega=omegao, If=abs(Efo) ** 2, phif=phifo, omega_0_ind=abs(omegao - self.omega_0).argmin())
        for name in ('Ef', 'Et', 'fwhm', 'omega_centroid', 'omega_hm0', 'omega_hm1',):
            del kwargs[name]
        return self.__class__(**kwargs)

    def oversample_t(self, factor):
        # kwargs=self.temporal_oversampled_class.kwargs(self)#,self.temporal_oversampled_class)
        kwargs = {attr: getattr(self, attr) for attr in self.temporal_oversampled_class.axis_attrs}
        if factor > 0:
            num_pad = int(len(self.omega) * factor / 2)
            omegao, Efo = usv.pad_sampled(self.omega, self.Ef, num_pad, num_pad, AXIS_TEMP)
            kwargs.update(omega=omegao, Ef=Efo)
        else:
            kwargs.update(omega=self.omega, Ef=self.Ef)
        return self.temporal_oversampled_class(**kwargs)

    def apply_refractive_index(self, n, l):
        """
        Args:
            n (function): refractive index given wavelength in nm
            l (float): material thickness in mm
        """
        pass

    # def same_axes(self,other):
    #     return super().same_axes(other) and self.omega_0_ind==other.omega_0_ind
    #     
    # def setup_arrays(self):
    #     shape=self.array_shape
    #     self.If=np.zeros(shape)
    #     self.phif=np.zeros(shape)
    #     self.Et=np.zeros(shape)
    #     self.Ef=np.zeros(shape)
    #     
    # def set_axes(self,other):
    #     super().set_axes(other)
    #     self.omega_0_ind=other.omega_0_ind
    #     self.omega_0=omega[omega_0_ind]
    #     
    # def set_pulse_slice(self,slc,pulse):
    #     self.If[slc]=pulse.If
    #     self.phif[slc]=pulse.phif
    #     self.Ef[slc]=pulse.Ef
    #     self.Et[slc]=pulse.Et


class PolarSpectral1D(PolarSpectral):
    temporal_oversampled_class = Temporal1D

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for attr in self.array_attrs:
            squeezed = np.squeeze(getattr(self, attr))
            assert squeezed.ndim <= 1
            setattr(self, attr, squeezed)

    def set_usp_plot(self, plot, key, **kwargs):
        plot.set_pulse(key, omega=self.omega, If=self.If, phif=self.phif, **kwargs)


class PolarSpectralYDep(PolarSpectral, YDependent):
    axis_attrs = tuple(set(PolarSpectral.axis_attrs + YDependent.axis_attrs))
    array_attrs = tuple(set(PolarSpectral.array_attrs + YDependent.array_attrs))

    temporal_oversampled_class = Temporal2D
    y_picked_class = PolarSpectral1D
    y_dependent_kwargs = 'If', 'phif', 'Ef', 'Et', 'omega_centroid', 'fwhm', 'omega_hm0', 'omega_hm1', 'phif_taylor'

    def oversample_y(self, factor):
        if factor == 0:
            return self
        ypo, Efo, _ = ft.oversample(self.yp, self.Ef, factor)
        Ifo = abs(Efo) ** 2
        phifo = np.angle(Efo)
        phifo_orig = interp1d(self.yp.squeeze(), self.phif, copy=False, axis=AXIS_Y, fill_value='extrapolate')(
            ypo.squeeze())
        phifo += 2 * math.pi * ((phifo_orig - phifo) / (2 * math.pi)).round()
        return self.__class__(omega=self.omega, If=Ifo, phif=phifo, yp=ypo, omega_0_ind=self.omega_0_ind)


class PolarSpectral2D(PolarSpectralYDep):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.If.shape == (len(self.yp), len(self.t)) and self.phif.shape == self.If.shape

    def set_usp_plot(self, plot, **kwargs):
        plot.set_pulse_2D(omega=self.omega, y=self.yp, If=self.If, phif=self.phif, **kwargs)


class MultiFrame(Result):
    def __init__(self, *args, **kwargs):
        """
        Args:
            usp (tuple[usp_plot,key])
        """
        super().__init__(*args, **kwargs)
        # if been_set is None:
        #     been_set=[]
        # self.been_set=set(been_set)
        # # initialize been_set based on any nonzero elements
        # self.been_set=np.any(np.rollaxis(self.Ef,AXIS_SEQUENCE).reshape(self.length,-1),1)

    @property
    def length(self):
        return self.Ef.shape[AXIS_FRAME]

    def set_pulse(self, n, pulse):
        """
        Args:
            pulse: the class of self, minus the Sequence part
        """
        assert self.same_axes_as(pulse)
        assert pulse.__class__ == self.single_frame_class
        assert n < self.length
        slc = mathx.slice_dim([n], AXIS_FRAME)
        for attr in self.array_attrs:
            getattr(self, attr)[slc] = getattr(pulse, attr)
        # self.been_set.add(n)
        # self.update()

    def pick_frame(self, n):
        kwargs = self.single_frame_class.kwargs(self)
        for attr in self.array_attrs:
            kwargs[attr] = mathx.squeeze_leading(kwargs[attr].take([n], AXIS_FRAME))
        return self.single_frame_class(**kwargs)

    # @staticmethod
    # def set_pulse(sequence,n,pulse,usp=None):
    #     if sequence is None or not sequence.same_axes_as(pulse):
    #         if usp_plot is not None:
    #             sequence.clear_usp_plot(*usp)
    #         axis_kwargs={attr:getattr(pulse,attr) for attr in pulse.axis_attrs}
    #         def make_sequence_array(pulse_array):
    #             return np.zeros(mathx.set_shape_element(pulse_array.shape,AXIS_SEQUENCE,self.length),dtype=pulse_array.dtype)
    #         array_kwargs={attr:make_sequence_array(getattr(pulse,attr))}
    #         sequence=self.__class__(**axis_kwargs,**array_kwargs)
    #     sequence._set_pulse(n,pulse)
    #     return sequence

    @classmethod
    def make_from_pulse(cls, length, n, pulse):
        axis_kwargs = {attr: getattr(pulse, attr) for attr in pulse.axis_attrs}

        def make_sequence_array(pulse_array):
            pulse_array = np.asarray(pulse_array)
            array = np.empty(mathx.set_shape_element(pulse_array.shape, AXIS_FRAME, length), dtype=pulse_array.dtype)
            array[:] = np.nan
            return array

        array_kwargs = {attr: make_sequence_array(getattr(pulse, attr)) for attr in pulse.array_attrs}
        sequence = cls(**axis_kwargs, **array_kwargs)
        # Removed 17/2/2017
        # sequence.set_pulse(n,pulse)
        return sequence

    # def update(self):
    #     pass


class MultiFrame1D(MultiFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # def check_invariant(self):
    #     for attr in self.array_attrs:
    #         shape=getattr(self,attr).shape
    #         assert shape==(self.length,1,len(self.omega))

    def set_usp_plot(self, plot, key, **kwargs):
        for n in range(self.length):
            if np.any(self.Ef[n]):
                self.pick_frame(n).set_usp_plot(plot, (key, n), **kwargs)


class Temporal1DMultiFrame(Temporal, MultiFrame1D):
    single_frame_class = Temporal1D

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.check_invariant() # todo: make more systematic


class PolarSpectral1DMultiFrame(PolarSpectral, MultiFrame1D):
    temporal_oversampled_class = Temporal1DMultiFrame
    single_frame_class = PolarSpectral1D

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.check_invariant()


class MultiFrame2D(MultiFrame, YDependent):
    pass
    # def check_invariant(self):
    #     for attr in self.array_attrs:
    #         assert getattr(self,attr).shape==(self.length,len(self.yp),len(self.omega))


class Temporal2DMultiFrame(TemporalYDep, MultiFrame2D):
    single_frame_class = Temporal2D
    y_picked_class = Temporal1DMultiFrame

    # def __init__(self,*args,**kwargs):
    #     super().__init__(*args,**kwargs)
    #     self.check_invariant()


class PolarSpectral2DMultiFrame(PolarSpectralYDep, MultiFrame2D):
    single_frame_class = PolarSpectral2D
    temporal_oversampled_class = Temporal2DMultiFrame
    y_picked_class = PolarSpectral1DMultiFrame

    # def __init__(self,*args,**kwargs):
    #     super().__init__(*args,**kwargs)
    #     self.check_invariant()
