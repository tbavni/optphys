"""For plotting ultrashort pulses.
Pulses addressed by dictionary.

Complicated example: comparing SPIDER measurement with simulation
One MultiPulsePlot, with StackedTemporalPlot and StackedSpectralPlot.

All plots: time/frequency spans -1
2D: y spans -2
"""
import math
from collections import namedtuple

import numpy as np
import pyqtgraph_extended as pg
from PyQt5 import QtGui
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d


class Multi:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pulses = {}
        self.plots = {}

    def set_pulse(self, key, **kwargs):
        if key in self.pulses:
            self.pulses[key].update(kwargs)
        else:
            self.pulses[key] = kwargs
        for plot in self.plots.values():
            plot.set_pulse(key, **kwargs)

    def add_plot(self, key, plot):
        self.plots[key] = plot
        for k, v in self.pulses:
            plot.set_pulse(k, **v)


class Single:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def default_line_kwargs(self, type):
        return dict(pen='b')


class IntensityPhase(Single):
    Pair = namedtuple('Pair', ['intensity', 'phase'])

    def __init__(self, gl=None, row=None, col=None, **kwargs):
        """
        Args:
            gl (pyqtgraph GraphicsLayout): to which the plot is added. If
                None, creates one.
            row_0 (int): row
        """
        super().__init__(**kwargs)
        if gl is None:
            self.glw = pg.GraphicsLayoutWidget()
            self.glw.show()
            self.gl = self.glw.ci
        else:
            self.glw = None
            self.gl = gl
        if row is None:
            row = self.gl.currentRow
        if col is None:
            col = self.gl.currentCol
        self.row = row
        self.col = col
        self.lines = self.Pair({}, {})  # self.line_x_bounds={}

    def set_pulse(self, key, **kwargs):
        name = kwargs.pop('name', None)
        if key not in self.lines.intensity:
            # The lines are created as items and then added to the plot, rather than
            # using the PlotItem's addItem directly. With the latter method ther
            # are problems with the' _boundsCache' when using our dataBounds method.

            # Intensity
            line = pg.PlotCurveItem(**self.default_line_kwargs('intensity'), name=name)
            self.plots.intensity.addItem(line)
            # dataBounds used to be set to make autozoom show the main energy
            # of the pulse. But it causes drawing errors.
            # line.dataBounds=partial(self.get_line_data_bounds,key,line)
            self.lines.intensity[key] = line
            # Phase
            line = pg.PlotCurveItem(**self.default_line_kwargs('phase'), name=name)
            self.plots.phase.addItem(line)
            # This is necessary for Stacked plots because the bottom AxisItem
            # applies to the phase plot.
            # line.dataBounds=partial(self.get_line_data_bounds,key,line)
            self.lines.phase[key] = line
        elif name is not None:
            raise ValueError(
                'Can only specify name in first call to set_pulse for a key due to a limitation in pyqtgraph - hope to fix')
        ikwargs, pkwargs = self.process_pulse_kwargs(**kwargs)
        intensity_line = self.lines.intensity[key]
        intensity_line.setData(**ikwargs)
        self.lines.phase[key].setData(**pkwargs)

    def remove_pulse(self, key, allow_absent=False):
        if key in self.lines.intensity or not allow_absent:
            self.plots.intensity.removeItem(self.lines.intensity[key])
            del self.lines.intensity[key]
            self.plots.phase.removeItem(self.lines.phase[key])
            del self.lines.phase[key]  # self.line_x_bounds.pop(key,None)

    def clear_pulse(self, key):
        self.lines.intensity[key].clear()
        self.lines.phase[key].clear()

    def get_line_data_bounds(self, key, line, ax, frac=1.0, orthoRange=None):
        """Range occupied by the data.
        
        Uses percentiles of intensity integral for horizontal axis. For vertical
        axis, uses the PlotDataCurve method.
        
        Unfortunately this causes drawing errors, so is not used. Will leave code
        for now in case I come up with solution.
        
        See pyqtgraph's documentation for more information.
        
        Args:
            key (str): the key into self.lines.intensity/phase
            line (pg.PlotCurveItem)
            ax (int): 0 for x, 1 for y
            frac, orthoRange: ignored
        
        Returns:
            sequence of min, max
        """
        if ax == 0:
            # If x axis, use predetermined sensible intensity range
            try:
                return self.line_x_bounds[key]
            except:
                intensity_line = self.lines.intensity[key]
                x = intensity_line.xData
                y = intensity_line.yData
                if x is not None and len(x) == len(y) and len(x) > 0:
                    inty = cumtrapz(y, x, initial=0)
                    inty /= inty[-1]
                    x = interp1d(inty, x, copy=False, assume_sorted=True)([0.01, 0.99])
                    self.line_x_bounds[key] = x
                    return x
                return pg.PlotCurveItem.dataBounds(line, ax, frac, orthoRange)
        else:
            # If y axis, use default dataBounds
            return pg.PlotCurveItem.dataBounds(line, ax, frac, orthoRange)


class Temporal(IntensityPhase):
    default_bottom_label = 't (fs)'

    def process_pulse_kwargs(self, **kwargs):
        ikwargs = {}
        pkwargs = {}
        unused_keys = {}
        for k, v in kwargs.items():
            iv = v
            pv = v
            if k == 't':
                k = 'x'
            elif k == 'Et':
                k = 'y'
                iv = abs(v)**2
                pv = np.angle(v)/math.pi
            elif k == 'It':
                k = 'y'
                pv = None
            elif k == 'phit':
                k = 'y'
                pv = pv/math.pi
                iv = None
            if iv is not None:
                ikwargs[k] = iv
            if pv is not None:
                pkwargs[k] = pv

        return ikwargs, pkwargs


class Frequency(IntensityPhase):
    def __init__(self, domain='omega', **kwargs):
        if domain == 'omega':
            self.default_bottom_label = '&omega; (rad/fs)'
        elif domain == 'lambda':
            self.default_bottom_label = '&lambda; (nm)'
        self.domain = domain
        self.c = 2.998e2
        super().__init__(**kwargs)

    def process_pulse_kwargs(self, **kwargs):
        ikwargs = {}
        pkwargs = {}
        for k, v in kwargs.items():
            iv = v
            pv = v
            if k == 'omega':
                k = 'x'
                if self.domain == 'lambda':
                    v = 2*math.pi*self.c/v
                iv = v
                pv = v
            elif k == 'Ef':
                k = 'y'
                iv = abs(v)**2
                pv = np.angle(v)/math.pi
            elif k == 'If':
                k = 'y'
                pv = None
            elif k == 'phif':
                k = 'y'
                pv = pv/math.pi
                iv = None
            if iv is not None:
                ikwargs[k] = iv
            if pv is not None:
                pkwargs[k] = pv

        return ikwargs, pkwargs


class Stacked(IntensityPhase):
    def __init__(self, bottom_label=None, intensity_label='intensity', phase_label='phase (&pi; rad)', **kwargs):
        super().__init__(**kwargs)
        if bottom_label is None:
            bottom_label = self.default_bottom_label
        intensity = self.gl.addAlignedPlot(labels={'left': intensity_label}, row=self.row, col=self.col)
        intensity.hideAxis('bottom')
        self.gl.addVerticalSpacer(10, row=self.row + 4, col=self.col)
        phase = self.gl.addAlignedPlot(labels={'left': phase_label, 'bottom': bottom_label}, row=self.row + 5,
                                       col=self.col)
        phase.setXLink(intensity)
        self.plots = self.Pair(intensity, phase)
        # Set current position to 'end of line'
        self.gl.currentRow = self.row  # +8
        self.gl.currentCol = self.col + 4


class Overlaid(IntensityPhase):
    def __init__(self, bottom_label=None, intensity_label='intensity', phase_label='phase', pens=None, **kwargs):
        super().__init__(**kwargs)
        if bottom_label is None:
            bottom_label = self.default_bottom_label
        if pens is None:
            pens = {}
        pens.setdefault('intensity', 'b')
        pens.setdefault('phase', 'r')
        self.pens = pens
        intensity = self.gl.addAlignedPlot(labels={'left': intensity_label, 'bottom': bottom_label}, row=self.row,
                                           col=self.col)
        intensity.getAxis('left').setPen(self.pens['intensity'])
        phase = pg.add_right_axis(intensity, self.pens['phase'], label=phase_label)
        phase.setXLink(intensity)
        self.plots = self.Pair(intensity, phase)

    def default_line_kwargs(self, type):
        return {'pen': self.pens[type]}

    # def make_plots(self,x_axis,row_0,col_0,labels,pens={}):  #     """Called by setup_plots."""  #     intensity=self.gl.addAlignedPlot(labels={'left':intensity_label,'bottom':labels[x_axis]},row=row_0,col=col_0)  #     intensity.getAxis('left').setPen(self.pens['intensity'])  #     phase=pg.add_right_axis(intensity,self.pens['phase'],label=phase_label)  #     self.plots=self.Pair(intensity,phase)


class Stacked2D(Stacked):
    """Has a background image for spatial plots.
    
    Attributes:
        image_plots (IntensityPhase.Pair of plot items)
        images (IntensityPhase.Pair of ImageItems)
        image_annotations (IntensityPhase.Pair of dictionaries of QGraphicsItems)
    """

    def __init__(self, y_label='y', color_bar=True, symmetric_phase_colormap=False, luts=None, **kwargs):
        if luts is None:
            luts = {}
        super().__init__(**kwargs)
        intensity_plot = pg.add_right_axis(self.plots.intensity, 'k', y_label)
        phase_plot = pg.add_right_axis(self.plots.phase, 'k', y_label)
        phase_plot.setYLink(intensity_plot)
        self.image_plots = self.Pair(intensity_plot, phase_plot)
        intensity_image = pg.ImageItem(np.zeros((2, 2)), lut=luts.pop('intensity', pg.get_colormap_lut()))
        intensity_plot.addItem(intensity_image)
        phase_image = pg.ImageItem(np.zeros((2, 2)), lut=luts.pop('phase', pg.get_colormap_lut('bipolar')))
        phase_plot.addItem(phase_image)
        self.images = self.Pair(intensity_image, phase_image)
        self.symmetric_phase_colormap = symmetric_phase_colormap
        if color_bar:
            self.gl.addHorizontalSpacer(30, row=self.row + 2, col=self.col + 3)
            # labels on color bar unnecessary
            self.gl.addColorBar(image=intensity_image, row=self.row + 2, col=self.col + 4)  # label=intensity_label,
            self.gl.addColorBar(image=phase_image, row=self.row + 7, col=self.col + 4)  # label=phase_label,
        # Seems necessary for autorange at startup
        self.plots.intensity.enableAutoRange()
        self.gl.currentRow = self.row  # +8
        self.gl.currentCol = self.col + 5
        self.image_annotations = self.Pair({}, {})

    def set_pulse_2D(self, **kwargs):
        axes, intensity, phase = self.process_pulse_2D_kwargs(**kwargs)
        if intensity is not None:
            self.images.intensity.setImage(intensity.T, autoLevels=False)
        if phase is not None:
            kwargs = {}
            if self.symmetric_phase_colormap:
                mn = phase.min()
                mx = phase.max()
                rng = max(mn, mx)
                kwargs['levels'] = -rng, rng
            self.images.phase.setImage(phase.T, autoLevels=False, **kwargs)
        if axes is not None:
            rect = pg.axes_to_rect(*axes)
            self.images.intensity.setRect(rect)
            self.images.phase.setRect(rect)

    def add_image_annotation(self, type, key, item):
        d = getattr(self.image_annotations, type)
        assert isinstance(item, QtGui.QGraphicsItem)
        assert key not in d
        d[key] = item
        getattr(self.image_plots, type).addItem(item)

    def clear_2D(self):
        self.images.intensity.clear()
        self.images.phase.clear()
        for plot, annotations in zip(self.image_plots, self.image_annotations):
            for annotation in annotations:
                plot.removeItem(annotation)
            annotations.clear


class Frequency2D(Frequency):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image_axes = [None, None]

    def process_pulse_2D_kwargs(self, **kwargs):
        axes = [None, None]
        intensity = None
        phase = None
        for k, v in kwargs.items():
            if k == 'omega':
                axes[0] = v
            elif k == 'y':
                axes[1] = v
            elif k == 'Ef':
                intensity = abs(v)**2
                phase = np.angle(v)/math.pi
            elif k == 'If':
                intensity = v
            elif k == 'phif':
                phase = v/math.pi
        # If only one axis given, replace other with stored value
        if any(axis is not None for axis in axes):
            for n in range(2):
                if axes[n] is None:
                    axes[n] = self.image_axes[n]
                else:
                    self.image_axes[n] = axes[n]
            axes = [axis if axis is not None else image_axis for axis, image_axis in zip(axes, self.image_axes)]
        # Must have x and y
        if any(axis is None for axis in axes):
            axes = None
        return axes, intensity, phase

    def remove_pulse(self, key, allow_absent=False):
        if key in self.lines.intensity or not allow_absent:
            self.plots.intensity.removeItem(self.lines.intensity[key])
            del self.lines.intensity[key]
            self.plots.phase.removeItem(self.lines.phase[key])
            del self.lines.phase[key]


class Temporal2D(Temporal):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image_axes = [None, None]

    def process_pulse_2D_kwargs(self, **kwargs):
        axes = [None, None]
        intensity = None
        phase = None
        for k, v in kwargs.items():
            if k == 't':
                axes[0] = v
            elif k == 'y':
                axes[1] = v
            elif k == 'Et':
                intensity = abs(v)**2
                phase = np.angle(v)/math.pi
            elif k == 'It':
                intensity = v
            elif k == 'phit':
                phase = v/math.pi
        # If only one axis given, replace other with stored value
        if any(axis is not None for axis in axes):
            for n in range(2):
                if axes[n] is None:
                    axes[n] = self.image_axes[n]
                else:
                    self.image_axes[n] = axes[n]
            axes = [axis if axis is not None else image_axis for axis, image_axis in zip(axes, self.image_axes)]
        # Must have x and y
        if any(axis is None for axis in axes):
            axes = None
        return axes, intensity, phase


class StackedFrequency(Stacked, Frequency):
    pass  # def __init__(self,**kwargs):  #     super().__init__(**kwargs)


class OverlaidFrequency(Overlaid, Frequency):
    pass  # def __init__(self,**kwargs):  #     super().__init__(**kwargs)


class StackedTemporal(Stacked, Temporal):
    pass  # def __init__(self,**kwargs):  #     super().__init__(**kwargs)


class OverlaidTemporal(Overlaid, Temporal):
    pass  # def __init__(self,**kwargs):  #     super().__init__(**kwargs)


class StackedFrequency2D(Stacked2D, Frequency2D):
    pass


class StackedTemporal2D(Stacked2D, Temporal2D):
    pass


def time_and_freq_stacked(temp_kwargs=None, freq_kwargs=None, gl=None, legend=None):
    if gl is None:
        glw = pg.GraphicsLayoutWidget()
        gl = glw.ci
    row_0 = gl.currentRow
    col_0 = gl.currentCol
    mpp = Multi()
    mpp.add_plot('time', StackedTemporal(gl=gl, row=row_0, col=col_0, **temp_kwargs))
    mpp.add_plot('freq', StackedFrequency(gl=gl, row=row_0 + 9, col=col_0, **freq_kwargs))
    gl.currentRow = row_0 + 18
    gl.currentCol = col_0
    if legend is not None:
        domain = legend.pop('domain', 'time')
        type = legend.pop('type', 'intensity')
        leg_plot = getattr(mpp.plots[domain].plots, type)
        pg.addLegend(leg_plot, **legend)
    if 'glw' in locals():
        glw.show()
        mpp.glw = glw
    return mpp
