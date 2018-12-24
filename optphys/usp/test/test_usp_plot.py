import numpy as np
import pyqtgraph_extended as pg
from mathx import sft as fourier

from optphys.usp import plot as uspp

t = np.linspace(-200, 200, 1000)
omega_0 = 2.35
T0 = 20
w = 0.3
Et = np.exp(-0.5*(t/T0)**2 - 1j*omega_0*t)
omega = fourier.conj_axis(t, 0.5, 'start')
Ef = fourier.trans(t, 1, Et, omega)
y = np.linspace(-1, 1, 100)[:, None]
My = np.exp(-(y/w)**2)
Eyt = Et*My
Eyf = Ef*My


def test_StackedTemoral(qtbot):
    stp = uspp.StackedTemporal(bottom_label='TIME', intensity_label='INTENSITY', phase_label='PHASE (pi)')
    stp.set_pulse(0, t=t, Et=Et, pen='b')
    stp.glw.setWindowTitle('StackedTemporal')
    if qtbot is not None:
        qtbot.addWidget(stp.glw)
    return stp


def test_OverlaidFrequency(qtbot):
    stp = uspp.OverlaidFrequency(bottom_label='TIME', intensity_label='INTENSITY', phase_label='PHASE (pi)')
    stp.set_pulse(0, omega=omega, Ef=Ef, pen='r')
    stp.glw.setWindowTitle('OverlaidFrequency')

    stp2 = uspp.OverlaidFrequency(domain='lambda')
    stp2.set_pulse(0, omega=omega, Ef=Ef, pen='r')
    stp2.glw.setWindowTitle('OverlaidFrequency')
    if qtbot is not None:
        qtbot.addWidget(stp.glw)
    return stp


def test_Multi(qtbot):
    mpp = uspp.Multi()

    glw1 = pg.GraphicsLayoutWidget(window_title='Multi 1')
    gl = glw1.ci
    mpp.add_plot('temp', uspp.OverlaidTemporal(gl=gl, pens={'intensity': 'b', 'phase': 'r'}, bottom_label='t (cows)',
                                               intensity_label='intensity (fish)'))
    mpp.add_plot('freq', uspp.OverlaidFrequency(gl=gl, phase_label='phase (pi rad)'))
    glw1.show()

    glw2 = pg.GraphicsLayoutWidget(window_title='Multi 2')
    gl = glw2.ci
    mpp.add_plot('temp2', uspp.StackedTemporal(gl=gl, bottom_label='bottom'))
    # gl.currentCol=3
    # gl.currentRow=0
    mpp.add_plot('freq2', uspp.StackedFrequency(gl=gl))
    glw2.show()
    mpp.set_pulse(0, t=t, Et=Et, omega=omega, Ef=Ef)

    if qtbot is not None:
        qtbot.addWidget(glw1)
        qtbot.addWidget(glw2)
    return mpp, glw1, glw2


def test_time_and_freq_stacked(qtbot):
    plt = uspp.time_and_freq_stacked(temp_kwargs={'bottom_label': 'timelike', 'intensity_label': 'intensitylike'},
                                     freq_kwargs={'bottom_label': 'freqons', 'phase_label': 'anglelike'},
                                     legend={'domain': 'time', 'type': 'phase'})
    plt.set_pulse(0, t=t, Et=Et, omega=omega, Ef=Ef, name='the name', pen='b')
    if qtbot is not None:
        qtbot.addWidget(plt)
    return plt


def test_StackedTemporal2D(qtbot):
    st = uspp.StackedTemporal2D(y_label='y (pix)')
    st.set_pulse(0, t=t, Et=Et, pen='r')
    st.set_pulse_2D(t=t, y=y, Et=Eyt)
    if qtbot is not None:
        qtbot.addWidget(st.glw)
    return st


def test_StackedFrequency2D(qtbot):
    sf = uspp.StackedFrequency2D(y_label='y (pix)')
    sf.set_pulse(0, omega=omega, Ef=Ef, pen='r')
    sf.set_pulse_2D(omega=omega, y=y, Ef=Eyf)
    if qtbot is not None:
        qtbot.addWidget(sf.glw)
    return sf
