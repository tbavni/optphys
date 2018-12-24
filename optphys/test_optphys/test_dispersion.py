import numpy as np
from optphys.dispersion import *


def test_refractive_index():
    assert (abs(refractive_index(name='BBOo', lamb=800e-9) - 1.6614) < 1e-3)
    assert (abs(refractive_index(800e-9, name='BBOe') - 1.5462) < 1e-3)
    assert (abs(material_dispersion(0.5e-6, name='BBOo')*1e27 - 145.53) < 1e-2)
    assert (abs(material_dispersion(0.8e-6, book='SiO2')*1e27 - 36.1618) < 1e-3)


def test_optimal_propagation():
    omega = np.linspace(2, 2.7, 100)
    omega_0 = 2.35

    def test(phi_2, beta_2):
        Ef = np.exp(-((omega - omega_0)*20)**2 + 0.5j*phi_2*(omega - omega_0)**2)
        betaf = 0.5*beta_2*(omega - omega_0)**2
        assert np.isclose(optimal_propagation(omega, Ef, betaf, -10, 10), -phi_2/beta_2, rtol=1e-3)

    test(300, 150)
    test(-100, -50)
