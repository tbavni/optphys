from math import pi

import mathx
from mathx import matseq
import numpy as np

from optphys import abcd, dispersion

##
lamb = 800e-9
w = 3e-3
f = 100e-3
q = abcd.Gaussian_wR_to_q(w, float('inf'), lamb)


def test_perfect_focusing():
    m = matseq.mult_mat_mat(abcd.propagation(f), abcd.thin_lens(f), abcd.propagation(f))
    qp = abcd.transform_Gaussian(m, q)
    wf, Rf = abcd.Gaussian_q_to_wR(qp, lamb)
    # Formula for focusing Gaussian beam
    wf_theor = f * lamb / (pi * w)
    assert np.isclose(wf, wf_theor)
    assert Rf == float('inf')


def test_lens_focusing():
    ml = abcd.thick_lens(dispersion.refractive_index(lamb, book='BK7'), float('inf'), 51.5e-3, 3.6e-3)
    fa = -1 / ml[1][0]
    m = matseq.mult_mat_mat(abcd.propagation(fa), ml, abcd.propagation(fa))
    qp = abcd.transform_Gaussian(m, q)
    wf, Rf = abcd.Gaussian_q_to_wR(qp, lamb)
    wf_theor = fa * lamb / (pi * w)
    assert np.isclose(wf, wf_theor)

# if __name__ == "__main__":
#     test_perfect_focusing()
#     test_lens_focusing()
