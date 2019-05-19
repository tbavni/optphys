import numpy as np
from optphys import GaussianBeam

# todo make proper tests
w_0 = 1
b = GaussianBeam(1, w_0=w_0)

print(b.roc(-0.01), b.roc(0), b.roc(0.01), b.roc(1))
print(b.Gouy(0), b.Gouy(100))
print(b.waist(0), b.waist(1))
print(b.calc_all(0, 0), b.calc_all(0, 10e-6), b.calc_all(0.1, 0))

r = np.linspace(0, w_0 * 5, 100)
Dr = r[1] - r[0]
absE, phi = b.absEphi(0, r)
print(((abs(absE) ** 2 * r ** 2 * r).sum() / (abs(absE) ** 2 * r).sum()) ** 0.5)
