#!/usr/bin/env python
"""Simple example AR(p) fitting."""

import numpy as np
from matplotlib import pyplot as plt

from nitime import utils
from nitime import algorithms as alg
reload(utils)

npts = 2048*10
sigma = 1
drop_transients = 1024
coefs = np.array([0.9, -0.5])

# Generate AR(2) time series
X, v, _ = utils.ar_generator(npts, sigma, coefs, drop_transients)

# Visualize
plt.figure()
plt.plot(v)
plt.title('noise')
plt.figure()
plt.plot(X)
plt.title('AR signal')

# Estimate the model parameters
sigma_est, coefs_est = alg.yule_AR_est(X, 2, 2*npts, system=True)

print 'coefs    :', coefs
print 'coefs est:', coefs_est

plt.show()
