"""
=================
Simple AR fitting
=================

This example demonstrates simple AR(p) fitting with the Yule Walker equations.

We start off with imports from numpy, matplotlib and import :mod:`nitime.utils` as
well as :mod:`nitime.algorithms:`

"""

import numpy as np
from matplotlib import pyplot as plt

from nitime import utils
from nitime import algorithms as alg


"""
We define some variables, which will be used in generating the AR process:
"""

npts = 2048 * 10
sigma = 1
drop_transients = 1024
coefs = np.array([0.9, -0.5])


"""

This generates an AR(2) time series:

"""

X, v, _ = utils.ar_generator(npts, sigma, coefs, drop_transients)


"""We use the plot_tseries function in order to visualize the process: """

import nitime.timeseries as ts
from nitime.viz import plot_tseries

fig_noise = plot_tseries(ts.TimeSeries(v, sampling_rate=1000, time_unit='s'))
fig_noise.suptitle('noise')

"""

.. image:: fig/simple_ar_01.png

"""

fig_ar = plot_tseries(ts.TimeSeries(X, sampling_rate=1000, time_unit='s'))
fig_ar.suptitle('AR signal')

"""

.. image:: fig/simple_ar_02.png

Now we estimate back the model parameters:


"""

coefs_est, sigma_est = alg.AR_est_YW(X, 2)
# no rigorous purpose behind 100 transients
X_hat, _, _ = utils.ar_generator(
    N=npts, sigma=sigma_est, coefs=coefs_est, drop_transients=100, v=v
    )
fig_ar_est = plt.figure()
ax = fig_ar_est.add_subplot(111)
ax.plot(np.arange(100, len(X_hat)+100), X_hat, label='estimated process')
ax.plot(X, 'g--', label='original process')
ax.legend()
err = X_hat - X[100:]
mse = np.dot(err, err)/len(X_hat)
ax.set_title('Mean Square Error: %1.3e'%mse)

"""

.. image:: fig/simple_ar_03.png

Reconstructed AR sequence based on estimated AR coefs


"""

plt.show()
