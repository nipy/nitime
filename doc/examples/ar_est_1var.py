"""

.. _ar:

=============================
Auto-regressive model fitting
=============================

Auto-regressive (AR) processes are processes that follow the following equation:

.. math::

   x_t = \sum_{i=1}^{n}a_i * x_{t-i} + \epsilon_t

In this example, we will demonstrate the estimation of the AR model coefficients and the
estimation of the AR process spectrum, based on the estimation of the
coefficients.

We start with imports from numpy, matplotlib and import :mod:`nitime.utils` as
well as :mod:`nitime.algorithms:`

"""

import numpy as np
from matplotlib import pyplot as plt

from nitime import utils
from nitime import algorithms as alg
from nitime.timeseries import TimeSeries
from nitime.viz import plot_tseries

"""

We define some variables, which will be used in generating the AR process:

"""

npts = 2048
sigma = 0.1
drop_transients = 128

"""

In this case, we generate an order 2 AR process, with the following coefficients:


"""


coefs = np.array([2.7607, -3.8106, 2.6535, -0.9238])


"""

This generates the AR(2) time series:

"""

X, noise, _ = utils.ar_generator(npts, sigma, coefs, drop_transients)

ts_x = TimeSeries(X,sampling_rate=1000,time_unit='s')
ts_noise = TimeSeries(noise,sampling_rate=1000,time_unit='s')

"""

We use the plot_tseries function in order to visualize the process:


"""

fig01 = plot_tseries(ts_x,label='AR signal')
fig01 = plot_tseries(ts_noise,fig=fig01,label='Noise')
fig01.axes[0].legend()

"""

.. image:: fig/ar_est_1var_01.*


Now we estimate back the model parameters, using two different estimation
algorithms.


"""
fig02
for order in [1,2,3,4]:
    sigma_est, coefs_est = alg.AR_est_YW(X, 2)
    plot

plt.show()
