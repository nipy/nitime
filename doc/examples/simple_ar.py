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

npts = 2048*10
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

fig_noise = plot_tseries(ts.TimeSeries(v,sampling_rate=1000,time_unit='s'))
fig_noise.suptitle('noise')

"""

.. image:: fig/simple_ar_01.*

"""

fig_ar = plot_tseries(ts.TimeSeries(X,sampling_rate=1000,time_unit='s'))
fig_ar.suptitle('AR signal')

"""

.. image:: fig/simple_ar_02.*

Now we estimate back the model parameters:


"""

sigma_est, coefs_est = alg.yule_AR_est(X, 2, 2*npts, system=True)

plt.show()
