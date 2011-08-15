""" .. _model_fit:

========================================
Fitting an MAR model: analyzer interface
========================================

In this example, we will use the Analyzer interface to fit a multi-variate
auto-regressive model with two time-series influencing each other.

We start by importing 3rd party modules:

"""

import numpy as np
import matplotlib.pyplot as plt

"""

And then by importing Granger analysis sub-module, which we will use for fitting the MAR
model:

"""

import nitime.analysis.granger as gc

"""

The utils sub-module includes a function to generate auto-regressive processes
based on provided coefficients:

"""

import nitime.utils as utils


"""

Generate some MAR processes (according to Ding and Bressler [Ding2006]_),

"""

a1 = np.array([[0.9, 0],
               [0.16, 0.8]])

a2 = np.array([[-0.5, 0],
               [-0.2, -0.5]])

am = np.array([-a1, -a2])

x_var = 1
y_var = 0.7
xy_cov = 0.4
cov = np.array([[x_var, xy_cov],
                [xy_cov, y_var]])


"""

Number of realizations of the process

"""

N = 500

"""

Length of each realization:

"""

L = 1024

order = am.shape[0]
n_lags = order + 1

n_process = am.shape[-1]

z = np.empty((N, n_process, L))
nz = np.empty((N, n_process, L))

np.random.seed(1981)
for i in xrange(N):
    z[i], nz[i] = utils.generate_mar(am, cov, L)


"""

We start by estimating the order of the model from the data:

"""

est_order = []
for i in xrange(N):
    this_order, this_Rxx, this_coef, this_ecov = gc.fit_model(z[i][0], z[i][1])
    est_order.append(this_order)

order = int(np.round(np.mean(est_order)))

"""

Once we have estimated the order, we  go ahead and fit each realization of the
MAR model, constraining the model order accordingly (by setting the order
key-word argument) to be always equal to the model order estimated above.

"""

Rxx = np.empty((N, n_process, n_process, n_lags))
coef = np.empty((N, n_process, n_process, order))
ecov = np.empty((N, n_process, n_process))

for i in xrange(N):
    this_order, this_Rxx, this_coef, this_ecov = gc.fit_model(z[i][0], z[i][1], order=order)
    Rxx[i] = this_Rxx
    coef[i] = this_coef
    ecov[i] = this_ecov

"""

We generate a time-series from the recovered coefficients, using the same
randomization seed as the first mar. These should look pretty similar to each other:

"""

np.random.seed(1981)
est_ts, _ = utils.generate_mar(np.mean(coef, axis=0), np.mean(ecov, axis=0), L)

fig01 = plt.figure()
ax = fig01.add_subplot(1, 1, 1)

ax.plot(est_ts[0][0:100])
ax.plot(z[0][0][0:100], 'g--')

"""

.. image:: fig/ar_model_fit_01.png


"""

plt.show()

"""

.. [Ding2006] M. Ding, Y. Chen and S.L. Bressler (2006) Granger causality:
   basic theory and application to neuroscience. In Handbook of Time Series
   Analysis, ed. B. Schelter, M. Winterhalder, and J. Timmer, Wiley-VCH
   Verlage, 2006: 451-474

"""
