
"""

.. _mar:

=====================================
Mulitvariate auto-regressive modeling
=====================================


This example is based on Ding, Chen and Bressler 2006 [Ding2006]_.


We start by importing the required libraries:


"""

import numpy as np
import matplotlib.pyplot as plt


"""

From nitime, we import the algorithms and the utils:

"""

import nitime.algorithms as alg
import nitime.utils as utils


"""

Setting the random seed assures that we always get the same 'random' answer:

"""

np.random.seed(1981)

"""

We will generate an AR(2) model, with the following coefficients (taken from
[Ding2006]_, eq. 55):

.. math::
   :label:eqn_ar

   x_t & = & 0.9x_{t-1} - 0.5 x_{t-2} + \epsilon_t\\
   y_t & =& 0.8Y_{t-1} - 0.5 y_{t-2} + 0.16 x_{t-1} - 0.2 x_{t-2} + \eta_t

Or more succinctly, if we define:

.. math::

    Z_{t}=\left(\begin{array}{c}
    x_{t}\\
    y_{t}\end{array}\right),\,E_t=\left(\begin{array}{c}
    \epsilon_{t}\\
    \eta_{t}\end{array}\right)

then:

.. math::

  Z_t = A_1 Z_{t-1} + A_2 Z_{t-2} + E_t

where:

.. math::

   E_t \sim {\cal N} (\mu,\Sigma) \mathrm{, where} \,\, \Sigma=\left(\begin{array}{cc}var_{\epsilon} & cov_{xy}\\ cov_{xy} & var_{\eta}\end{array}\right)


We now build the two :math:`A_i` matrices with the values indicated above:

"""

a1 = np.array([[0.9, 0],
               [0.16, 0.8]])

a2 = np.array([[-0.5, 0],
               [-0.2, -0.5]])


"""

For implementation reasons, we rewrite the equation (:ref:`eqn_ar`) as follows:

.. math::

    Z_t + \sum_{i=1}^2 a_i Z_{t-i} = E_t

where: $a_i = - A_i$:

"""

am = np.array([-a1, -a2])


"""


The variances and covariance of the processes are known (provided as part of
the example in [Ding2006]_, after eq. 55):


"""

x_var = 1
y_var = 0.7
xy_cov = 0.4
cov = np.array([[x_var, xy_cov],
                [xy_cov, y_var]])


"""

We can calculate the spectral matrix analytically, based on the known
coefficients, for 1024 frequency bins:

"""

n_freqs = 1024

w, Hw = alg.transfer_function_xy(am, n_freqs=n_freqs)
Sw_true = alg.spectral_matrix_xy(Hw, cov)

"""

Next, we will generate 500 example sets of 100 points of these processes, to analyze:


"""

#Number of realizations of the process
N = 500
#Length of each realization:
L = 1024

order = am.shape[0]
n_lags = order + 1

n_process = am.shape[-1]

z = np.empty((N, n_process, L))
nz = np.empty((N, n_process, L))

for i in xrange(N):
    z[i], nz[i] = utils.generate_mar(am, cov, L)

"""

We can estimate the 2nd order AR coefficients, by averaging together N
estimates of auto-covariance at lags k=0,1,2

Each $R^{xx}(k)$ has the shape (2,2), where:

.. math::

   \begin{array}{ccc}
   R^{xx}_{00}(k) &=& E( Z_0(t)Z_0^*(t-k) )\\
   R^{xx}_{01}(k) &=& E( Z_0(t)Z_1^*(t-k) )\\
   R^{xx}_{10}(k) &=& E( Z_1(t)Z_0^*(t-k) )\\
   R^{xx}_{11}(k) &=& E( Z_1(t)Z_1^*(t-k) )\end{array}


Where $E$ is the expected value and $^*$ marks the conjugate transpose. Thus, only $R^{xx}(0)$ is symmetric.

This is calculated by using the function :func:`utils.autocov_vector`:

"""

Rxx = np.empty((N, n_process, n_process, n_lags))

for i in xrange(N):
    Rxx[i] = utils.autocov_vector(z[i], nlags=n_lags)

Rxx = Rxx.mean(axis=0)

R0 = Rxx[..., 0]
Rm = Rxx[..., 1:]

Rxx = Rxx.transpose(2, 0, 1)


"""

We use the Levinson-Whittle(-Wiggins) and Robinson algorithm, as described in
[Morf1978]_

"""

a, ecov = alg.lwr_recursion(Rxx)

"""

Calculate Granger causality:


"""

w, f_x2y, f_y2x, f_xy, Sw = alg.granger_causality_xy(a,
                                                     ecov,
                                                     n_freqs=n_freqs)

f = plt.figure()
c_x = np.empty((L, w.shape[0]))
c_y = np.empty((L, w.shape[0]))

"""

Plot the results:

"""

for i in xrange(N):
    frex, c_x[i], nu = alg.multi_taper_psd(z[i][0])
    frex, c_y[i], nu = alg.multi_taper_psd(z[i][1])

# power plot
ax = f.add_subplot(321)
# correct for one-sided spectral density functions
Sxx_true = Sw_true[0, 0].real
Syy_true = Sw_true[1, 1].real
Sxx_est = np.abs(Sw[0, 0])
Syy_est = np.abs(Sw[1, 1])

#ax.plot(w, Sxx_true, 'b', label='true Sxx(w)')
ax.plot(w, Sxx_est, 'b--', label='estimated Sxx(w)')
#ax.plot(w, Syy_true, 'g', label='true Syy(w)')
ax.plot(w, Syy_est, 'g--', label='estimated Syy(w)')

#scaler = np.mean(Sxx_est/np.mean(c_x,0))
ax.plot(w, np.mean(c_x, 0), 'r', label='Sxx(w) - MT PSD')
ax.plot(w, np.mean(c_y, 0), 'r--', label='Syy(w) - MT PSD')

ax.legend()

ax.set_title('power spectra')

# interdependence plot
ax = f.add_subplot(322)

f_id = alg.interdependence_xy(Sw)
ax.plot(w, f_id)
ax.set_title('interdependence')
ax.set_ylim([0, 2.2])

# x causes y plot
ax = f.add_subplot(323)
ax.plot(w, f_x2y)
ax.set_title('g. causality X on Y')
ax.set_ylim([0, 0.1])

# y causes x plot
ax = f.add_subplot(324)
ax.plot(w, f_y2x)
ax.set_title('g. causality Y on X')
ax.set_ylim([0, 0.01])

# instantaneous causality
ax = f.add_subplot(325)
ax.plot(w, f_xy)
ax.set_title('instantaneous causality')
ax.set_ylim([0, 2.2])

# total causality
ax = f.add_subplot(326)
ax.plot(w, f_xy + f_x2y + f_y2x)
ax.set_title('total causality')
ax.set_ylim([0, 2.2])

"""

.. image:: fig/ar_est_2vars_01.png


"""


plt.show()

"""




.. [Ding2008] M. Ding, Y. Chen and S.L. Bressler (2006) Granger causality:
   basic theory and application to neuroscience. In Handbook of Time Series
   Analysis, ed. B. Schelter, M. Winterhalder, and J. Timmer, Wiley-VCH
   Verlage, 2006: 451-474

.. [Morf1978] M. Morf, A. Vieira and T. Kailath (1978) Covariance
   Characterization by Partial Autocorrelation Matrices. The Annals of Statistics,
   6: 643-648


"""
