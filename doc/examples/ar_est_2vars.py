
"""

.. _mar:

=====================================
Mulitvariate auto-regressive modeling
=====================================


This example is based on Ding, Chen and Bressler 2006 [Ding2006]_. 


We start by importing the required libraries: 


"""

import numpy as np
import matplotlib.pyplot as pp


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

The example data from _[Ding2006] pg 18 (eq 55)
# X[t] = 0.9X[t-1] - 0.5X[t-2] + err_x
# Y[t] = 0.8Y[t-1] - 0.5Y[t-2] + 0.16X[t-1] - 0.2X[t-2] + err_y

"""

a1 = np.array([ [0.9, 0],
                [0.16, 0.8] ])

a2 = np.array([ [-0.5, 0],
                [-0.2, -0.5] ])

"""

Re-balance the equation to satisfy the relationship

.. math::

      Z[t] + sum_{i=1}^2 a[i]Z[t-i] = Err[t] ,

where $Z[t] = [X[t]$, $Y[t]]^t$ and $Err[t] ~ N(mu, cov=[ [x_var, xy_cov], [xy_cov, y_var] ])$




"""

am = np.array([ -a1, -a2 ])

x_var = 1
y_var = 0.7
xy_cov = 0.4
cov = np.array([ [x_var, xy_cov],
                 [xy_cov, y_var] ])


"""

Calculate the spectral matrix analytically ( z-transforms evaluated at
z=exp(j*omega) from omega in [0,pi] )


"""

n_freqs=1024

w, Hw = alg.transfer_function_xy(am, n_freqs=n_freqs)
Sw_true = alg.spectral_matrix_xy(Hw, cov)

"""

generate 500 sets of 100 points

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

# try to estimate the 2nd order (m)AR coefficients--
# average together N estimates of auto-covariance at lags k=0,1,2

# each Rxx(k) is shape (2,2), where
# Rxx_00(k) is E{ z0(t)z0*(t-k) }
# Rxx_01(k) is E{ z0(t)z1*(t-k) }
# Rxx_10(k) is E{ z1(t)z0*(t-k) }
# Rxx_11(k) is E{ z1(t)z1*(t-k) }
# So only Rxx(0) is symmetric

Rxx = np.empty((N,n_process,n_process,n_lags))

for i in xrange(N):
    Rxx[i] = utils.autocov_vector(z[i],nlags=n_lags)

Rxx = Rxx.mean(axis=0)

R0 = Rxx[...,0]
Rm = Rxx[...,1:]

Rxx = Rxx.transpose(2,0,1)

a, ecov = utils.lwr(Rxx)

print 'compare coefficients to estimate:'
print a - am
print 'compare covariance to estimate:'
print ecov - cov

w, f_x2y, f_y2x, f_xy, Sw = alg.granger_causality_xy(a,ecov,n_freqs=n_freqs)

f = pp.figure()
c_x = np.empty((L,w.shape[0]))
c_y = np.empty((L,w.shape[0]))

for i in xrange(N):
    frex,c_x[i],nu = alg.multi_taper_psd(z[i][0])
    frex,c_y[i],nu = alg.multi_taper_psd(z[i][1])

# power plot
ax = f.add_subplot(321)
# correct for one-sided spectral density functions
Sxx_true = Sw_true[0,0].real; Syy_true = Sw_true[1,1].real
Sxx_est = np.abs(Sw[0,0]); Syy_est = np.abs(Sw[1,1])
#ax.plot(w, Sxx_true, 'b', label='true Sxx(w)')
ax.plot(w, Sxx_est, 'b--', label='estimated Sxx(w)')
#ax.plot(w, Syy_true, 'g', label='true Syy(w)')
ax.plot(w, Syy_est, 'g--', label='estimated Syy(w)')

#scaler = np.mean(Sxx_est/np.mean(c_x,0))
ax.plot(w,np.mean(c_x,0),'r',label='Sxx(w) - MT PSD')
ax.plot(w,np.mean(c_y,0),'r--',label='Syy(w) - MT PSD')

ax.legend()

ax.set_title('power spectra')

# interdependence plot
ax = f.add_subplot(322)

f_id = alg.interdependence_xy(Sw)
ax.plot(w, f_id)
ax.set_title('interdependence')
ax.set_ylim([0,2.2])

# x causes y plot
ax = f.add_subplot(323)
ax.plot(w, f_x2y)
ax.set_title('g. causality X on Y')
ax.set_ylim([0,0.1])

# y causes x plot
ax = f.add_subplot(324)
ax.plot(w, f_y2x)
ax.set_title('g. causality Y on X')
ax.set_ylim([0,0.01])

# instantaneous causality
ax = f.add_subplot(325)
ax.plot(w, f_xy)
ax.set_title('instantaneous causality')
ax.set_ylim([0,2.2])

# total causality
ax = f.add_subplot(326)
ax.plot(w, f_xy + f_x2y + f_y2x)
ax.set_title('total causality')
ax.set_ylim([0,2.2])

"""

.. image:: fig/ar_est_2vars_01.png



"""


pp.show()

"""




.. [Ding2008] M. Ding, Y. Chen and S.L. Bressler (2006) Granger causality:
   basic theory and application to neuroscience. In Handbook of Time Series
   Analysis, ed. B. Schelter, M. Winterhalder, and J. Timmer, Wiley-VCH
   Verlage, 2006: 451-474



"""
