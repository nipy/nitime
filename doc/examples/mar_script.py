import numpy as np
import matplotlib.pyplot as pp

import nitime.mar.mar_xy_analysis as mar_xy_ana
import nitime.mar.mar_tools as mar_tools

np.random.seed(1981)

# example data from Ding, Chen, Bressler 2008 pg 18 (eq 55)
# X[t] = 0.9X[t-1] - 0.5X[t-2] + err_x
# Y[t] = 0.8Y[t-1] - 0.5Y[t-2] + 0.16X[t-1] - 0.2X[t-2] + err_y

a1 = np.array([ [0.9, 0],
                [0.16, 0.8] ])

a2 = np.array([ [-0.5, 0],
                [-0.2, -0.5] ])

# re-balance the equation to satisfying the relationship
# Z[t] + sum_{i=1}^2 a[i]Z[t-i] = Err[t],
# where Z[t] = [X[t], Y[t]]^t
# and Err[t] ~ N(mu, cov=[ [x_var, xy_cov], [xy_cov, y_var] ])

am = np.array([ -a1, -a2 ])

x_var = 1
y_var = 0.7
xy_cov = 0.4
cov = np.array([ [x_var, xy_cov],
                 [xy_cov, y_var] ])


# calculate the spectral matrix analytically ( z-transforms
# evaluated at z=exp(j*omega) from omega in [0,pi] )
w, Hw = mar_xy_ana.transfer_function_xy(am)
Sw_true = mar_xy_ana.spectral_matrix_xy(Hw, cov)

# generate 500 sets of 100 points
N = 500
L = 100

z = np.empty((N, 2, L))
nz = np.empty((N, 2, L))
for i in xrange(N):
    z[i], nz[i] = mar_tools.generate_mar(am, cov, L)


# try to estimate the 2nd order (m)AR coefficients--
# average together N estimates of auto-covariance at lags k=0,1,2

# each Rxx(k) is shape (2,2), where
# Rxx_00(k) is E{ z0(t)z0*(t-k) }
# Rxx_01(k) is E{ z0(t)z1*(t-k) }
# Rxx_10(k) is E{ z1(t)z0*(t-k) }
# Rxx_11(k) is E{ z1(t)z1*(t-k) }
# So only Rxx(0) is symmetric

Rxx = np.empty((500,2,2,3))
for i in xrange(N):
    Rxx[i] = mar_tools.autocov_vector(z[i], nlags=3)

Rxx = Rxx.mean(axis=0)

R0 = Rxx[...,0]
Rm = Rxx[...,1:]

Rxx = Rxx.transpose(2,0,1)

a, ecov = mar_tools.lwr(Rxx)

print a - am

w, f_x2y, f_y2x, f_xy, Sw = mar_xy_ana.granger_causality_xy(a, ecov)

f = pp.figure()

# power plot
ax = f.add_subplot(321)
# correct for one-sided spectral density functions
Sxx_true = 2*Sw_true[0,0].real; Syy_true = 2*Sw_true[1,1].real
Sxx_est = 2*Sw[0,0].real; Syy_est = 2*Sw[1,1].real
ax.plot(w, Sxx_true, 'b', label='true Sxx(w)')
ax.plot(w, Sxx_est, 'b--', label='estimated Sxx(w)')
ax.plot(w, Syy_true, 'g', label='true Syy(w)')
ax.plot(w, Syy_est, 'g--', label='estimated Syy(w)')
ax.legend()
ax.set_title('power spectra')

# interdependence plot
ax = f.add_subplot(322)

f_id = mar_xy_ana.interdependence_xy(Sw)
ax.plot(w, f_id)
ax.set_title('interdependence')

# x causes y plot
ax = f.add_subplot(323)

ax.plot(w, f_x2y)
ax.set_title('g. causality X on Y')

# y causes x plot
ax = f.add_subplot(324)

ax.plot(w, f_y2x)
ax.set_title('g. causality Y on X')

# instantaneous causality
ax = f.add_subplot(325)

ax.plot(w, f_xy)
ax.set_title('instantaneous causality')

# total causality
ax = f.add_subplot(326)

ax.plot(w, f_xy + f_x2y + f_y2x)
ax.set_title('total causality')

pp.show()
