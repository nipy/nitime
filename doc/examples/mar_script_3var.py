import numpy as np
import matplotlib.pyplot as pp

import nitime.mar.mar_xy_analysis as mar_xy_ana
import nitime.mar.mar_tools as mar_tools
import nitime.algorithms as alg

np.random.seed(1981)

# simulate two mAR systems:

cov = np.diag( [0.3, 1.0, 0.2] )

# X(t) = 0.8X(t-1) - 0.5X(t-2) + 0.4Z(t-1) + Ex(t)
# Y(t) = 0.9Y(t-1) - 0.8Y(t-2) + Ey(t)
# Z(t) = 0.5Z(t-1) - 0.2Z(t-2) + 0.5Y(t-1) + Ez(t)


a1 = -np.array([ [0.8, 0.0, 0.4],
                 [0.0, 0.9, 0.0],
                 [0.0, 0.5, 0.5] ])
a2 = -np.array([ [-0.5, 0.0, 0.0],
                 [0.0, -0.8, 0.0],
                 [0.0, 0.0, -0.2] ])
a = np.array([a1.copy(), a2.copy()])

# X(t) = 0.8X(t-1) - 0.5X(t-2) + 0.4Z(t-1) + 0.2Y(t-2) + Ex(t)
# Y(t) = 0.9Y(t-1) - 0.8Y(t-2) + Ey(t)
# Z(t) = 0.5Z(t-1) -0.2Z(t-2) + 0.5Y(t-1) + Ez(t)
# just add some feedback from Y to X at 2 lags

a2[0,1] = -0.2

b = np.array([a1.copy(), a2.copy()])

def extract_ij(i, j, m):
    m_ij_rows = m[[i,j]]
    return m_ij_rows[:,[i,j]]

w, Haw = transfer_function(a)
w, Hbw = transfer_function(b)

# generate 500 sets of 100 points
N = 500
L = 100

za = np.empty((N, 3, L))
zb = np.empty((N, 3, L))
ea = np.empty((N, 3, L))
eb = np.empty((N, 3, L))
for i in xrange(N):
    za[i], ea[i] = mar_tools.generate_mar(a, cov, L)
    zb[i], eb[i] = mar_tools.generate_mar(b, cov, L)


# try to estimate the 2nd order (m)AR coefficients--
# average together N estimates of auto-covariance at lags k=0,1,2

Raxx = np.empty((N,3,3,3))
Rbxx = np.empty((N,3,3,3))

for i in xrange(N):
    Raxx[i] = mar_tools.autocov_vector(za[i], nlags=3)
    Rbxx[i] = mar_tools.autocov_vector(zb[i], nlags=3)

# average trials together to find autocovariance estimate,
# and extract pairwise components from the ac sequence
Raxx = Raxx.mean(axis=0)
xyRa = extract_ij(0, 1, Raxx)
xzRa = extract_ij(0, 2, Raxx)
yzRa = extract_ij(1, 2, Raxx)

Rbxx = Rbxx.mean(axis=0)
xyRb = extract_ij(0, 1, Rbxx)
xzRb = extract_ij(0, 2, Rbxx)
yzRb = extract_ij(1, 2, Rbxx)

# now estimate mAR coefficients and covariance from the full and
# pairwise relationships
Raxx = Raxx.transpose(2,0,1)
a_est, cov_est1 = mar_tools.lwr(Raxx)
a_xy_est, cov_xy_est1 = mar_tools.lwr(xyRa.transpose(2,0,1))
a_xz_est, cov_xz_est1 = mar_tools.lwr(xzRa.transpose(2,0,1))
a_yz_est, cov_yz_est1 = mar_tools.lwr(yzRa.transpose(2,0,1))

Rbxx = Rbxx.transpose(2,0,1)
b_est, cov_est2 = mar_tools.lwr(Rbxx)
b_xy_est, cov_xy_est2 = mar_tools.lwr(xyRb.transpose(2,0,1))
b_xz_est, cov_xz_est2 = mar_tools.lwr(xzRb.transpose(2,0,1))
b_yz_est, cov_yz_est2 = mar_tools.lwr(yzRb.transpose(2,0,1))

fig = pp.figure()

## w, x2y_a, y2x_a = pairwise_causality(0, 1, a_est, cov_est1)
## w, x2y_b, y2x_b = pairwise_causality(0, 1, b_est, cov_est2)
w, x2y_a, y2x_a, _, _ = mar_xy_ana.granger_causality_xy(a_xy_est, cov_xy_est1)
w, x2y_b, y2x_b, _, _ = mar_xy_ana.granger_causality_xy(b_xy_est, cov_xy_est2)
ax = fig.add_subplot(321)
ax.plot(w, x2y_a, 'b--')
ax.plot(w, x2y_b, 'b')
ax.set_title('x to y')
ax.set_ylim((0,6))
ax = fig.add_subplot(322)
ax.plot(w, y2x_a, 'b--')
ax.plot(w, y2x_b, 'b')
ax.set_title('y to x')
ax.set_ylim((0,6))

## w, y2z_a, z2y_a = pairwise_causality(1, 2, a_est, cov_est1)
## w, y2z_b, z2y_b = pairwise_causality(1, 2, b_est, cov_est2)
w, y2z_a, z2y_a, _, _ = mar_xy_ana.granger_causality_xy(a_yz_est, cov_yz_est1)
w, y2z_b, z2y_b, _, _ = mar_xy_ana.granger_causality_xy(b_yz_est, cov_yz_est2)
ax = fig.add_subplot(323)
ax.plot(w, y2z_a, 'b--')
ax.plot(w, y2z_b, 'b')
ax.set_title('y to z')
ax.set_ylim((0,6))
ax = fig.add_subplot(324)
ax.plot(w, z2y_a, 'b--')
ax.plot(w, z2y_b, 'b')
ax.set_title('z to y')
ax.set_ylim((0,6))

## w, x2z_a, z2x_a = pairwise_causality(0, 2, a_est, cov_est1)
## w, x2z_b, z2x_b = pairwise_causality(0, 2, b_est, cov_est2)
w, x2z_a, z2x_a, _, _ = mar_xy_ana.granger_causality_xy(a_xz_est, cov_xz_est1)
w, x2z_b, z2x_b, _, _ = mar_xy_ana.granger_causality_xy(b_xz_est, cov_xz_est2)
ax = fig.add_subplot(325)
ax.plot(w, x2z_a, 'b--')
ax.plot(w, x2z_b, 'b')
ax.set_title('x to z')
ax.set_ylim((0,6))
ax = fig.add_subplot(326)
ax.plot(w, z2x_a, 'b--')
ax.plot(w, z2x_b, 'b')
ax.set_title('z to x')
ax.set_ylim((0,6))

pp.show()
