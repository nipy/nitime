"""This illustrates multi-taper spectral density function estimation.

References:

[1] D. Thomson, "Spectrum estimation and harmonic analysis," Proceedings of the
IEEE, vol. 70, 1982.

[2] D.J. Thomson, "Jackknifing Multitaper Spectrum Estimates [Identifying
variances of complicated estimation procedures]," IEEE Signal Processing
Magazine, 2007, pp. 20-30.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats.distributions as dist

import nitime.algorithms as alg
import nitime.utils as utils

def dB(x, out=None):
    if out is None:
        return 10 * np.log10(x)
    else:
        np.log10(x, out)
        np.multiply(out, 10, out)

def plot_estimate(ax, f, sdf_ests, limits=None, elabels=()):
    ax.plot(f, sdf, 'c', label='True S(f)')
    if not elabels:
        elabels = ('',) * len(sdf_ests)
    colors = 'bgkmy'
    for e, l, c in zip(sdf_ests, elabels, colors):
        ax.plot(f, e, color=c, linewidth=2, label=l)

    if limits is not None:
        ax.fill_between(f, limits[0], y2=limits[1], color=(1,0,0,.3))
    ax.set_ylim(ax_limits)
    ax.figure.set_size_inches([8,6])
    ax.legend()
    

### Log-to-dB conversion factor ###
ln2db = dB(np.e)

N = 512
import os
#Load AR process from file:
if os.path.exists('example_arrs.npz'):
    foo = np.load('example_arrs.npz')
    ar_seq = foo['arr_0']
    nz = foo['arr_1']
    alpha = foo['arr_2']
#Otherwise, generate AR process:
else:
    ar_seq, nz, alpha = utils.ar_generator(N=N, drop_transients=10)
    ar_seq -= ar_seq.mean()
    np.savez('example_arrs', ar_seq, nz, alpha)

# --- True SDF
fgrid, hz = alg.my_freqz(1.0, a=np.r_[1, -alpha], Nfreqs=N)
sdf = (hz*hz.conj()).real
# onesided spectrum, so double the power
sdf[1:-1] *= 2
dB(sdf, sdf)

# --- Direct Spectral Estimator
freqs, d_sdf = alg.periodogram(ar_seq)
dB(d_sdf, d_sdf)

#For plotting: 
ax_limits = 2*sdf.min(), 1.25*sdf.max()

f = plt.figure()
ax = f.add_subplot(1,1,1)
plot_estimate(ax, freqs, (d_sdf,), elabels=("Periodogram",))


# --- Welch's Overlapping Periodogram Method via mlab
welch_sdf, welch_freqs = plt.mlab.psd(ar_seq, NFFT=N)
welch_freqs *= (np.pi/welch_freqs.max())
welch_sdf = welch_sdf.squeeze()
dB(welch_sdf, welch_sdf)

f = plt.figure()
ax = f.add_subplot(1,1,1)
plot_estimate(ax, freqs, (welch_sdf,), elabels=("Welch",))

# --- Regular Multitaper Estimate
f, sdf_mt, nu = alg.multi_taper_psd(
    ar_seq, adaptive=False, jackknife=False
    )
dB(sdf_mt, sdf_mt)

# Get the number of tapers used from here
Kmax = nu[0]/2

# --- Hypothetical intervals with chi2(2Kmax) --------------------------------
# from Percival and Walden eq 258
p975 = dist.chi2.ppf(.975, 2*Kmax)
p025 = dist.chi2.ppf(.025, 2*Kmax)

l1 = ln2db * np.log(2*Kmax/p975)
l2 = ln2db * np.log(2*Kmax/p025)

hyp_limits = ( sdf_mt + l1, sdf_mt + l2 )

f = plt.figure()
ax = f.add_subplot(1,1,1)
plot_estimate(ax, freqs, (sdf_mt,), hyp_limits,
              elabels=('MT with hypothetical 5% interval',))

# --- Adaptively Weighted Multitapter Estimate
# -- Adaptive weighting from Thomson 1982, or Percival and Walden 1993
f, adaptive_sdf_mt, nu = alg.multi_taper_psd(
    ar_seq,  adaptive=True, jackknife=False
    )
dB(adaptive_sdf_mt, adaptive_sdf_mt)

# --- Jack-knifed intervals for regular weighting-----------------------------
# currently returns log-variance
_, _, jk_var = alg.multi_taper_psd(
    ar_seq, adaptive=False, jackknife=True
    )

# the Jackknife mean is approximately distributed about the true log-sdf
# as a Student's t distribution with variance jk_var ... but in
# fact the jackknifed variance better describes the normal
# multitaper estimator [Thomson2007]

# find 95% confidence limits from inverse of t-dist CDF
jk_p = (dist.t.ppf(.975, Kmax-1) * np.sqrt(jk_var)) * ln2db

jk_limits = ( sdf_mt - jk_p, sdf_mt + jk_p )


f = plt.figure()
ax = f.add_subplot(1,1,1)
plot_estimate(ax, freqs, (sdf_mt,),
              jk_limits,
              elabels=('MT with JK 5% interval',))


# --- Jack-knifed intervals for adaptive weighting----------------------------
_, _, adaptive_jk_var = alg.multi_taper_psd(
    ar_seq, adaptive=True, jackknife=True
    )

# find 95% confidence limits from inverse of t-dist CDF
jk_p = (dist.t.ppf(.975, Kmax-1)*np.sqrt(adaptive_jk_var)) * ln2db

adaptive_jk_limits = ( adaptive_sdf_mt - jk_p, adaptive_sdf_mt + jk_p )


f = plt.figure()
ax = f.add_subplot(1,1,1)
plot_estimate(ax, freqs, (adaptive_sdf_mt, ),
              adaptive_jk_limits,
              elabels=('(a)MT with JK 5% interval',))
