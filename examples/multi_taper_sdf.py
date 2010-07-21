import numpy as np
import matplotlib.pyplot as pp
import scipy.stats.distributions as dist

import nitime.algorithms as alg
import nitime.utils as utils

def dB(x):
    return 10 * np.log10(x)

### Log-to-dB conversion factor ###
ln2db = dB(np.e)

N = 512
ar_seq, nz, alpha = utils.ar_generator(N=N, drop_transients=10)
# --- True SDF
fgrid, hz = alg.my_freqz(1.0, a=np.r_[1, -alpha], Nfreqs=N)
sdf = (hz*hz.conj()).real
# onesided spectrum, so double the power
sdf[1:-1] *= 2
sdf = dB( sdf )

# --- Direct Spectral Estimator
freqs, d_sdf = alg.periodogram(ar_seq)
d_sdf = dB(d_sdf)

# --- Welch's Overlapping Periodogram Method via mlab
mlab_sdf, mlab_freqs = pp.mlab.psd(ar_seq, NFFT=N)
mlab_sdf = dB(mlab_sdf.squeeze())
mlab_freqs *= (np.pi/mlab_freqs.max())


### Taper Bandwidth Adjustments
NW = 2; Kmax = int(2*NW)

tapers, v = alg.DPSS_windows(N, NW, Kmax)

freqs, p_sdfs = alg.periodogram( tapers * ar_seq, normalize=False )

# --- Regular Multitaper Estimate
sdf_mt = (p_sdfs * v[:,None]).sum(axis=0)
sdf_mt /= v.sum()
sdf_mt = dB(sdf_mt)

# --- Adaptively Weighted Multitapter Estimate
weights, nu_f = utils.adaptive_weights(p_sdfs, v, N)
adaptive_sdf_mt = (p_sdfs * weights**2).sum(axis=0)
adaptive_sdf_mt /= (weights**2).sum(axis=0)
adaptive_sdf_mt = dB(adaptive_sdf_mt)

# --- Jack-knifed intervals for regular weighting-----------------------------

# returns log-variance
jn_var, jn_mean = utils.jackknifed_sdf_variance(
    p_sdfs, np.sqrt(v[:,None])
    )
# convert sigma and mu to dB
jn_sigma_db = ln2db * np.sqrt(jn_var)
jn_mu_db = ln2db * jn_mean

# jn_mean is approximately distributed about the true log-sdf
# as a Student's t distribution with variance jn_var 
jn_p = dist.t.ppf(.975, Kmax-1) * jn_sigma_db

# the limits are only really valid around jn_mu_db
jn_limits = ( jn_mu_db - jn_p, jn_mu_db + jn_p )

# --- Jack-knifed intervals for adaptive weighting----------------------------

adaptive_jn_var, adaptive_jn_mean = utils.jackknifed_sdf_variance(
    p_sdfs, weights
    )
# convert sigma and mu to dB
jn_sigma_db = ln2db * np.sqrt(adaptive_jn_var)
adaptive_jn_mu_db = ln2db * adaptive_jn_mean

jn_p = dist.t.ppf(.975, Kmax-1) * jn_sigma_db

adaptive_jn_limits = ( adaptive_jn_mu_db - jn_p, adaptive_jn_mu_db + jn_p )

# --- Hypothetical intervals with chi2(2Kmax) --------------------------------

# from Percival and Walden eq 258
p975 = dist.chi2.ppf(.975, 2*Kmax)
p025 = dist.chi2.ppf(.025, 2*Kmax)

l1 = ln2db * np.log(2*Kmax/p975)
l2 = ln2db * np.log(2*Kmax/p025)

hyp_limits = ( sdf_mt + l1, sdf_mt + l2 )

# --- Hypothetical intervals with chi2(nu(f)) --------------------------------

p975 = dist.chi2.ppf(.975, nu_f)
p025 = dist.chi2.ppf(.025, nu_f)

l1 = ln2db * np.log(nu_f/p975)
l2 = ln2db * np.log(nu_f/p025)

adaptive_hyp_limits = ( adaptive_sdf_mt + l1, adaptive_sdf_mt + l2 )

# --- Plotting ---------------------------------------------------------------
ax_limits = 2*sdf.min(), 1.25*sdf.max()

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
    ax.legend()

f = pp.figure()
ax = f.add_subplot(611)
plot_estimate(ax, freqs, (d_sdf,), elabels=("Periodogram",))
ax = f.add_subplot(612)
plot_estimate(ax, mlab_freqs, (mlab_sdf,), elabels=("Welch's method",))
ax = f.add_subplot(613)
plot_estimate(ax, freqs, (sdf_mt,), hyp_limits,
              elabels=('MT with hypothetical 5% interval',))
ax = f.add_subplot(614)
plot_estimate(ax, freqs, (sdf_mt, jn_mu_db),
              jn_limits,
              elabels=('MT with JN 5% interval',
                       'JN MT with JN 5% interval'))
ax = f.add_subplot(615)
plot_estimate(ax, freqs, (adaptive_sdf_mt,),
              adaptive_hyp_limits,
              elabels=('(a)MT with hypothetical 5% interval',))
ax = f.add_subplot(616)
plot_estimate(ax, freqs, (adaptive_sdf_mt, adaptive_jn_mu_db),
              adaptive_jn_limits,
              elabels=('(a)MT with JN 5% interval',
                       'JN (a)MT with JN 5% interval'))
f.text(.5, .9, '%d Tapers'%Kmax)
pp.show()
