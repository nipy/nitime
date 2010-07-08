import numpy as np
import matplotlib.pyplot as pp
import scipy.stats.distributions as dist

import nitime.algorithms as alg
import nitime.utils as utils

def dB(x):
    return 10 * np.log10(x)

N = 512
ar_seq, nz, alpha = utils.ar_generator(N=N, drop_transients=10)
fgrid, hz = alg.my_freqz(1.0, a=np.r_[1, -alpha], Nfreqs=N)
sdf = dB( (hz*hz.conj()).real )

# --- Direct Spectral Estimator
freqs, d_sdf = alg.periodogram(ar_seq)
d_sdf = dB(d_sdf)

NW = 2.5; Kmax = int(2*NW)

tapers, v = alg.DPSS_windows(N, NW, Kmax)

freqs, p_sdfs = alg.periodogram( tapers * ar_seq, normalize=False )

# --- Regular Multitaper Estimate
sdf_mt = (p_sdfs * v[:,None]).sum(axis=0)
sdf_mt /= v.sum()
sdf_mt = dB(sdf_mt)

# --- Adaptively Weighted Multitapter Estimate
weights, nu_f = utils.adaptive_weights(p_sdfs, tapers, v, N)
adaptive_sdf_mt = (p_sdfs * weights**2).sum(axis=0)
adaptive_sdf_mt /= (weights**2).sum(axis=0)
adaptive_sdf_mt = dB(adaptive_sdf_mt)

### Log-to-dB conversion factor ###
ln2db = dB(np.e)

# --- Jack-knifed intervals --------------------------------------------------

jn_var = utils.jackknifed_sdf_variance(p_sdfs, weights=weights)
# convert sigma to dB (?? or var??)
jn_sigma_db = ln2db * np.sqrt(jn_var)

jn_p = dist.t.ppf(.975, Kmax-1) * jn_sigma_db

jn_limits = ( adaptive_sdf_mt - jn_p, adaptive_sdf_mt + jn_p )

# --- Hypothetical intervals with chi2(2Kmax) --------------------------------

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

ax_limits = 2*sdf.min(), 1.25*sdf.max()

def plot_estimate(ax, f, sdf_est, limits=None, elabel=''):
    ax.plot(f, sdf, 'c', label='True S(f)')
    ax.plot(f, sdf_est, 'b', linewidth=2, label=elabel)
    if limits is not None:
        ax.fill_between(f, limits[0], y2=limits[1], color=(1,0,0,.3))
    ax.set_ylim(ax_limits)
    ax.legend()

f = pp.figure()
ax = f.add_subplot(411)
plot_estimate(ax, freqs, d_sdf, elabel='Periodogram')
ax = f.add_subplot(412)
plot_estimate(ax, freqs, sdf_mt, hyp_limits,
              elabel='MT with hypothetical 5% interval')
ax = f.add_subplot(413)
plot_estimate(ax, freqs, adaptive_sdf_mt, adaptive_hyp_limits,
              elabel='(a)MT with hypothetical 5% interval')
ax = f.add_subplot(414)
plot_estimate(ax, freqs, adaptive_sdf_mt, jn_limits,
              elabel='(a)MT with jackknifed 5% interval')
f.text(.5, .9, '%d Tapers'%Kmax)
pp.show()
