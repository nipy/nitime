import numpy as np
import matplotlib.pyplot as plt
import scipy.stats.distributions as dist

import nitime.algorithms as alg
import nitime.utils as utils

#nitime/doc/examples/spectral_examples_helper.py
from spectral_examples_helper import plot_estimate,dB,ln2db

#Generate a sequence with known spectral properties:
N = 512
ar_seq, nz, alpha = utils.ar_generator(N=N, drop_transients=10)
ar_seq -= ar_seq.mean()

# --- True SDF
fgrid, hz = alg.my_freqz(1.0, a=np.r_[1, -alpha], Nfreqs=N)
sdf = (hz*hz.conj()).real

# onesided spectrum, so double the power
sdf *= 2
dB(sdf, sdf)

# --- Direct Spectral Estimator
freqs, d_sdf = alg.periodogram(ar_seq)
dB(d_sdf, d_sdf)

plot_estimate(freqs, sdf, (d_sdf,), elabels=("Periodogram",))

# --- Welch's Overlapping Periodogram Method:
welch_freqs, welch_sdf = alg.get_spectra(ar_seq,
                                         method=dict(this_method='welch',NFFT=N))
welch_freqs *= (np.pi/welch_freqs.max())
welch_sdf = welch_sdf.squeeze()
dB(welch_sdf, welch_sdf)

plot_estimate(freqs, sdf, (welch_sdf,), elabels=("Welch",))

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

hyp_limits = (sdf_mt + l1, sdf_mt + l2 )

plot_estimate(freqs, sdf, (sdf_mt,), hyp_limits,
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


plot_estimate(freqs, sdf, (sdf_mt,),
              jk_limits,
              elabels=('MT with JK 5% interval',))


# --- Jack-knifed intervals for adaptive weighting----------------------------
_, _, adaptive_jk_var = alg.multi_taper_psd(
    ar_seq, adaptive=True, jackknife=True
    )

# find 95% confidence limits from inverse of t-dist CDF
jk_p = (dist.t.ppf(.975, Kmax-1)*np.sqrt(adaptive_jk_var)) * ln2db

adaptive_jk_limits = ( adaptive_sdf_mt - jk_p, adaptive_sdf_mt + jk_p )

plot_estimate(freqs, sdf,(adaptive_sdf_mt, ),
              adaptive_jk_limits,
              elabels=('adaptive-MT with JK 5% interval',))
