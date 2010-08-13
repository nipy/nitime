"""
This diagnostic module evaluates two issues:

* the effect of re-calculating adaptive weights for spectra of tapered
  data in a jackknife process (as compared to jackknifing pre-computed weights)
* the difference in weight calculations when performed frequency-by-frequency
  in Cython, and when performed in a vectorized fashion (in Python).

Findings
--------

1. For data with 'colored' spectra, there is a smallish error between recomputed
   weights and jackknifed weights (rms error across K-1 values ~ 10**-1).

2. There appears to be little difference ( < 10**-7) between the weight
   calculation methods, and the vectorized method is 1-2 orders of magnitude
   faster.


"""
import numpy as np
import matplotlib.pyplot as pp
from matplotlib.mlab import csv2rec
import nitime.algorithms as alg
from nitime import utils

import scipy.stats.distributions as dist

def generate_subset_weights_and_err(spectra, eigvals):
    K, L = spectra.shape

    all_weights, _ = utils.adaptive_weights_cython(
        spectra, eigvals, L
        )

    # the K delete-one adaptive weightings
    jackknifed_weights = np.empty( (K, K-1, L) )
    # the K RMS errors between original w{k != i} and delete-one w{k != i}
    rms_err = np.empty( (K, L) )
    
    full_set = set(range(K))
    for i in full_set:
        sub = list( full_set.difference(set([i])) )
        
        ts_i = np.take(spectra, sub, axis=0)
        eigs_i = np.take(eigvals, sub)
        jackknifed_weights[i], _ = utils.adaptive_weights_cython(
            ts_i, eigs_i, L
            )
        orig_weights = np.take(all_weights, sub, axis=0)
        err = orig_weights - jackknifed_weights[i]
        rms_err[i] = (err**2).mean(axis=0)**0.5

    return rms_err, jackknifed_weights, all_weights

def compare_weight_methods(spectra, eigvals):
    L = spectra.shape[-1]
    fxf_weights, _ = utils.adaptive_weights_cython(spectra, eigvals, L)
    vec_weights, _ = utils.adaptive_weights(spectra, eigvals, L)
    err = np.abs(fxf_weights - vec_weights)
    return err

def plot_err(err, title):
    f = pp.figure()
    ax = f.add_subplot(111)
    ax.plot(err.T)
    ax.set_title(title)
    
NW = 4
K = 7
tapers, l = alg.DPSS_windows(256, NW, K)

# ---- ARTIFICIAL SPECTRA BASED ON CHI-2 WITH 2K DEGREES OF FREEDOM
artificial_tapered_spectra = dist.chi2.rvs(2*K, size=(K, 256))

rms_err_art, jk_w_art, w_art = generate_subset_weights_and_err(
    artificial_tapered_spectra, l
    )
w_err_art = compare_weight_methods(artificial_tapered_spectra, l)
plot_err(rms_err_art, 'chi2 re-weight err')
plot_err(w_err_art, 'chi2 weights err')
pp.show()

# ---- ACTUAL FMRI DATA
TR=1.89
data_rec = csv2rec('../doc/examples/data/fmri_timeseries.csv')
roi_names= np.array(data_rec.dtype.names)
nseq = len(roi_names)
n_samples = data_rec.shape[0]
data = np.zeros((nseq, n_samples))

for n_idx, roi in enumerate(roi_names):
   data[n_idx] = data_rec[roi]

tapers, l = alg.DPSS_windows(n_samples, NW, K)
pdata = utils.percent_change(data)
tdata = tapers[None,:,:] * pdata[:,None,:]

tspectra = np.fft.fft(tdata)
mag_sqr_spectra = np.abs(tspectra)
np.power(mag_sqr_spectra, 2, mag_sqr_spectra)

rms_err_fmri, _, _ = generate_subset_weights_and_err(
    mag_sqr_spectra[10], l
    )
w_err_fmri = compare_weight_methods(mag_sqr_spectra[10], l)
plot_err(rms_err_fmri, 'FMRI re-weight err')
plot_err(w_err_fmri, 'FMRI weights err')
pp.show()

# ---- AUTOREGRESSIVE SEQUENCE

ar_seq, nz, alpha = utils.ar_generator(N=n_samples, drop_transients=10)
ar_seq -= ar_seq.mean()

ar_spectra = np.abs(np.fft.fft(tapers * ar_seq))
np.power(ar_spectra, 2, ar_spectra)

rms_err_arseq, _, _ = generate_subset_weights_and_err(
    ar_spectra, l
    )
w_err_arseq = compare_weight_methods(ar_spectra, l)
plot_err(rms_err_arseq, 'AR re-weight err')
plot_err(w_err_arseq, 'AR weights err')
pp.show()



