#!/usr/bin/python

#Imports as before:
import numpy as np
import matplotlib.pyplot as pp
from matplotlib.mlab import csv2rec
from nitime.timeseries import TimeSeries
from nitime import utils
import nitime.algorithms as alg
import nitime.viz
reload(nitime.viz)
from nitime.viz import drawmatrix_channels
import scipy.stats.distributions as dist
#This time Import the coherence analyzer 
from nitime.analysis import CoherenceAnalyzer

#This part is the same as before
TR=1.89
data_rec = csv2rec('data/fmri_timeseries.csv')
roi_names= np.array(data_rec.dtype.names)
nseq = len(roi_names)
n_samples = data_rec.shape[0]
data = np.zeros((nseq, n_samples))

for n_idx, roi in enumerate(roi_names):
   data[n_idx] = data_rec[roi]

pdata = utils.percent_change(data)
T = TimeSeries(pdata,sampling_interval=TR)
T.metadata['roi'] = roi_names

NW = 5
K = 2*NW-1
tapers, eigs = alg.DPSS_windows(n_samples, NW, 2*NW-1)

tdata = tapers[None,:,:] * pdata[:,None,:]

tspectra = np.fft.fft(tdata)
mag_sqr_spectra = np.abs(tspectra)
np.power(mag_sqr_spectra, 2, mag_sqr_spectra)
# Only compute half the spectrum.. coherence for real sequences is symmetric
L = n_samples/2 + 1
#L = n_samples
w = np.empty( (nseq, K, L) )
for i in xrange(nseq):
   w[i], _ = utils.adaptive_weights_cython(mag_sqr_spectra[i], eigs, L)

# calculate the coherence
csd_mat = np.zeros((nseq, nseq, L), 'D')
psd_mat = np.zeros((2, nseq, nseq, L), 'd')
coh_mat = np.zeros((nseq, nseq, L), 'd')
coh_var = np.zeros_like(coh_mat)
for i in xrange(nseq):
   for j in xrange(i):
      sxy = alg.mtm_cross_spectrum(
         tspectra[i], tspectra[j], (w[i], w[j]), sides='onesided'
         )
      sxx = alg.mtm_cross_spectrum(
         tspectra[i], tspectra[i], (w[i], w[i]), sides='onesided'
         ).real
      syy = alg.mtm_cross_spectrum(
         tspectra[j], tspectra[j], (w[i], w[j]), sides='onesided'
         ).real
      psd_mat[0,i,j] = sxx
      psd_mat[1,i,j] = syy
      coh_mat[i,j] = np.abs(sxy)**2
      coh_mat[i,j] /= (sxx * syy)
      csd_mat[i,j] = sxy
      if i != j:
         coh_var[i,j] = utils.jackknifed_coh_variance(
            tspectra[i], tspectra[j], weights=(w[i], w[j]), last_freq=L
            )
upper_idc = utils.triu_indices(nseq, k=1)
lower_idc = utils.tril_indices(nseq, k=-1)
coh_mat[upper_idc] = coh_mat[lower_idc]
coh_var[upper_idc] = coh_var[lower_idc]

# convert this measure with the normalizing function
coh_mat_xform = utils.normalize_coherence(coh_mat, 2*K-2)

t025_limit = coh_mat_xform + dist.t.ppf(.025, K-1)*np.sqrt(coh_var)
t975_limit = coh_mat_xform + dist.t.ppf(.975, K-1)*np.sqrt(coh_var)


utils.normal_coherence_to_unit(t025_limit, 2*K-2, t025_limit)
utils.normal_coherence_to_unit(t975_limit, 2*K-2, t975_limit)

if L < n_samples:
   freqs = np.linspace(0, 1/(2*TR), L)
else:
   freqs = np.linspace(0, 1/TR, L, endpoint=False)


#We look only at frequencies between 0.02 and 0.15 (the physiologically
#relevant band, see http://imaging.mrc-cbu.cam.ac.uk/imaging/DesignEfficiency:
freq_idx = np.where((freqs>0.02) * (freqs<0.15))[0]

#Extract the coherence and average across these frequency bands: 
coh = np.mean(coh_mat[:,:,freq_idx],-1) #Averaging on the last dimension 
drawmatrix_channels(coh,roi_names,size=[10.,10.],color_anchor=0,
                    title='MTM Coherence')

C = CoherenceAnalyzer(T)

#We look only at frequencies between 0.02 and 0.15 (the physiologically
#relevant band, see http://imaging.mrc-cbu.cam.ac.uk/imaging/DesignEfficiency:
freq_idx = np.where((C.frequencies>0.02) * (C.frequencies<0.15))[0]

#Extract the coherence and average across these frequency bands: 
coh = np.mean(C.coherence[:,:,freq_idx],-1) #Averaging on the last dimension 
drawmatrix_channels(coh,roi_names,size=[10.,10.],color_anchor=0,
                    title='CoherenceAnalyzer')

pp.show()
