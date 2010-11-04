import numpy as np
import matplotlib.pyplot as plt
import scipy.stats.distributions as dist

import nitime.algorithms as alg
import nitime.utils as utils
import nitime.timeseries as ts
import nitime.viz as viz

def dB(x, out=None):
    if out is None:
        return 10 * np.log10(x)
    else:
        np.log10(x, out)
        np.multiply(out, 10, out)

def plot_estimate(f, sdf, sdf_ests, limits=None, elabels=()):

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax_limits = 2*sdf.min(), 1.25*sdf.max()
    ax.plot(f, sdf, 'r', label='True S(f)')

    if not elabels:
        elabels = ('',) * len(sdf_ests)
    colors = 'bgkmy'
    for e, l, c in zip(sdf_ests, elabels, colors):
        ax.plot(f, e, color=c, linewidth=2, label=l)

    if limits is not None:
        ax.fill_between(f, limits[0], y2=limits[1], color=(1,0,0,.3),
                        alpha=0.5)
        
    ax.set_ylim(ax_limits)
    ax.figure.set_size_inches([8,6])
    ax.legend()
    return fig

### Log-to-dB conversion factor ###
ln2db = dB(np.e)
##

#Generate a sequence with known spectral properties:
N = 30

ar_seq, nz, alpha = utils.ar_generator(N=N, drop_transients=10)
ar_seq -= ar_seq.mean()

# --- True SDF
fgrid, hz = alg.my_freqz(1.0, a=np.r_[1, -alpha], Nfreqs=N)
sdf = (hz*hz.conj()).real

# onesided spectrum, so double the power
sdf *= 2
dB(sdf, sdf)

freqs, d_sdf = alg.periodogram(ar_seq,Fs=1.0)
dB(d_sdf, d_sdf)

fig = plot_estimate(freqs, sdf, (d_sdf,), elabels=("Periodogram",))

n_trials = 12

sample = []
for idx,noise in enumerate([1,10,50,100]):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    #Make n_trials copies of the original sequence:
    sample.append(np.ones((n_trials,ar_seq.shape[-1]))+ar_seq)
    n_points = sample[-1].shape[-1]

    #Add noise:
    for trial in  xrange(n_trials):
        sample[-1][trial] += np.random.randn(sample[-1][trial].shape[0]) * noise
    sample_mean = np.mean(sample[-1],0)

    ax.plot(sample[-1].T)
    ax.plot(ar_seq,'b',linewidth=4)
    ax.plot(sample_mean,'r',linewidth=4)
    tseries = ts.TimeSeries(sample[-1],sampling_rate=1.)

    fig_snr = viz.plot_snr(tseries)

ts1 = ts.TimeSeries(sample[-1],sampling_rate=1.)
ts2 = ts.TimeSeries(sample[-2],sampling_rate=1.)
fig_compare = viz.plot_snr_diff(ts1,ts2)
                
    
