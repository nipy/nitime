import numpy as np
from nitime.lazy import scipy_stats as stats

from nitime import descriptors as desc
from nitime import algorithms as tsa
from nitime import timeseries as ts

from nitime.index_utils import tril_indices_from


from .base import BaseAnalyzer


def signal_noise(response):
    """
    Signal and noise as defined in Borst and Theunissen 1999, Figure 2

    Parameters
    ----------

    response: nitime TimeSeries object
       The data here are individual responses of a single unit to the same
       stimulus, with repetitions being the first dimension and time as the
       last dimension
    """

    signal = np.mean(response.data, 0)  # The estimate of the signal is the
                                       # average response

    noise = response.data - signal  # Noise is the individual
                               # repetition's deviation from the
                               # estimate of the signal

    # Return TimeSeries objects with the sampling rate of the input:
    return  (ts.TimeSeries(signal, sampling_rate=response.sampling_rate),
             ts.TimeSeries(noise, sampling_rate=response.sampling_rate))


class SNRAnalyzer(BaseAnalyzer):
    """
    Calculate SNR for a response to repetitions of the same stimulus, according
    to (Borst, 1999) (Figure 2) and (Hsu, 2004).

    Hsu A, Borst A and Theunissen, FE (2004) Quantifying variability in neural
    responses ans its application for the validation of model
    predictions. Network: Comput Neural Syst 15:91-109

    Borst A and Theunissen FE (1999) Information theory and neural coding. Nat
    Neurosci 2:947-957
    """
    def __init__(self, input=None, bandwidth=None, adaptive=False,
                 low_bias=False):
        """
        Initializer for the multi_taper_SNR object

        Parameters
        ----------
        input: TimeSeries object

        bandwidth: float,
           The bandwidth of the windowing function will determine the number
           tapers to use. This parameters represents trade-off between
           frequency resolution (lower main lobe bandwidth for the taper) and
           variance reduction (higher bandwidth and number of averaged
           estimates). Per default will be set to 4 times the fundamental
           frequency, such that NW=4

        adaptive: bool, default to False
            Whether to set the weights for the tapered spectra according to the
            adaptive algorithm (Thompson, 2007).

        low_bias : bool, default to False
            Rather than use 2NW tapers, only use the tapers that have better
            than 90% spectral concentration within the bandwidth (still using a
            maximum of 2NW tapers)

        Notes
        -----

        Thompson, DJ (2007) Jackknifing multitaper spectrum estimates. IEEE
        Signal Processing Magazing. 24: 20-30

        """
        self.input = input
        self.signal, self.noise = signal_noise(input)
        self.bandwidth = bandwidth
        self.adaptive = adaptive
        self.low_bias = low_bias

    @desc.setattr_on_read
    def mt_frequencies(self):
        return np.linspace(0, self.input.sampling_rate / 2,
                           self.input.data.shape[-1] // 2 + 1)

    @desc.setattr_on_read
    def mt_signal_psd(self):
        _, p, _ = tsa.multi_taper_psd(self.signal.data,
                                    Fs=self.input.sampling_rate,
                                    BW=self.bandwidth,
                                    adaptive=self.adaptive,
                                    low_bias=self.low_bias)
        return p

    @desc.setattr_on_read
    def mt_noise_psd(self):
        p = np.empty((self.noise.data.shape[0],
                     self.noise.data.shape[-1] // 2 + 1))

        for i in range(p.shape[0]):
            _, p[i], _ = tsa.multi_taper_psd(self.noise.data[i],
                                    Fs=self.input.sampling_rate,
                                    BW=self.bandwidth,
                                    adaptive=self.adaptive,
                                    low_bias=self.low_bias)
        return np.mean(p, 0)

    @desc.setattr_on_read
    def mt_coherence(self):
        """ """
        return self.mt_signal_psd / (self.mt_signal_psd + self.mt_noise_psd)

    @desc.setattr_on_read
    def mt_information(self):
        df = self.mt_frequencies[1] - self.mt_frequencies[0]
        return -1 * np.log2(1 - self.mt_coherence) * df
        #These two formulations should be equivalent
        #return np.log2(1+self.mt_snr)

    @desc.setattr_on_read
    def mt_snr(self):
        return self.mt_signal_psd / self.mt_noise_psd

    @desc.setattr_on_read
    def correlation(self):
        """
        The correlation between all combinations of trials

        Returns
        -------
        (r,e) : tuple
           r is the mean correlation and e is the mean error of the correlation
           (with df = n_trials - 1)
        """

        c = np.corrcoef(self.input.data)
        c = c[tril_indices_from(c, -1)]

        return np.mean(c), stats.sem(c)
