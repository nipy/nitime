import warnings

import numpy as np
from nitime.lazy import scipy_stats_distributions as dist
from nitime.lazy import scipy_fftpack as fftpack

from nitime import descriptors as desc
from nitime import utils as tsu
from nitime import algorithms as tsa

# To support older versions of numpy that don't have tril_indices:
from nitime.index_utils import tril_indices, triu_indices

from .base import BaseAnalyzer


class CoherenceAnalyzer(BaseAnalyzer):
    """Analyzer object for coherence/coherency analysis """

    def __init__(self, input=None, method=None, unwrap_phases=False):
        """

        Parameters
        ----------

        input : TimeSeries object
           Containing the data to analyze.

        method : dict, optional,
            This is the method used for spectral analysis of the signal for the
            coherence caclulation. See :func:`algorithms.get_spectra`
            documentation for details.

        unwrap_phases : bool, optional
           Whether to unwrap the phases. This should be True if you assume that
           the time-delay is the same for all the frequency bands. See
           _[Sun2005] for details. Default : False

        Examples
        --------
        >>> import nitime.timeseries as ts
        >>> np.set_printoptions(precision=4)  # for doctesting
        >>> t1 = ts.TimeSeries(data = np.arange(0,1024,1).reshape(2,512),
        ...                                 sampling_rate=np.pi)
        >>> c1 = CoherenceAnalyzer(t1)
        >>> c1.method['Fs'] # doctest: +ELLIPSIS
        3.1415926535... Hz
        >>> c1.method['this_method']
        'welch'
        >>> c1.coherence[0,1]
        array([ 0.9024,  0.9027,  0.9652,  0.9433,  0.9297,  0.9213,  0.9161,
                0.9126,  0.9102,  0.9085,  0.9072,  0.9063,  0.9055,  0.905 ,
                0.9045,  0.9041,  0.9038,  0.9036,  0.9034,  0.9032,  0.9031,
                0.9029,  0.9028,  0.9027,  0.9027,  0.9026,  0.9026,  0.9025,
                0.9025,  0.9025,  0.9025,  0.9026,  1.    ])
        >>> c1.phase[0,1]
        array([ 0.    , -0.035 , -0.4839, -0.4073, -0.3373, -0.2828, -0.241 ,
               -0.2085, -0.1826, -0.1615, -0.144 , -0.1292, -0.1164, -0.1054,
               -0.0956, -0.0869, -0.0791, -0.072 , -0.0656, -0.0596, -0.0541,
               -0.0489, -0.0441, -0.0396, -0.0353, -0.0314, -0.0277, -0.0244,
               -0.0216, -0.0197, -0.0198, -0.028 ,  0.    ])

        """
        BaseAnalyzer.__init__(self, input)

        # Set the variables for spectral estimation (can also be entered by
        # user):
        if method is None:
            self.method = {'this_method': 'welch',
                           'Fs': self.input.sampling_rate}
        else:
            self.method = method

        # If an input is provided, get the sampling rate from there, if you
        # want to over-ride that, input a method with a 'Fs' field specified:
        self.method['Fs'] = self.method.get('Fs', self.input.sampling_rate)

        self._unwrap_phases = unwrap_phases

        # The following only applies to the welch method:
        if (self.method.get('this_method') == 'welch' or
            self.method.get('this_method') is None):

            # If the input is shorter than NFFT, all the coherences will be
            # 1 per definition. Throw a warning about that:
            self.method['NFFT'] = self.method.get('NFFT', tsa.default_nfft)
            self.method['n_overlap'] = self.method.get('n_overlap',
                                                       tsa.default_n_overlap)
            if (self.input.shape[-1] <
                            (self.method['NFFT'] + self.method['n_overlap'])):
                e_s = "In nitime.analysis, the provided input time-series is"
                e_s += " shorter than the requested NFFT + n_overlap. All "
                e_s += "coherence values will be set to 1."
                warnings.warn(e_s, RuntimeWarning)

    @desc.setattr_on_read
    def coherency(self):
        """The standard output for this kind of analyzer is the coherency """
        data = self.input.data
        tseries_length = data.shape[0]
        spectrum_length = self.spectrum.shape[-1]

        coherency = np.zeros((tseries_length,
                              tseries_length,
                              spectrum_length), dtype=complex)

        for i in range(tseries_length):
            for j in range(i, tseries_length):
                coherency[i][j] = tsa.coherency_spec(self.spectrum[i][j],
                                                     self.spectrum[i][i],
                                                     self.spectrum[j][j])

        idx = tril_indices(tseries_length, -1)
        coherency[idx[0], idx[1], ...] = coherency[idx[1], idx[0], ...].conj()

        return coherency

    @desc.setattr_on_read
    def spectrum(self):
        """
        The spectra of each of the channels and cross-spectra between
        different channels  in the input TimeSeries object
        """
        f, spectrum = tsa.get_spectra(self.input.data, method=self.method)
        return spectrum

    @desc.setattr_on_read
    def frequencies(self):
        """
        The central frequencies in the bands
        """

        #XXX Use NFFT in the method in order to calculate these, without having
        #to calculate the spectrum:
        f, spectrum = tsa.get_spectra(self.input.data, method=self.method)
        return f

    @desc.setattr_on_read
    def coherence(self):
        """
        The coherence between the different channels in the input TimeSeries
        object
        """

        #XXX Calculate this from the standard output, instead of recalculating
        #the coherence:

        tseries_length = self.input.data.shape[0]
        spectrum_length = self.spectrum.shape[-1]
        coherence = np.zeros((tseries_length,
                              tseries_length,
                              spectrum_length))

        for i in range(tseries_length):
            for j in range(i, tseries_length):
                coherence[i][j] = tsa.coherence_spec(self.spectrum[i][j],
                                                     self.spectrum[i][i],
                                                     self.spectrum[j][j])

        idx = tril_indices(tseries_length, -1)
        coherence[idx[0], idx[1], ...] = coherence[idx[1], idx[0], ...].conj()

        return coherence

    @desc.setattr_on_read
    def phase(self):
        """ The frequency-dependent phase relationship between all the pairwise
        combinations of time-series in the data"""

        #XXX calcluate this from the standard output, instead of recalculating:

        tseries_length = self.input.data.shape[0]
        spectrum_length = self.spectrum.shape[-1]

        phase = np.zeros((tseries_length,
                            tseries_length,
                            spectrum_length))

        for i in range(tseries_length):
            for j in range(i, tseries_length):
                phase[i][j] = np.angle(
                    self.spectrum[i][j])

                phase[j][i] = np.angle(
                    self.spectrum[i][j].conjugate())
        return phase

    @desc.setattr_on_read
    def delay(self):
        """ The delay in seconds between the two time series """
        p_shape = self.phase.shape[:-1]
        delay = np.zeros(self.phase.shape)
        for i in range(p_shape[0]):
            for j in range(p_shape[1]):
                this_phase = self.phase[i, j]
                #If requested, unwrap the phases:
                if self._unwrap_phases:
                    this_phase = tsu.unwrap_phases(this_phase)

                delay[i, j] = this_phase / (2 * np.pi * self.frequencies)

        return delay

    @desc.setattr_on_read
    def coherence_partial(self):
        """The partial coherence between data[i] and data[j], given data[k], as
        a function of frequency band"""

        tseries_length = self.input.data.shape[0]
        spectrum_length = self.spectrum.shape[-1]

        p_coherence = np.zeros((tseries_length,
                                tseries_length,
                                tseries_length,
                                spectrum_length))

        for i in range(tseries_length):
            for j in range(tseries_length):
                for k in range(tseries_length):
                    if j == k or i == k:
                        pass
                    else:
                        p_coherence[i][j][k] = tsa.coherence_partial_spec(
                            self.spectrum[i][j],
                            self.spectrum[i][i],
                            self.spectrum[j][j],
                            self.spectrum[i][k],
                            self.spectrum[j][k],
                            self.spectrum[k][k])

        idx = tril_indices(tseries_length, -1)
        p_coherence[idx[0], idx[1], ...] =\
                            p_coherence[idx[1], idx[0], ...].conj()

        return p_coherence


class MTCoherenceAnalyzer(BaseAnalyzer):
    """ Analyzer for multi-taper coherence analysis, including jack-knife
    estimate of confidence interval """
    def __init__(self, input=None, bandwidth=None, alpha=0.05, adaptive=True):

        """
        Initializer function for the MTCoherenceAnalyzer

        Parameters
        ----------

        input : TimeSeries object

        bandwidth : float,
           The bandwidth of the windowing function will determine the number
           tapers to use. This parameters represents trade-off between
           frequency resolution (lower main lobe bandwidth for the taper) and
           variance reduction (higher bandwidth and number of averaged
           estimates). Per default will be set to 4 times the fundamental
           frequency, such that NW=4

        alpha : float, default =0.05
            This is the alpha used to construct a confidence interval around
            the multi-taper csd estimate, based on a jack-knife estimate of the
            variance [Thompson2007]_.

        adaptive : bool, default to True
            Whether to set the weights for the tapered spectra according to the
            adaptive algorithm (Thompson, 2007).

        Notes
        -----

        Thompson, DJ (2007) Jackknifing multitaper spectrum estimates. IEEE
        Signal Processing Magazing. 24: 20-30

        """

        BaseAnalyzer.__init__(self, input)

        if input is None:
            self.NW = 4
            self.bandwidth = None
        else:
            N = input.shape[-1]
            Fs = self.input.sampling_rate
            if bandwidth is not None:
                self.NW = bandwidth / (2 * Fs) * N
            else:
                self.NW = 4
                self.bandwidth = self.NW * (2 * Fs) / N

        self.alpha = alpha
        self._L = self.input.data.shape[-1] // 2 + 1
        self._adaptive = adaptive

    @desc.setattr_on_read
    def tapers(self):
        return tsa.dpss_windows(self.input.shape[-1], self.NW,
                                2 * self.NW - 1)[0]

    @desc.setattr_on_read
    def eigs(self):
        return tsa.dpss_windows(self.input.shape[-1], self.NW,
                                      2 * self.NW - 1)[1]

    @desc.setattr_on_read
    def df(self):
        # The degrees of freedom:
        return 2 * self.NW - 1

    @desc.setattr_on_read
    def spectra(self):
        tdata = self.tapers[None, :, :] * self.input.data[:, None, :]
        tspectra = fftpack.fft(tdata)
        return tspectra

    @desc.setattr_on_read
    def weights(self):
        channel_n = self.input.data.shape[0]
        w = np.empty((channel_n, self.df, self._L))

        if self._adaptive:
            for i in range(channel_n):
                # this is always a one-sided spectrum?
                w[i] = tsu.adaptive_weights(self.spectra[i],
                                            self.eigs,
                                            sides='onesided')[0]

        # Set the weights to be the square root of the eigen-values:
        else:
            wshape = [1] * len(self.spectra.shape)
            wshape[0] = channel_n
            wshape[-2] = int(self.df)
            pre_w = np.sqrt(self.eigs) + np.zeros((wshape[0],
                                                    self.eigs.shape[0]))

            w = pre_w.reshape(*wshape)

        return w

    @desc.setattr_on_read
    def coherence(self):
        nrows = self.input.data.shape[0]
        psd_mat = np.zeros((2, nrows, nrows, self._L), 'd')
        coh_mat = np.zeros((nrows, nrows, self._L), 'd')

        for i in range(self.input.data.shape[0]):
            for j in range(i):
                sxy = tsa.mtm_cross_spectrum(self.spectra[i], self.spectra[j],
                                           (self.weights[i], self.weights[j]),
                                           sides='onesided')
                sxx = tsa.mtm_cross_spectrum(self.spectra[i], self.spectra[i],
                                             self.weights[i],
                                             sides='onesided')
                syy = tsa.mtm_cross_spectrum(self.spectra[j], self.spectra[j],
                                             self.weights[i],
                                             sides='onesided')
                psd_mat[0, i, j] = sxx
                psd_mat[1, i, j] = syy
                coh_mat[i, j] = np.abs(sxy) ** 2
                coh_mat[i, j] /= (sxx * syy)

        idx = triu_indices(self.input.data.shape[0], 1)
        coh_mat[idx[0], idx[1], ...] = coh_mat[idx[1], idx[0], ...].conj()

        return coh_mat

    @desc.setattr_on_read
    def confidence_interval(self):
        """The size of the 1-alpha confidence interval"""
        coh_var = np.zeros((self.input.data.shape[0],
                            self.input.data.shape[0],
                            self._L), 'd')
        for i in range(self.input.data.shape[0]):
            for j in range(i):
                if i != j:
                    coh_var[i, j] = tsu.jackknifed_coh_variance(
                        self.spectra[i],
                        self.spectra[j],
                        self.eigs,
                        adaptive=self._adaptive
                        )

        idx = triu_indices(self.input.data.shape[0], 1)
        coh_var[idx[0], idx[1], ...] = coh_var[idx[1], idx[0], ...].conj()

        coh_mat_xform = tsu.normalize_coherence(self.coherence,
                                                2 * self.df - 2)

        lb = coh_mat_xform + dist.t.ppf(self.alpha / 2,
                                        self.df - 1) * np.sqrt(coh_var)
        ub = coh_mat_xform + dist.t.ppf(1 - self.alpha / 2,
                                        self.df - 1) * np.sqrt(coh_var)

        # convert this measure with the normalizing function
        tsu.normal_coherence_to_unit(lb, 2 * self.df - 2, lb)
        tsu.normal_coherence_to_unit(ub, 2 * self.df - 2, ub)

        return ub - lb

    @desc.setattr_on_read
    def frequencies(self):
        return np.linspace(0, self.input.sampling_rate / 2, self._L)


class SparseCoherenceAnalyzer(BaseAnalyzer):
    """
    This analyzer is intended for analysis of large sets of data, in which
    possibly only a subset of combinations of time-series needs to be compared.
    The constructor for this class receives as input not only a time-series
    object, but also a list of tuples with index combinations (i,j) for the
    combinations. Importantly, this class implements only the mlab csd function
    and cannot use other methods of spectral estimation
    """

    def __init__(self, time_series=None, ij=(0, 0), method=None, lb=0, ub=None,
                 prefer_speed_over_memory=True, scale_by_freq=True):
        """The constructor for the SparseCoherenceAnalyzer

        Parameters
        ----------

        time_series : a time-series object

        ij : a list of tuples, each containing a pair of indices.
           The resulting cache will contain the fft of time-series in the rows
           indexed by the unique elements of the union of i and j

        lb,ub : float,optional, default: lb=0, ub=None (max frequency)

            define a frequency band of interest

        prefer_speed_over_memory: Boolean, optional, default=True

            Does exactly what the name implies. If you have enough memory

        method : optional, dict
             The method for spectral estimation (see
             :func:`algorithms.get_spectra`)

        """

        BaseAnalyzer.__init__(self, time_series)
        #Initialize variables from the time series
        self.ij = ij

        #Set the variables for spectral estimation (can also be entered by
        #user):
        if method is None:
            self.method = {'this_method': 'welch'}

        else:
            self.method = method

        if self.method['this_method'] != 'welch':
            e_s = "For SparseCoherenceAnalyzer, "
            e_s += "spectral estimation method must be welch"
            raise ValueError(e_s)

        self.method['Fs'] = self.method.get('Fs', self.input.sampling_rate)

        #Additional parameters for the coherency estimation:
        self.lb = lb
        self.ub = ub
        self.prefer_speed_over_memory = prefer_speed_over_memory
        self.scale_by_freq = scale_by_freq

    @desc.setattr_on_read
    def coherency(self):
        """ The default behavior is to calculate the cache, extract it and then
        output the coherency"""
        coherency = tsa.cache_to_coherency(self.cache, self.ij)

        return coherency

    @desc.setattr_on_read
    def coherence(self):
        """ The coherence values for the output"""
        coherence = np.abs(self.coherency ** 2)

        return coherence

    @desc.setattr_on_read
    def cache(self):
        """Caches the fft windows required by the other methods of the
        SparseCoherenceAnalyzer. Calculate only once and reuse
        """
        data = self.input.data
        f, cache = tsa.cache_fft(data,
                                self.ij,
                                lb=self.lb,
                                ub=self.ub,
                                method=self.method,
                        prefer_speed_over_memory=self.prefer_speed_over_memory,
                                scale_by_freq=self.scale_by_freq)

        return cache

    @desc.setattr_on_read
    def spectrum(self):
        """get the spectrum for the collection of time-series in this analyzer
        """
        spectrum = tsa.cache_to_psd(self.cache, self.ij)

        return spectrum

    @desc.setattr_on_read
    def phases(self):
        """The frequency-band dependent phases of the spectra of each of the
           time -series i,j in the analyzer"""

        phase = tsa.cache_to_phase(self.cache, self.ij)

        return phase

    @desc.setattr_on_read
    def relative_phases(self):
        """The frequency-band dependent relative phase between the two
        time-series """
        return np.angle(self.coherency)

    @desc.setattr_on_read
    def delay(self):
        """ The delay in seconds between the two time series """
        return self.relative_phases / (2 * np.pi * self.frequencies)

    @desc.setattr_on_read
    def frequencies(self):
        """Get the central frequencies for the frequency bands, given the
           method of estimating the spectrum """

        self.method['Fs'] = self.method.get('Fs', self.input.sampling_rate)
        NFFT = self.method.get('NFFT', 64)
        Fs = self.method.get('Fs')
        freqs = tsu.get_freqs(Fs, NFFT)
        lb_idx, ub_idx = tsu.get_bounds(freqs, self.lb, self.ub)

        return freqs[lb_idx:ub_idx]


class SeedCoherenceAnalyzer(object):
    """
    This analyzer takes two time-series. The first is designated as a
    time-series of seeds. The other is designated as a time-series of targets.
    The analyzer performs a coherence analysis between each of the channels in
    the seed time-series and *all* of the channels in the target time-series.

    Note
    ----

    This is a convenience class, which provides a convenient-to-use interface
    to the SparseCoherenceAnalyzer

    """

    def __init__(self, seed_time_series=None, target_time_series=None,
                 method=None, lb=0, ub=None, prefer_speed_over_memory=True,
                 scale_by_freq=True):

        """

        The constructor for the SeedCoherenceAnalyzer

        Parameters
        ----------

        seed_time_series: a time-series object

        target_time_series: a time-series object

        lb,ub: float,optional, default: lb=0, ub=None (max frequency)

            define a frequency band of interest

        prefer_speed_over_memory: Boolean, optional, default=True

            Makes things go a bit faster, if you have enough memory


        """

        self.seed = seed_time_series
        self.target = target_time_series

        # Check that the seed and the target have the same sampling rate:
        if self.seed.sampling_rate != self.target.sampling_rate:
            e_s = "The sampling rate for the seed time-series and the target"
            e_s += " time-series need to be identical."
            raise ValueError(e_s)

        #Set the variables for spectral estimation (can also be entered by
        #user):
        if method is None:
            self.method = {'this_method': 'welch'}

        else:
            self.method = method

        if ('this_method' in self.method.keys() and
            self.method['this_method'] != 'welch'):
            e_s = "For SeedCoherenceAnalyzer, "
            e_s += "spectral estimation method must be welch"
            raise ValueError(e_s)

        #Additional parameters for the coherency estimation:
        self.lb = lb
        self.ub = ub
        self.prefer_speed_over_memory = prefer_speed_over_memory
        self.scale_by_freq = scale_by_freq

    @desc.setattr_on_read
    def coherence(self):
        """
        The coherence between each of the channels of the seed time series and
        all the channels of the target time-series.

        """
        return np.abs(self.coherency) ** 2

    @desc.setattr_on_read
    def frequencies(self):
        """Get the central frequencies for the frequency bands, given the
           method of estimating the spectrum """

        # Get the sampling rate from the seed time-series:
        self.method['Fs'] = self.method.get('Fs', self.seed.sampling_rate)
        NFFT = self.method.get('NFFT', 64)
        Fs = self.method.get('Fs')
        freqs = tsu.get_freqs(Fs, NFFT)
        lb_idx, ub_idx = tsu.get_bounds(freqs, self.lb, self.ub)

        return freqs[lb_idx:ub_idx]

    @desc.setattr_on_read
    def target_cache(self):
        data = self.target.data

        #Make a cache with all the fft windows for each of the channels in the
        #target.

        #This is the kind of input that cache_fft expects:
        ij = list(zip(np.arange(data.shape[0]), np.arange(data.shape[0])))

        f, cache = tsa.cache_fft(data, ij, lb=self.lb, ub=self.ub,
                                 method=self.method,
                        prefer_speed_over_memory=self.prefer_speed_over_memory,
                        scale_by_freq=self.scale_by_freq)

        return cache

    @desc.setattr_on_read
    def coherency(self):

        #Pre-allocate the final result:
        if len(self.seed.shape) > 1:
            Cxy = np.empty((self.seed.data.shape[0],
                            self.target.data.shape[0],
                            self.frequencies.shape[0]), dtype=np.complex)
        else:
            Cxy = np.empty((self.target.data.shape[0],
                            self.frequencies.shape[0]), dtype=np.complex)

        #Get the fft window cache for the target time-series:
        cache = self.target_cache

        #A list of indices for the target:
        target_chan_idx = np.arange(self.target.data.shape[0])

        #This is a list of indices into the cached fft window libraries,
        #setting the index of the seed to be -1, so that it is easily
        #distinguished from the target indices:
        ij = list(zip(np.ones_like(target_chan_idx) * -1, target_chan_idx))

        #If there is more than one channel in the seed time-series:
        if len(self.seed.shape) > 1:
            for seed_idx, this_seed in enumerate(self.seed.data):
                #Here ij is 0, because it is just one channel and we stack the
                #channel onto itself in order for the input to the function to
                #make sense:
                f, seed_cache = tsa.cache_fft(
                    np.vstack([this_seed, this_seed]),
                    [(0, 0)],
                    lb=self.lb,
                    ub=self.ub,
                    method=self.method,
                    prefer_speed_over_memory=self.prefer_speed_over_memory,
                    scale_by_freq=self.scale_by_freq)

                #Insert the seed_cache into the target_cache:
                cache['FFT_slices'][-1] = seed_cache['FFT_slices'][0]

                #If this is true, the cache contains both FFT_slices and
                #FFT_conj_slices:
                if self.prefer_speed_over_memory:
                    cache['FFT_conj_slices'][-1] = \
                                            seed_cache['FFT_conj_slices'][0]

                #This performs the caclulation for this seed:
                Cxy[seed_idx] = tsa.cache_to_coherency(cache, ij)

        #In the case where there is only one channel in the seed time-series:
        else:
            f, seed_cache = tsa.cache_fft(
                np.vstack([self.seed.data,
                           self.seed.data]),
                [(0, 0)],
                lb=self.lb,
                ub=self.ub,
                method=self.method,
                prefer_speed_over_memory=self.prefer_speed_over_memory,
                scale_by_freq=self.scale_by_freq)

            cache['FFT_slices'][-1] = seed_cache['FFT_slices'][0]

            if self.prefer_speed_over_memory:
                cache['FFT_conj_slices'][-1] = \
                                            seed_cache['FFT_conj_slices'][0]

            Cxy = tsa.cache_to_coherency(cache, ij)

        return Cxy.squeeze()

    @desc.setattr_on_read
    def relative_phases(self):
        """The frequency-band dependent relative phase between the two
        time-series """
        return np.angle(self.coherency)

    @desc.setattr_on_read
    def delay(self):
        """ The delay in seconds between the two time series """
        return self.relative_phases / (2 * np.pi * self.frequencies)
