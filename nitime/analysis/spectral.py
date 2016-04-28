import numpy as np
from nitime.lazy import scipy
from nitime.lazy import scipy_signal as signal
from nitime.lazy import scipy_fftpack as fftpack

from nitime import descriptors as desc
from nitime import utils as tsu
from nitime import algorithms as tsa
from nitime import timeseries as ts

from .base import BaseAnalyzer


class SpectralAnalyzer(BaseAnalyzer):
    """ Analyzer object for spectral analysis"""
    def __init__(self, input=None, method=None, BW=None, adaptive=False,
                 low_bias=False):
        """
        The initialization of the

        Parameters
        ----------
        input: time-series objects

        method: dict (optional),
           The method spec used in calculating 'psd' see
           :func:`algorithms.get_spectra` for details.

        BW: float (optional),
            In 'spectrum_multi_taper' The bandwidth of the windowing function
            will determine the number tapers to use. This parameters represents
            trade-off between frequency resolution (lower main lobe BW for the
            taper) and variance reduction (higher BW and number of averaged
            estimates).

        adaptive : {True/False}
           In 'spectrum_multi_taper', use an adaptive weighting routine to
           combine the PSD estimates of different tapers.

        low_bias: {True/False}
           In spectrum_multi_taper, use bias correction


        Examples
        --------
        >>> np.set_printoptions(precision=4)  # for doctesting
        >>> t1 = ts.TimeSeries(data = np.arange(0,1024,1).reshape(2,512),
        ... sampling_rate=np.pi)
        >>> s1 = SpectralAnalyzer(t1)
        >>> s1.method['this_method']
        'welch'
        >>> s1.method['Fs'] # doctest: +ELLIPSIS
        3.1415926535... Hz
        >>> f,s = s1.psd
        >>> f
        array([ 0.    ,  0.0491,  0.0982,  0.1473,  0.1963,  0.2454,  0.2945,
                0.3436,  0.3927,  0.4418,  0.4909,  0.54  ,  0.589 ,  0.6381,
                0.6872,  0.7363,  0.7854,  0.8345,  0.8836,  0.9327,  0.9817,
                1.0308,  1.0799,  1.129 ,  1.1781,  1.2272,  1.2763,  1.3254,
                1.3744,  1.4235,  1.4726,  1.5217,  1.5708])
        >>> s[0,0]   # doctest: +ELLIPSIS
        1128276.92538360...
        """
        BaseAnalyzer.__init__(self, input)

        self.method = method

        if self.method is None:
            self.method = {'this_method': 'welch',
                           'Fs': self.input.sampling_rate}

        self.BW = BW
        self.adaptive = adaptive
        self.low_bias = low_bias

    @desc.setattr_on_read
    def psd(self):
        """
        The standard output for this analyzer is a tuple f,s, where: f is the
        frequency bands associated with the discrete spectral components
        and s is the PSD calculated using :func:`mlab.psd`.

        """
        NFFT = self.method.get('NFFT', 64)
        Fs = self.input.sampling_rate
        detrend = self.method.get('detrend', tsa.mlab.detrend_none)
        window = self.method.get('window', tsa.mlab.window_hanning)
        n_overlap = self.method.get('n_overlap', int(np.ceil(NFFT / 2.0)))

        if np.iscomplexobj(self.input.data):
            psd_len = NFFT
            dt = complex
        else:
            psd_len = NFFT // 2 + 1
            dt = float

        #If multi-channel data:
        if len(self.input.data.shape) > 1:
            psd_shape = (self.input.shape[:-1] + (psd_len,))
            flat_data = np.reshape(self.input.data, (-1,
                                                     self.input.data.shape[-1]))
            flat_psd = np.empty((flat_data.shape[0], psd_len), dtype=dt)
            for i in range(flat_data.shape[0]):
                #'f' are the center frequencies of the frequency bands
                #represented in the psd. These are identical in each iteration
                #of the loop, so they get reassigned into the same variable in
                #each iteration:
                temp, f = tsa.mlab.psd(flat_data[i],
                            NFFT=NFFT,
                            Fs=Fs,
                            detrend=detrend,
                            window=window,
                            noverlap=n_overlap)
                flat_psd[i] = temp.squeeze()
            psd = np.reshape(flat_psd, psd_shape).squeeze()

        else:
            psd, f = tsa.mlab.psd(self.input.data,
                            NFFT=NFFT,
                            Fs=Fs,
                            detrend=detrend,
                            window=window,
                            noverlap=n_overlap)

        return f, psd

    @desc.setattr_on_read
    def cpsd(self):
        """
        This outputs both the PSD and the CSD calculated using
        :func:`algorithms.get_spectra`.

        Returns
        -------

        (f,s): tuple
           f: Frequency bands over which the psd/csd are calculated and
           s: the n by n by len(f) matrix of PSD (on the main diagonal) and CSD
           (off diagonal)
        """
        self.welch_method = self.method
        self.welch_method['this_method'] = 'welch'
        self.welch_method['Fs'] = self.input.sampling_rate
        f, spectrum_welch = tsa.get_spectra(self.input.data,
                                           method=self.welch_method)

        return f, spectrum_welch

    @desc.setattr_on_read
    def periodogram(self):
        """

        This is the spectrum estimated as the FFT of the time-series

        Returns
        -------
        (f,spectrum): f is an array with the frequencies and spectrum is the
        complex-valued FFT.
        """
        return tsa.periodogram(self.input.data,
                               Fs=self.input.sampling_rate)

    @desc.setattr_on_read
    def spectrum_fourier(self):
        """

        This is the spectrum estimated as the FFT of the time-series

        Returns
        -------
        (f,spectrum): f is an array with the frequencies and spectrum is the
        complex-valued FFT.

        """

        data = self.input.data
        sampling_rate = self.input.sampling_rate

        fft = fftpack.fft
        if np.any(np.iscomplex(data)):
            # Get negative frequencies, as well as positive:
            f = np.linspace(-sampling_rate/2., sampling_rate/2., data.shape[-1])
            spectrum_fourier = np.fft.fftshift(fft(data))
        else:
            f = tsu.get_freqs(sampling_rate, data.shape[-1])
            spectrum_fourier = fft(data)[..., :f.shape[0]]

        return f, spectrum_fourier

    @desc.setattr_on_read
    def spectrum_multi_taper(self):
        """

        The spectrum and cross-spectra, computed using
        :func:`multi_taper_csd'

        """
        if np.iscomplexobj(self.input.data):
            psd_len = self.input.shape[-1]
            dt = complex
        else:
            psd_len = self.input.shape[-1] // 2 + 1
            dt = float

        #Initialize the output
        spectrum_multi_taper = np.empty((self.input.shape[:-1] + (psd_len,)),
                                         dtype=dt)

        #If multi-channel data:
        if len(self.input.data.shape) > 1:
            for i in range(self.input.data.shape[0]):
                # 'f' are the center frequencies of the frequency bands
                # represented in the MT psd. These are identical in each
                # iteration of the loop, so they get reassigned into the same
                # variable in each iteration:
                f, spectrum_multi_taper[i], _ = tsa.multi_taper_psd(
                    self.input.data[i],
                    Fs=self.input.sampling_rate,
                    BW=self.BW,
                    adaptive=self.adaptive,
                    low_bias=self.low_bias)
        else:
            f, spectrum_multi_taper, _ = tsa.multi_taper_psd(self.input.data,
                                                  Fs=self.input.sampling_rate,
                                                  BW=self.BW,
                                                  adaptive=self.adaptive,
                                                  low_bias=self.low_bias)

        return f, spectrum_multi_taper


class FilterAnalyzer(desc.ResetMixin):
    """ A class for performing filtering operations on time-series and
    producing the filtered versions of the time-series

    Parameters
    ----------

    time_series: A nitime TimeSeries object.

    lb,ub: float (optional)
       Lower and upper band of a pass-band into which the data will be
       filtered. Default: 0, Nyquist

    boxcar_iterations: int (optional)
       For box-car filtering, how many times to iterate over the data while
       convolving with a box-car function. Default: 2

    gpass: float (optional)
       For iir filtering, the pass-band maximal ripple loss (default: 1)

    gstop: float (optional)
       For iir filtering, the stop-band minimal attenuation (default: 60).

    filt_order: int (optional)
        For iir/fir filtering, the order of the filter. Note for fir filtering,
        this needs to be an even number. Default: 64

    iir_ftype: str (optional)
        The type of filter to be used in iir filtering (see
        scipy.signal.iirdesign for details). Default 'ellip'

    fir_win: str
        The window to be used in fir filtering (see scipy.signal.firwin for
        details). Default: 'hamming'

    Note
    ----
    All filtering methods used here keep the original DC component of the
    signal.

    """
    def __init__(self, time_series, lb=0, ub=None, boxcar_iterations=2,
                 filt_order=64, gpass=1, gstop=60, iir_ftype='ellip',
                 fir_win='hamming'):

        #Initialize all the local variables you will need for all the different
        #filtering methods:
        self._ts = time_series
        self.data = self._ts.data
        self.sampling_rate = self._ts.sampling_rate
        self.ub = ub
        self.lb = lb
        self.time_unit = self._ts.time_unit
        self._boxcar_iterations = boxcar_iterations
        self._gstop = gstop
        self._gpass = gpass
        self._filt_order = filt_order
        self._ftype = iir_ftype
        self._win = fir_win

    def filtfilt(self, b, a, in_ts=None):

        """
        Zero-phase delay filtering (either iir or fir).

        Parameters
        ----------

        a,b: filter coefficients

        in_ts: time-series object.
           This allows to replace the input. Instead of analyzing this
           analyzers input data, analyze some other time-series object

        Note
        ----

        This is a wrapper around scipy.signal.filtfilt

        """
        # Switch in the new in_ts:
        if in_ts is not None:
            data = in_ts.data
            Fs = in_ts.sampling_rate
            t0 = in_ts.t0
            time_unit = in_ts.time_unit
        else:
            data = self._ts.data
            Fs = self._ts.sampling_rate
            t0 = self._ts.t0
            time_unit = self._ts.time_unit

        # filtfilt only operates channel-by-channel, so we need to loop over
        # the channels, if the data is multi-channel data:
        if len(data.shape) > 1:
            out_data = np.empty(data.shape, dtype=data.dtype)
            for i in range(data.shape[0]):
                out_data[i] = signal.filtfilt(b, a, data[i])
                # Make sure to preserve the DC:
                dc = np.mean(data[i])
                out_data[i] = out_data[i] - np.mean(out_data[i])
                out_data[i] = out_data[i] + dc
        else:
            out_data = signal.filtfilt(b, a, data)
            # Make sure to preserve the DC:
            dc = np.mean(data)
            out_data -= np.mean(out_data)
            out_data += dc

        return ts.TimeSeries(out_data,
                             sampling_rate=Fs,
                             time_unit=time_unit,
                             t0=t0)

    @desc.setattr_on_read
    def fir(self):
        """
        Filter the time-series using an FIR digital filter. Filtering is done
        back and forth (using scipy.signal.filtfilt) to achieve zero phase
        delay
        """
        #Passband and stop-band are expressed as fraction of the Nyquist
        #frequency:
        if self.ub is not None:
            ub_frac = self.ub / (self.sampling_rate / 2.)
        else:
            ub_frac = 1.0

        lb_frac = self.lb / (self.sampling_rate / 2.)

        if lb_frac < 0 or ub_frac > 1:
            e_s = "The lower-bound or upper bound used to filter"
            e_s += " are beyond the range 0-Nyquist. You asked for"
            e_s += " a filter between"
            e_s += "%s and %s percent of" % (lb_frac * 100, ub_frac * 100)
            e_s += "the Nyquist frequency"
            raise ValueError(e_s)

        n_taps = self._filt_order + 1

        # This means the filter order you chose was too large (needs to be
        # shorter than a 1/3 of your time-series )
        if n_taps > self.data.shape[-1] * 3:
            e_s = "The filter order chosen is too large for this time-series"
            raise ValueError(e_s)

        # a is always 1:
        a = [1]

        sig = ts.TimeSeries(data=self._ts.data,
                            sampling_rate=self._ts.sampling_rate,
                            t0=self._ts.t0)

        # Lowpass:
        if ub_frac < 1:
            b = signal.firwin(n_taps, ub_frac, window=self._win)
            sig = self.filtfilt(b, a, sig)

        # High-pass
        if lb_frac > 0:
            #Includes a spectral inversion:
            b = -1 * signal.firwin(n_taps, lb_frac, window=self._win)
            b[n_taps // 2] = b[n_taps // 2] + 1
            sig = self.filtfilt(b, a, sig)

        return sig

    @desc.setattr_on_read
    def iir(self):
        """
        Filter the time-series using an IIR filter. Filtering is done back and
        forth (using scipy.signal.filtfilt) to achieve zero phase delay

        """

        #Passband and stop-band are expressed as fraction of the Nyquist
        #frequency:
        if self.ub is not None:
            ub_frac = self.ub / (self.sampling_rate / 2.)
        else:
            ub_frac = 1.0

        lb_frac = self.lb / (self.sampling_rate / 2.)

        # For the band-pass:
        if lb_frac > 0 and ub_frac < 1:

            wp = [lb_frac, ub_frac]

            ws = [np.max([lb_frac - 0.1, 0]),
                  np.min([ub_frac + 0.1, 1.0])]

        # For the lowpass:
        elif lb_frac == 0:
            wp = ub_frac
            ws = np.min([ub_frac + 0.1, 0.9])

        # For the highpass:
        elif ub_frac == 1:
            wp = lb_frac
            ws = np.max([lb_frac - 0.1, 0.1])

        b, a = signal.iirdesign(wp, ws, self._gpass, self._gstop,
                                ftype=self._ftype)

        return self.filtfilt(b, a)

    @desc.setattr_on_read
    def filtered_fourier(self):
        """

        Filter the time-series by passing it to the Fourier domain and null
        out the frequency bands outside of the range [lb,ub]

        """

        freqs = tsu.get_freqs(self.sampling_rate, self.data.shape[-1])

        if self.ub is None:
            self.ub = freqs[-1]

        power = fftpack.fft(self.data)
        idx_0 = np.hstack([np.where(freqs < self.lb)[0],
                           np.where(freqs > self.ub)[0]])

        #Make sure that you keep the DC component:
        keep_dc = np.copy(power[..., 0])
        power[..., idx_0] = 0
        power[..., -1 * idx_0] = 0  # Take care of the negative frequencies
        power[..., 0] = keep_dc  # And put the DC back in when you're done:

        data_out = fftpack.ifft(power)

        data_out = np.real(data_out)  # In order to make sure that you are not
                                      # left with float-precision residual
                                      # complex parts

        return ts.TimeSeries(data=data_out,
                             sampling_rate=self.sampling_rate,
                             time_unit=self.time_unit,
                             t0=self._ts.t0)

    @desc.setattr_on_read
    def filtered_boxcar(self):
        """
        Filter the time-series by a boxcar filter.

        The low pass filter is implemented by convolving with a boxcar function
        of the right length and amplitude and the high-pass filter is
        implemented by subtracting a low-pass version (as above) from the
        signal
        """

        if self.ub is not None:
            ub = self.ub / self.sampling_rate
        else:
            ub = 1.0

        lb = self.lb / self.sampling_rate

        data_out = tsa.boxcar_filter(np.copy(self.data),
                                     lb=lb, ub=ub,
                                     n_iterations=self._boxcar_iterations)

        return ts.TimeSeries(data=data_out,
                             sampling_rate=self.sampling_rate,
                             time_unit=self.time_unit,
                             t0=self._ts.t0)


class HilbertAnalyzer(BaseAnalyzer):

    """Analyzer class for extracting the Hilbert transform """

    def __init__(self, input=None):
        """Constructor function for the Hilbert analyzer class.

        Parameters
        ----------

        input: TimeSeries

        """
        BaseAnalyzer.__init__(self, input)

    @desc.setattr_on_read
    def analytic(self):
        """The natural output for this analyzer is the analytic signal """
        data = self.input.data
        sampling_rate = self.input.sampling_rate
        hilbert = signal.hilbert

        return ts.TimeSeries(data=hilbert(data),
                             sampling_rate=sampling_rate)

    @desc.setattr_on_read
    def amplitude(self):
        return ts.TimeSeries(data=np.abs(self.analytic.data),
                             sampling_rate=self.analytic.sampling_rate)

    @desc.setattr_on_read
    def phase(self):
        return ts.TimeSeries(data=np.angle(self.analytic.data),
                             sampling_rate=self.analytic.sampling_rate)

    @desc.setattr_on_read
    def real(self):
        return ts.TimeSeries(data=self.analytic.data.real,
                             sampling_rate=self.analytic.sampling_rate)

    @desc.setattr_on_read
    def imag(self):
        return ts.TimeSeries(data=self.analytic.data.imag,
                             sampling_rate=self.analytic.sampling_rate)


class MorletWaveletAnalyzer(BaseAnalyzer):

    """Analyzer class for extracting the (complex) Morlet wavelet transform """

    def __init__(self, input=None, freqs=None, sd_rel=.2, sd=None, f_min=None,
                 f_max=None, nfreqs=None, log_spacing=False, log_morlet=False):
        """Constructor function for the Wavelet analyzer class.

        Parameters
        ----------

        freqs: list or float
          List of center frequencies for the wavelet transform, or a scalar
          for a single band-passed signal.

        sd: list or float
          List of filter bandwidths, given as standard-deviation of center
          frequencies. Alternatively sd_rel can be specified.

        sd_rel: float
          Filter bandwidth, given as a fraction of the center frequencies.

        f_min: float
          Minimal frequency.

        f_max: float
          Maximal frequency.

        nfreqs: int
          Number of frequencies.

        log_spacing: bool
          If true, frequencies will be evenly spaced on a log-scale.
          Default: False

        log_morlet: bool
          If True, a log-Morlet wavelet is used, if False, a regular Morlet
          wavelet is used. Default: False
        """
        BaseAnalyzer.__init__(self, input)
        self.freqs = freqs
        self.sd_rel = sd_rel
        self.sd = sd
        self.f_min = f_min
        self.f_max = f_max
        self.nfreqs = nfreqs
        self.log_spacing = log_spacing
        self.log_morlet = log_morlet

        if log_morlet:
            self.wavelet = tsa.wlogmorlet
        else:
            self.wavelet = tsa.wmorlet

        if freqs is not None:
            self.freqs = np.array(freqs)
        elif f_min is not None and f_max is not None and nfreqs is not None:
            if log_spacing:
                self.freqs = np.logspace(np.log10(f_min), np.log10(f_max),
                                         num=nfreqs, endpoint=True)
            else:
                self.freqs = np.linspace(f_min, f_max, num=nfreqs,
                                         endpoint=True)
        else:
            raise NotImplementedError

        if sd is None:
            self.sd = self.freqs * self.sd_rel

    @desc.setattr_on_read
    def analytic(self):
        """The natural output for this analyzer is the analytic signal"""
        data = self.input.data
        sampling_rate = self.input.sampling_rate

        a_signal =\
    ts.TimeSeries(data=np.zeros(self.freqs.shape + data.shape,
                                dtype='D'), sampling_rate=sampling_rate)
        if self.freqs.ndim == 0:
            w = self.wavelet(self.freqs, self.sd,
                             sampling_rate=sampling_rate, ns=5,
                             normed='area')

            # nd = (w.shape[0] - 1) / 2
            a_signal.data[...] = (np.convolve(data, np.real(w), mode='same') +
                            1j * np.convolve(data, np.imag(w), mode='same'))
        else:
            for i, (f, sd) in enumerate(zip(self.freqs, self.sd)):
                w = self.wavelet(f, sd, sampling_rate=sampling_rate,
                                 ns=5, normed='area')

                # nd = (w.shape[0] - 1) / 2
                a_signal.data[i, ...] = (
                    np.convolve(data, np.real(w), mode='same') +
                    1j * np.convolve(data, np.imag(w), mode='same'))

        return a_signal

    @desc.setattr_on_read
    def amplitude(self):
        return ts.TimeSeries(data=np.abs(self.analytic.data),
                             sampling_rate=self.analytic.sampling_rate)

    @desc.setattr_on_read
    def phase(self):
        return ts.TimeSeries(data=np.angle(self.analytic.data),
                             sampling_rate=self.analytic.sampling_rate)

    @desc.setattr_on_read
    def real(self):
        return ts.TimeSeries(data=self.analytic.data.real,
                             sampling_rate=self.analytic.sampling_rate)

    @desc.setattr_on_read
    def imag(self):
        return ts.TimeSeries(data=self.analytic.data.imag,
                             sampling_rate=self.analytic.sampling_rate)
