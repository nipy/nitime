"""

These functions are taken from mlab and we have them here, to support systmes
with elderly versions of matplotlib, which may contain certain bugs in scaling
of PSD

"""
import numpy as np
import matplotlib.cbook as cbook

def detrend_none(x):
    "Return x: no detrending"
    return x

def window_hanning(x):
    "return x times the hanning window of len(x)"
    return np.hanning(len(x))*x

#This is a helper function that implements the commonality between the
#psd, csd, and spectrogram.  It is *NOT* meant to be used outside of mlab
def _spectral_helper(x, y, NFFT=256, Fs=2, detrend=detrend_none,
        window=window_hanning, noverlap=0, pad_to=None, sides='default',
        scale_by_freq=None):
    #The checks for if y is x are so that we can use the same function to
    #implement the core of psd(), csd(), and spectrogram() without doing
    #extra calculations.  We return the unaveraged Pxy, freqs, and t.
    same_data = y is x

    #Make sure we're dealing with a numpy array. If y and x were the same
    #object to start with, keep them that way

    x = np.asarray(x)
    if not same_data:
        y = np.asarray(y)

    # zero pad x and y up to NFFT if they are shorter than NFFT
    if len(x)<NFFT:
        n = len(x)
        x = np.resize(x, (NFFT,))
        x[n:] = 0

    if not same_data and len(y)<NFFT:
        n = len(y)
        y = np.resize(y, (NFFT,))
        y[n:] = 0

    if pad_to is None:
        pad_to = NFFT

    if scale_by_freq is None:
        scale_by_freq = True

    # For real x, ignore the negative frequencies unless told otherwise
    if (sides == 'default' and np.iscomplexobj(x)) or sides == 'twosided':
        numFreqs = pad_to
        scaling_factor = 1.
    elif sides in ('default', 'onesided'):
        numFreqs = pad_to//2 + 1
        scaling_factor = 2.
    else:
        raise ValueError("sides must be one of: 'default', 'onesided', or "
            "'twosided'")

    # MATLAB divides by the sampling frequency so that density function
    # has units of dB/Hz and can be integrated by the plotted frequency
    # values. Perform the same scaling here.
    if scale_by_freq:
        scaling_factor /= Fs

    if cbook.iterable(window):
        assert(len(window) == NFFT)
        windowVals = window
    else:
        windowVals = window(np.ones((NFFT,), x.dtype))

    step = NFFT - noverlap
    ind = np.arange(0, len(x) - NFFT + 1, step)
    n = len(ind)
    Pxy = np.zeros((numFreqs,n), np.complex_)

    # do the ffts of the slices
    for i in range(n):
        thisX = x[ind[i]:ind[i]+NFFT]
        thisX = windowVals * detrend(thisX)
        fx = np.fft.fft(thisX, n=pad_to)

        if same_data:
            fy = fx
        else:
            thisY = y[ind[i]:ind[i]+NFFT]
            thisY = windowVals * detrend(thisY)
            fy = np.fft.fft(thisY, n=pad_to)
        Pxy[:,i] = np.conjugate(fx[:numFreqs]) * fy[:numFreqs]

    # Scale the spectrum by the norm of the window to compensate for
    # windowing loss; see Bendat & Piersol Sec 11.5.2.
    Pxy *= 1 / (np.abs(windowVals)**2).sum()

    # Also include scaling factors for one-sided densities and dividing by the
    # sampling frequency, if desired. Scale everything, except the DC component
    # and the NFFT/2 component:
    Pxy[1:-1] *= scaling_factor

    #But do scale those components by Fs, if required
    if scale_by_freq:
        Pxy[[0,-1]] /= Fs

    t = 1./Fs * (ind + NFFT / 2.)
    freqs = float(Fs) / pad_to * np.arange(numFreqs)

    if (np.iscomplexobj(x) and sides == 'default') or sides == 'twosided':
        # center the frequency range at zero
        freqs = np.concatenate((freqs[numFreqs//2:] - Fs, freqs[:numFreqs//2]))
        Pxy = np.concatenate((Pxy[numFreqs//2:, :], Pxy[:numFreqs//2, :]), 0)

    return Pxy, freqs, t


#Split out these keyword docs so that they can be used elsewhere
docstring.interpd.update(PSD=cbook.dedent("""

"""))

def psd(x, NFFT=256, Fs=2, detrend=detrend_none, window=window_hanning,
        noverlap=0, pad_to=None, sides='default', scale_by_freq=None):
    """
    The power spectral density by Welch's average periodogram method.
    The vector *x* is divided into *NFFT* length blocks.  Each block
    is detrended by the function *detrend* and windowed by the function
    *window*.  *noverlap* gives the length of the overlap between blocks.
    The absolute(fft(block))**2 of each segment are averaged to compute
    *Pxx*, with a scaling to correct for power loss due to windowing.

    If len(*x*) < *NFFT*, it will be zero padded to *NFFT*.

    *x*
        Array or sequence containing the data

    Keyword arguments:

      *NFFT*: integer
          The number of data points used in each block for the FFT.
          Must be even; a power 2 is most efficient.  The default value is 256.

      *Fs*: scalar
          The sampling frequency (samples per time unit).  It is used
          to calculate the Fourier frequencies, freqs, in cycles per time
          unit. The default value is 2.

      *detrend*: callable
          The function applied to each segment before fft-ing,
          designed to remove the mean or linear trend.  Unlike in
          MATLAB, where the *detrend* parameter is a vector, in
          matplotlib is it a function.  The :mod:`~matplotlib.pylab`
          module defines :func:`~matplotlib.pylab.detrend_none`,
          :func:`~matplotlib.pylab.detrend_mean`, and
          :func:`~matplotlib.pylab.detrend_linear`, but you can use
          a custom function as well.

      *window*: callable or ndarray
          A function or a vector of length *NFFT*. To create window
          vectors see :func:`window_hanning`, :func:`window_none`,
          :func:`numpy.blackman`, :func:`numpy.hamming`,
          :func:`numpy.bartlett`, :func:`scipy.signal`,
          :func:`scipy.signal.get_window`, etc. The default is
          :func:`window_hanning`.  If a function is passed as the
          argument, it must take a data segment as an argument and
          return the windowed version of the segment.

      *noverlap*: integer
          The number of points of overlap between blocks.  The default value
          is 0 (no overlap).

      *pad_to*: integer
          The number of points to which the data segment is padded when
          performing the FFT.  This can be different from *NFFT*, which
          specifies the number of data points used.  While not increasing
          the actual resolution of the psd (the minimum distance between
          resolvable peaks), this can give more points in the plot,
          allowing for more detail. This corresponds to the *n* parameter
          in the call to fft(). The default is None, which sets *pad_to*
          equal to *NFFT*

      *sides*: [ 'default' | 'onesided' | 'twosided' ]
          Specifies which sides of the PSD to return.  Default gives the
          default behavior, which returns one-sided for real data and both
          for complex data.  'onesided' forces the return of a one-sided PSD,
          while 'twosided' forces two-sided.

      *scale_by_freq*: boolean
          Specifies whether the resulting density values should be scaled
          by the scaling frequency, which gives density in units of Hz^-1.
          This allows for integration over the returned frequency values.
          The default is True for MATLAB compatibility.
          
    Returns
    -------

    The tuple (*Pxx*, *freqs*).

    Notes
    -----

    For details, see: Bendat & Piersol -- Random Data: Analysis and Measurement
    Procedures, John Wiley & Sons (1986)

    """
    Pxx,freqs = csd(x, x, NFFT, Fs, detrend, window, noverlap, pad_to, sides,
        scale_by_freq)
    return Pxx.real,freqs

def csd(x, y, NFFT=256, Fs=2, detrend=detrend_none, window=window_hanning,
        noverlap=0, pad_to=None, sides='default', scale_by_freq=None):
    """
    The cross power spectral density by Welch's average periodogram
    method.  The vectors *x* and *y* are divided into *NFFT* length
    blocks.  Each block is detrended by the function *detrend* and
    windowed by the function *window*.  *noverlap* gives the length
    of the overlap between blocks.  The product of the direct FFTs
    of *x* and *y* are averaged over each segment to compute *Pxy*,
    with a scaling to correct for power loss due to windowing.

    If len(*x*) < *NFFT* or len(*y*) < *NFFT*, they will be zero
    padded to *NFFT*.

    *x*, *y*
        Array or sequence containing the data

    Keyword arguments:

      *NFFT*: integer
          The number of data points used in each block for the FFT.
          Must be even; a power 2 is most efficient.  The default value is 256.

      *Fs*: scalar
          The sampling frequency (samples per time unit).  It is used
          to calculate the Fourier frequencies, freqs, in cycles per time
          unit. The default value is 2.

      *detrend*: callable
          The function applied to each segment before fft-ing,
          designed to remove the mean or linear trend.  Unlike in
          MATLAB, where the *detrend* parameter is a vector, in
          matplotlib is it a function.  The :mod:`~matplotlib.pylab`
          module defines :func:`~matplotlib.pylab.detrend_none`,
          :func:`~matplotlib.pylab.detrend_mean`, and
          :func:`~matplotlib.pylab.detrend_linear`, but you can use
          a custom function as well.

      *window*: callable or ndarray
          A function or a vector of length *NFFT*. To create window
          vectors see :func:`window_hanning`, :func:`window_none`,
          :func:`numpy.blackman`, :func:`numpy.hamming`,
          :func:`numpy.bartlett`, :func:`scipy.signal`,
          :func:`scipy.signal.get_window`, etc. The default is
          :func:`window_hanning`.  If a function is passed as the
          argument, it must take a data segment as an argument and
          return the windowed version of the segment.

      *noverlap*: integer
          The number of points of overlap between blocks.  The default value
          is 0 (no overlap).

      *pad_to*: integer
          The number of points to which the data segment is padded when
          performing the FFT.  This can be different from *NFFT*, which
          specifies the number of data points used.  While not increasing
          the actual resolution of the psd (the minimum distance between
          resolvable peaks), this can give more points in the plot,
          allowing for more detail. This corresponds to the *n* parameter
          in the call to fft(). The default is None, which sets *pad_to*
          equal to *NFFT*

      *sides*: [ 'default' | 'onesided' | 'twosided' ]
          Specifies which sides of the PSD to return.  Default gives the
          default behavior, which returns one-sided for real data and both
          for complex data.  'onesided' forces the return of a one-sided PSD,
          while 'twosided' forces two-sided.

      *scale_by_freq*: boolean
          Specifies whether the resulting density values should be scaled
          by the scaling frequency, which gives density in units of Hz^-1.
          This allows for integration over the returned frequency values.
          The default is True for MATLAB compatibility.
          
    Returns
    -------
    The tuple (*Pxy*, *freqs*).


    Notes
    -----
    For details see: Bendat & Piersol -- Random Data: Analysis and Measurement
    Procedures, John Wiley & Sons (1986)
    """
    Pxy, freqs, t = _spectral_helper(x, y, NFFT, Fs, detrend, window,
        noverlap, pad_to, sides, scale_by_freq)

    if len(Pxy.shape) == 2 and Pxy.shape[1]>1:
        Pxy = Pxy.mean(axis=1)
    return Pxy, freqs
