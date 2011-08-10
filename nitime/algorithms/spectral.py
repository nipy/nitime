"""

Spectral transforms are used in order to estimate the frequency-domain
representation of time-series. Several methods can be used and this module
contains implementations of several algorithms for the calculation of spectral
transforms.

The functions in this module are:

XXX

"""

import numpy as np
from matplotlib import mlab
from scipy import linalg
from scipy import signal as sig
from scipy import interpolate
import nitime.utils as utils

# To suppport older versions of numpy that don't have tril_indices:
from nitime.index_utils import tril_indices, triu_indices


# Set global variables for the default NFFT to be used in spectral analysis and
# the overlap:
default_nfft = 64
default_n_overlap = int(np.ceil(default_nfft / 2.0))

def get_spectra(time_series, method=None):
    r"""
    Compute the spectra of an n-tuple of time series and all of
    the pairwise cross-spectra.

    Parameters
    ----------
    time_series: float array
        The time-series, where time is the last dimension

    method: dict, optional

        contains: this_method:'welch'
           indicates that :func:`mlab.psd` will be used in
           order to calculate the psd/csd, in which case, additional optional
           inputs (and default values) are:

               NFFT=64

               Fs=2pi

               detrend=mlab.detrend_none

               window=mlab.window_hanning

               n_overlap=0

        this_method:'periodogram_csd'
           indicates that :func:`periodogram` will
           be used in order to calculate the psd/csd, in which case, additional
           optional inputs (and default values) are:

               Skx=None

               Sky=None

               N=None

               sides='onesided'

               normalize=True

               Fs=2pi

        this_method:'multi_taper_csd'
           indicates that :func:`multi_taper_psd` used in order to calculate
           psd/csd, in which case additional optional inputs (and default
           values) are:

               BW=0.01

               Fs=2pi

               sides = 'onesided'

    Returns
    -------

    f: float array
        The central frequencies for the frequency bands for which the spectra
        are estimated

    fxy: float array
        A semi-filled matrix with the cross-spectra of the signals. The csd of
        signal i and signal j is in f[j][i], but not in f[i][j] (which will be
        filled with zeros). For i=j fxy[i][j] is the psd of signal i.

    """
    if method is None:
        method = {'this_method': 'welch'}  # The default
    # If no choice of method was explicitely set, but other parameters were
    # passed, assume that the method is mlab:
    this_method = method.get('this_method', 'welch')

    if this_method == 'welch':
        NFFT = method.get('NFFT', default_nfft)
        Fs = method.get('Fs', 2 * np.pi)
        detrend = method.get('detrend', mlab.detrend_none)
        window = method.get('window', mlab.window_hanning)
        n_overlap = method.get('n_overlap', int(np.ceil(NFFT / 2.0)))

        # The length of the spectrum depends on how many sides are taken, which
        # depends on whether or not this is a complex object:
        if np.iscomplexobj(time_series):
            fxy_len = NFFT
        else:
            fxy_len = NFFT / 2.0 + 1

        # If there is only 1 channel in the time-series:
        if len(time_series.shape) == 1 or time_series.shape[0] == 1:
            temp, f = mlab.csd(time_series, time_series,
                               NFFT, Fs, detrend, window, n_overlap,
                               scale_by_freq=True)

            fxy = temp.squeeze()  # the output of mlab.csd has a weird
                                  # shape
        else:
            fxy = np.zeros((time_series.shape[0],
                            time_series.shape[0],
                            fxy_len), dtype=complex)  # Make sure it's complex

            for i in xrange(time_series.shape[0]):
                for j in xrange(i, time_series.shape[0]):
                    #Notice funny indexing, in order to conform to the
                    #conventions of the other methods:
                    temp, f = mlab.csd(time_series[j], time_series[i],
                                       NFFT, Fs, detrend, window, n_overlap,
                                       scale_by_freq=True)

                    fxy[i][j] = temp.squeeze()  # the output of mlab.csd has a
                                                # wierd shape
    elif this_method in ('multi_taper_csd', 'periodogram_csd'):
        # these methods should work with similar signatures
        mdict = method.copy()
        func = eval(mdict.pop('this_method'))
        freqs, fxy = func(time_series, **mdict)
        f = utils.circle_to_hz(freqs, mdict.get('Fs', 2 * np.pi))

    else:
        raise ValueError("Unknown method provided")

    return f, fxy.squeeze()


def get_spectra_bi(x, y, method=None):
    r"""
    Computes the spectra of two timeseries and the cross-spectrum between them

    Parameters
    ----------

    x,y : float arrays
        Time-series data

    method: dict, optional
       See :func:`get_spectra` documentation for details

    Returns
    -------
    f: float array
        The central frequencies for the frequency
        bands for which the spectra are estimated
    fxx: float array
         The psd of the first signal
    fyy: float array
        The psd of the second signal
    fxy: float array
        The cross-spectral density of the two signals

    """
    f, fij = get_spectra(np.vstack((x, y)), method=method)
    fxx = fij[0, 0].real
    fyy = fij[1, 1].real
    fxy = fij[0, 1]
    return f, fxx, fyy, fxy


# The following spectrum estimates are normalized to the following convention..
# By definition, Sxx(w) = DTFT{Rxx(n)}, where Rxx(n) is the autocovariance
# function of x(n). Therefore the integral from
# [-PI, PI] of Sxx(w)/(2PI) is Rxx(0)
# And from the definition of Rxx(n),
# Rxx(0) = Expected-Value{x(n)x*(n)} = Expected-Value{ |x|^2 },
# which is estimated as (x*x.conj()).mean()


def periodogram(s, Fs=2 * np.pi, Sk=None, N=None,
                sides='default', normalize=True):
    """Takes an N-point periodogram estimate of the PSD function. The
    number of points N, or a precomputed FFT Sk may be provided. By default,
    the PSD function returned is normalized so that the integral of the PSD
    is equal to the mean squared amplitude (mean energy) of s (see Notes).

    Parameters
    ----------
    s : ndarray
        Signal(s) for which to estimate the PSD, time dimension in the last
        axis

    Fs: float (optional)
       The sampling rate. Defaults to 2*pi

    Sk : ndarray (optional)
        Precomputed FFT of s

    N : int (optional)
        Indicates an N-point FFT where N != s.shape[-1]

    sides : str (optional) [ 'default' | 'onesided' | 'twosided' ]
         This determines which sides of the spectrum to return.
         For complex-valued inputs, the default is two-sided, for real-valued
         inputs, default is one-sided Indicates whether to return a one-sided
         or two-sided

    PSD normalize : boolean (optional, default=True) Normalizes the PSD

    Returns
    -------
    (f, psd): tuple
       f: The central frequencies for the frequency bands
       PSD estimate for each row of s

    Notes
    -----
    setting dw = 2*PI/N, then the integral from -PI, PI (or 0,PI) of PSD/(2PI)
    will be nearly equal to sxx(0), where sxx is the autocovariance function
    of s(n). By definition, sxx(0) = E{s(n)s*(n)} ~ (s*s.conj()).mean()
    """
    if Sk is not None:
        N = Sk.shape[-1]
    else:
        N = s.shape[-1] if not N else N
        Sk = np.fft.fft(s, n=N)
    pshape = list(Sk.shape)
    norm = float(s.shape[-1])

    # if the time series is a complex vector, a one sided PSD is invalid:
    if (sides == 'default' and np.iscomplexobj(s)) or sides == 'twosided':
        sides = 'twosided'
    elif sides in ('default', 'onesided'):
        sides = 'onesided'

    if sides == 'onesided':
        # putative Nyquist freq
        Fn = N / 2 + 1
        # last duplicate freq
        Fl = (N + 1) / 2
        pshape[-1] = Fn
        P = np.zeros(pshape, 'd')
        freqs = np.linspace(0, Fs / 2, Fn)
        P[..., 0] = (Sk[..., 0] * Sk[..., 0].conj()).real
        P[..., 1:Fl] = 2 * (Sk[..., 1:Fl] * Sk[..., 1:Fl].conj()).real
        if Fn > Fl:
            P[..., Fn - 1] = (Sk[..., Fn - 1] * Sk[..., Fn - 1].conj()).real
    else:
        P = (Sk * Sk.conj()).real
        freqs = np.linspace(0, Fs, N, endpoint=False)
    if normalize:
        P /= norm
    return freqs, P


def periodogram_csd(s, Fs=2 * np.pi, Sk=None, NFFT=None, sides='default',
                    normalize=True):
    """Takes an N-point periodogram estimate of all the cross spectral
    density functions between rows of s.

    The number of points N, or a precomputed FFT Sk may be provided. By
    default, the CSD function returned is normalized so that the integral of
    the PSD is equal to the mean squared amplitude (mean energy) of s (see
    Notes).

    Parameters
    ---------

    s : ndarray
        Signals for which to estimate the CSD, time dimension in the last axis

    Fs: float (optional)
       The sampling rate. Defaults to 2*pi

    Sk : ndarray (optional)
        Precomputed FFT of rows of s

    NFFT : int (optional)
        Indicates an N-point FFT where N != s.shape[-1]

    sides : str (optional)   [ 'default' | 'onesided' | 'twosided' ]
        This determines which sides of the spectrum to return.
        For complex-valued inputs, the default is two-sided, for real-valued
        inputs, default is one-sided Indicates whether to return a one-sided
        or two-sided

    normalize : boolean (optional)
        Normalizes the PSD

    Returns
    -------

    freqs, csd_est : ndarrays
        The estimatated CSD and the frequency points vector.
        The CSD{i,j}(f) are returned in a square "matrix" of vectors
        holding Sij(f). For an input array that is reshaped to (M,N),
        the output is (M,M,N)

    Notes
    -----
    setting dw = 2*PI/N, then the integral from -PI, PI (or 0,PI) of PSD/(2PI)
    will be nearly equal to sxy(0), where sxx is the crosscovariance function
    of s1(n), s2(n). By definition, sxy(0) = E{s1(n)s2*(n)} ~
    (s1*s2.conj()).mean()
    """
    s_shape = s.shape
    s.shape = (np.prod(s_shape[:-1]), s_shape[-1])
    # defining an Sk_loc is a little opaque, but it avoids having to
    # reset the shape of any user-given Sk later on
    if Sk is not None:
        Sk_shape = Sk.shape
        N = Sk.shape[-1]
        Sk_loc = Sk.reshape(np.prod(Sk_shape[:-1]), N)
    else:
        if NFFT is not None:
            N = NFFT
        else:
            N = s.shape[-1]
        Sk_loc = np.fft.fft(s, n=N)
    # reset s.shape
    s.shape = s_shape

    M = Sk_loc.shape[0]
    norm = float(s.shape[-1])

    # if the time series is a complex vector, a one sided PSD is invalid:
    if (sides == 'default' and np.iscomplexobj(s)) or sides == 'twosided':
        sides = 'twosided'
    elif sides in ('default', 'onesided'):
        sides = 'onesided'

    if sides == 'onesided':
        # putative Nyquist freq
        Fn = N / 2 + 1
        # last duplicate freq
        Fl = (N + 1) / 2
        csd_mat = np.empty((M, M, Fn), 'D')
        freqs = np.linspace(0, Fs / 2, Fn)
        for i in xrange(M):
            for j in xrange(i + 1):
                csd_mat[i, j, 0] = Sk_loc[i, 0] * Sk_loc[j, 0].conj()
                csd_mat[i, j, 1:Fl] = 2 * (Sk_loc[i, 1:Fl] *
                                           Sk_loc[j, 1:Fl].conj())
                if Fn > Fl:
                    csd_mat[i, j, Fn - 1] = (Sk_loc[i, Fn - 1] *
                                             Sk_loc[j, Fn - 1].conj())

    else:
        csd_mat = np.empty((M, M, N), 'D')
        freqs = np.linspace(0, Fs / 2, N, endpoint=False)
        for i in xrange(M):
            for j in xrange(i + 1):
                csd_mat[i, j] = Sk_loc[i] * Sk_loc[j].conj()
    if normalize:
        csd_mat /= norm

    upper_idc = triu_indices(M, k=1)
    lower_idc = tril_indices(M, k=-1)
    csd_mat[upper_idc] = csd_mat[lower_idc].conj()
    return freqs, csd_mat


def dpss_windows(N, NW, Kmax, interp_from=None, interp_kind='linear'):
    """
    Returns the Discrete Prolate Spheroidal Sequences of orders [0,Kmax-1]
    for a given frequency-spacing multiple NW and sequence length N.

    Paramters
    ---------
    N : int
        sequence length
    NW : float, unitless
        standardized half bandwidth corresponding to 2NW = BW*f0 = BW*N/dt
        but with dt taken as 1
    Kmax : int
        number of DPSS windows to return is Kmax (orders 0 through Kmax-1)
    interp_from: int (optional)
        The dpss will can calculated using interpolation from a set of dpss
        with the same NW and Kmax, but shorter N. This is the length of this
        shorter set of dpss windows.
    interp_kind: str (optional)
        This input variable is passed to scipy.interpolate.interp1d and
        specifies the kind of interpolation as a string ('linear', 'nearest',
        'zero', 'slinear', 'quadratic, 'cubic') or as an integer specifying the
        order of the spline interpolator to use.


    Returns
    -------
    v, e : tuple,
        v is an array of DPSS windows shaped (Kmax, N)
        e are the eigenvalues

    Notes
    -----
    Tridiagonal form of DPSS calculation from:

    Slepian, D. Prolate spheroidal wave functions, Fourier analysis, and
    uncertainty V: The discrete case. Bell System Technical Journal,
    Volume 57 (1978), 1371430
    """
    Kmax = int(Kmax)
    W = float(NW) / N
    nidx = np.arange(N, dtype='d')

    # In this case, we create the dpss windows of the smaller size
    # (interp_from) and then interpolate to the larger size (N)
    if interp_from is not None:
        if interp_from > N:
            e_s = 'In dpss_windows, interp_from is: %s ' % interp_from
            e_s += 'and N is: %s. ' % N
            e_s += 'Please enter interp_from smaller than N.'
            raise ValueError(e_s)
        dpss = []
        d, e = dpss_windows(interp_from, NW, Kmax)
        for this_d in d:
            x = np.arange(this_d.shape[-1])
            I = interpolate.interp1d(x, this_d, kind=interp_kind)
            d_temp = I(np.arange(0, this_d.shape[-1] - 1,
                                 float(this_d.shape[-1] - 1) / N))

            # Rescale:
            d_temp = d_temp / np.sqrt(np.sum(d_temp ** 2))

            dpss.append(d_temp)

        dpss = np.array(dpss)

    else:
        # here we want to set up an optimization problem to find a sequence
        # whose energy is maximally concentrated within band [-W,W].
        # Thus, the measure lambda(T,W) is the ratio between the energy within
        # that band, and the total energy. This leads to the eigen-system
        # (A - (l1)I)v = 0, where the eigenvector corresponding to the largest
        # eigenvalue is the sequence with maximally concentrated energy. The
        # collection of eigenvectors of this system are called Slepian
        # sequences, or discrete prolate spheroidal sequences (DPSS). Only the
        # first K, K = 2NW/dt orders of DPSS will exhibit good spectral
        # concentration
        # [see http://en.wikipedia.org/wiki/Spectral_concentration_problem]

        # Here I set up an alternative symmetric tri-diagonal eigenvalue
        # problem such that
        # (B - (l2)I)v = 0, and v are our DPSS (but eigenvalues l2 != l1)
        # the main diagonal = ([N-1-2*t]/2)**2 cos(2PIW), t=[0,1,2,...,N-1]
        # and the first off-diagonal = t(N-t)/2, t=[1,2,...,N-1]
        # [see Percival and Walden, 1993]
        diagonal = ((N - 1 - 2 * nidx) / 2.) ** 2 * np.cos(2 * np.pi * W)
        off_diag = np.zeros_like(nidx)
        off_diag[:-1] = nidx[1:] * (N - nidx[1:]) / 2.
        # put the diagonals in LAPACK "packed" storage
        ab = np.zeros((2, N), 'd')
        ab[1] = diagonal
        ab[0, 1:] = off_diag[:-1]
        # only calculate the highest Kmax eigenvalues
        w = linalg.eigvals_banded(ab, select='i',
                                  select_range=(N - Kmax, N - 1))
        w = w[::-1]

        # find the corresponding eigenvectors via inverse iteration
        t = np.linspace(0, np.pi, N)
        dpss = np.zeros((Kmax, N), 'd')
        for k in xrange(Kmax):
            dpss[k] = utils.tridi_inverse_iteration(
                diagonal, off_diag, w[k], x0=np.sin((k + 1) * t)
                )

    # By convention (Percival and Walden, 1993 pg 379)
    # * symmetric tapers (k=0,2,4,...) should have a positive average.
    # * antisymmetric tapers should begin with a positive lobe
    fix_symmetric = (dpss[0::2].sum(axis=1) < 0)
    for i, f in enumerate(fix_symmetric):
        if f:
            dpss[2 * i] *= -1
    fix_skew = (dpss[1::2, 1] < 0)
    for i, f in enumerate(fix_skew):
        if f:
            dpss[2 * i + 1] *= -1

    # Now find the eigenvalues of the original spectral concentration problem
    # Use the autocorr sequence technique from Percival and Walden, 1993 pg 390
    dpss_rxx = utils.autocorr(dpss) * N
    r = 4 * W * np.sinc(2 * W * nidx)
    r[0] = 2 * W
    eigvals = np.dot(dpss_rxx, r)

    return dpss, eigvals


def mtm_cross_spectrum(tx, ty, weights, sides='twosided'):
    r"""

    The cross-spectrum between two tapered time-series, derived from a
    multi-taper spectral estimation.

    Parameters
    ----------

    tx, ty: ndarray (K, ..., N)
       The complex DFTs of the tapered sequence

    weights: ndarray, or 2-tuple or list
       Weights can be specified as a length-2 list of weights for spectra tx
       and ty respectively. Alternatively, if tx is ty and this function is
       computing the spectral density function of a single sequence, the
       weights can be given as an ndarray of weights for the spectrum.
       Weights may be

       * scalars, if the shape of the array is (K, ..., 1)
       * vectors, with the shape of the array being the same as tx or ty

    sides: str in {'onesided', 'twosided'}
       For the symmetric spectra of a real sequence, optionally combine half
       of the frequencies and scale the duplicate frequencies in the range
       (0, F_nyquist).

    Notes
    -----

    spectral densities are always computed as

    :math:`S_{xy}^{mt}(f) = \frac{\sum_k
    [d_k^x(f)y_k^x(f)][d_k^y(f)(y_k^y(f))^{*}]}{[\sum_k
    d_k^x(f)^2]^{\frac{1}{2}}[\sum_k d_k^y(f)^2]^{\frac{1}{2}}}`

    """
    N = tx.shape[-1]
    if ty.shape != tx.shape:
        raise ValueError('shape mismatch between tx, ty')

    # pshape = list(tx.shape)

    if isinstance(weights, (list, tuple)):
        autospectrum = False
        weights_x = weights[0]
        weights_y = weights[1]
        denom = (np.abs(weights_x) ** 2).sum(axis=0) ** 0.5
        denom *= (np.abs(weights_y) ** 2).sum(axis=0) ** 0.5
    else:
        autospectrum = True
        weights_x = weights
        weights_y = weights
        denom = (np.abs(weights) ** 2).sum(axis=0)

    if sides == 'onesided':
        # where the nyq freq should be
        Fn = N / 2 + 1
        truncated_slice = [slice(None)] * len(tx.shape)
        truncated_slice[-1] = slice(0, Fn)
        tsl = tuple(truncated_slice)
        tx = tx[tsl]
        ty = ty[tsl]
        # if weights.shape[-1] > 1 then make sure weights are truncated too
        if weights_x.shape[-1] > 1:
            weights_x = weights_x[tsl]
            weights_y = weights_y[tsl]
            denom = denom[tsl[1:]]

    sf = weights_x * tx
    sf *= (weights_y * ty).conj()
    sf = sf.sum(axis=0)
    sf /= denom

    if sides == 'onesided':
        # dbl power at duplicated freqs
        Fl = (N + 1) / 2
        sub_slice = [slice(None)] * len(sf.shape)
        sub_slice[-1] = slice(1, Fl)
        sf[tuple(sub_slice)] *= 2

    if autospectrum:
        return sf.real
    return sf


def multi_taper_psd(s, Fs=2 * np.pi, BW=None,  adaptive=False,
                    jackknife=True, low_bias=True, sides='default', NFFT=None):
    """Returns an estimate of the PSD function of s using the multitaper
    method. If the NW product, or the BW and Fs in Hz are not specified
    by the user, a bandwidth of 4 times the fundamental frequency,
    corresponding to NW = 4 will be used.

    Parameters
    ----------
    s : ndarray
       An array of sampled random processes, where the time axis is assumed to
       be on the last axis

    Fs: float
        Sampling rate of the signal

    BW: float
        The bandwidth of the windowing function will determine the number
        tapers to use. This parameters represents trade-off between frequency
        resolution (lower main lobe BW for the taper) and variance reduction
        (higher BW and number of averaged estimates).

    adaptive : {True/False}
       Use an adaptive weighting routine to combine the PSD estimates of
       different tapers.
    jackknife : {True/False}
       Use the jackknife method to make an estimate of the PSD variance
       at each point.
    low_bias : {True/False}
       Rather than use 2NW tapers, only use the tapers that have better than
       90% spectral concentration within the bandwidth (still using
       a maximum of 2NW tapers)
    sides : str (optional)   [ 'default' | 'onesided' | 'twosided' ]
         This determines which sides of the spectrum to return.
         For complex-valued inputs, the default is two-sided, for real-valued
         inputs, default is one-sided Indicates whether to return a one-sided
         or two-sided

    Returns
    -------
    (freqs, psd_est, var_or_nu) : ndarrays
        The first two arrays are the frequency points vector and the
        estimatated PSD. The last returned array differs depending on whether
        the jackknife was used. It is either

        * The jackknife estimated variance of the log-psd, OR
        * The degrees of freedom in a chi2 model of how the estimated
          PSD is distributed about the true log-PSD (this is either
          2*floor(2*NW), or calculated from adaptive weights)
    """
    # have last axis be time series for now
    N = s.shape[-1] if not NFFT else NFFT
    rest_of_dims = s.shape[:-1]

    s = s.reshape(int(np.product(rest_of_dims)), N)
    # de-mean this sucker
    s = utils.remove_bias(s, axis=-1)

    # Get the number of tapers from the sampling rate and the bandwidth:
    if BW is not None:
        NW = BW / (2 * Fs) * N
    else:
        NW = 4

    Kmax = int(2 * NW)

    dpss, eigs = dpss_windows(N, NW, Kmax)
    if low_bias:
        keepers = (eigs > 0.9)
        dpss = dpss[keepers]
        eigs = eigs[keepers]
        Kmax = len(dpss)

    # if the time series is a complex vector, a one sided PSD is invalid:
    if (sides == 'default' and np.iscomplexobj(s)) or sides == 'twosided':
        sides = 'twosided'
    elif sides in ('default', 'onesided'):
        sides = 'onesided'

    sig_sl = [slice(None)] * len(s.shape)
    sig_sl.insert(-1, np.newaxis)

    # tapered.shape is (..., Kmax, N)
    tapered = s[sig_sl] * dpss
    # Find the direct spectral estimators S_k(f) for k tapered signals..
    # don't normalize the periodograms by 1/N as normal.. since the taper
    # windows are orthonormal, they effectively scale the signal by 1/N

    # XXX: scipy fft is faster
    tapered_spectra = np.fft.fft(tapered)

    last_freq = N / 2 + 1 if sides == 'onesided' else N

    # degrees of freedom at each timeseries, at each freq
    nu = np.empty((s.shape[0], last_freq))
    if adaptive:
        weights = np.empty(tapered_spectra.shape[:-1] + (last_freq,))
        for i in xrange(s.shape[0]):
            weights[i], nu[i] = utils.adaptive_weights(
                tapered_spectra[i], eigs, sides=sides
                )
    else:
        # let the weights simply be the square-root of the eigenvalues.
        # repeat these values across all n_chan channels of data
        n_chan = tapered.shape[0]
        weights = np.tile(np.sqrt(eigs), n_chan).reshape(n_chan, Kmax, 1)
        nu.fill(2 * Kmax)

    if jackknife:
        jk_var = np.empty_like(nu)
        for i in xrange(s.shape[0]):
            jk_var[i] = utils.jackknifed_sdf_variance(
                tapered_spectra[i], eigs, sides=sides, adaptive=adaptive
                )

    # Compute the unbiased spectral estimator for S(f) as the sum of
    # the S_k(f) weighted by the function w_k(f)**2, all divided by the
    # sum of the w_k(f)**2 over k

    # 1st, roll the tapers axis forward
    tapered_spectra = np.rollaxis(tapered_spectra, 1, start=0)
    weights = np.rollaxis(weights, 1, start=0)
    sdf_est = mtm_cross_spectrum(
        tapered_spectra, tapered_spectra, weights, sides=sides
        )

    if sides == 'onesided':
        freqs = np.linspace(0, Fs / 2, N / 2 + 1)
    else:
        freqs = np.linspace(0, Fs, N, endpoint=False)

    out_shape = rest_of_dims + (len(freqs),)
    sdf_est.shape = out_shape
    # XXX: always return nu and jk_var
    if jackknife:
        jk_var.shape = out_shape
        return freqs, sdf_est, jk_var
    else:
        nu.shape = out_shape
        return freqs, sdf_est, nu


def multi_taper_csd(s, Fs=2 * np.pi, BW=None, low_bias=True,
                    adaptive=False, sides='default'):
    """Returns an estimate of the Cross Spectral Density (CSD) function
    between all (N choose 2) pairs of timeseries in s, using the multitaper
    method. If the NW product, or the BW and Fs in Hz are not specified by
    the user, a bandwidth of 4 times the fundamental frequency, corresponding
    to NW = 4 will be used.

    Parameters
    ----------
    s : ndarray
        An array of sampled random processes, where the time axis is
        assumed to be on the last axis. If ndim > 2, the number of time
        series to compare will still be taken as prod(s.shape[:-1])

    Fs: float, Sampling rate of the signal

    BW: float,
       The bandwidth of the windowing function will determine the number tapers
       to use. This parameters represents trade-off between frequency
       resolution (lower main lobe BW for the taper) and variance reduction
       (higher BW and number of averaged estimates).

    adaptive : {True, False}
       Use adaptive weighting to combine spectra
    low_bias : {True, False}
       Rather than use 2NW tapers, only use the tapers that have better than
       90% spectral concentration within the bandwidth (still using
       a maximum of 2NW tapers)
    sides : str (optional)   [ 'default' | 'onesided' | 'twosided' ]
         This determines which sides of the spectrum to return.  For
         complex-valued inputs, the default is two-sided, for real-valued
         inputs, default is one-sided Indicates whether to return a one-sided
         or two-sided

    Returns
    -------
    (freqs, csd_est) : ndarrays
        The estimatated CSD and the frequency points vector.
        The CSD{i,j}(f) are returned in a square "matrix" of vectors
        holding Sij(f). For an input array of (M,N), the output is (M,M,N)
    """
    # have last axis be time series for now
    N = s.shape[-1]
    rest_of = s.shape[:-1]
    M = int(np.product(rest_of))

    s = s.reshape(M, N)
    # de-mean this sucker
    s = utils.remove_bias(s, axis=-1)

    #Get the number of tapers from the sampling rate and the bandwidth:
    if BW is not None:
        NW = BW / (2 * Fs) * N
    else:
        NW = 4

    Kmax = int(2 * NW)

    dpss, eigvals = dpss_windows(N, NW, Kmax)
    if low_bias:
        keepers = (eigvals > 0.9)
        dpss = dpss[keepers]
        eigvals = eigvals[keepers]
        Kmax = len(dpss)

    # if the time series is a complex vector, a one sided PSD is invalid:
    if (sides == 'default' and np.iscomplexobj(s)) or sides == 'twosided':
        sides = 'twosided'
    elif sides in ('default', 'onesided'):
        sides = 'onesided'

    sig_sl = [slice(None)] * len(s.shape)
    sig_sl.insert(len(s.shape) - 1, np.newaxis)

    # tapered.shape is (M, Kmax, N)
    tapered = s[sig_sl] * dpss

    # compute the y_{i,k}(f)
    tapered_spectra = np.fft.fft(tapered)

    # compute the cross-spectral density functions
    last_freq = N / 2 + 1 if sides == 'onesided' else N

    if adaptive:
        w = np.empty(tapered_spectra.shape[:-1] + (last_freq,))
        nu = np.empty((M, last_freq))
        for i in xrange(M):
            w[i], nu[i] = utils.adaptive_weights(
                tapered_spectra[i], eigvals, sides=sides
                )
    else:
        weights = np.sqrt(eigvals).reshape(Kmax, 1)

    csdfs = np.empty((M, M, last_freq), 'D')
    for i in xrange(M):
        if adaptive:
            wi = w[i]
        else:
            wi = weights
        for j in xrange(i + 1):
            if adaptive:
                wj = w[j]
            else:
                wj = weights
            ti = tapered_spectra[i]
            tj = tapered_spectra[j]
            csdfs[i, j] = mtm_cross_spectrum(ti, tj, (wi, wj), sides=sides)

    upper_idc = triu_indices(M, k=1)
    lower_idc = tril_indices(M, k=-1)
    csdfs[upper_idc] = csdfs[lower_idc].conj()

    if sides == 'onesided':
        freqs = np.linspace(0, Fs / 2, N / 2 + 1)
    else:
        freqs = np.linspace(0, Fs, N, endpoint=False)

    return freqs, csdfs


def freq_response(b, a=1., n_freqs=1024, sides='onesided'):
    """
    Returns the frequency response of the IIR or FIR filter described
    by beta and alpha coefficients.

    Parameters
    ----------

    b : beta sequence (moving average component)
    a : alpha sequence (autoregressive component)
    n_freqs : size of frequency grid
    sides : {'onesided', 'twosided'}
       compute frequencies between [-PI,PI), or from [0, PI]

    Returns
    -------

    fgrid, H(e^jw)

    Notes
    -----
    For a description of the linear constant-coefficient difference equation,
    see
    http://en.wikipedia.org/wiki/Z-transform
    """
    # transitioning to scipy freqz
    real_n = n_freqs / 2 + 1 if sides == 'onesided' else n_freqs
    return sig.freqz(b, a=a, worN=real_n, whole=sides != 'onesided')
