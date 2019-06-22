"""

Coherency is an analogue of correlation, calculated in the frequency
domain. This is a useful quantity for describing a system of oscillators
coupled with delay. This is because the coherency captures not only the
magnitude of the time-shift-independent correlation between the time-series
(termed 'coherence'), but can also be used in order to estimate the size of the
time-delay (the phase-delay between the time-series in a particular frequency
band).

"""

import numpy as np
from nitime.lazy import scipy_fftpack as fftpack
from nitime.lazy import matplotlib_mlab as mlab

from .spectral import get_spectra, get_spectra_bi
import nitime.utils as utils

# To support older versions of numpy that don't have tril_indices:
from nitime.index_utils import tril_indices


def coherency(time_series, csd_method=None):
    r"""
    Compute the coherency between the spectra of n-tuple of time series.
    Input to this function is in the time domain

    Parameters
    ----------

    time_series : n*t float array
       an array of n different time series of length t each

    csd_method : dict, optional.
       See :func:`get_spectra` documentation for details

    Returns
    -------

    f : float array
        The central frequencies for the frequency bands for which the spectra
        are estimated

    c : float array
        This is a symmetric matrix with the coherencys of the signals. The
        coherency of signal i and signal j is in f[i][j]. Note that f[i][j] =
        f[j][i].conj()

    Notes
    -----

    This is an implementation of equation (1) of Sun (2005):

    .. math::

        R_{xy} (\lambda) = \frac{f_{xy}(\lambda)}
        {\sqrt{f_{xx} (\lambda) \cdot f_{yy}(\lambda)}}

    F.T. Sun and L.M. Miller and M. D'Esposito (2005). Measuring temporal
    dynamics of functional networks using phase spectrum of fMRI
    data. Neuroimage, 28: 227-37.

    """
    if csd_method is None:
        csd_method = {'this_method': 'welch'}  # The default

    f, fxy = get_spectra(time_series, csd_method)

    # A container for the coherencys, with the size and shape of the expected
    # output:
    c = np.zeros((time_series.shape[0],
                  time_series.shape[0],
                  f.shape[0]), dtype=complex)  # Make sure it's complex

    for i in range(time_series.shape[0]):
        for j in range(i, time_series.shape[0]):
            c[i][j] = coherency_spec(fxy[i][j], fxy[i][i], fxy[j][j])

    idx = tril_indices(time_series.shape[0], -1)
    c[idx[0], idx[1], ...] = c[idx[1], idx[0], ...].conj()  # Make it symmetric

    return f, c


def coherency_spec(fxy, fxx, fyy):
    r"""
    Compute the coherency between the spectra of two time series.

    Input to this function is in the frequency domain.

    Parameters
    ----------

    fxy : float array
         The cross-spectrum of the time series

    fyy,fxx : float array
         The spectra of the signals

    Returns
    -------

    complex array
        the frequency-band-dependent coherency

    See also
    --------
    :func:`coherency`
    """
    return fxy / np.sqrt(fxx * fyy)


def coherence(time_series, csd_method=None):
    r"""Compute the coherence between the spectra of an n-tuple of time_series.

    Parameters of this function are in the time domain.

    Parameters
    ----------
    time_series : float array
       an array of different time series with time as the last dimension

    csd_method : dict, optional
       See :func:`algorithms.spectral.get_spectra` documentation for details

    Returns
    -------
    f : float array
        The central frequencies for the frequency bands for which the spectra
        are estimated

    c : float array
        This is a symmetric matrix with the coherencys of the signals. The
        coherency of signal i and signal j is in f[i][j].

    Notes
    -----

    This is an implementation of equation (2) of Sun (2005):

    .. math::

        Coh_{xy}(\lambda) = |{R_{xy}(\lambda)}|^2 =
        \frac{|{f_{xy}(\lambda)}|^2}{f_{xx}(\lambda) \cdot f_{yy}(\lambda)}

    F.T. Sun and L.M. Miller and M. D'Esposito (2005). Measuring temporal
    dynamics of functional networks using phase spectrum of fMRI data.
    Neuroimage, 28: 227-37.

    """
    if csd_method is None:
        csd_method = {'this_method': 'welch'}  # The default

    f, fxy = get_spectra(time_series, csd_method)

    # A container for the coherences, with the size and shape of the expected
    # output:
    c = np.zeros((time_series.shape[0],
                  time_series.shape[0],
                  f.shape[0]))

    for i in range(time_series.shape[0]):
        for j in range(i, time_series.shape[0]):
            c[i][j] = coherence_spec(fxy[i][j], fxy[i][i], fxy[j][j])

    idx = tril_indices(time_series.shape[0], -1)
    c[idx[0], idx[1], ...] = c[idx[1], idx[0], ...].conj()  # Make it symmetric

    return f, c


def coherence_spec(fxy, fxx, fyy):
    r"""
    Compute the coherence between the spectra of two time series.

    Parameters of this function are in the frequency domain.

    Parameters
    ----------

    fxy : array
         The cross-spectrum of the time series

    fyy, fxx : array
         The spectra of the signals

    Returns
    -------

    float : a frequency-band-dependent measure of the linear association
        between the two time series

    See also
    --------
    :func:`coherence`
    """
    if not np.isrealobj(fxx):
        fxx = np.real(fxx)
    if not np.isrealobj(fyy):
        fyy = np.real(fyy)
    c = np.abs(fxy) ** 2 / (fxx * fyy)
    return c


def coherency_regularized(time_series, epsilon, alpha, csd_method=None):
    r"""
    Compute a regularized measure of the coherence.

    Regularization may be needed in order to overcome numerical imprecisions

    Parameters
    ----------

    time_series: float array
        The time series data for which the regularized coherence is
        calculated. Time as the last dimension.

    epsilon: float
        Small regularization parameter. Should be much smaller than any
        meaningful value of coherence you might encounter

    alpha: float
        Large regularization parameter. Should be much larger than any
        meaningful value of coherence you might encounter (preferably much
        larger than 1).

    csd_method: dict, optional.
        See :func:`get_spectra` documentation for details

    Returns
    -------
    f: float array
        The central frequencies for the frequency bands for which the spectra
        are estimated

    c: float array
        This is a symmetric matrix with the coherencys of the signals. The
        coherency of signal i and signal j is in f[i][j]. Note that f[i][j] =
        f[j][i].conj()


    Notes
    -----
    The regularization scheme is as follows:

    .. math::

        Coh_{xy}^R = \frac{(\alpha f_{xx} + \epsilon) ^2}
                          {\alpha^{2}(f_{xx}+\epsilon)(f_{yy}+\epsilon)}


    """
    if csd_method is None:
        csd_method = {'this_method': 'welch'}  # The default

    f, fxy = get_spectra(time_series, csd_method)

    # A container for the coherences, with the size and shape of the expected
    # output:
    c = np.zeros((time_series.shape[0],
                  time_series.shape[0],
                  f.shape[0]), dtype=complex)  # Make sure it's complex

    for i in range(time_series.shape[0]):
        for j in range(i, time_series.shape[0]):
            c[i][j] = _coherency_reqularized(fxy[i][j], fxy[i][i],
                                             fxy[j][j], epsilon, alpha)

    idx = tril_indices(time_series.shape[0], -1)
    c[idx[0], idx[1], ...] = c[idx[1], idx[0], ...].conj()  # Make it symmetric

    return f, c


def _coherency_reqularized(fxy, fxx, fyy, epsilon, alpha):
    r"""
    A regularized version of the calculation of coherency, which is more
    robust to numerical noise than the standard calculation

    Input to this function is in the frequency domain.

    Parameters
    ----------

    fxy, fxx, fyy: float arrays
        The cross- and power-spectral densities of the two signals x and y

    epsilon: float
        First regularization parameter. Should be much smaller than any
        meaningful value of coherence you might encounter

    alpha: float
        Second regularization parameter. Should be much larger than any
        meaningful value of coherence you might encounter (preferably much
        larger than 1).

    Returns
    -------
    float array
        The coherence values

    """
    return (((alpha * fxy + epsilon)) /
            np.sqrt(((alpha ** 2) * (fxx + epsilon) * (fyy + epsilon))))


def coherence_regularized(time_series, epsilon, alpha, csd_method=None):
    r"""
    Same as coherence, except regularized in order to overcome numerical
    imprecisions

    Parameters
    ----------

    time_series: n-d float array
       The time series data for which the regularized coherence is calculated

    epsilon: float
       Small regularization parameter. Should be much smaller than any
       meaningful value of coherence you might encounter

    alpha: float
       large regularization parameter. Should be much larger than any
       meaningful value of coherence you might encounter (preferably much
       larger than 1).

    csd_method: dict, optional.
       See :func:`get_spectra` documentation for details

    Returns
    -------
    f: float array
       The central frequencies for the frequency bands for which the spectra
       are estimated

    c: n-d array
       This is a symmetric matrix with the coherencys of the signals. The
       coherency of signal i and signal j is in f[i][j].


    Notes
    -----
    The regularization scheme is as follows:

    .. math::

        C_{x,y} = \frac{(\alpha f_{xx} + \epsilon)^2}
        {\alpha^{2}((f_{xx}+\epsilon)(f_{yy}+\epsilon))}

    """
    if csd_method is None:
        csd_method = {'this_method': 'welch'}  # The default

    f, fxy = get_spectra(time_series, csd_method)

    # A container for the coherences, with the size and shape of the expected
    # output:
    c = np.zeros((time_series.shape[0],
                  time_series.shape[0],
                  f.shape[0]), complex)

    for i in range(time_series.shape[0]):
        for j in range(i, time_series.shape[0]):
            c[i][j] = _coherence_reqularized(fxy[i][j], fxy[i][i],
                                             fxy[j][j], epsilon, alpha)

    idx = tril_indices(time_series.shape[0], -1)
    c[idx[0], idx[1], ...] = c[idx[1], idx[0], ...].conj()  # Make it symmetric

    return f, c


def _coherence_reqularized(fxy, fxx, fyy, epsilon, alpha):
    r"""A regularized version of the calculation of coherence, which is more
    robust to numerical noise than the standard calculation.

    Input to this function is in the frequency domain

    Parameters
    ----------

    fxy, fxx, fyy: float arrays
        The cross- and power-spectral densities of the two signals x and y

    epsilon: float
        First regularization parameter. Should be much smaller than any
        meaningful value of coherence you might encounter

    alpha: float
        Second regularization parameter. Should be much larger than any
        meaningful value of coherence you might encounter (preferably much
        larger than 1)

    Returns
    -------
    float array
       The coherence values

    """
    return (((alpha * np.abs(fxy) + epsilon) ** 2) /
            ((alpha ** 2) * (fxx + epsilon) * (fyy + epsilon)))


def coherency_bavg(time_series, lb=0, ub=None, csd_method=None):
    r"""
    Compute the band-averaged coherency between the spectra of two time series.

    Input to this function is in the time domain.

    Parameters
    ----------
    time_series: n*t float array
       an array of n different time series of length t each

    lb, ub: float, optional
       the upper and lower bound on the frequency band to be used in averaging
       defaults to 1,max(f)

    csd_method: dict, optional.
       See :func:`get_spectra` documentation for details

    Returns
    -------
    c: float array
        This is an upper-diagonal array, where c[i][j] is the band-averaged
        coherency between time_series[i] and time_series[j]

    Notes
    -----

    This is an implementation of equation (A4) of Sun(2005):

    .. math::

        \bar{Coh_{xy}} (\bar{\lambda}) =
        \frac{\left|{\sum_\lambda{\hat{f_{xy}}}}\right|^2}
        {\sum_\lambda{\hat{f_{xx}}}\cdot sum_\lambda{\hat{f_{yy}}}}

    F.T. Sun and L.M. Miller and M. D'Esposito (2005). Measuring
    temporal dynamics of functional networks using phase spectrum of fMRI
    data. Neuroimage, 28: 227-37.
    """
    if csd_method is None:
        csd_method = {'this_method': 'welch'}  # The default

    f, fxy = get_spectra(time_series, csd_method)

    lb_idx, ub_idx = utils.get_bounds(f, lb, ub)

    if lb == 0:
        lb_idx = 1  # The lowest frequency band should be f0

    c = np.zeros((time_series.shape[0],
                  time_series.shape[0]), dtype=complex)

    for i in range(time_series.shape[0]):
        for j in range(i, time_series.shape[0]):
            c[i][j] = _coherency_bavg(fxy[i][j][lb_idx:ub_idx],
                                      fxy[i][i][lb_idx:ub_idx],
                                      fxy[j][j][lb_idx:ub_idx])

    idx = tril_indices(time_series.shape[0], -1)
    c[idx[0], idx[1], ...] = c[idx[1], idx[0], ...].conj()  # Make it symmetric

    return c


def _coherency_bavg(fxy, fxx, fyy):
    r"""
    Compute the band-averaged coherency between the spectra of two time series.

    Input to this function is in the frequency domain.

    Parameters
    ----------

    fxy : float array
         The cross-spectrum of the time series

    fyy,fxx : float array
         The spectra of the signals

    Returns
    -------

    float
        the band-averaged coherency

    Notes
    -----

    This is an implementation of equation (A4) of [Sun2005]_:

    .. math::

        \bar{Coh_{xy}} (\bar{\lambda}) =
        \frac{\left|{\sum_\lambda{\hat{f_{xy}}}}\right|^2}
        {\sum_\lambda{\hat{f_{xx}}}\cdot sum_\lambda{\hat{f_{yy}}}}

    .. [Sun2005] F.T. Sun and L.M. Miller and M. D'Esposito(2005). Measuring
        temporal dynamics of functional networks using phase spectrum of fMRI
        data. Neuroimage, 28: 227-37.
    """
    # Average the phases and the magnitudes separately and then recombine:

    p = np.angle(fxy)
    p_bavg = np.mean(p)

    m = np.abs(coherency_spec(fxy, fxx, fyy))
    m_bavg = np.mean(m)

    # Recombine according to z = r(cos(phi)+sin(phi)i):
    return m_bavg * (np.cos(p_bavg) + np.sin(p_bavg) * 1j)


def coherence_bavg(time_series, lb=0, ub=None, csd_method=None):
    r"""
    Compute the band-averaged coherence between the spectra of two time series.

    Input to this function is in the time domain.

    Parameters
    ----------
    time_series : float array
       An array of time series, time as the last dimension.

    lb, ub: float, optional
       The upper and lower bound on the frequency band to be used in averaging
       defaults to 1,max(f)

    csd_method: dict, optional.
       See :func:`get_spectra` documentation for details

    Returns
    -------
    c : float
       This is an upper-diagonal array, where c[i][j] is the band-averaged
       coherency between time_series[i] and time_series[j]
    """
    if csd_method is None:
        csd_method = {'this_method': 'welch'}  # The default

    f, fxy = get_spectra(time_series, csd_method)

    lb_idx, ub_idx = utils.get_bounds(f, lb, ub)

    if lb == 0:
        lb_idx = 1  # The lowest frequency band should be f0

    c = np.zeros((time_series.shape[0],
                  time_series.shape[0]))

    for i in range(time_series.shape[0]):
        for j in range(i, time_series.shape[0]):
            c[i][j] = _coherence_bavg(fxy[i][j][lb_idx:ub_idx],
                                      fxy[i][i][lb_idx:ub_idx],
                                      fxy[j][j][lb_idx:ub_idx])

    idx = tril_indices(time_series.shape[0], -1)
    c[idx[0], idx[1], ...] = c[idx[1], idx[0], ...].conj()  # Make it symmetric

    return c


def _coherence_bavg(fxy, fxx, fyy):
    r"""
    Compute the band-averaged coherency between the spectra of two time series.

    Input to this function is in the frequency domain

    Parameters
    ----------

    fxy : float array
         The cross-spectrum of the time series

    fyy,fxx : float array
         The spectra of the signals

    Returns
    -------

    float :
        the band-averaged coherence
    """
    if not np.isrealobj(fxx):
        fxx = np.real(fxx)
    if not np.isrealobj(fyy):
        fyy = np.real(fyy)

    return (np.abs(fxy.sum()) ** 2) / (fxx.sum() * fyy.sum())


def coherence_partial(time_series, r, csd_method=None):
    r"""
    Compute the band-specific partial coherence between the spectra of
    two time series.

    The partial coherence is the part of the coherence between x and
    y, which cannot be attributed to a common cause, r.

    Input to this function is in the time domain.

    Parameters
    ----------

    time_series: float array
       An array of time-series, with time as the last dimension.

    r: float array
        This array represents the temporal sequence of the common cause to be
        partialed out, sampled at the same rate as time_series

    csd_method: dict, optional
       See :func:`get_spectra` documentation for details


    Returns
    -------
    f: array,
        The mid-frequencies of the frequency bands in the spectral
        decomposition

    c: float array
       The frequency dependent partial coherence between time_series i and
       time_series j in c[i][j] and in c[j][i], with r partialed out


    Notes
    -----

    This is an implementation of equation (2) of Sun (2004):

    .. math::

        Coh_{xy|r} = \frac{|{R_{xy}(\lambda) - R_{xr}(\lambda)
        R_{ry}(\lambda)}|^2}{(1-|{R_{xr}}|^2)(1-|{R_{ry}}|^2)}

    F.T. Sun and L.M. Miller and M. D'Esposito (2004). Measuring interregional
    functional connectivity using coherence and partial coherence analyses of
    fMRI data Neuroimage, 21: 647-58.
    """
    if csd_method is None:
        csd_method = {'this_method': 'welch'}  # The default

    f, fxy = get_spectra(time_series, csd_method)

    # Initialize c according to the size of f:
    c = np.zeros((time_series.shape[0],
                  time_series.shape[0],
                  f.shape[0]), dtype=complex)

    for i in range(time_series.shape[0]):
        for j in range(i, time_series.shape[0]):
            f, fxx, frr, frx = get_spectra_bi(time_series[i], r, csd_method)
            f, fyy, frr, fry = get_spectra_bi(time_series[j], r, csd_method)
            c[i, j] = coherence_partial_spec(fxy[i][j],
                                             fxy[i][i],
                                             fxy[j][j],
                                             frx,
                                             fry,
                                             frr)

    idx = tril_indices(time_series.shape[0], -1)
    c[idx[0], idx[1], ...] = c[idx[1], idx[0], ...].conj()  # Make it symmetric

    return f, c


def coherence_partial_spec(fxy, fxx, fyy, fxr, fry, frr):
    r"""
    Compute the band-specific partial coherence between the spectra of
    two time series. See :func:`partial_coherence`.

    Input to this function is in the frequency domain.

    Parameters
    ----------
    fxy : float array
         The cross-spectrum of the time series

    fyy, fxx : float array
         The spectra of the signals

    fxr, fry : float array
         The cross-spectra of the signals with the event

    Returns
    -------
    float
        the band-averaged coherency
    """
    coh = coherency_spec
    Rxr = coh(fxr, fxx, frr)
    Rry = coh(fry, fyy, frr)
    Rxy = coh(fxy, fxx, fyy)

    return (((np.abs(Rxy - Rxr * Rry)) ** 2) /
            ((1 - ((np.abs(Rxr)) ** 2)) * (1 - ((np.abs(Rry)) ** 2))))


def coherency_phase_spectrum(time_series, csd_method=None):
    r"""
    Compute the phase spectrum of the cross-spectrum between two time series.

    The parameters of this function are in the time domain.

    Parameters
    ----------

    time_series : n*t float array
    The time series, with t, time, as the last dimension

    Returns
    -------

    f : mid frequencies of the bands

    p : an array with the pairwise phase spectrum between the time
    series, where p[i][j] is the phase spectrum between time series[i] and
    time_series[j]

    Notes
    -----

    This is an implementation of equation (3) of Sun et al. (2005) [Sun2005]_:

    .. math::

        \phi(\lambda) = arg [R_{xy} (\lambda)] = arg [f_{xy} (\lambda)]

    F.T. Sun and L.M. Miller and M. D'Esposito (2005). Measuring temporal
    dynamics of functional networks using phase spectrum of fMRI data.
    Neuroimage, 28: 227-37.
    """
    if csd_method is None:
        csd_method = {'this_method': 'welch'}  # The default

    f, fxy = get_spectra(time_series, csd_method)

    p = np.zeros((time_series.shape[0],
                  time_series.shape[0],
                  f.shape[0]))

    for i in range(time_series.shape[0]):
        for j in range(i + 1, time_series.shape[0]):
            p[i][j] = np.angle(fxy[i][j])
            p[j][i] = np.angle(fxy[i][j].conjugate())

    return f, p


def coherency_phase_delay(time_series, lb=0, ub=None, csd_method=None):
    """
    The temporal delay calculated from the coherency phase spectrum.

    Parameters
    ----------

    time_series: float array
       The time-series data for which the delay is calculated.

    lb, ub: float
       Frequency boundaries (in Hz), for the domain over which the delays are
       calculated. Defaults to 0-max(f)

    csd_method : dict, optional.
       See :func:`get_spectra`

    Returns
    -------
    f : float array
       The mid-frequencies for the frequency bands over which the calculation
       is done.
    p : float array
       Pairwise temporal delays between time-series (in seconds).

    """
    if csd_method is None:
        csd_method = {'this_method': 'welch'}  # The default

    f, fxy = get_spectra(time_series, csd_method)

    lb_idx, ub_idx = utils.get_bounds(f, lb, ub)

    if lb_idx == 0:
        lb_idx = 1

    p = np.zeros((time_series.shape[0], time_series.shape[0],
                  f[lb_idx:ub_idx].shape[-1]))

    for i in range(time_series.shape[0]):
        for j in range(i, time_series.shape[0]):
            p[i][j] = _coherency_phase_delay(f[lb_idx:ub_idx],
                                             fxy[i][j][lb_idx:ub_idx])
            p[j][i] = _coherency_phase_delay(
                                f[lb_idx:ub_idx],
                                fxy[i][j][lb_idx:ub_idx].conjugate())

    return f[lb_idx:ub_idx], p


def _coherency_phase_delay(f, fxy):
    r"""
    Compute the phase delay between the spectra of two signals. The input to
    this function is in the frequency domain.

    Parameters
    ----------

    f: float array
         The frequencies

    fxy : float array
         The cross-spectrum of the time series

    Returns
    -------

    float array
        the phase delay (in sec) for each frequency band.

    """
    return np.angle(fxy) / (2 * np.pi * f)


def correlation_spectrum(x1, x2, Fs=2 * np.pi, norm=False):
    """
    Calculate the spectral decomposition of the correlation.

    Parameters
    ----------
    x1,x2: ndarray
       Two arrays to be correlated. Same dimensions

    Fs: float, optional
       Sampling rate in Hz. If provided, an array of
       frequencies will be returned.Defaults to 2

    norm: bool, optional
       When this is true, the spectrum is normalized to sum to 1

    Returns
    -------
    f: ndarray
       ndarray with the frequencies

    ccn: ndarray
       The spectral decomposition of the correlation

    Notes
    -----

    This method is described in full in: D Cordes, V M Haughton, K Arfanakis, G
    J Wendt, P A Turski, C H Moritz, M A Quigley, M E Meyerand (2000). Mapping
    functionally related regions of brain with functional connectivity MR
    imaging. AJNR American journal of neuroradiology 21:1636-44
    """
    x1 = x1 - np.mean(x1)
    x2 = x2 - np.mean(x2)
    x1_f = fftpack.fft(x1)
    x2_f = fftpack.fft(x2)
    D = np.sqrt(np.sum(x1 ** 2) * np.sum(x2 ** 2))
    n = x1.shape[0]

    ccn = ((np.real(x1_f) * np.real(x2_f) +
           np.imag(x1_f) * np.imag(x2_f)) /
           (D * n))

    if norm:
        # Only half of the sum is sent back because of the freq domain
        # symmetry.
        ccn = ccn / np.sum(ccn) * 2
        # XXX Does normalization make this strictly positive?

    f = utils.get_freqs(Fs, n)
    return f, ccn[0:(n // 2 + 1)]


# -----------------------------------------------------------------------
# Coherency calculated using cached spectra
# -----------------------------------------------------------------------
"""The idea behind this set of functions is to keep a cache of the windowed fft
calculations of each time-series in a massive collection of time-series, so
that this calculation doesn't have to be repeated each time a cross-spectrum is
calculated. The first function creates the cache and then, another function
takes the cached spectra and calculates PSDs and CSDs, which are then passed to
coherency_spec and organized in a data structure similar to the one
created by coherence"""


def cache_fft(time_series, ij, lb=0, ub=None,
              method=None, prefer_speed_over_memory=False,
              scale_by_freq=True):
    """compute and cache the windowed FFTs of the time_series, in such a way
    that computing the psd and csd of any combination of them can be done
    quickly.

    Parameters
    ----------

    time_series : float array
       An ndarray with time-series, where time is the last dimension

    ij: list of tuples
      Each tuple in this variable should contain a pair of
      indices of the form (i,j). The resulting cache will contain the fft of
      time-series in the rows indexed by the unique elements of the union of i
      and j

    lb,ub: float
       Define a frequency band of interest, for which the fft will be cached

    method: dict, optional
        See :func:`get_spectra` for details on how this is used. For this set
        of functions, 'this_method' has to be 'welch'


    Returns
    -------
    freqs, cache

        where: cache =
             {'FFT_slices':FFT_slices,'FFT_conj_slices':FFT_conj_slices,
             'norm_val':norm_val}

    Notes
    -----

    - For these functions, only the Welch windowed periodogram ('welch') is
      available.

    - Detrending the input is not an option here, in order to save
      time on an empty function call.

    """
    if method is None:
        method = {'this_method': 'welch'}  # The default

    this_method = method.get('this_method', 'welch')

    if this_method == 'welch':
        NFFT = method.get('NFFT', 64)
        Fs = method.get('Fs', 2 * np.pi)
        window = method.get('window', mlab.window_hanning)
        n_overlap = method.get('n_overlap', int(np.ceil(NFFT / 2.0)))
    else:
        e_s = "For cache_fft, spectral estimation method must be welch"
        raise ValueError(e_s)
    time_series = utils.zero_pad(time_series, NFFT)

    # The shape of the zero-padded version:
    n_channels, n_time_points = time_series.shape

    # get all the unique channels in time_series that we are interested in by
    # checking the ij tuples
    all_channels = set()
    for i, j in ij:
        all_channels.add(i)
        all_channels.add(j)

    # for real time_series, ignore the negative frequencies
    if np.iscomplexobj(time_series):
        n_freqs = NFFT
    else:
        n_freqs = NFFT // 2 + 1

    # Which frequencies
    freqs = utils.get_freqs(Fs, NFFT)

    # If there are bounds, limit the calculation to within that band,
    # potentially include the DC component:
    lb_idx, ub_idx = utils.get_bounds(freqs, lb, ub)

    n_freqs = ub_idx - lb_idx
    # Make the window:
    if np.iterable(window):
        assert(len(window) == NFFT)
        window_vals = window
    else:
        window_vals = window(np.ones(NFFT, time_series.dtype))

    # Each fft needs to be normalized by the square of the norm of the window
    # and, for consistency with newer versions of mlab.csd (which, in turn, are
    # consistent with Matlab), normalize also by the sampling rate:

    if scale_by_freq:
        # This is the normalization factor for one-sided estimation, taking
        # into account the sampling rate. This makes the PSD a density
        # function, with units of dB/Hz, so that integrating over
        # frequencies gives you the RMS. (XXX this should be in the tests!).
        norm_val = (np.abs(window_vals) ** 2).sum() * (Fs / 2)

    else:
        norm_val = (np.abs(window_vals) ** 2).sum() / 2

    # cache the FFT of every windowed, detrended NFFT length segment
    # of every channel.  If prefer_speed_over_memory, cache the conjugate
    # as well

    i_times = list(range(0, n_time_points - NFFT + 1, NFFT - n_overlap))
    n_slices = len(i_times)
    FFT_slices = {}
    FFT_conj_slices = {}

    for i_channel in all_channels:
        Slices = np.zeros((n_slices, n_freqs), dtype=np.complex)
        for iSlice in range(n_slices):
            thisSlice = time_series[i_channel,
                                    i_times[iSlice]:i_times[iSlice] + NFFT]

            # Windowing:
            thisSlice = window_vals * thisSlice  # No detrending
            # Derive the fft for that slice:
            Slices[iSlice, :] = (fftpack.fft(thisSlice)[lb_idx:ub_idx])

        FFT_slices[i_channel] = Slices

        if prefer_speed_over_memory:
            FFT_conj_slices[i_channel] = np.conjugate(Slices)

    cache = {'FFT_slices': FFT_slices, 'FFT_conj_slices': FFT_conj_slices,
             'norm_val': norm_val, 'Fs': Fs, 'scale_by_freq': scale_by_freq}

    return freqs, cache


def cache_to_psd(cache, ij):
    """
    From a set of cached windowed fft, calculate the psd

    Parameters
    ----------
    cache : dict
        Return value from :func:`cache_fft`

    ij : list
        A list of tuples of the form (i,j).

    Returns
    -------
    Pxx : dict
        The phases for the intersection of (time_series[i],time_series[j]). The
        keys are the intersection of i,j values in the parameter ij

    """
    # This is the way it is saved by cache_spectra:
    FFT_slices = cache['FFT_slices']
    FFT_conj_slices = cache['FFT_conj_slices']
    norm_val = cache['norm_val']
    # Fs = cache['Fs']

    # This is where the output goes to:
    Pxx = {}
    all_channels = set()
    for i, j in ij:
        all_channels.add(i)
        all_channels.add(j)

    for i in all_channels:
        # If we made the conjugate slices:
        if FFT_conj_slices:
            Pxx[i] = FFT_slices[i] * FFT_conj_slices[i]
        else:
            Pxx[i] = FFT_slices[i] * np.conjugate(FFT_slices[i])

        # If there is more than one window
        if FFT_slices[i].shape[0] > 1:
            Pxx[i] = np.mean(Pxx[i], 0)

        Pxx[i] /= norm_val
        # Correct for the NFFT/2 and DC components:
        Pxx[i][[0, -1]] /= 2

    return Pxx


def cache_to_phase(cache, ij):
    """ From a set of cached set of windowed fft's, calculate the
    frequency-band dependent phase for each of the channels in ij.
    Note that this returns the absolute phases of the time-series, not the
    relative phases between them. In order to get relative phases, use
    cache_to_relative_phase

    Parameters
    ----------
    cache : dict
         The return value of  :func:`cache_fft`

    ij: list
       A list of tuples of the form (i,j) for all the indices for which to
       calculate the phases

    Returns
    -------

    Phase : dict
         The individual phases, keys are all the i and j in ij, such that
         Phase[i] gives you the phase for the time-series i in the input to
         :func:`cache_fft`

    """
    FFT_slices = cache['FFT_slices']

    Phase = {}

    all_channels = set()
    for i, j in ij:
        all_channels.add(i)
        all_channels.add(j)

    for i in all_channels:
        Phase[i] = np.angle(FFT_slices[i])
        # If there is more than one window, average over all the windows:
        if FFT_slices[i].shape[0] > 1:
            Phase[i] = np.mean(Phase[i], 0)

    return Phase


def cache_to_relative_phase(cache, ij):
    """ From a set of cached set of windowed fft's, calculate the
    frequency-band dependent relative phase for the combinations ij.

    Parameters
    ----------
    cache: dict
        The return value from :func:`cache_fft`

    ij: list
       A list of tuples of the form (i,j), all the pairs of indices for which
       to calculate the relative phases

    Returns
    -------

    Phi_xy : dict
        The relative phases between the time-series i and j. Such that
        Phi_xy[i,j] is the phase from time_series[i] to time_series[j].

    Note
    ----

    This function will give you a different result than using
    :func:`coherency_phase_spectrum`. This is because
    :func:`coherency_phase_spectrum` calculates the angle based on the average
    psd, whereas this function calculates the average of the angles calculated
    on individual windows.

    """
    # This is the way it is saved by cache_spectra:
    FFT_slices = cache['FFT_slices']
    FFT_conj_slices = cache['FFT_conj_slices']
    # norm_val = cache['norm_val']

    freqs = cache['FFT_slices'][ij[0][0]].shape[-1]

    ij_array = np.array(ij)

    channels_i = max(1, max(ij_array[:, 0]) + 1)
    channels_j = max(1, max(ij_array[:, 1]) + 1)
    # Pre-allocate for speed:
    Phi_xy = np.zeros((channels_i, channels_j, freqs), dtype=np.complex)

    # These checks take time, so do them up front, not in every iteration:
    if list(FFT_slices.items())[0][1].shape[0] > 1:
        if FFT_conj_slices:
            for i, j in ij:
                phi = np.angle(FFT_slices[i] * FFT_conj_slices[j])
                Phi_xy[i, j] = np.mean(phi, 0)

        else:
            for i, j in ij:
                phi = np.angle(FFT_slices[i] * np.conjugate(FFT_slices[j]))
                Phi_xy[i, j] = np.mean(phi, 0)

    else:
        if FFT_conj_slices:
            for i, j in ij:
                Phi_xy[i, j] = np.angle(FFT_slices[i] * FFT_conj_slices[j])

        else:
            for i, j in ij:
                Phi_xy[i, j] = np.angle(FFT_slices[i] *
                                        np.conjugate(FFT_slices[j]))

    return Phi_xy


def cache_to_coherency(cache, ij):
    """From a set of cached spectra, calculate the coherency
    relationships

    Parameters
    ----------
    cache: dict
        the return value from :func:`cache_fft`

    ij: list
      a list of (i,j) tuples, the pairs of indices for which the
      cross-coherency is to be calculated

    Returns
    -------
    Cxy: dict
       coherence values between the time-series ij. Indexing into this dict
       takes the form Cxy[i,j] in order to extract the coherency between
       time-series i and time-series j in the original input to
       :func:`cache_fft`
    """

    #This is the way it is saved by cache_spectra:
    FFT_slices = cache['FFT_slices']
    FFT_conj_slices = cache['FFT_conj_slices']
    norm_val = cache['norm_val']

    freqs = cache['FFT_slices'][ij[0][0]].shape[-1]

    ij_array = np.array(ij)

    channels_i = max(1, max(ij_array[:, 0]) + 1)
    channels_j = max(1, max(ij_array[:, 1]) + 1)
    Cxy = np.zeros((channels_i, channels_j, freqs), dtype=np.complex)

    #These checks take time, so do them up front, not in every iteration:
    if list(FFT_slices.items())[0][1].shape[0] > 1:
        if FFT_conj_slices:
            for i, j in ij:
                #dbg:
                #print i,j
                Pxy = FFT_slices[i] * FFT_conj_slices[j]
                Pxx = FFT_slices[i] * FFT_conj_slices[i]
                Pyy = FFT_slices[j] * FFT_conj_slices[j]
                Pxx = np.mean(Pxx, 0)
                Pyy = np.mean(Pyy, 0)
                Pxy = np.mean(Pxy, 0)
                Pxy /= norm_val
                Pxx /= norm_val
                Pyy /= norm_val
                Cxy[i, j] = Pxy / np.sqrt(Pxx * Pyy)

        else:
            for i, j in ij:
                Pxy = FFT_slices[i] * np.conjugate(FFT_slices[j])
                Pxx = FFT_slices[i] * np.conjugate(FFT_slices[i])
                Pyy = FFT_slices[j] * np.conjugate(FFT_slices[j])
                Pxx = np.mean(Pxx, 0)
                Pyy = np.mean(Pyy, 0)
                Pxy = np.mean(Pxy, 0)
                Pxy /= norm_val
                Pxx /= norm_val
                Pyy /= norm_val
                Cxy[i, j] = Pxy / np.sqrt(Pxx * Pyy)
    else:
        if FFT_conj_slices:
            for i, j in ij:
                Pxy = FFT_slices[i] * FFT_conj_slices[j]
                Pxx = FFT_slices[i] * FFT_conj_slices[i]
                Pyy = FFT_slices[j] * FFT_conj_slices[j]
                Pxy /= norm_val
                Pxx /= norm_val
                Pyy /= norm_val
                Cxy[i, j] = Pxy / np.sqrt(Pxx * Pyy)

        else:
            for i, j in ij:
                Pxy = FFT_slices[i] * np.conjugate(FFT_slices[j])
                Pxx = FFT_slices[i] * np.conjugate(FFT_slices[i])
                Pyy = FFT_slices[j] * np.conjugate(FFT_slices[j])
                Pxy /= norm_val
                Pxx /= norm_val
                Pyy /= norm_val
                Cxy[i, j] = Pxy / np.sqrt(Pxx * Pyy)

    return Cxy
