"""

Event-related analysis

"""

import numpy as np
from nitime.lazy import scipy_linalg as linalg
from nitime.lazy import scipy_fftpack as fftpack


def fir(timeseries, design):
    """
    Calculate the FIR (finite impulse response) HRF, according to [Burock2000]_

    Parameters
    ----------

    timeseries : float array
            timeseries data

    design : int array
          This is a design matrix.  It has to have shape = (number
          of TRS, number of conditions * length of HRF)

          The form of the matrix is:

              A B C ...

          where A is a (number of TRs) x (length of HRF) matrix with a unity
          matrix placed with its top left corner placed in each TR in which
          event of type A occurred in the design. B is the equivalent for
          events of type B, etc.

    Returns
    -------

    HRF: float array
        HRF is a numpy array of 1X(length of HRF * number of conditions)
        with the HRFs for the different conditions concatenated. This is an
        estimate of the linear filters between the time-series and the events
        described in design.

    Notes
    -----

    Implements equation 4 in Burock(2000):

    .. math::

        \hat{h} = (X^T X)^{-1} X^T y

    M.A. Burock and A.M.Dale (2000). Estimation and Detection of Event-Related
    fMRI Signals with Temporally Correlated Noise: A Statistically Efficient
    and Unbiased Approach. Human Brain Mapping, 11:249-260

    """
    return linalg.pinv(design.T @design) @ design.T @ timeseries.T


def freq_domain_xcorr(tseries, events, t_before, t_after, Fs=1):
    """
    Calculates the  event related timeseries, using a cross-correlation in the
    frequency domain.

    Parameters
    ----------
    tseries: float array
       Time series data with time as the last dimension

    events: float array
       An array with time-resolved events, at the same sampling rate as tseries

    t_before: float
       Time before the event to include

    t_after: float
       Time after the event to include

    Fs: float
       Sampling rate of the time-series (in Hz)

    Returns
    -------
    xcorr: float array
        The correlation function between the tseries and the events. Can be
        interperted as a linear filter from events to responses (the
        time-series) of an LTI.

    """
    fft = fftpack.fft
    ifft = fftpack.ifft
    fftshift = fftpack.fftshift

    xcorr = np.real(fftshift(ifft(fft(tseries) *
                                  fft(np.fliplr([events])))))

    return xcorr[0][int(np.ceil(len(xcorr[0]) // 2) - t_before * Fs):
                    int(np.ceil(len(xcorr[0]) // 2) + t_after // 2 * Fs)] / np.sum(events)


def freq_domain_xcorr_zscored(tseries, events, t_before, t_after, Fs=1):
    """
    Calculates the z-scored event related timeseries, using a cross-correlation
    in the frequency domain.

    Parameters
    ----------
    tseries: float array
       Time series data with time as the last dimension

    events: float array
       An array with time-resolved events, at the same sampling rate as tseries

    t_before: float
       Time before the event to include

    t_after: float
       Time after the event to include

    Fs: float
       Sampling rate of the time-series (in Hz)

    Returns
    -------
    xcorr: float array
        The correlation function between the tseries and the events. Can be
        interperted as a linear filter from events to responses (the
        time-series) of an LTI. Because it is normalized to its own mean and
        variance, it can be interperted as measuring statistical significance
        relative to all time-shifted versions of the events.

    """

    fft = fftpack.fft
    ifft = fftpack.ifft
    fftshift = fftpack.fftshift

    xcorr = np.real(fftshift(ifft(fft(tseries) * fft(np.fliplr([events])))))

    meanSurr = np.mean(xcorr)
    stdSurr = np.std(xcorr)

    return (((xcorr[0][int(np.ceil(len(xcorr[0]) // 2) - t_before * Fs):
                       int(np.ceil(len(xcorr[0]) // 2) + t_after * Fs)]) -
             meanSurr) /
            stdSurr)
