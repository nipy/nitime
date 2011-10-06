"""

Wavelets

"""

import numpy as np
from nitime.lazy import scipy_fftpack as fftpack


def wfmorlet_fft(f0, sd, sampling_rate, ns=5, nt=None):
    """
    returns a complex morlet wavelet in the frequency domain

    Parameters
    ----------
        f0 : center frequency
        sd : standard deviation of center frequency
        sampling_rate : samplingrate
        ns : window length in number of standard deviations
        nt : window length in number of sample points
    """
    if nt == None:
        st = 1. / (2. * np.pi * sd)
        nt = 2 * int(ns * st * sampling_rate) + 1
    f = fftpack.fftfreq(nt, 1. / sampling_rate)
    wf = 2 * np.exp(-(f - f0) ** 2 / (2 * sd ** 2)) * np.sqrt(sampling_rate /
                                                    (np.sqrt(np.pi) * sd))
    wf[f < 0] = 0
    wf[f == 0] /= 2
    return wf


def wmorlet(f0, sd, sampling_rate, ns=5, normed='area'):
    """
    returns a complex morlet wavelet in the time domain

    Parameters
    ----------
        f0 : center frequency
        sd : standard deviation of frequency
        sampling_rate : samplingrate
        ns : window length in number of standard deviations
    """
    st = 1. / (2. * np.pi * sd)
    w_sz = float(int(ns * st * sampling_rate))  # half time window size
    t = np.arange(-w_sz, w_sz + 1, dtype=float) / sampling_rate
    if normed == 'area':
        w = np.exp(-t ** 2 / (2. * st ** 2)) * np.exp(
            2j * np.pi * f0 * t) / np.sqrt(np.sqrt(np.pi) * st * sampling_rate)
    elif normed == 'max':
        w = np.exp(-t ** 2 / (2. * st ** 2)) * np.exp(
            2j * np.pi * f0 * t) * 2 * sd * np.sqrt(2 * np.pi) / sampling_rate
    else:
        assert 0, 'unknown norm %s' % normed
    return w


def wlogmorlet_fft(f0, sd, sampling_rate, ns=5, nt=None):
    """
    returns a complex log morlet wavelet in the frequency domain

    Parameters
    ----------
        f0 : center frequency
        sd : standard deviation
        sampling_rate : samplingrate
        ns : window length in number of standard deviations
        nt : window length in number of sample points
    """
    if nt == None:
        st = 1. / (2. * np.pi * sd)
        nt = 2 * int(ns * st * sampling_rate) + 1
    f = fftpack.fftfreq(nt, 1. / sampling_rate)

    sfl = np.log(1 + 1. * sd / f0)
    wf = (2 * np.exp(-(np.log(f) - np.log(f0)) ** 2 / (2 * sfl ** 2)) *
          np.sqrt(sampling_rate / (np.sqrt(np.pi) * sd)))
    wf[f < 0] = 0
    wf[f == 0] /= 2
    return wf


def wlogmorlet(f0, sd, sampling_rate, ns=5, normed='area'):
    """
    returns a complex log morlet wavelet in the time domain

    Parameters
    ----------
        f0 : center frequency
        sd : standard deviation of frequency
        sampling_rate : samplingrate
        ns : window length in number of standard deviations
    """
    st = 1. / (2. * np.pi * sd)
    w_sz = int(ns * st * sampling_rate)  # half time window size
    wf = wlogmorlet_fft(f0, sd, sampling_rate=sampling_rate, nt=2 * w_sz + 1)
    w = fftpack.fftshift(fftpack.ifft(wf))
    if normed == 'area':
        w /= w.real.sum()
    elif normed == 'max':
        w /= w.real.max()
    elif normed == 'energy':
        w /= np.sqrt((w ** 2).sum())
    else:
        assert 0, 'unknown norm %s' % normed
    return w
