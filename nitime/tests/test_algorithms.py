import os

import numpy as np
import numpy.testing as npt
from scipy.signal import signaltools

import nitime
from nitime import algorithms as tsa
from nitime import utils as ut

#Define globally
test_dir_path = os.path.join(nitime.__path__[0], 'tests')


def test_scipy_resample():
    """ Tests scipy signal's resample function
    """
    # create a freq list with max freq < 16 Hz
    freq_list = np.random.randint(0, high=15, size=5)
    # make a test signal with sampling freq = 64 Hz
    a = [np.sin(2 * np.pi * f * np.linspace(0, 1, 64, endpoint=False))
         for f in freq_list]
    tst = np.array(a).sum(axis=0)
    # interpolate to 128 Hz sampling
    t_up = signaltools.resample(tst, 128)
    np.testing.assert_array_almost_equal(t_up[::2], tst)
    # downsample to 32 Hz
    t_dn = signaltools.resample(tst, 32)
    np.testing.assert_array_almost_equal(t_dn, tst[::2])

    # downsample to 48 Hz, and compute the sampling analytically for comparison
    dn_samp_ana = np.array([np.sin(2 * np.pi * f * np.linspace(0, 1, 48, endpoint=False))
                            for f in freq_list]).sum(axis=0)
    t_dn2 = signaltools.resample(tst, 48)
    npt.assert_array_almost_equal(t_dn2, dn_samp_ana)


def test_dpss_windows():
    "Are the eigenvalues representing spectral concentration near unity"
    # these values from Percival and Walden 1993
    _, l = tsa.dpss_windows(31, 6, 4)
    unos = np.ones(4)
    yield npt.assert_array_almost_equal, l, unos
    _, l = tsa.dpss_windows(31, 7, 4)
    yield npt.assert_array_almost_equal, l, unos
    _, l = tsa.dpss_windows(31, 8, 4)
    yield npt.assert_array_almost_equal, l, unos
    _, l = tsa.dpss_windows(31, 8, 4.2)
    yield npt.assert_array_almost_equal, l, unos


def test_dpss_matlab():
    """Do the dpss windows resemble the equivalent matlab result

    The variable b is read in from a text file generated by issuing:

    dpss(100,2)

    in matlab

    """
    a, _ = tsa.dpss_windows(100, 2, 4)
    b = np.loadtxt(os.path.join(test_dir_path, 'dpss_matlab.txt'))
    npt.assert_almost_equal(a, b.T)


def test_periodogram():
    arsig, _, _ = ut.ar_generator(N=512)
    avg_pwr = (arsig * arsig.conjugate()).mean()
    f, psd = tsa.periodogram(arsig, N=2048)
    # for efficiency, let's leave out the 2PI in the numerator and denominator
    # for the following integral
    dw = 1. / 2048
    avg_pwr_est = np.trapz(psd, dx=dw)
    npt.assert_almost_equal(avg_pwr, avg_pwr_est, decimal=1)


def permutation_system(N):
    p = np.zeros((N, N))
    targets = range(N)
    for i in xrange(N):
        popper = np.random.randint(0, high=len(targets))
        j = targets.pop(popper)
        p[i, j] = 1
    return p


def test_boxcar_filter():
    a = np.random.rand(100)
    b = tsa.boxcar_filter(a)
    npt.assert_equal(a, b)

    #Should also work for odd number of elements:
    a = np.random.rand(99)
    b = tsa.boxcar_filter(a)
    npt.assert_equal(a, b)

    b = tsa.boxcar_filter(a, ub=0.25)
    npt.assert_equal(a.shape, b.shape)

    b = tsa.boxcar_filter(a, lb=0.25)
    npt.assert_equal(a.shape, b.shape)


def test_get_spectra():
    """Testing get_spectra"""
    t = np.linspace(0, 16 * np.pi, 2 ** 10)
    x = (np.sin(t) + np.sin(2 * t) + np.sin(3 * t) +
         0.1 * np.random.rand(t.shape[-1]))

    #First test for 1-d data:
    NFFT = 64
    N = x.shape[-1]
    f_welch = tsa.get_spectra(x, method={'this_method': 'welch', 'NFFT': NFFT})
    f_periodogram = tsa.get_spectra(x, method={'this_method': 'periodogram_csd'})
    f_multi_taper = tsa.get_spectra(x, method={'this_method': 'multi_taper_csd'})

    npt.assert_equal(f_welch[0].shape, (NFFT / 2 + 1,))
    npt.assert_equal(f_periodogram[0].shape, (N / 2 + 1,))
    npt.assert_equal(f_multi_taper[0].shape, (N / 2 + 1,))

    #Test for multi-channel data
    x = np.reshape(x, (2, x.shape[-1] / 2))
    N = x.shape[-1]

    #Make sure you get back the expected shape for different spectra:
    NFFT = 64
    f_welch = tsa.get_spectra(x, method={'this_method': 'welch', 'NFFT': NFFT})
    f_periodogram = tsa.get_spectra(x, method={'this_method': 'periodogram_csd'})
    f_multi_taper = tsa.get_spectra(x, method={'this_method': 'multi_taper_csd'})

    npt.assert_equal(f_welch[0].shape[0], NFFT / 2 + 1)
    npt.assert_equal(f_periodogram[0].shape[0], N / 2 + 1)
    npt.assert_equal(f_multi_taper[0].shape[0], N / 2 + 1)


def test_psd_matlab():

    """ Test the results of mlab csd/psd against saved results from Matlab"""

    from matplotlib import mlab

    test_dir_path = os.path.join(nitime.__path__[0], 'tests')

    ts = np.loadtxt(os.path.join(test_dir_path, 'tseries12.txt'))

    #Complex signal!
    ts0 = ts[1] + ts[0] * np.complex(0, 1)

    NFFT = 256
    Fs = 1.0
    noverlap = NFFT / 2

    fxx, f = mlab.psd(ts0, NFFT=NFFT, Fs=Fs, noverlap=noverlap,
                      scale_by_freq=True)

    fxx_mlab = np.fft.fftshift(fxx).squeeze()

    fxx_matlab = np.loadtxt(os.path.join(test_dir_path, 'fxx_matlab.txt'))

    npt.assert_almost_equal(fxx_mlab, fxx_matlab, decimal=5)
