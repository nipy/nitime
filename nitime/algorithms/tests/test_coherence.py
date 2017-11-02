"""

Tests of functions under algorithms.coherence

"""
import os
import warnings

import numpy as np
import numpy.testing as npt
from scipy.signal import signaltools
import pytest

import matplotlib
import matplotlib.mlab as mlab
has_mpl = True
# Matplotlib older than 0.99 will have some issues with the normalization
# of t:
if float(matplotlib.__version__[:3]) < 0.99:
    w_s = "You have a relatively old version of Matplotlib. "
    w_s += " Estimation of the PSD DC component might not be as expected."
    w_s +=" Consider updating Matplotlib: http://matplotlib.sourceforge.net/"
    warnings.warn(w_s, Warning)
    old_mpl = True
else:
    old_mpl = False

from scipy import fftpack

import nitime
import nitime.algorithms as tsa
import nitime.utils as utils

#Define globally
test_dir_path = os.path.join(nitime.__path__[0], 'tests')

# Define these once globally:
t = np.linspace(0, 16 * np.pi, 1024)
x = np.sin(t) + np.sin(2 * t) + np.sin(3 * t) + np.random.rand(t.shape[-1])
y = x + np.random.rand(t.shape[-1])

tseries = np.vstack([x, y])

methods = [None,
           {"this_method": 'multi_taper_csd', "Fs": 2 * np.pi},
           {"this_method": 'periodogram_csd', "Fs": 2 * np.pi, "NFFT": 256}]

if has_mpl:
    methods.append({"this_method": 'welch', "NFFT": 256, "Fs": 2 * np.pi,
                    "window": mlab.window_hanning(np.ones(256))})
    methods.append({"this_method": 'welch', "NFFT": 256, "Fs": 2 * np.pi})

def test_coherency():
    """
    Tests that the coherency algorithm runs smoothly, using the different
    csd routines, that the resulting matrix is symmetric and for the welch
    method, that the frequency bands in the output make sense
    """

    for method in methods:
        f, c = tsa.coherency(tseries, csd_method=method)

        npt.assert_array_almost_equal(c[0, 1], c[1, 0].conjugate())
        npt.assert_array_almost_equal(c[0, 0], np.ones(f.shape))

        if method is not None and method['this_method'] != "multi_taper_csd":
            f_theoretical = utils.get_freqs(method['Fs'], method['NFFT'])
            npt.assert_array_almost_equal(f, f_theoretical)


def test_coherence():
    """
    Tests that the coherency algorithm runs smoothly, using the different csd
    routines and that the result is symmetrical:
    """

    for method in methods:
        f, c = tsa.coherence(tseries, csd_method=method)
        npt.assert_array_almost_equal(c[0, 1], c[1, 0])
        npt.assert_array_almost_equal(c[0, 0], np.ones(f.shape))


def test_coherency_regularized():
    """
    Tests that the regularized coherency algorithm runs smoothly, using the
    different csd routines and that the result is symmetrical:
    """

    for method in methods:
        f, c = tsa.coherency_regularized(tseries, 0.05, 1000,
                                         csd_method=method)
        npt.assert_array_almost_equal(c[0, 1], c[1, 0].conjugate())


def test_coherence_regularized():
    """

    Tests that the regularized coherence algorithm runs smoothly, using the
    different csd routines and that the result is symmetrical:

    """
    for method in methods:
        f, c = tsa.coherence_regularized(tseries, 0.05, 1000,
                                         csd_method=method)
        npt.assert_array_almost_equal(c[0, 1], c[1, 0])


# Define as global for the following functions:

def test_coherency_bavg():
    ub = [np.pi / 2, None]
    lb = [0, 0.2]
    for method in methods:
        for this_lb in lb:
            for this_ub in ub:
                c = tsa.coherency_bavg(tseries, lb=this_lb, ub=this_ub,
                                       csd_method=method)

                #Test that this gets rid of the frequency axis:
                npt.assert_equal(len(c.shape), 2)
                # And that the result is equal
                npt.assert_almost_equal(c[0, 1], c[1, 0].conjugate())


def test_coherence_bavg():
    ub = [np.pi / 2, None]
    lb = [0, 0.2]
    for method in methods:
        for this_lb in lb:
            for this_ub in ub:
                c = tsa.coherence_bavg(tseries, lb=this_lb, ub=this_ub,
                                       csd_method=method)

                #Test that this gets rid of the frequency axis:
                npt.assert_equal(len(c.shape), 2)
                # And that the result is equal
                npt.assert_almost_equal(c[0, 1], c[1, 0].conjugate())


# XXX FIXME: This doesn't work for the periodogram method:
def test_coherence_partial():
    """ Test partial coherence"""

    x = np.sin(t) + np.sin(2 * t) + np.sin(3 * t) + np.random.rand(t.shape[-1])
    y = x + np.random.rand(t.shape[-1])
    z = y + np.random.rand(t.shape[-1])

    for method in methods:
        if (method is None) or method['this_method'] == 'welch':
            f, c = tsa.coherence_partial(np.vstack([x, y]), z,
                                         csd_method=method)
            npt.assert_array_almost_equal(c[0, 1], c[1, 0].conjugate())


def test_coherence_phase_delay():
    """

    Test the phase spectrum calculation

    """

    # Set up two time-series with a known phase delay:
    nz = np.random.rand(t.shape[-1])
    x = np.sin(t) + nz
    y = np.sin(t + np.pi) + nz

    tseries = np.vstack([x, y])
    for method in methods:
        f1, pdelay = tsa.coherency_phase_spectrum(tseries, csd_method=method)
        f2, tdelay = tsa.coherency_phase_delay(tseries, csd_method=method)
        npt.assert_almost_equal(pdelay[0, 1], -pdelay[1, 0])
        npt.assert_almost_equal(tdelay[0, 1], -tdelay[1, 0])
        # This is the relationship between these two quantities:
        npt.assert_almost_equal(tdelay[0, 1],
                                pdelay[0, 1][1:] / (2 * np.pi * f2))


def test_coherency_cached():
    """Tests that the cached coherency gives the same result as the standard
    coherency"""

    f1, c1 = tsa.coherency(tseries)

    ij = [(0, 1), (1, 0)]
    f2, cache = tsa.cache_fft(tseries, ij)

    c2 = tsa.cache_to_coherency(cache, ij)

    npt.assert_array_almost_equal(c1[1, 0], c2[1, 0])
    npt.assert_array_almost_equal(c1[0, 1], c2[0, 1])


def test_correlation_spectrum():
    """

    Test the correlation spectrum method

    """
    # Smoke-test for now - unclear what to test here...
    f, c = tsa.correlation_spectrum(x, y, norm=True)


# XXX FIXME: http://github.com/nipy/nitime/issues/issue/1
@pytest.mark.skipif(True, reason="http://github.com/nipy/nitime/issues/issue/1")
def test_coherence_linear_dependence():
    """
    Tests that the coherence between two linearly dependent time-series
    behaves as expected.

    From William Wei's book, according to eq. 14.5.34, if two time-series are
    linearly related through:

    y(t)  = alpha*x(t+time_shift)

    then the coherence between them should be equal to:

    .. :math:

    C(\nu) = \frac{1}{1+\frac{fft_{noise}(\nu)}{fft_{x}(\nu) \cdot \alpha^2}}

    """
    t = np.linspace(0, 16 * np.pi, 2 ** 14)
    x = (np.sin(t) + np.sin(2 * t) + np.sin(3 * t) +
         0.1 * np.random.rand(t.shape[-1]))
    N = x.shape[-1]

    alpha = 10
    m = 3
    noise = 0.1 * np.random.randn(t.shape[-1])
    y = alpha * np.roll(x, m) + noise

    f_noise = fftpack.fft(noise)[0:N // 2]
    f_x = fftpack.fft(x)[0:N // 2]

    c_t = (1 / (1 + (f_noise / (f_x * (alpha ** 2)))))

    method = {"this_method": 'welch',
              "NFFT": 2048,
              "Fs": 2 * np.pi}

    f, c = tsa.coherence(np.vstack([x, y]), csd_method=method)
    c_t = np.abs(signaltools.resample(c_t, c.shape[-1]))

    npt.assert_array_almost_equal(c[0, 1], c_t, 2)


def test_coherence_matlab():

    """ Test against coherence values calculated with matlab's mscohere"""

    ts = np.loadtxt(os.path.join(test_dir_path, 'tseries12.txt'))

    ts0 = ts[1]
    ts1 = ts[0]

    method = {}
    method['this_method'] = 'welch'
    method['NFFT'] = 64
    method['Fs'] = 1.0
    method['noverlap'] = method['NFFT'] // 2

    ttt = np.vstack([ts0, ts1])
    f, cxy_mlab = tsa.coherence(ttt, csd_method=method)
    cxy_matlab = np.loadtxt(os.path.join(test_dir_path, 'cxy_matlab.txt'))

    npt.assert_almost_equal(cxy_mlab[0][1], cxy_matlab, decimal=5)

@pytest.mark.skipif(old_mpl, reason="MPL version before 0.99")
def test_cached_coherence():
    """Testing the cached coherence functions """
    NFFT = 64  # This is the default behavior
    n_freqs = NFFT // 2 + 1
    ij = [(0, 1), (1, 0)]
    ts = np.loadtxt(os.path.join(test_dir_path, 'tseries12.txt'))
    freqs, cache = tsa.cache_fft(ts, ij)

    # Are the frequencies the right ones?
    npt.assert_equal(freqs, utils.get_freqs(2 * np.pi, NFFT))

    # Check that the fft of the first window is what we expect:
    hann = mlab.window_hanning(np.ones(NFFT))
    w_ts = ts[0][:NFFT] * hann
    w_ft = fftpack.fft(w_ts)[0:n_freqs]

    # This is the result of the function:
    first_window_fft = cache['FFT_slices'][0][0]

    npt.assert_equal(w_ft, first_window_fft)

    coh_cached = tsa.cache_to_coherency(cache, ij)[0, 1]
    f, c = tsa.coherency(ts)
    coh_direct = c[0, 1]

    npt.assert_almost_equal(coh_direct, coh_cached)

    # Only welch PSD works and an error is thrown otherwise. This tests that
    # the error is thrown:
    with pytest.raises(ValueError) as e_info:
        tsa.cache_fft(ts, ij, method=methods[2])

    # Take the method in which the window is defined on input:
    freqs, cache1 = tsa.cache_fft(ts, ij, method=methods[3])
    # And compare it to the method in which it isn't:
    freqs, cache2 = tsa.cache_fft(ts, ij, method=methods[4])
    npt.assert_equal(cache1, cache2)

    # Do the same, while setting scale_by_freq to False:
    freqs, cache1 = tsa.cache_fft(ts, ij, method=methods[3],
                                  scale_by_freq=False)
    freqs, cache2 = tsa.cache_fft(ts, ij, method=methods[4],
                                  scale_by_freq=False)
    npt.assert_equal(cache1, cache2)

    # Test cache_to_psd:
    psd1 = tsa.cache_to_psd(cache, ij)[0]
    # Against the standard get_spectra:
    f, c = tsa.get_spectra(ts)
    psd2 = c[0][0]

    npt.assert_almost_equal(psd1, psd2)

    # Test that prefer_speed_over_memory doesn't change anything:
    freqs, cache1 = tsa.cache_fft(ts, ij)
    freqs, cache2 = tsa.cache_fft(ts, ij, prefer_speed_over_memory=True)
    psd1 = tsa.cache_to_psd(cache1, ij)[0]
    psd2 = tsa.cache_to_psd(cache2, ij)[0]
    npt.assert_almost_equal(psd1, psd2)


# XXX This is not testing anything substantial for now - I am not sure what to
# test here...
def test_cache_to_phase():
    """
    Test phase calculations from cached windowed FFT

    """
    ij = [(0, 1), (1, 0)]
    x = np.sin(t) + np.sin(2 * t) + np.sin(3 * t) + np.random.rand(t.shape[-1])
    y = np.sin(t) + np.sin(2 * t) + np.sin(3 * t) + np.random.rand(t.shape[-1])
    ts = np.vstack([x, y])
    freqs, cache = tsa.cache_fft(ts, ij)
    ph = tsa.cache_to_phase(cache, ij)


def test_cache_to_coherency():
    """

    Test cache_to_coherency against the standard coherency calculation

    """
    ij = [(0, 1), (1, 0)]
    ts = np.loadtxt(os.path.join(test_dir_path, 'tseries12.txt'))
    freqs, cache = tsa.cache_fft(ts, ij)
    Cxy = tsa.cache_to_coherency(cache, ij)
    f, c = tsa.coherency(ts)
    npt.assert_almost_equal(Cxy[0][1], c[0, 1])

    # Check that it doesn't matter if you prefer_speed_over_memory:
    freqs, cache2 = tsa.cache_fft(ts, ij, prefer_speed_over_memory=True)
    Cxy2 = tsa.cache_to_coherency(cache2, ij)

    npt.assert_equal(Cxy2, Cxy)

    # XXX Calculating the angle of the averaged psd and calculating the average
    # of the angles calculated over different windows does not yield exactly
    # the same number, because the angle is not a linear functions (arctan),
    # so it is unclear how to test this, but we make sure that it runs,
    # whether or not you prefer_speed_over_memory:
    freqs, cache = tsa.cache_fft(ts, ij)
    tsa.cache_to_relative_phase(cache, ij)

    freqs, cache = tsa.cache_fft(ts, ij, prefer_speed_over_memory=True)
    tsa.cache_to_relative_phase(cache, ij)

    # Check that things run alright, even if there is just one window for the
    # entire ts:
    freqs, cache = tsa.cache_fft(ts, ij, method=dict(this_method='welch',
                                                   NFFT=ts.shape[-1],
                                                   n_overlap=0))

    cxy_one_window = tsa.cache_to_coherency(cache, ij)
    ph_one_window = tsa.cache_to_relative_phase(cache, ij)

    # And whether or not you prefer_speed_over_memory
    freqs, cache = tsa.cache_fft(ts, ij, method=dict(this_method='welch',
                                                   NFFT=ts.shape[-1],
                                                   n_overlap=0),
                                prefer_speed_over_memory=True)

    cxy_one_window = tsa.cache_to_coherency(cache, ij)
    ph_one_window = tsa.cache_to_relative_phase(cache, ij)
