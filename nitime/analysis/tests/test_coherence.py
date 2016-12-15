import warnings

import numpy as np
import numpy.testing as npt
import matplotlib
import matplotlib.mlab as mlab
import pytest

import nitime.timeseries as ts
import nitime.analysis as nta

import platform

# Some tests might require python version 2.5 or above:
if float(platform.python_version()[:3]) < 2.5:
    old_python = True
else:
    old_python = False

# Matplotlib older than 0.99 will have some issues with the normalization of t

if float(matplotlib.__version__[:3]) < 0.99:
    w_s = "You have a relatively old version of Matplotlib. "
    w_s += " Estimation of the PSD DC component might not be as expected"
    w_s += " Consider updating Matplotlib: http://matplotlib.sourceforge.net/"
    warnings.warn(w_s, Warning)
    old_mpl = True
else:
    old_mpl = False

def test_CoherenceAnalyzer():
    methods = (None,
           {"this_method": 'welch', "NFFT": 256},
           {"this_method": 'multi_taper_csd'},
           {"this_method": 'periodogram_csd', "NFFT": 256})

    Fs = np.pi
    t = np.arange(1024)
    x = np.sin(10 * t) + np.random.rand(t.shape[-1])
    y = np.sin(10 * t) + np.random.rand(t.shape[-1])
    # Third time-series used for calculation of partial coherence:
    z = np.sin(10 * t)
    T = ts.TimeSeries(np.vstack([x, y, z]), sampling_rate=np.pi)
    n_series = T.shape[0]
    for unwrap in [True, False]:
        for method in methods:
            C = nta.CoherenceAnalyzer(T, method, unwrap_phases=unwrap)
            if method is None:
                # This is the default behavior (grab the NFFT from the number
                # of frequencies):
                npt.assert_equal(C.coherence.shape, (n_series, n_series,
                                                     C.frequencies.shape[0]))

            elif (method['this_method'] == 'welch' or
                  method['this_method'] == 'periodogram_csd'):
                npt.assert_equal(C.coherence.shape, (n_series, n_series,
                                                     method['NFFT'] // 2 + 1))
            else:
                npt.assert_equal(C.coherence.shape, (n_series, n_series,
                                                     len(t) // 2 + 1))

            # Coherence symmetry:
            npt.assert_equal(C.coherence[0, 1], C.coherence[1, 0])

            # Phase/delay asymmetry:
            npt.assert_equal(C.phase[0, 1], -1 * C.phase[1, 0])

            # The very first one is a nan, test from second and onwards:
            npt.assert_almost_equal(C.delay[0, 1][1:], -1 * C.delay[1, 0][1:])

            if method is not None and method['this_method'] == 'welch':
                S = nta.SpectralAnalyzer(T, method)
                npt.assert_almost_equal(S.cpsd[0], C.frequencies)
                npt.assert_almost_equal(S.cpsd[1], C.spectrum)
            # Test that partial coherence runs through and has the right number
            # of dimensions:
            npt.assert_equal(len(C.coherence_partial.shape), 4)


@pytest.mark.skipif(old_mpl, reason="Old MPL")
def test_SparseCoherenceAnalyzer():
    Fs = np.pi
    t = np.arange(256)
    x = np.sin(10 * t) + np.random.rand(t.shape[-1])
    y = np.sin(10 * t) + np.random.rand(t.shape[-1])
    T = ts.TimeSeries(np.vstack([x, y]), sampling_rate=Fs)
    C1 = nta.SparseCoherenceAnalyzer(T, ij=((0, 1), (1, 0)))
    C2 = nta.CoherenceAnalyzer(T)

    # Coherence symmetry:
    npt.assert_almost_equal(np.abs(C1.coherence[0, 1]),
                            np.abs(C1.coherence[1, 0]))
    npt.assert_almost_equal(np.abs(C1.coherency[0, 1]),
                            np.abs(C1.coherency[1, 0]))

    # Make sure you get the same answers as you would from the standard
    # CoherenceAnalyzer:

    npt.assert_almost_equal(C2.coherence[0, 1], C1.coherence[0, 1])
    # This is the PSD (for the first time-series in the object):
    npt.assert_almost_equal(C2.spectrum[0, 0], C1.spectrum[0])
    # And the second (for good measure):
    npt.assert_almost_equal(C2.spectrum[1, 1], C1.spectrum[1])

    # The relative phases should be equal
    npt.assert_almost_equal(C2.phase[0, 1], C1.relative_phases[0, 1])
    # But not the absolute phases (which have the same shape):
    npt.assert_equal(C1.phases[0].shape, C1.relative_phases[0, 1].shape)

    # The delay is equal:
    npt.assert_almost_equal(C2.delay[0, 1], C1.delay[0, 1])
    # Make sure that you would get an error if you provided a method other than
    # 'welch':
    with pytest.raises(ValueError) as e_info:
        nta.SparseCoherenceAnalyzer(T, method=dict(this_method='foo'))


def test_MTCoherenceAnalyzer():
    """Test the functionality of the multi-taper spectral coherence """

    Fs = np.pi
    t = np.arange(256)
    x = np.sin(10 * t) + np.random.rand(t.shape[-1])
    y = np.sin(10 * t) + np.random.rand(t.shape[-1])
    T = ts.TimeSeries(np.vstack([x, y]), sampling_rate=Fs)
    n_series = T.shape[0]
    NFFT = t.shape[0] // 2 + 1
    for adaptive in [True, False]:
        C = nta.MTCoherenceAnalyzer(T, adaptive=adaptive)
        npt.assert_equal(C.frequencies.shape[0], NFFT)
        npt.assert_equal(C.coherence.shape, (n_series, n_series, NFFT))
        npt.assert_equal(C.confidence_interval.shape, (n_series, n_series,
                                                       NFFT))


@pytest.mark.skipif(old_python, reason="Old Python")
def test_warn_short_tseries():
    """

    A warning is provided when the time-series is shorter than
    the NFFT + n_overlap.

    The implementation of this test is based on this:
    http://docs.python.org/library/warnings.html#testing-warnings

    """

    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        # The following should throw a warning, because 70 is smaller than the
        # default NFFT=64 + n_overlap=32:
        nta.CoherenceAnalyzer(ts.TimeSeries(np.random.rand(2, 70),
                                            sampling_rate=1))
        # Verify some things
        npt.assert_equal(len(w), 1)


def test_SeedCoherenceAnalyzer():
    """ Test the SeedCoherenceAnalyzer """
    methods = (None,
           {"this_method": 'welch', "NFFT": 256},
           {"this_method": 'multi_taper_csd'},
           {"this_method": 'periodogram_csd', "NFFT": 256})

    Fs = np.pi
    t = np.arange(256)
    seed1 = np.sin(10 * t) + np.random.rand(t.shape[-1])
    seed2 = np.sin(10 * t) + np.random.rand(t.shape[-1])
    target = np.sin(10 * t) + np.random.rand(t.shape[-1])
    T = ts.TimeSeries(np.vstack([seed1, target]), sampling_rate=Fs)
    T_seed1 = ts.TimeSeries(seed1, sampling_rate=Fs)
    T_seed2 = ts.TimeSeries(np.vstack([seed1, seed2]), sampling_rate=Fs)
    T_target = ts.TimeSeries(np.vstack([seed1, target]), sampling_rate=Fs)
    for this_method in methods:
        if this_method is None or this_method['this_method'] == 'welch':
            C1 = nta.CoherenceAnalyzer(T, method=this_method)
            C2 = nta.SeedCoherenceAnalyzer(T_seed1, T_target,
                                           method=this_method)
            C3 = nta.SeedCoherenceAnalyzer(T_seed2, T_target,
                                           method=this_method)

            npt.assert_almost_equal(C1.coherence[0, 1], C2.coherence[1])
            npt.assert_almost_equal(C2.coherence[1], C3.coherence[0, 1])
            npt.assert_almost_equal(C1.phase[0, 1], C2.relative_phases[1])
            npt.assert_almost_equal(C1.delay[0, 1], C2.delay[1])

        else:
            with pytest.raises(ValueError) as e_info:
                nta.SeedCoherenceAnalyzer(T_seed1, T_target, this_method)


def test_SeedCoherenceAnalyzer_same_Fs():
    """

    Providing two time-series with different sampling rates throws an error

    """

    Fs1 = np.pi
    Fs2 = 2 * np.pi
    t = np.arange(256)

    T1 = ts.TimeSeries(np.random.rand(t.shape[-1]),
                       sampling_rate=Fs1)

    T2 = ts.TimeSeries(np.random.rand(t.shape[-1]),
                       sampling_rate=Fs2)
    with pytest.raises(ValueError) as e_info:
        nta.SeedCoherenceAnalyzer(T1, T2)
