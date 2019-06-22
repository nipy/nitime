"""
Tests for the algorithms.spectral submodule

"""

import numpy as np
import scipy
from scipy import fftpack
import numpy.testing as npt
import pytest

import nitime.algorithms as tsa
import nitime.utils as utils


def test_get_spectra():
    """

    Testing spectral estimation

    """

    methods = (None,
           {"this_method": 'welch', "NFFT": 256, "Fs": 2 * np.pi},
           {"this_method": 'welch', "NFFT": 1024, "Fs": 2 * np.pi})

    for method in methods:
        avg_pwr1 = []
        avg_pwr2 = []
        est_pwr1 = []
        est_pwr2 = []
        arsig1, _, _ = utils.ar_generator(N=2 ** 16)  # needs to be that long
                                                  # for the answers to converge
        arsig2, _, _ = utils.ar_generator(N=2 ** 16)

        avg_pwr1.append((arsig1 ** 2).mean())
        avg_pwr2.append((arsig2 ** 2).mean())

        tseries = np.vstack([arsig1, arsig2])

        f, c = tsa.get_spectra(tseries, method=method)

        # \sum_{\omega} psd d\omega:
        est_pwr1.append(np.sum(c[0, 0]) * (f[1] - f[0]))
        est_pwr2.append(np.sum(c[1, 1]) * (f[1] - f[0]))

        # Get it right within the order of magnitude:
        npt.assert_array_almost_equal(est_pwr1, avg_pwr1, decimal=-1)
        npt.assert_array_almost_equal(est_pwr2, avg_pwr2, decimal=-1)


def test_get_spectra_complex():
    """

    Testing spectral estimation

    """

    methods = (None,
           {"this_method": 'welch', "NFFT": 256, "Fs": 2 * np.pi},
           {"this_method": 'welch', "NFFT": 1024, "Fs": 2 * np.pi})

    for method in methods:
        avg_pwr1 = []
        avg_pwr2 = []
        est_pwr1 = []
        est_pwr2 = []

        # Make complex signals:
        r, _, _ = utils.ar_generator(N=2 ** 16)  # It needs to be that long for
                                                 # the answers to converge
        c, _, _ = utils.ar_generator(N=2 ** 16)
        arsig1 = r + c * scipy.sqrt(-1)

        r, _, _ = utils.ar_generator(N=2 ** 16)
        c, _, _ = utils.ar_generator(N=2 ** 16)

        arsig2 = r + c * scipy.sqrt(-1)
        avg_pwr1.append((arsig1 * arsig1.conjugate()).mean())
        avg_pwr2.append((arsig2 * arsig2.conjugate()).mean())

        tseries = np.vstack([arsig1, arsig2])

        f, c = tsa.get_spectra(tseries, method=method)

        # \sum_{\omega} psd d\omega:
        est_pwr1.append(np.sum(c[0, 0]) * (f[1] - f[0]))
        est_pwr2.append(np.sum(c[1, 1]) * (f[1] - f[0]))

        # Get it right within the order of magnitude:
        npt.assert_array_almost_equal(est_pwr1, avg_pwr1, decimal=-1)
        npt.assert_array_almost_equal(est_pwr2, avg_pwr2, decimal=-1)


def test_get_spectra_unknown_method():
    """
    Test that providing an unknown method to get_spectra rasies a ValueError

    """
    tseries = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError) as e_info:
        tsa.get_spectra(tseries, method=dict(this_method='foo'))


def test_periodogram():
    """Test some of the inputs to periodogram """

    arsig, _, _ = utils.ar_generator(N=1024)
    Sk = fftpack.fft(arsig)

    f1, c1 = tsa.periodogram(arsig)
    f2, c2 = tsa.periodogram(arsig, Sk=Sk)

    npt.assert_equal(c1, c2)

    # Check that providing a complex signal does the right thing
    # (i.e. two-sided spectrum):
    N = 1024
    r, _, _ = utils.ar_generator(N=N)
    c, _, _ = utils.ar_generator(N=N)
    arsig = r + c * scipy.sqrt(-1)

    f, c = tsa.periodogram(arsig)
    npt.assert_equal(f.shape[0], N)  # Should be N, not the one-sided N/2 + 1


def test_periodogram_csd():
    """Test corner cases of  periodogram_csd"""

    arsig1, _, _ = utils.ar_generator(N=1024)
    arsig2, _, _ = utils.ar_generator(N=1024)

    tseries = np.vstack([arsig1, arsig2])

    Sk = fftpack.fft(tseries)

    f1, c1 = tsa.periodogram_csd(tseries)
    f2, c2 = tsa.periodogram_csd(tseries, Sk=Sk)
    npt.assert_equal(c1, c2)

    # Check that providing a complex signal does the right thing
    # (i.e. two-sided spectrum):
    N = 1024
    r, _, _ = utils.ar_generator(N=N)
    c, _, _ = utils.ar_generator(N=N)
    arsig1 = r + c * scipy.sqrt(-1)

    r, _, _ = utils.ar_generator(N=N)
    c, _, _ = utils.ar_generator(N=N)
    arsig2 = r + c * scipy.sqrt(-1)

    tseries = np.vstack([arsig1, arsig2])

    f, c = tsa.periodogram_csd(tseries)
    npt.assert_equal(f.shape[0], N)  # Should be N, not the one-sided N/2 + 1


def test_dpss_windows():
    """ Test a couple of funky corner cases of DPSS_windows """

    N = 1024
    NW = 0  # Setting NW to 0 triggers the weird corner case in which some of
            # the symmetric tapers have a negative average
    Kmax = 7

    # But that's corrected by the algorithm:
    d, w = tsa.dpss_windows(1024, 0, 7)
    for this_d in d[0::2]:
        npt.assert_equal(this_d.sum(axis=-1) < 0, False)

    # Make sure we interpolate to the proper number of points
    d, w = tsa.dpss_windows(245411, 4, 8, 1000)
    npt.assert_equal(d.shape[-1], 245411)


def test_dpss_properties():
    """ Test conventions of Slepian eigenvectors """

    N = 2000
    NW = 200
    d, lam = tsa.dpss_windows(N, NW, 2*NW-2)
    # 2NW-2 lamdas should be all > 0.9
    npt.assert_(
        (lam > 0.9).all(), 'Eigenvectors show poor spectral concentration'
        )
    # test orthonomality
    err = np.linalg.norm(d.dot(d.T) - np.eye(2*NW-2), ord='fro')
    npt.assert_(err**2 < 1e-16, 'Eigenvectors not numerically orthonormal')
    # test positivity of even functions
    npt.assert_(
        (d[::2].sum(axis=1) > 0).all(),
        'Even Slepian sequences should have positive DC'
        )
    # test positive initial slope of odd functions
    # (this tests the sign of a linear slope)
    pk = np.argmax(np.abs(d[1::2, :N//2]), axis=1)
    t = True
    for p, f in zip(pk, d[1::2]):
        t = t and np.sum( np.arange(1,p+1) * f[:p] ) >= 0
    npt.assert_(t, 'Odd Slepians should begin positive-going')

def test_get_spectra_bi():
    """

    Test the bi-variate get_spectra function

    """

    methods = (None,
           {"this_method": 'welch', "NFFT": 256, "Fs": 2 * np.pi},
           {"this_method": 'welch', "NFFT": 1024, "Fs": 2 * np.pi})

    for method in methods:
        arsig1, _, _ = utils.ar_generator(N=2 ** 16)
        arsig2, _, _ = utils.ar_generator(N=2 ** 16)

        avg_pwr1 = (arsig1 ** 2).mean()
        avg_pwr2 = (arsig2 ** 2).mean()
        avg_xpwr = (arsig1 * arsig2.conjugate()).mean()

        tseries = np.vstack([arsig1, arsig2])

        f, fxx, fyy, fxy = tsa.get_spectra_bi(arsig1, arsig2, method=method)

        # \sum_{\omega} PSD(\omega) d\omega:
        est_pwr1 = np.sum(fxx * (f[1] - f[0]))
        est_pwr2 = np.sum(fyy * (f[1] - f[0]))
        est_xpwr = np.sum(fxy * (f[1] - f[0])).real

        # Test that we have the right order of magnitude:
        npt.assert_array_almost_equal(est_pwr1, avg_pwr1, decimal=-1)
        npt.assert_array_almost_equal(est_pwr2, avg_pwr2, decimal=-1)
        npt.assert_array_almost_equal(np.mean(est_xpwr),
                                      np.mean(avg_xpwr),
                                      decimal=-1)


def test_mtm_lin_combo():
    "Test the functionality of cross and autospectrum MTM combinations"
    spec1 = np.random.randn(5, 100) + 1j * np.random.randn(5, 100)
    spec2 = np.random.randn(5, 100) + 1j * np.random.randn(5, 100)
    # test on both broadcasted weights and per-point weights
    for wshape in ((2, 5, 1), (2, 5, 100)):
        weights = np.random.randn(*wshape)
        sides = 'onesided'
        mtm_cross = tsa.mtm_cross_spectrum(
            spec1, spec2, (weights[0], weights[1]), sides=sides
            )
        npt.assert_(mtm_cross.dtype in np.sctypes['complex'],
               'Wrong dtype for crossspectrum')
        npt.assert_(len(mtm_cross) == 51,
               'Wrong length for halfband spectrum')
        sides = 'twosided'
        mtm_cross = tsa.mtm_cross_spectrum(
            spec1, spec2, (weights[0], weights[1]), sides=sides
            )
        npt.assert_(len(mtm_cross) == 100,
               'Wrong length for fullband spectrum')
        sides = 'onesided'
        mtm_auto = tsa.mtm_cross_spectrum(
            spec1, spec1, weights[0], sides=sides
            )
        npt.assert_(mtm_auto.dtype in np.sctypes['float'],
               'Wrong dtype for autospectrum')
        npt.assert_(len(mtm_auto) == 51,
               'Wrong length for halfband spectrum')
        sides = 'twosided'
        mtm_auto = tsa.mtm_cross_spectrum(
            spec1, spec2, weights[0], sides=sides
            )
        npt.assert_(len(mtm_auto) == 100,
               'Wrong length for fullband spectrum')


def test_mtm_cross_spectrum():
    """

    Test the multi-taper cross-spectral estimation. Based on the example in
    doc/examples/multi_taper_coh.py

    """
    NW = 4
    K = 2 * NW - 1

    N = 2 ** 10
    n_reps = 10
    n_freqs = N

    tapers, eigs = tsa.dpss_windows(N, NW, 2 * NW - 1)

    est_psd = []
    for k in range(n_reps):
        data, nz, alpha = utils.ar_generator(N=N)
        fgrid, hz = tsa.freq_response(1.0, a=np.r_[1, -alpha], n_freqs=n_freqs)
        # 'one-sided', so multiply by 2:
        psd = 2 * (hz * hz.conj()).real

        tdata = tapers * data

        tspectra = fftpack.fft(tdata)

        L = N / 2 + 1
        sides = 'onesided'
        w, _ = utils.adaptive_weights(tspectra, eigs, sides=sides)

        sxx = tsa.mtm_cross_spectrum(tspectra, tspectra, w, sides=sides)
        est_psd.append(sxx)

    fxx = np.mean(est_psd, 0)

    psd_ratio = np.mean(fxx / psd)

    # This is a rather lenient test, making sure that the average ratio is 1 to
    # within an order of magnitude. That is, that they are equal on average:
    npt.assert_array_almost_equal(psd_ratio, 1, decimal=1)

    # Test raising of error in case the inputs don't make sense:
    with pytest.raises(ValueError) as e_info:
        tsa.mtm_cross_spectrum(tspectra, np.r_[tspectra, tspectra], (w, w))


def test_multi_taper_psd_csd():
    """

    Test the multi taper psd and csd estimation functions.
    Based on the example in
    doc/examples/multi_taper_spectral_estimation.py

    """

    N = 2 ** 10
    n_reps = 10

    psd = []
    est_psd = []
    est_csd = []
    for jk in [True, False]:
        for k in range(n_reps):
            for adaptive in [True, False]:
                ar_seq, nz, alpha = utils.ar_generator(N=N, drop_transients=10)
                ar_seq -= ar_seq.mean()
                fgrid, hz = tsa.freq_response(1.0, a=np.r_[1, -alpha],
                                              n_freqs=N)
                psd.append(2 * (hz * hz.conj()).real)
                f, psd_mt, nu = tsa.multi_taper_psd(ar_seq, adaptive=adaptive,
                                                    jackknife=jk)
                est_psd.append(psd_mt)
                f, csd_mt = tsa.multi_taper_csd(np.vstack([ar_seq, ar_seq]),
                                               adaptive=adaptive)
                # Symmetrical in this case, so take one element out:
                est_csd.append(csd_mt[0][1])

        fxx = np.mean(psd, axis=0)
        fxx_est1 = np.mean(est_psd, axis=0)
        fxx_est2 = np.mean(est_csd, axis=0)

        # Tests the psd:
        psd_ratio1 = np.mean(fxx_est1 / fxx)
        npt.assert_array_almost_equal(psd_ratio1, 1, decimal=-1)
        # Tests the csd:
        psd_ratio2 = np.mean(fxx_est2 / fxx)
        npt.assert_array_almost_equal(psd_ratio2, 1, decimal=-1)


def test_gh57():
    """
    https://github.com/nipy/nitime/issues/57
    """
    data = np.random.randn(10, 1000)
    for jk in [True, False]:
        for adaptive in [True, False]:
            f, psd, sigma = tsa.multi_taper_psd(data, adaptive=adaptive,
                                                jackknife=jk)


def test_hermitian_periodogram_csd():
    """
    Make sure CSD matrices returned by various methods have
    Hermitian symmetry.
    """

    sig = np.random.randn(4,256)

    _, csd1 = tsa.periodogram_csd(sig)

    for i in range(4):
        for j in range(i+1):
            xc1 = csd1[i,j]
            xc2 = csd1[j,i]
            npt.assert_equal(
                xc1, xc2.conj(), err_msg='Periodogram CSD not Hermitian'
                )

    _, psd = tsa.periodogram(sig)
    for i in range(4):
        npt.assert_almost_equal(
            psd[i], csd1[i,i].real,
            err_msg='Periodgram CSD diagonal inconsistent with real PSD'
            )

def test_hermitian_multitaper_csd():
    """
    Make sure CSD matrices returned by various methods have
    Hermitian symmetry.
    """

    sig = np.random.randn(4,256)

    _, csd1 = tsa.multi_taper_csd(sig, adaptive=False)

    for i in range(4):
        for j in range(i+1):
            xc1 = csd1[i,j]
            xc2 = csd1[j,i]
            npt.assert_equal(
                xc1, xc2.conj(), err_msg='MTM CSD not Hermitian'
                )

    _, psd, _ = tsa.multi_taper_psd(sig, adaptive=False)
    for i in range(4):
        npt.assert_almost_equal(
            psd[i], csd1[i,i].real,
            err_msg='MTM CSD diagonal inconsistent with real PSD'
            )

def test_periodogram_spectral_normalization():
    """
    Check that the spectral estimators are normalized in the
    correct Watts/Hz fashion
    """

    x = np.random.randn(1024)
    f1, Xp1 = tsa.periodogram(x)
    f2, Xp2 = tsa.periodogram(x, Fs=100)
    f3, Xp3 = tsa.periodogram(x, N=2**12)

    p1 = np.sum(Xp1) * 2 * np.pi / 2**10
    p2 = np.sum(Xp2) * 100 / 2**10
    p3 = np.sum(Xp3) * 2 * np.pi / 2**12
    npt.assert_( np.abs(p1 - p2) < 1e-14,
                    'Inconsistent frequency normalization in periodogram (1)' )
    npt.assert_( np.abs(p3 - p2) < 1e-8,
                    'Inconsistent frequency normalization in periodogram (2)' )

    td_var = np.var(x)
    # assure that the estimators are at least in the same
    # order of magnitude as the time-domain variance
    npt.assert_( np.abs(np.log10(p1/td_var)) < 1,
                    'Incorrect frequency normalization in periodogram' )

    # check the freq vector while we're here
    npt.assert_( f2.max() == 50, 'Periodogram returns wrong frequency bins' )

def test_multitaper_spectral_normalization():
    """
    Check that the spectral estimators are normalized in the
    correct Watts/Hz fashion
    """

    x = np.random.randn(1024)
    f1, Xp1, _ = tsa.multi_taper_psd(x)
    f2, Xp2, _ = tsa.multi_taper_psd(x, Fs=100)
    f3, Xp3, _ = tsa.multi_taper_psd(x, NFFT=2**12)

    p1 = np.sum(Xp1) * 2 * np.pi / 2**10
    p2 = np.sum(Xp2) * 100 / 2**10
    p3 = np.sum(Xp3) * 2 * np.pi / 2**12
    npt.assert_( np.abs(p1 - p2) < 1e-14,
                    'Inconsistent frequency normalization in MTM PSD (1)' )
    npt.assert_( np.abs(p3 - p2) < 1e-8,
                    'Inconsistent frequency normalization in MTM PSD (2)' )

    td_var = np.var(x)
    # assure that the estimators are at least in the same
    # order of magnitude as the time-domain variance
    npt.assert_( np.abs(np.log10(p1/td_var)) < 1,
                    'Incorrect frequency normalization in MTM PSD' )

    # check the freq vector while we're here
    npt.assert_( f2.max() == 50, 'MTM PSD returns wrong frequency bins' )
