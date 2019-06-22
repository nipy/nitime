import numpy as np
import numpy.testing as npt

import nitime.algorithms as tsa
import nitime.utils as utils

# Set the random seed:
np.random.seed(1)


def _random_poles(half_poles=3):
    poles_rp = np.random.rand(half_poles * 20)
    poles_ip = np.random.rand(half_poles * 20)

    # get real/imag parts of some poles such that magnitudes bounded
    # away from 1
    stable_pole_idx = np.where(poles_rp ** 2 + poles_ip ** 2 < .75 ** 2)[0]
    # keep 3 of these, and supplement with complex conjugate
    stable_poles = poles_rp[stable_pole_idx[:half_poles]] + \
        1j * poles_ip[stable_pole_idx[:half_poles]]
    stable_poles = np.r_[stable_poles, stable_poles.conj()]
    # we have the roots, now find the polynomial
    ak = np.poly(stable_poles)
    return ak


def test_AR_est_consistency():
    order = 10  # some even number
    ak = _random_poles(order // 2)
    x, v, _ = utils.ar_generator(N=512, coefs=-ak[1:], drop_transients=100)
    ak_yw, ssq_yw = tsa.AR_est_YW(x, order)
    ak_ld, ssq_ld = tsa.AR_est_LD(x, order)
    npt.assert_almost_equal(ak_yw, ak_ld)
    npt.assert_almost_equal(ssq_yw, ssq_ld)


def test_AR_YW():
    arsig, _, _ = utils.ar_generator(N=512)
    avg_pwr = (arsig * arsig.conjugate()).mean()
    order = 8
    ak, sigma_v = tsa.AR_est_YW(arsig, order)
    w, psd = tsa.AR_psd(ak, sigma_v)
    # the psd is a one-sided power spectral density, which has been
    # multiplied by 2 to preserve the property that
    # 1/2pi int_{-pi}^{pi} Sxx(w) dw = Rxx(0)

    # evaluate this integral numerically from 0 to pi
    dw = np.pi / len(psd)
    avg_pwr_est = np.trapz(psd, dx=dw) / (2 * np.pi)
    # consistency on the order of 10**0 is pretty good for this test
    npt.assert_almost_equal(avg_pwr, avg_pwr_est, decimal=0)

    # Test for providing the autocovariance as an input:
    ak, sigma_v = tsa.AR_est_YW(arsig, order, utils.autocov(arsig))
    w, psd = tsa.AR_psd(ak, sigma_v)
    avg_pwr_est = np.trapz(psd, dx=dw) / (2 * np.pi)
    npt.assert_almost_equal(avg_pwr, avg_pwr_est, decimal=0)


def test_AR_LD():
    """

    Test the Levinson Durbin estimate of the AR coefficients against the
    expercted PSD

    """
    arsig, _, _ = utils.ar_generator(N=512)
    avg_pwr = (arsig * arsig.conjugate()).real.mean()
    order = 8
    ak, sigma_v = tsa.AR_est_LD(arsig, order)
    w, psd = tsa.AR_psd(ak, sigma_v)

    # the psd is a one-sided power spectral density, which has been
    # multiplied by 2 to preserve the property that
    # 1/2pi int_{-pi}^{pi} Sxx(w) dw = Rxx(0)

    # evaluate this integral numerically from 0 to pi
    dw = np.pi / len(psd)
    avg_pwr_est = np.trapz(psd, dx=dw) / (2 * np.pi)
    npt.assert_almost_equal(avg_pwr, avg_pwr_est, decimal=0)

    # Test for providing the autocovariance as an input:
    ak, sigma_v = tsa.AR_est_LD(arsig, order, utils.autocov(arsig))
    w, psd = tsa.AR_psd(ak, sigma_v)
    avg_pwr_est = np.trapz(psd, dx=dw) / (2 * np.pi)
    npt.assert_almost_equal(avg_pwr, avg_pwr_est, decimal=0)


def test_MAR_est_LWR():
    """

    Test the LWR MAR estimator against the power of the signal

    This also tests the functions: transfer_function_xy, spectral_matrix_xy,
    coherence_from_spectral and granger_causality_xy

    """

    # This is the same processes as those in doc/examples/ar_est_2vars.py:
    a1 = np.array([[0.9, 0],
                   [0.16, 0.8]])

    a2 = np.array([[-0.5, 0],
                   [-0.2, -0.5]])

    am = np.array([-a1, -a2])

    x_var = 1
    y_var = 0.7
    xy_cov = 0.4
    cov = np.array([[x_var, xy_cov],
                    [xy_cov, y_var]])

    n_freqs = 1024
    w, Hw = tsa.transfer_function_xy(am, n_freqs=n_freqs)
    Sw = tsa.spectral_matrix_xy(Hw, cov)

    # This many realizations of the process:
    N = 500
    # Each one this long
    L = 1024

    order = am.shape[0]
    n_lags = order + 1

    n_process = am.shape[-1]

    z = np.empty((N, n_process, L))
    nz = np.empty((N, n_process, L))

    for i in range(N):
        z[i], nz[i] = utils.generate_mar(am, cov, L)

    a_est = []
    cov_est = []

    # This loop runs MAR_est_LWR:
    for i in range(N):
        Rxx = (tsa.MAR_est_LWR(z[i], order=n_lags))
        a_est.append(Rxx[0])
        cov_est.append(Rxx[1])

    a_est = np.mean(a_est, 0)
    cov_est = np.mean(cov_est, 0)

    # This tests transfer_function_xy and spectral_matrix_xy:
    w, Hw_est = tsa.transfer_function_xy(a_est, n_freqs=n_freqs)
    Sw_est = tsa.spectral_matrix_xy(Hw_est, cov_est)

    # coherence_from_spectral:
    c = tsa.coherence_from_spectral(Sw)
    c_est = tsa.coherence_from_spectral(Sw_est)

    # granger_causality_xy:

    w, f_x2y, f_y2x, f_xy, Sw = tsa.granger_causality_xy(am,
                                                         cov,
                                                         n_freqs=n_freqs)

    w, f_x2y_est, f_y2x_est, f_xy_est, Sw_est = tsa.granger_causality_xy(a_est,
                                                             cov_est,
                                                             n_freqs=n_freqs)

    # interdependence_xy
    i_xy = tsa.interdependence_xy(Sw)
    i_xy_est = tsa.interdependence_xy(Sw_est)

    # This is all very approximate:
    npt.assert_almost_equal(Hw, Hw_est, decimal=1)
    npt.assert_almost_equal(Sw, Sw_est, decimal=1)
    npt.assert_almost_equal(c, c_est, 1)
    npt.assert_almost_equal(f_xy, f_xy_est, 1)
    npt.assert_almost_equal(f_x2y, f_x2y_est, 1)
    npt.assert_almost_equal(f_y2x, f_y2x_est, 1)
    npt.assert_almost_equal(i_xy, i_xy_est, 1)


def test_lwr():
    "test solution of lwr recursion"
    for trial in range(3):
        nc = np.random.randint(2, high=10)
        P = np.random.randint(2, high=6)
        # nc is channels, P is lags (order)
        r = np.random.randn(P + 1, nc, nc)
        r[0] = np.dot(r[0], r[0].T)  # force r0 to be symmetric

        a, Va = tsa.lwr_recursion(r)
        # Verify the "orthogonality" principle of the mAR system
        # Set up a system in blocks to compute, for each k
        #   sum_{i=1}^{P} A(i)R(k-i) = -R(k) k > 0
        # = sum_{i=1}^{P} R(k-i)^T A(i)^T = -R(k)^T
        # = sum_{i=1}^{P} R(i-k)A(i)^T = -R(k)^T
        rmat = np.zeros((nc * P, nc * P))
        for k in range(1, P + 1):
            for i in range(1, P + 1):
                im = i - k
                if im < 0:
                    r1 = r[-im].T
                else:
                    r1 = r[im]
                rmat[(k - 1) * nc:k * nc, (i - 1) * nc:i * nc] = r1

        rvec = np.zeros((nc * P, nc))
        avec = np.zeros((nc * P, nc))
        for m in range(P):
            rvec[m * nc:(m + 1) * nc] = -r[m + 1].T
            avec[m * nc:(m + 1) * nc] = a[m].T

        l2_d = np.dot(rmat, avec) - rvec
        l2_d = (l2_d ** 2).sum() ** 0.5
        l2_r = (rvec ** 2).sum() ** 0.5

        # compute |Ax-b| / |b| metric
        npt.assert_almost_equal(l2_d / l2_r, 0, decimal=5)
