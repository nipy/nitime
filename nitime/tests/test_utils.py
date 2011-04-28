import numpy as np
import numpy.testing as npt
import nose.tools as nt

from nitime import utils


def test_zscore():

    x = np.array([[1, 1, 3, 3],
                  [4, 4, 6, 6]])

    z = utils.zscore(x)
    yield npt.assert_equal, x.shape, z.shape

    #Default axis is -1
    yield npt.assert_equal, utils.zscore(x), np.array([[-1., -1., 1., 1.],
                                                      [-1., -1., 1., 1.]])

    #Test other axis:
    yield npt.assert_equal, utils.zscore(x, 0), np.array([[-1., -1., -1., -1.],
                                                        [1., 1., 1., 1.]])


def test_percent_change():
    x = np.array([[99, 100, 101], [4, 5, 6]])
    p = utils.percent_change(x)

    yield npt.assert_equal, x.shape, p.shape
    yield npt.assert_almost_equal, p[0, 2], 1.0

    ts = np.arange(4 * 5).reshape(4, 5)
    ax = 0
    yield npt.assert_almost_equal, utils.percent_change(ts, ax), np.array(
        [[-100., -88.23529412, -78.94736842, -71.42857143, -65.2173913],
        [-33.33333333, -29.41176471, -26.31578947, -23.80952381, -21.73913043],
        [33.33333333,   29.41176471,   26.31578947,   23.80952381, 21.73913043],
        [100., 88.23529412, 78.94736842, 71.42857143, 65.2173913]])

    ax = 1
    yield npt.assert_almost_equal, utils.percent_change(ts, ax), np.array(
        [[-100., -50., 0., 50., 100.],
         [-28.57142857, -14.28571429, 0., 14.28571429, 28.57142857],
          [-16.66666667, -8.33333333, 0., 8.33333333, 16.66666667],
          [-11.76470588, -5.88235294, 0., 5.88235294, 11.76470588]])


def test_debias():
    x = np.arange(64).reshape(4, 4, 4)
    x0 = utils.remove_bias(x, axis=1)
    npt.assert_equal((x0.mean(axis=1) == 0).all(), True)


def ref_crosscov(x, y, all_lags=True):
    "Computes sxy[k] = E{x[n]*y[n+k]}"
    x = utils.remove_bias(x, 0)
    y = utils.remove_bias(y, 0)
    lx, ly = len(x), len(y)
    pad_len = lx + ly - 1
    sxy = np.correlate(x, y, mode='full') / lx
    if all_lags:
        return sxy
    c_idx = pad_len / 2
    return sxy[c_idx:]


def test_crosscov():
    N = 128
    ar_seq1, _, _ = utils.ar_generator(N=N)
    ar_seq2, _, _ = utils.ar_generator(N=N)

    for all_lags in (True, False):
        sxy = utils.crosscov(ar_seq1, ar_seq2, all_lags=all_lags)
        sxy_ref = ref_crosscov(ar_seq1, ar_seq2, all_lags=all_lags)
        err = sxy_ref - sxy
        mse = np.dot(err, err) / N
        yield nt.assert_true, mse < 1e-12, \
               'Large mean square error w.r.t. reference cross covariance'


def test_autocorr():
    N = 128
    ar_seq, _, _ = utils.ar_generator(N=N)
    rxx = utils.autocorr(ar_seq)
    yield nt.assert_true, rxx[0] == 1, \
          'Zero lag autocorrelation is not equal to 1'
    rxx = utils.autocorr(ar_seq, all_lags=True)
    yield nt.assert_true, rxx[127] == 1, \
          'Zero lag autocorrelation is not equal to 1'


# Should this really be in the tests?
def plot_savings():
    import matplotlib.pyplot as pp
    from IPython.Magic import timings
    f1 = ref_crosscov
    f2 = utils.crosscov
    times1 = []
    times2 = []
    for N in map(lambda x: 2 ** x, range(8, 13)):
        a = np.random.randn(N)
        args = (a, a)
        kws = dict()
        _, t_est1 = timings(1000, f1, *args, **kws)
        _, t_est2 = timings(1000, f2, *args, **kws)
        times1.append(t_est1)
        times2.append(t_est2)
    pp.figure()
    pp.plot(range(8, 13), times1, label='tdomain crosscov')
    pp.plot(range(8, 13), times2, label='fdomain crosscov')
    pp.legend()
    pp.show()


def test_lwr():
    "test solution of lwr recursion"
    for trial in xrange(3):
        nc = np.random.randint(2, high=10)
        P = np.random.randint(2, high=6)
        # nc is channels, P is lags (order)
        r = np.random.randn(P + 1, nc, nc)
        r[0] = np.dot(r[0], r[0].T)  # force r0 to be symmetric

        a, Va = utils.lwr(r)
        # Verify the "orthogonality" principle of the mAR system
        # Set up a system in blocks to compute, for each k
        #   sum_{i=1}^{P} A(i)R(k-i) = -R(k) k > 0
        # = sum_{i=1}^{P} R(k-i)^T A(i)^T = -R(k)^T
        # = sum_{i=1}^{P} R(i-k)A(i)^T = -R(k)^T
        rmat = np.zeros((nc * P, nc * P))
        for k in xrange(1, P + 1):
            for i in xrange(1, P + 1):
                im = i - k
                if im < 0:
                    r1 = r[-im].T
                else:
                    r1 = r[im]
                rmat[(k - 1) * nc:k * nc, (i - 1) * nc:i * nc] = r1

        rvec = np.zeros((nc * P, nc))
        avec = np.zeros((nc * P, nc))
        for m in xrange(P):
            rvec[m * nc:(m + 1) * nc] = -r[m + 1].T
            avec[m * nc:(m + 1) * nc] = a[m].T

        l2_d = np.dot(rmat, avec) - rvec
        l2_d = (l2_d ** 2).sum() ** 0.5
        l2_r = (rvec ** 2).sum() ** 0.5

        # compute |Ax-b| / |b| metric
        npt.assert_almost_equal(l2_d / l2_r, 0, decimal=5)


def test_lwr_alternate():
    "test solution of lwr recursion"

    for trial in xrange(3):
        nc = np.random.randint(2, high=10)
        P = np.random.randint(2, high=6)
        # nc is channels, P is lags (order)
        r = np.random.randn(P + 1, nc, nc)
        r[0] = np.dot(r[0], r[0].T)  # force r0 to be symmetric

        a, Va = utils.lwr_alternate(r)
        # Verify the "orthogonality" principle of the mAR system
        # Set up a system in blocks to compute, for each k
        #   sum_{i=1}^{P} A(i)R(-k+i) = -R(-k)  k > 0
        # = sum_{i=1}^{P} (R(-k+i)^T A^T(i))^T = -R(-k) = -R(k)^T
        # = sum_{i=1}^{P} R(k-i)A.T(i) = -R(k)
        rmat = np.zeros((nc * P, nc * P))
        for k in xrange(1, P + 1):
            for i in xrange(1, P + 1):
                im = k - i
                if im < 0:
                    r1 = r[-im].T
                else:
                    r1 = r[im]
                rmat[(k - 1) * nc:k * nc, (i - 1) * nc:i * nc] = r1

        rvec = np.zeros((nc * P, nc))
        avec = np.zeros((nc * P, nc))
        for m in xrange(P):
            rvec[m * nc:(m + 1) * nc] = -r[m + 1]
            avec[m * nc:(m + 1) * nc] = a[m].T

        l2_d = np.dot(rmat, avec) - rvec
        l2_d = (l2_d ** 2).sum() ** 0.5
        l2_r = (rvec ** 2).sum() ** 0.5

        # compute |Ax-b| / |b| metric
        yield nt.assert_almost_equal, l2_d / l2_r, 0


def test_information_criteria():
    """

    Test the implementation of information criteria:

    """
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

    N = 10
    L = 100

    z = np.empty((N, 2, L))
    nz = np.empty((N, 2, L))
    for i in xrange(N):
        z[i], nz[i] = utils.generate_mar(am, cov, L)

    AIC = []
    BIC = []
    AICc = []
    for i in range(10):
        AIC.append(utils.akaike_information_criterion(z, i))
        AICc.append(utils.akaike_information_criterion_c(z, i))
        BIC.append(utils.bayesian_information_criterion(z, i))

    # The model has order 2, so this should minimize on 2:
    #nt.assert_equal(np.argmin(AIC),2)
    #nt.assert_equal(np.argmin(AICc),2)
    nt.assert_equal(np.argmin(BIC), 2)
