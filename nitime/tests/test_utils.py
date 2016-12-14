import numpy as np
import numpy.testing as npt

from nitime import utils
import nitime.algorithms as alg


def test_zscore():

    x = np.array([[1, 1, 3, 3],
                  [4, 4, 6, 6]])

    z = utils.zscore(x)
    npt.assert_equal(x.shape, z.shape)

    # Default axis is -1
    npt.assert_equal(utils.zscore(x), np.array([[-1., -1., 1., 1.],
                                                [-1., -1., 1., 1.]]))

    # Test other axis:
    npt.assert_equal(utils.zscore(x, 0), np.array([[-1., -1., -1., -1.],
                                                   [1., 1., 1., 1.]]))


def test_percent_change():
    x = np.array([[99, 100, 101], [4, 5, 6]])
    p = utils.percent_change(x)

    npt.assert_equal(x.shape, p.shape)
    npt.assert_almost_equal(p[0, 2], 1.0)

    ts = np.arange(4 * 5).reshape(4, 5)
    ax = 0
    npt.assert_almost_equal(utils.percent_change(ts, ax), np.array(
        [[-100., -88.23529412, -78.94736842, -71.42857143, -65.2173913],
        [-33.33333333, -29.41176471, -26.31578947, -23.80952381, -21.73913043],
        [33.33333333,   29.41176471,   26.31578947,   23.80952381, 21.73913043],
        [100., 88.23529412, 78.94736842, 71.42857143, 65.2173913]]))

    ax = 1
    npt.assert_almost_equal(utils.percent_change(ts, ax), np.array(
        [[-100., -50., 0., 50., 100.],
         [-28.57142857, -14.28571429, 0., 14.28571429, 28.57142857],
          [-16.66666667, -8.33333333, 0., 8.33333333, 16.66666667],
          [-11.76470588, -5.88235294, 0., 5.88235294, 11.76470588]]))

def test_tridi_inverse_iteration():
    import scipy.linalg as la
    from scipy.sparse import spdiags
    # set up a spectral concentration eigenvalue problem for testing
    N = 2000
    NW = 4
    K = 8
    W = float(NW) / N
    nidx = np.arange(N, dtype='d')
    ab = np.zeros((2, N), 'd')
    # store this separately for tridisolve later
    sup_diag = np.zeros((N,), 'd')
    sup_diag[:-1] = nidx[1:] * (N - nidx[1:]) / 2.
    ab[0, 1:] = sup_diag[:-1]
    ab[1] = ((N - 1 - 2 * nidx) / 2.) ** 2 * np.cos(2 * np.pi * W)
    # only calculate the highest Kmax-1 eigenvalues
    w = la.eigvals_banded(ab, select='i', select_range=(N - K, N - 1))
    w = w[::-1]
    E = np.zeros((K, N), 'd')
    t = np.linspace(0, np.pi, N)
    # make sparse tridiagonal matrix for eigenvector check
    sp_data = np.zeros((3,N), 'd')
    sp_data[0, :-1] = sup_diag[:-1]
    sp_data[1] = ab[1]
    sp_data[2, 1:] = sup_diag[:-1]
    A = spdiags(sp_data, [-1, 0, 1], N, N)
    E = np.zeros((K,N), 'd')
    for j in range(K):
        e = utils.tridi_inverse_iteration(
            ab[1], sup_diag, w[j], x0=np.sin((j+1)*t)
            )
        b = A*e
        npt.assert_(
               np.linalg.norm(np.abs(b) - np.abs(w[j]*e)) < 1e-8,
               'Inverse iteration eigenvector solution is inconsistent with '\
               'given eigenvalue'
               )
        E[j] = e

    # also test orthonormality of the eigenvectors
    ident = np.dot(E, E.T)
    npt.assert_almost_equal(ident, np.eye(K))

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
    c_idx = pad_len // 2
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
        npt.assert_(mse < 1e-12, \
               'Large mean square error w.r.t. reference cross covariance')


def test_autocorr():
    N = 128
    ar_seq, _, _ = utils.ar_generator(N=N)
    rxx = utils.autocorr(ar_seq)
    npt.assert_(rxx[0] == rxx.max(), \
          'Zero lag autocorrelation is not maximum autocorrelation')
    rxx = utils.autocorr(ar_seq, all_lags=True)
    npt.assert_(rxx[127] == rxx.max(), \
          'Zero lag autocorrelation is not maximum autocorrelation')

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

    #Number of realizations of the process
    N = 500
    #Length of each realization:
    L = 1024

    order = am.shape[0]
    n_process = am.shape[-1]

    z = np.empty((N, n_process, L))
    nz = np.empty((N, n_process, L))

    for i in range(N):
        z[i], nz[i] = utils.generate_mar(am, cov, L)

    AIC = []
    BIC = []
    AICc = []

    # The total number data points available for estimation:
    Ntotal = L * n_process

    for n_lags in range(1, 10):

        Rxx = np.empty((N, n_process, n_process, n_lags))

        for i in range(N):
            Rxx[i] = utils.autocov_vector(z[i], nlags=n_lags)

        Rxx = Rxx.mean(axis=0)
        Rxx = Rxx.transpose(2, 0, 1)

        a, ecov = alg.lwr_recursion(Rxx)

        IC = utils.akaike_information_criterion(ecov, n_process, n_lags, Ntotal)
        AIC.append(IC)

        IC = utils.akaike_information_criterion(ecov, n_process, n_lags, Ntotal, corrected=True)
        AICc.append(IC)

        IC = utils.bayesian_information_criterion(ecov, n_process, n_lags, Ntotal)
        BIC.append(IC)

    # The model has order 2, so this should minimize on 2:

    # We do not test this for AIC/AICc, because these sometimes do not minimize
    # (see Ding and Bressler)
    npt.assert_equal(np.argmin(BIC), 2)


def test_multi_intersect():
    """
    Testing the multi-intersect utility function
    """

    arr1 = np.array(np.arange(1000).reshape(2,500))
    arr2 = np.array([[1,0.1,0.2],[0.3,0.4, 0.5]])
    arr3 = np.array(1)
    npt.assert_equal(1, utils.multi_intersect([arr1, arr2, arr3]))


def test_zero_pad():
    """
    Test the zero_pad function
    """
    # Freely assume that time is the last dimension:
    ts1 = np.empty((64, 64, 35, 32))
    NFFT = 64
    zp1 = utils.zero_pad(ts1, NFFT)
    npt.assert_equal(zp1.shape[-1], NFFT)

    # Try this with something with only 1 dimension:
    ts2 = np.empty(64)
    zp2 = utils.zero_pad(ts2, NFFT)
    npt.assert_equal(zp2.shape[-1], NFFT)


def test_detect_lines():
    """
    Tests detect_lines utility in the reliable low-SNR scenario.
    """
    N = 1000
    fft_pow = int( np.ceil(np.log2(N) + 2) )
    NW = 4
    lines = np.sort(np.random.randint(100, 2**(fft_pow-4), size=(3,)))
    while np.any( np.diff(lines) < 2*NW ):
        lines = np.sort(np.random.randint(2**(fft_pow-4), size=(3,)))
    lines = lines.astype('d')
    #lines += np.random.randn(3) # displace from grid locations
    lines /= 2.0**(fft_pow-2) # ensure they are well separated

    phs = np.random.rand(3) * 2 * np.pi
    # amps approximately such that RMS power = 1 +/- N(0,1)
    amps = np.sqrt(2)/2 + np.abs( np.random.randn(3) )

    nz_sig = 0.05
    tx = np.arange(N)

    harmonics = amps[:,None]*np.cos( 2*np.pi*tx*lines[:,None] + phs[:,None] )
    harmonic = np.sum(harmonics, axis=0)
    nz = np.random.randn(N) * nz_sig
    sig = harmonic + nz

    f, b = utils.detect_lines(sig, (NW, 2*NW), low_bias=True, NFFT=2**fft_pow)
    h_est = 2*(b[:,None]*np.exp(2j*np.pi*tx*f[:,None])).real

    npt.assert_(
        len(f) == 3, 'The wrong number of harmonic components were detected'
        )

    err = harmonic - np.sum(h_est, axis=0)
    phs_est = np.angle(b)
    phs_est[phs_est < 0] += 2*np.pi

    phs_err = np.linalg.norm(phs_est - phs)**2
    amp_err = np.linalg.norm(amps - 2*np.abs(b))**2 / np.linalg.norm(amps)**2
    freq_err = np.linalg.norm(lines - f)**2

    # FFT bin detections should be exact
    npt.assert_equal(lines, f)
    # amplitude estimation should be pretty good
    npt.assert_(amp_err < 1e-4, 'Harmonic amplitude was poorly estimated')
    # phase estimation should be decent
    npt.assert_(phs_err < 1e-3, 'Harmonic phase was poorly estimated')
    # the error relative to the noise power should be below 1
    rel_mse = np.mean(err**2)/nz_sig**2
    npt.assert_(
        rel_mse < 1,
        'The error in detected harmonic components is too high relative to '\
        'the noise level: %1.2e'%rel_mse
        )

def test_detect_lines_2dmode():
    """
    Test multi-sequence operation
    """

    N = 1000

    sig = np.cos( 2*np.pi*np.arange(N) * 20./N ) + np.random.randn(N) * .01

    sig2d = np.row_stack( (sig, sig, sig) )

    lines = utils.detect_lines(sig2d, (4, 8), low_bias=True, NFFT=2**12)

    npt.assert_(len(lines)==3, 'Detect lines failed multi-sequence mode')

    consistent1 = (lines[0][0] == lines[1][0]).all() and \
      (lines[1][0] == lines[2][0]).all()
    consistent2 = (lines[0][1] == lines[1][1]).all() and \
      (lines[1][1] == lines[2][1]).all()

    npt.assert_(consistent1 and consistent2, 'Inconsistent results')
