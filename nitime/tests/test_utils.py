import numpy as np
import numpy.testing as npt
import nitime.timeseries as ts
import decotest
import nose.tools as nt

from scipy.signal import signaltools

from nitime import utils

def test_zscore():

    x = np.array([[1,2,3],[4,5,6]])
    z = utils.zscore(x)

    npt.assert_equal(x.shape,z.shape)
    
def test_percent_change():
    x = np.array([[99,100,101],[4,5,6]])
    p = utils.percent_change(x)

    npt.assert_equal(x.shape,p.shape)
    npt.assert_almost_equal(p[0,2],1.0)

def test_debias():
    x = np.arange(64).reshape(4,4,4)
    x0 = utils.remove_bias(x, axis=1)
    assert (x0.mean(axis=1)==0).all(), \
           'did not remove the bias from axis 1'

def ref_crosscov(x, y, all_lags=True):
    "Computes sxy[k] = E{x[n]*y[n+k]}"
    x = utils.remove_bias(x, 0)
    y = utils.remove_bias(y, 0)
    lx, ly = len(x), len(y)
    pad_len = lx + ly - 1
    sxy = np.correlate(x, y, mode='full')/lx
    if all_lags:
        return sxy
    c_idx = pad_len/2
    return sxy[c_idx:]

def test_crosscov():
    N = 128
    ar_seq1, _, _ = utils.ar_generator(N=N)
    ar_seq2, _, _ = utils.ar_generator(N=N)

    for all_lags in (True, False):
        sxy = utils.crosscov(ar_seq1, ar_seq2, all_lags=all_lags)
        sxy_ref = ref_crosscov(ar_seq1, ar_seq2, all_lags=all_lags)
        err = sxy_ref - sxy
        mse = np.dot(err, err)/N
        yield nt.assert_true, mse < 1e-12, \
               'Large mean square error w.r.t. reference cross covariance'

def test_autocorr():
    N = 128
    ar_seq, _, _ = utils.ar_generator(N=N)
    rxx = utils.autocorr(ar_seq)
    yield nt.assert_true, rxx[0]==1, \
          'Zero lag autocorrelation is not equal to 1'
    rxx = utils.autocorr(ar_seq, all_lags=True)
    yield nt.assert_true, rxx[127]==1, \
          'Zero lag autocorrelation is not equal to 1'

def plot_savings():
    import matplotlib.pyplot as pp
    from IPython.Magic import timings
    f1 = ref_crosscov
    f2 = utils.crosscov
    times1 = []
    times2 = []
    for N in map(lambda x: 2**x, range(8,13)):
        a = np.random.randn(N)
        args = (a,a)
        kws = dict()
        _, t_est1 = timings(1000, f1, *args, **kws)
        _, t_est2 = timings(1000, f2, *args, **kws)
        times1.append(t_est1)
        times2.append(t_est2)
    pp.figure()
    pp.plot(range(8,13), times1, label='tdomain crosscov')
    pp.plot(range(8,13), times2, label='fdomain crosscov')
    pp.legend()
    pp.show()
