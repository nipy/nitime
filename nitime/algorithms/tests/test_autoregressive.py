import numpy as np
import numpy.testing as npt

import nitime.algorithms as tsa


def test_AR_YW():
    arsig,_,_ = ut.ar_generator(N=512)
    avg_pwr = (arsig*arsig.conjugate()).mean()
    ak,sigma_v = tsa.AR_est_YW(arsig, 8, 1024)
    w, psd = tsa.AR_psd(ak, sigma_v)
    # for efficiency, let's leave out the 2PI in the numerator and denominator
    # for the following integral
    dw = 1./1024
    avg_pwr_est = np.trapz(psd, dx=dw)
    npt.assert_almost_equal(avg_pwr, avg_pwr_est, decimal=0)

def test_AR_LD():
    """

    Test the Levinson Durbin estimate of the AR coefficients agains the
    expercted PSD

    """
    arsig,_,_ = ut.ar_generator(N=512)
    avg_pwr = (arsig*arsig.conjugate()).mean()
    ak, sigma_v = tsa.AR_est_LD(arsig, 8, 1024)
    w, psd = tsa.AR_psd(ak, sigma_v)

    # for efficiency, let's leave out the 2PI in the numerator and denominator
    # for the following integral
    dw = 1./1024
    avg_pwr_est = np.trapz(psd, dx=dw)
    npt.assert_almost_equal(avg_pwr, avg_pwr_est, decimal=0)

def test_MAR_est_LWR():
    """

    Test the LWR MAR estimator against the power of the signal

    """
    a1 = np.array([ [0.9, 0],
                [0.16, 0.8] ])

    a2 = np.array([ [-0.5, 0],
                [-0.2, -0.5] ])


    am = np.array([ -a1, -a2 ])

    x_var = 1
    y_var = 0.7
    xy_cov = 0.4
    cov = np.array([ [x_var, xy_cov],
                     [xy_cov, y_var] ])


    w, Hw = tsa.transfer_function_xy(am, Nfreqs=Nfreqs)
    Sw_true = tsa.spectral_matrix_xy(Hw, cov)

    # This many realizations of the process:
    N = 500
    # Each one this long
    L = 1024

    order = am.shape[0]
    n_lags = order + 1

    n_process = am.shape[-1]

    z = np.empty((N, n_process, L))
    nz = np.empty((N, n_process, L))

    for i in xrange(N):
        z[i], nz[i] = utils.generate_mar(am, cov, L)
        
    Rxx = np.empty((N,n_process,n_process,n_lags))
        
    for i in xrange(N):
        Rxx[i] = utils.autocov_vector(z[i],nlags=n_lags)

    Rxx = Rxx.mean(axis=0)
    Rxx = Rxx.transpose(2,0,1)

    a, ecov = utils.lwr(Rxx)

    w, f_x2y, f_y2x, f_xy, Sw = alg.granger_causality_xy(a,ecov,Nfreqs=Nfreqs)

    # correct for one-sided spectral density functions
    Sxx_true = Sw_true[0,0].real; Syy_true = Sw_true[1,1].real
    Sxx_est = np.abs(Sw[0,0]); Syy_est = np.abs(Sw[1,1])
