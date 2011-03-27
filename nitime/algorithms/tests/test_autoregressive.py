import numpy as np
import numpy.testing as npt

import nitime.algorithms as tsa
import nitime.utils as utils

def test_AR_YW():
    arsig,_,_ = utils.ar_generator(N=512)
    avg_pwr = (arsig*arsig.conjugate()).mean()
    order = 8
    ak,sigma_v = tsa.AR_est_YW(arsig, order) 
    w, psd = tsa.AR_psd(ak, sigma_v)
    # for efficiency, let's leave out the 2PI in the numerator and denominator
    # for the following integral
    dw = 1./1024
    avg_pwr_est = np.trapz(psd, dx=dw)
    npt.assert_almost_equal(avg_pwr, avg_pwr_est, decimal=0)

    # Test for providing the autocovariance as an input:
    ak,sigma_v = tsa.AR_est_YW(arsig, order, utils.autocov(arsig)[:order+1])
    w, psd = tsa.AR_psd(ak, sigma_v)
    avg_pwr_est = np.trapz(psd, dx=dw)
    npt.assert_almost_equal(avg_pwr, avg_pwr_est, decimal=0)


def test_AR_LD():
    """

    Test the Levinson Durbin estimate of the AR coefficients agains the
    expercted PSD

    """
    arsig,_,_ = utils.ar_generator(N=512)
    avg_pwr = (arsig*arsig.conjugate()).mean()
    order = 8
    ak, sigma_v = tsa.AR_est_LD(arsig, order)
    w, psd = tsa.AR_psd(ak, sigma_v)

    # for efficiency, let's leave out the 2PI in the numerator and denominator
    # for the following integral
    dw = 1./1024
    avg_pwr_est = np.trapz(psd, dx=dw)
    npt.assert_almost_equal(avg_pwr, avg_pwr_est, decimal=0)

    # Test for providing the autocovariance as an input:
    ak,sigma_v = tsa.AR_est_LD(arsig, order, utils.autocov(arsig)[:order+1])
    w, psd = tsa.AR_psd(ak, sigma_v)
    avg_pwr_est = np.trapz(psd, dx=dw)
    npt.assert_almost_equal(avg_pwr, avg_pwr_est, decimal=0)


def test_MAR_est_LWR():
    """

    Test the LWR MAR estimator against the power of the signal

    This also tests the functions: transfer_function_xy, spectral_matrix_xy,
    coherence_from_spectral and granger_causality_xy
    
    """

    # This is the same processes as those in doc/examples/ar_est_2vars.py: 
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


    Nfreqs = 1024
    w, Hw = tsa.transfer_function_xy(am, Nfreqs=Nfreqs)
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

    for i in xrange(N):
        z[i], nz[i] = utils.generate_mar(am, cov, L)

    a_est = []
    cov_est = []

    # This loop runs MAR_est_LWR:
    for i in xrange(N):
        Rxx = (tsa.MAR_est_LWR(z[i],order=n_lags))
        a_est.append(Rxx[0])
        cov_est.append(Rxx[1])

    a_est = np.mean(a_est,0)
    cov_est = np.mean(cov_est,0)

    # This tests transfer_function_xy and spectral_matrix_xy: 
    w, Hw_est = tsa.transfer_function_xy(a_est, Nfreqs=Nfreqs)
    Sw_est = tsa.spectral_matrix_xy(Hw_est, cov_est)

    # coherence_from_spectral:
    c = tsa.coherence_from_spectral(Sw)
    c_est = tsa.coherence_from_spectral(Sw_est)

    # granger_causality_xy:

    w, f_x2y, f_y2x, f_xy, Sw = tsa.granger_causality_xy(am,
                                                         cov,
                                                         Nfreqs=Nfreqs)

    w, f_x2y_est, f_y2x_est, f_xy_est, Sw_est = tsa.granger_causality_xy(a_est,
                                                                     cov_est,
                                                                     Nfreqs=Nfreqs)


    # interdependence_xy

    i_xy = tsa.interdependence_xy(Sw)
    i_xy_est = tsa.interdependence_xy(Sw_est)
    
    # This is all very approximate:
    npt.assert_almost_equal(Hw,Hw_est,decimal=1)
    npt.assert_almost_equal(Sw,Sw_est,decimal=1)
    npt.assert_almost_equal(c,c_est,1)
    npt.assert_almost_equal(f_xy,f_xy_est,1)
    npt.assert_almost_equal(f_x2y,f_x2y_est,1)
    npt.assert_almost_equal(f_y2x,f_y2x_est,1)
    npt.assert_almost_equal(i_xy,i_xy_est,1)
