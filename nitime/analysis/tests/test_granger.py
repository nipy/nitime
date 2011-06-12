"""
Testing the analysis.granger submodule

"""


import numpy as np
import nitime.analysis.granger as gc
import nitime.utils as utils
import numpy.testing as npt

def test_fit_model():
    """

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
    n_lags = order + 1

    n_process = am.shape[-1]

    z = np.empty((N, n_process, L))
    nz = np.empty((N, n_process, L))

    for i in xrange(N):
        z[i], nz[i] = utils.generate_mar(am, cov, L)

    Rxx = np.empty((N, n_process, n_process, n_lags))
    coef = np.empty((N, n_process, n_process, order))
    ecov = np.empty((N, n_process, n_process))

    for i in xrange(N):
        order, this_Rxx, this_coef, this_ecov = gc.fit_model(z[i][0], z[i][1], order=2)
        Rxx[i] = this_Rxx
        coef[i] = this_coef
        ecov[i] = this_ecov

    npt.assert_almost_equal(cov, np.mean(ecov,0), decimal=1)
    npt.assert_almost_equal(am, np.mean(coef,0), decimal=1)

    est_order = []
    for i in xrange(N):
        this_order, this_Rxx, this_coef, this_ecov = gc.fit_model(z[i][0], z[i][1])
        est_order.append(this_order)

    npt.assert_almost_equal(order, np.mean(est_order), decimal=1)
