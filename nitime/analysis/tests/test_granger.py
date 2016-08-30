"""
Testing the analysis.granger submodule

"""


import numpy as np
import numpy.testing as npt

import nitime.analysis.granger as gc
import nitime.utils as utils
import nitime.timeseries as ts


def test_model_fit():
    """
    Testing of model fitting procedure of the nitime.analysis.granger module.
    """
    # Start by generating some MAR processes (according to Ding and Bressler),
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

    for i in range(N):
        z[i], nz[i] = utils.generate_mar(am, cov, L)

    # First we test that the model fitting procedure recovers the coefficients,
    # on average:
    Rxx = np.empty((N, n_process, n_process, n_lags))
    coef = np.empty((N, n_process, n_process, order))
    ecov = np.empty((N, n_process, n_process))

    for i in range(N):
        this_order, this_Rxx, this_coef, this_ecov = gc.fit_model(z[i][0],
                                                                  z[i][1],
                                                                  order=2)
        Rxx[i] = this_Rxx
        coef[i] = this_coef
        ecov[i] = this_ecov

    npt.assert_almost_equal(cov, np.mean(ecov, axis=0), decimal=1)
    npt.assert_almost_equal(am, np.mean(coef, axis=0), decimal=1)

    # Next we test that the automatic model order estimation procedure works:
    est_order = []
    for i in range(N):
        this_order, this_Rxx, this_coef, this_ecov = gc.fit_model(z[i][0],
                                                                  z[i][1])
        est_order.append(this_order)

    npt.assert_almost_equal(order, np.mean(est_order), decimal=1)


def test_GrangerAnalyzer():
    """
    Testing the GrangerAnalyzer class, which simplifies calculations of related
    quantities
    """

    # Start by generating some MAR processes (according to Ding and Bressler),
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

    L = 1024
    z, nz = utils.generate_mar(am, cov, L)

    # Move on to testing the Analyzer object itself:
    ts1 = ts.TimeSeries(data=z, sampling_rate=np.pi)
    g1 = gc.GrangerAnalyzer(ts1)

    # Check that things have the right shapes:
    npt.assert_equal(g1.frequencies.shape[-1], g1._n_freqs // 2 + 1)
    npt.assert_equal(g1.causality_xy[0, 1].shape, g1.causality_yx[0, 1].shape)

    # Test inputting ij:
    g2 = gc.GrangerAnalyzer(ts1, ij=[(0, 1), (1, 0)])

    # g1 agrees with g2
    npt.assert_almost_equal(g1.causality_xy[0, 1], g2.causality_xy[0, 1])
    npt.assert_almost_equal(g1.causality_yx[0, 1], g2.causality_yx[0, 1])

    # x => y for one is like y => x for the other:
    npt.assert_almost_equal(g2.causality_yx[1, 0], g2.causality_xy[0, 1])
    npt.assert_almost_equal(g2.causality_xy[1, 0], g2.causality_yx[0, 1])
