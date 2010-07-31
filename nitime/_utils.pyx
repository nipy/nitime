""" -*- python -*- file
"""

import numpy as np
cimport numpy as cnp
cimport cython

@cython.boundscheck(False)
def adaptive_weights_cython(
    cnp.ndarray[cnp.npy_double, ndim=2] sdfs,
    cnp.ndarray[cnp.npy_double, ndim=1] eigvals,
    N,
    sigma_est = None
    ):
    r"""
    Perform an iterative procedure to find the optimal weights for K
    direct spectral estimators of DPSS tapered signals.

    Parameters
    ----------

    sdfs: ndarray, (K x L)
       The K estimators
    eigvals: ndarray, length-K
       The eigenvalues of the DPSS tapers
    N: int,
       length of the signal

    Returns
    -------

    weights, nu

       The weights (array like sdfs), and the
       "equivalent degrees of freedom" (array length-L)

    Notes
    -----

    The weights to use for making the multitaper estimate, such that
    :math:`S_{mt} = \sum_{k} w_k^2S_k^{mt} / \sum_{k} |w_k|^2`

    If there are less than 3 tapers, then the adaptive weights are not
    found. The square root of the eigenvalues are returned as weights,
    and the degrees of freedom are 2*K

    """
    if len(eigvals) < 3:
        print """
        Warning--not adaptively combining the spectral estimators
        due to a low number of tapers.
        """
        return np.sqrt(eigvals), len(eigvals)

    v = eigvals
    rt_v = np.sqrt(eigvals)

    cdef cnp.ndarray[cnp.npy_double, ndim=2] weights = np.empty_like(sdfs)
    cdef cnp.ndarray[cnp.npy_double, ndim=1] d_k = np.empty(len(eigvals))
    cdef cnp.ndarray[cnp.npy_double, ndim=1] err = np.empty_like(d_k)
    
    if sigma_est is None:
        # combine the SDFs in the traditional way in order to estimate
        # the variance of the timeseries
        sdf = (sdfs*eigvals[:,None]).sum(axis=0)
        sdf /= eigvals.sum()
        sigma_est = np.trapz(sdf, dx=1.0/N)
    
    # need to loop over freqs
    cdef Py_ssize_t f, k, n
    for f in xrange(sdfs.shape[1]):
        # (funky syntax because of cython slicing)
        # To begin iteration, average the 1st two tapered estimates
        sdf_iter = (sdfs[0,f]*eigvals[0] + sdfs[1,f]*eigvals[1])
        sdf_iter /= (eigvals[0] + eigvals[1])
        err[:] = 0
        n = 0
        while True:
            d_k = sdf_iter / (eigvals*sdf_iter + (1-eigvals)*sigma_est)
            d_k *= rt_v
            # subtract this d_k from the previous to find the convergence error
            err -= d_k
            # mse < 1e-10 ==> sse < 1e-10 * N
            n += 1
            if (err**2).sum() < 1e-10 or n > 20:
                break
            # update the iterative estimate of the sdf with this d_k
            sdf_iter = (d_k**2 * sdfs[:,f]).sum()
            sdf_iter /= (d_k**2).sum()
            err = d_k
        weights[:,f] = d_k
    # XXX: unsure about this measure
    nu = 2*(weights**2).sum(axis=0)
    return weights, nu
