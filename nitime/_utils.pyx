""" -*- python -*- file
"""

import numpy as np
cimport numpy as cnp
cimport cython

@cython.boundscheck(False)
def adaptive_weights_cython(
    cnp.ndarray[cnp.npy_double, ndim=2] sdfs,
    cnp.ndarray[cnp.npy_double, ndim=1] eigvals,
    last_freq = None,
    var_est = None
    ):
    r"""
    Perform an iterative procedure to find the optimal weights for K
    direct spectral estimators of DPSS tapered signals.

    Parameters
    ----------

    sdfs: ndarray, (K x N)
       The K estimators of the spectral density function
    eigvals: ndarray, length-K
       The eigenvalues of the DPSS tapers
    last_freq: int, optional
       The last frequency whose weight to compute (e.g., if the spectrum is
       symmetric)
       

    Returns
    -------

    weights, nu

       The weights (array like sdfs), and the
       "equivalent degrees of freedom" (array length-L)

    Notes
    -----

    The weights to use for making the multitaper estimate, such that
    :math:`S_{mt} = \sum_{k} w_k^2|y_k|^2 / \sum_{k} |w_k|^2`

    If there are less than 3 tapers, then the adaptive weights are not
    found. The square root of the eigenvalues are returned as weights,
    and the degrees of freedom are 2*K

    """
    if last_freq is None:
        last_freq = sdfs.shape[1]
    K, L = sdfs.shape[0], last_freq
    if len(eigvals) < 3:
        print """
        Warning--not adaptively combining the spectral estimators
        due to a low number of tapers.
        """
        return ( np.multiply.outer(np.sqrt(eigvals), np.ones(L)), 2*K )

    rt_l = np.sqrt(eigvals)
    Kmax = len(eigvals)

    cdef cnp.ndarray[cnp.npy_double, ndim=2] weights = np.empty((K,L))
    cdef cnp.ndarray[cnp.npy_double, ndim=1] d_k = np.empty((Kmax,))
    cdef cnp.ndarray[cnp.npy_double, ndim=1] err = np.zeros_like(d_k)
    
    if var_est is None:
        # combine the SDFs in the traditional way in order to estimate
        # the variance of the timeseries
        N = sdfs.shape[1]
        sdf = (sdfs*eigvals[:,None]).sum(axis=0)
        sdf /= eigvals.sum()
        var_est = np.trapz(sdf, dx=1.0/N)
    
    # need to loop over freqs
    cdef Py_ssize_t f, k, n
    cdef cnp.npy_double sqr_norm, sdf_iter
    for f in xrange(last_freq):
        # (funky syntax because of cython slicing)
        # To begin iteration, average the 1st two tapered estimates
        sdf_iter = (sdfs[0,f]*eigvals[0] + sdfs[1,f]*eigvals[1])
        sdf_iter /= (eigvals[0] + eigvals[1])
        # err is initialized to 0
        n = 0
        while True:
            d_k = sdf_iter / (eigvals*sdf_iter + (1-eigvals)*var_est)
            d_k *= rt_l
            # subtract this d_k from the previous to find the convergence error
            err -= d_k
            # mse < 1e-10 ==> sse < 1e-10 * K .. let's call it 1e-10 still
            n += 1
            if (err**2).sum() < 1e-10 or n > 20:
                break
            # update the iterative estimate of the sdf with this d_k
            sdf_iter = 0
            sqr_norm = 0
            for k in xrange(Kmax):
                sdf_iter += (d_k[k]**2 * sdfs[k,f])
                sqr_norm += d_k[k]**2
                err[k] = d_k[k]
            sdf_iter /= sqr_norm

        for k in xrange(Kmax):
            weights[k,f] = d_k[k]
    
    # XXX: unsure about this measure
    nu = 2*(weights**2).sum(axis=0)
    return weights, nu
