""" -*- python -*- file
C-level implementation of the following routines in utils.py:

  * tridisolve()

"""

import numpy as np
cimport numpy as cnp
cimport cython

@cython.boundscheck(False)
def tridisolve(cnp.ndarray[cnp.npy_double, ndim=1] d,
               cnp.ndarray[cnp.npy_double, ndim=1] e,
               cnp.ndarray[cnp.npy_double, ndim=1] b, overwrite_b=True):
    """
    Symmetric tridiagonal system solver, from Golub and Van Loan pg 157

    Parameters
    ----------

    d : ndarray
      main diagonal stored in d[:]
    e : ndarray
      superdiagonal stored in e[:-1]
    b : ndarray
      RHS vector

    Returns
    -------

    x : ndarray
      Solution to Ax = b (if overwrite_b is False). Otherwise solution is
      stored in previous RHS vector b

    """
    # indexing
    cdef int N = len(b)
    cdef int k

    # work vectors
    cdef cnp.ndarray[cnp.npy_double, ndim=1] dw
    cdef cnp.ndarray[cnp.npy_double, ndim=1] ew
    cdef cnp.ndarray[cnp.npy_double, ndim=1] x
    dw = d.copy()
    ew = e.copy()
    if overwrite_b:
        x = b
    else:
        x = b.copy()
    for k in xrange(1, N):
        # e^(k-1) = e(k-1) / d(k-1)
        # d(k) = d(k) - e^(k-1)e(k-1) / d(k-1)
        t = ew[k - 1]
        ew[k - 1] = t / dw[k - 1]
        dw[k] = dw[k] - t * ew[k - 1]
    for k in xrange(1, N):
        x[k] = x[k] - ew[k - 1] * x[k - 1]
    x[N - 1] = x[N - 1] / dw[N - 1]
    for k in xrange(N - 2, -1, -1):
        x[k] = x[k] / dw[k] - ew[k] * x[k + 1]

    if not overwrite_b:
        return x
