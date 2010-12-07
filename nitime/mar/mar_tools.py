import numpy as np
from numpy.linalg import inv
import nitime.algorithms as alg

def crosscov_vector(x, y, nlags=None):
    """
    This method computes the following function

    .. math::

    R_{xy}(k) = E{ x(t)y^{*}(t-k) } = E{ x(t+k)y^{*}(k) }
    k \in {0, 1, ..., nlags-1}

    (* := conjugate transpose)

    Note: In the case where x==y (autocovariance), this is related to
    the other commonly used definition for vector autocovariance

    .. math::

    R_{xx}^{(2)}(k) = E{ x(t-k)x^{*}(k) } = R_{xx}^{*}(k) = R_{xx}(-k)

    Parameters
    ----------

    x, y: ndarray (nc, N)

    nlags: int, optional
       compute lags for k in {0, ..., nlags-1}

    Returns
    -------

    rxy : ndarray (nc, nc, nlags)

    """
    N = x.shape[1]
    if nlags is None:
        nlags = N
    nc = x.shape[0]

    rxy = np.empty((nc, nc, nlags))

    # rxy(k) = E{ x(t)y*(t-k) } ( * = conj transpose )
    # Take the expectation over an outer-product
    # between x(t) and conj{y(t-k)} for each t

    for k in xrange(nlags):
        # rxy(k) = E{ x(t)y*(t-k) }
        prod = x[:,None,k:] * y[None,:,:N-k].conj()
##         # rxy(k) = E{ x(t)y*(t+k) }
##         prod = x[:,None,:N-k] * y[None,:,k:].conj()
        # Do a sample mean of N-k pts? or sum and divide by N?
        rxy[...,k] = prod.mean(axis=-1)
    return rxy

def autocov_vector(x, nlags=None):
    """
    This method computes the following function

    .. math::

    R_{xx}(k) = E{ x(t)x^{*}(t-k) } = E{ x(t+k)x^{*}(k) }
    k \in {0, 1, ..., nlags-1}

    (* := conjugate transpose)

    Note: this is related to
    the other commonly used definition for vector autocovariance

    .. math::

    R_{xx}^{(2)}(k) = E{ x(t-k)x^{*}(k) } = R_{xx}^{*}(k) = R_{xx}(-k)

    Parameters
    ----------

    x: ndarray (nc, N)

    nlags: int, optional
       compute lags for k in {0, ..., nlags-1}

    Returns
    -------

    rxx : ndarray (nc, nc, nlags)

    """
    return crosscov_vector(x, x, nlags=nlags)

def lwr(r):
    """Perform a Levinson-Wiggins[Whittle]-Robinson recursion to
    find the coefficients a(i) that satisfy the system of equations:

    sum_{k=0}^{p} a(k)r(j-k) = 0, for j = {1,2,...,p}

    with the additional equation

    sum_{k=0}^{p} a(k)r(-k) = V

    where V is the covariance matrix of the innovations process

    Also note that r is defined as:

    r(k) = E{ X(t)X*(t-k) } ( * = conjugate transpose )
    r(-k) = r(k).T


    This routine adapts the algorithm found in eqs (1)-(11)
    in Morf, Vieira, Kailath 1978

    Parameters
    ----------

    r : ndarray, shape (P+1, nc, nc)

    Returns
    -------

    a : ndarray (P,nc,nc)
      coefficient sequence of order P
    sigma : ndarray (nc,nc)
      covariance estimate

    """

    # r is (P+1, nc, nc)
    nc = r.shape[1]
    P = r.shape[0]-1

    a = np.zeros((P,nc,nc)) # ar coefs
    b = np.zeros_like(a) # lp coefs
    sigf = np.zeros_like(r[0]) # forward prediction error covariance
    sigb = np.zeros_like(r[0]) # backward prediction error covariance
    delta = np.zeros_like(r[0])

    # initialize
    idnt = np.eye(nc)
    sigb[:] = r[0]
    sigf[:] = r[0]

    # iteratively find sequences A_{p+1}(i) and B_{p+1}(i)
    for p in xrange(P):

        # calculate delta_{p+1}
        # delta_{p+1} = r(p+1) + sum_{i=1}^{p} a(i)r(p+1-i)
        delta[:] = r[p+1]
        for i in xrange(1,p+1):
            delta += np.dot(a[i-1], r[p+1-i])

        # intermediate values
        ka = np.dot(delta, inv(sigf))
        kb = np.dot(delta.T, inv(sigb))

        # store a_{p} before updating sequence to a_{p+1}
        ao = a.copy()
        # a_{p+1}(i) = a_{p}(i) - ka*b_{p}(p+1-i) for i in {1,2,...,p}
        # b_{p+1}(i) = b_{p}(i) - kb*a_{p}(p+1-i) for i in {1,2,...,p}
        for i in xrange(1,p+1):
            a[i-1] -= np.dot(ka, b[p-i])
        for i in xrange(1,p+1):
            b[i-1] -= np.dot(kb, ao[p-i])

        a[p] = -ka
        b[p] = -kb

        sigb = np.dot(idnt-np.dot(ka,kb), sigb)
        sigf = np.dot(idnt-np.dot(kb,ka), sigf)

    return a, sigb

def lwr_alternate(r):
    # r(k) = E{ X(t)X*(t+k) } ( * = conjugate transpose )
    # r(-k) = r(k).T
    # this routine solves the system of equations
    # sum_{k=0}^{p} A(k)r(k-j) = 0, for j = {1,2,...,p}
    # with the additional equation
    # sum_{k=0}^{p} A(k)r(k) = V
    # where V is the covariance matrix of the innovations process
    #
    # This routine adjusts the algorithm found in eqs (1)-(11)
    # in Morf, Vieira, Kailath 1978 to reflect that this system
    # is composed in a slightly different way (due to the conflicting
    # definition of autocovariance)

    # r is (P+1, nc, nc)
    nc = r.shape[1]
    P = r.shape[0]-1

    a = np.zeros((P,nc,nc)) # ar coefs
    b = np.zeros_like(a) # lp coefs
    sigf = np.zeros_like(r[0]) # forward prediction error covariance
    sigb = np.zeros_like(r[0]) # backward prediction error covariance
    delta = np.zeros_like(r[0])

    # initialize
    idnt = np.eye(nc)
    sigb[:] = r[0]
    sigf[:] = r[0]

    # iteratively find sequences A_{p+1}(i) and B_{p+1}(i)
    for p in xrange(P):

        # calculate delta_{p+1}
        delta[:] = r[p+1]
        for i in xrange(1,p+1):
            delta += np.dot(r[p+1-i], a[i-1].T)

        # intermediate values
        ka = np.dot(delta.T, inv(sigf).T) # (inv(sigf)*del)'
        kb = np.dot(delta, inv(sigb).T)   # (inv(sigb)*del')'

        # store a_{p} before updating sequence to a_{p+1}
        ao = a.copy()
        for i in xrange(1,p+1):
            a[i-1] -= np.dot(ka, b[p-i])
        for i in xrange(1,p+1):
            b[i-1] -= np.dot(kb, ao[p-i])

        a[p] = -ka
        b[p] = -kb

        sigb = np.dot(sigb, (idnt-np.dot(ka,kb).T))
        sigf = np.dot(sigf, (idnt-np.dot(kb,ka).T))

    return a, sigb

def generate_mar(a, cov, N):
    """
    Generates a multivariate autoregressive dataset given the formula:

    X(t) + sum_{i=1}^{P} a(i)X(t-i) = E(t)

    Where E(t) is a vector of samples from possibly covarying noise processes.

    Parameters
    ----------

    a : ndarray (P, nc, nc)
       An order P set of coefficient matrices, each shaped (nc, nc) for nchannel
       data
    cov : ndarray (nc, nc)
       The innovations process covariance
    N : int
       how many samples to generate

    Returns
    -------

    mar, nz

    mar and noise process shaped (nc, N)
    """
    n_seq = cov.shape[0]
    n_order = a.shape[0]

    nz = np.random.multivariate_normal(
        np.zeros(n_seq), cov, size=(N,)
        )

    # nz is a (N x n_seq) array

    mar = np.zeros((N, n_seq), 'd')

    # this looks like a redundant loop that can be rolled into a matrix-matrix
    # multiplication at each coef matrix a(i)

    # this rearranges the equation to read:
    # X(i) = E(i) - sum_{j=1}^{p} a(j)X(i-j)
    # where X(n) n < 0 is taken to be 0
    for i in xrange(N):
        mar[i,:] = nz[i,:]
        for j in xrange( min(i, n_order) ): # j logically in set {1, 2, ..., P}
            mar[i,:] += np.dot(-a[j], mar[i-j-1,:])
    return mar.transpose(), nz.transpose()
