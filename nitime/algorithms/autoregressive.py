r"""

Autoregressive (AR) processes are processes of the form:

.. math::

  x(n) = a(1)x(n-1) + a(2)x(n-2) + ... + a(P)x(n-P) + e(n)

where e(n) is a white noise process. The usage of 'e' suggests interpreting
the linear combination of P past values of x(n) as the minimum mean square
error linear predictor of x(n) Thus

.. math::

  e(n) = x(n) - a(1)x(n-1) - a(2)x(n-2) - ... - a(P)x(n-P)

Due to whiteness, e(n) is also pointwise uncorrelated--ie,

.. math::
   :nowrap:

   \begin{align*}
      \text{(i)}   && E\{e(n)e^{*}(n-m)\}& = \delta(n-m) &\\
      \text{(ii)}  && E\{e(n)x^{*}(m)\}  & = 0           & m\neq n\\
      \text{(iii)} && E\{|e|^{2}\} = E\{e(n)e^{*}(n)\} &= E\{e(n)x^{*}(n)\} &
   \end{align*}

These principles form the basis of the methods in this module for
estimating the AR coefficients and the error/innovations power.
"""


import numpy as np
from nitime.lazy import scipy_linalg as linalg

import nitime.utils as utils
from .spectral import freq_response


def AR_est_YW(x, order, rxx=None):
    r"""Determine the autoregressive (AR) model of a random process x using
    the Yule Walker equations. The AR model takes this convention:

    .. math::

      x(n) = a(1)x(n-1) + a(2)x(n-2) + \dots + a(p)x(n-p) + e(n)

    where e(n) is a zero-mean white noise process with variance sig_sq,
    and p is the order of the AR model. This method returns the a_i and
    sigma

    The orthogonality property of minimum mean square error estimates
    states that

    .. math::

      E\{e(n)x^{*}(n-k)\} = 0 \quad 1\leq k\leq p

    Inserting the definition of the error signal into the equations above
    yields the Yule Walker system of equations:

    .. math::

      R_{xx}(k) = \sum_{i=1}^{p}a(i)R_{xx}(k-i) \quad1\leq k\leq p

    Similarly, the variance of the error process is

    .. math::

      E\{e(n)e^{*}(n)\}   = E\{e(n)x^{*}(n)\} = R_{xx}(0)-\sum_{i=1}^{p}a(i)R^{*}(i)


    Parameters
    ----------
    x : ndarray
        The sampled autoregressive random process

    order : int
        The order p of the AR system

    rxx : ndarray (optional)
        An optional, possibly unbiased estimate of the autocorrelation of x

    Returns
    -------
    ak, sig_sq : The estimated AR coefficients and innovations variance

    """
    if rxx is not None and type(rxx) == np.ndarray:
        r_m = rxx[:order + 1]
    else:
        r_m = utils.autocorr(x)[:order + 1]

    Tm = linalg.toeplitz(r_m[:order])
    y = r_m[1:]
    ak = linalg.solve(Tm, y)
    sigma_v = r_m[0].real - np.dot(r_m[1:].conj(), ak).real
    return ak, sigma_v


def AR_est_LD(x, order, rxx=None):
    r"""Levinson-Durbin algorithm for solving the Hermitian Toeplitz
    system of Yule-Walker equations in the AR estimation problem

    .. math::

       T^{(p)}a^{(p)} = \gamma^{(p+1)}

    where

    .. math::
       :nowrap:

       \begin{align*}
       T^{(p)} &= \begin{pmatrix}
          R_{0} & R_{1}^{*} & \cdots & R_{p-1}^{*}\\
          R_{1} & R_{0} & \cdots & R_{p-2}^{*}\\
          \vdots & \vdots & \ddots & \vdots\\
          R_{p-1}^{*} & R_{p-2}^{*} & \cdots & R_{0}
       \end{pmatrix}\\
       a^{(p)} &=\begin{pmatrix} a_1 & a_2 & \cdots a_p \end{pmatrix}^{T}\\
       \gamma^{(p+1)}&=\begin{pmatrix}R_1 & R_2 & \cdots & R_p \end{pmatrix}^{T}
       \end{align*}

    and :math:`R_k` is the autocorrelation of the kth lag

    Parameters
    ----------

    x : ndarray
      the zero-mean stochastic process
    order : int
      the AR model order--IE the rank of the system.
    rxx : ndarray, optional
      (at least) order+1 samples of the autocorrelation sequence

    Returns
    -------

    ak, sig_sq
      The AR coefficients for 1 <= k <= p, and the variance of the
      driving white noise process

    """

    if rxx is not None and type(rxx) == np.ndarray:
        rxx_m = rxx[:order + 1]
    else:
        rxx_m = utils.autocorr(x)[:order + 1]
    w = np.zeros((order + 1, ), rxx_m.dtype)
    # initialize the recursion with the R[0]w[1]=r[1] solution (p=1)
    b = rxx_m[0].real
    w_k = rxx_m[1] / b
    w[1] = w_k
    p = 2
    while p <= order:
        b *= 1 - (w_k * w_k.conj()).real
        w_k = (rxx_m[p] - (w[1:p] * rxx_m[1:p][::-1]).sum()) / b
        # update w_k from k=1,2,...,p-1
        # with a correction from w*_i i=p-1,p-2,...,1
        w[1:p] = w[1:p] - w_k * w[1:p][::-1].conj()
        w[p] = w_k
        p += 1
    b *= 1 - (w_k * w_k.conj()).real
    return w[1:], b


def lwr_recursion(r):
    r"""Perform a Levinson-Wiggins[Whittle]-Robinson recursion to
    find the coefficients a(i) that satisfy the matrix version
    of the Yule-Walker system of P + 1 equations:

    sum_{i=0}^{P} a(i)r(k-i) = 0, for k = {1,2,...,P}

    with the additional equation

    sum_{i=0}^{P} a(i)r(-k) = V

    where V is the covariance matrix of the innovations process,
    and a(0) is fixed at the identity matrix

    Also note that r is defined as:

    r(k) = E{ X(t)X*(t-k) } ( * = conjugate transpose )
    r(-k) = r*(k)


    This routine adapts the algorithm found in eqs (1)-(11)
    in Morf, Vieira, Kailath 1978

    Parameters
    ----------

    r : ndarray, shape (P + 1, nc, nc)

    Returns
    -------

    a : ndarray (P,nc,nc)
      coefficient sequence of order P
    sigma : ndarray (nc,nc)
      covariance estimate

    """

    # r is (P+1, nc, nc)
    nc = r.shape[1]
    P = r.shape[0] - 1

    a = np.zeros((P, nc, nc))  # ar coefs
    b = np.zeros_like(a)  # lp coefs
    sigb = np.zeros_like(r[0])  # forward prediction error covariance
    sigf = np.zeros_like(r[0])  # backward prediction error covariance
    delta = np.zeros_like(r[0])

    # initialize
    idnt = np.eye(nc)
    sigf[:] = r[0]
    sigb[:] = r[0]

    # iteratively find sequences A_{p+1}(i) and B_{p+1}(i)
    for p in range(P):

        # calculate delta_{p+1}
        # delta_{p+1} = r(p+1) + sum_{i=1}^{p} a(i)r(p+1-i)
        delta[:] = r[p + 1]
        for i in range(1, p + 1):
            delta += np.dot(a[i - 1], r[p + 1 - i])

        # intermediate values XXX: should turn these into solution-problems
        ka = np.dot(delta, linalg.inv(sigb))
        kb = np.dot(delta.conj().T, linalg.inv(sigf))

        # store a_{p} before updating sequence to a_{p+1}
        ao = a.copy()
        # a_{p+1}(i) = a_{p}(i) - ka*b_{p}(p+1-i) for i in {1,2,...,p}
        # b_{p+1}(i) = b_{p}(i) - kb*a_{p}(p+1-i) for i in {1,2,...,p}
        for i in range(1, p + 1):
            a[i - 1] -= np.dot(ka, b[p - i])
        for i in range(1, p + 1):
            b[i - 1] -= np.dot(kb, ao[p - i])

        a[p] = -ka
        b[p] = -kb

        sigf = np.dot(idnt - np.dot(ka, kb), sigf)
        sigb = np.dot(idnt - np.dot(kb, ka), sigb)

    return a, sigf


def MAR_est_LWR(x, order, rxx=None):
    r"""
    MAR estimation, using the LWR algorithm, as in Morf et al.


    Parameters
    ----------
    x : ndarray
        The sampled autoregressive random process

    order : int
        The order P of the AR system

    rxx : ndarray (optional)
        An optional, possibly unbiased estimate of the autocovariance of x

    Returns
    -------
    a, ecov : The system coefficients and the estimated covariance
    """
    Rxx = utils.autocov_vector(x, nlags=order)
    a, ecov = lwr_recursion(Rxx.transpose(2, 0, 1))
    return a, ecov


def AR_psd(ak, sigma_v, n_freqs=1024, sides='onesided'):
    r"""
    Compute the PSD of an AR process, based on the process coefficients and
    covariance

    n_freqs : int
        The number of spacings on the frequency grid from [-PI,PI).
        If sides=='onesided', n_freqs/2+1 frequencies are computed from [0,PI]

    sides : str (optional)
        Indicates whether to return a one-sided or two-sided PSD

    Returns
    -------
    (w, ar_psd)
    w : Array of normalized frequences from [-.5, .5) or [0,.5]
    ar_psd : A PSD estimate computed by sigma_v / |1-a(f)|**2 , where
             a(f) = DTFT(ak)


    """
    # compute the psd as |H(f)|**2, where H(f) is the transfer function
    # for this model s[n] = a1*s[n-1] + a2*s[n-2] + ... aP*s[n-P] + v[n]
    # Taken as a IIR system with unit-variance white noise input e[n]
    # and output s[n],
    # b0*e[n] = w0*s[n] + w1*s[n-1] + w2*s[n-2] + ... + wP*s[n-P],
    # where b0 = sqrt(VAR{v[n]}), w0 = 1, and wk = -ak for k>0
    # the transfer function here is H(f) = DTFT(w)
    # leading to Sxx(f)/Exx(f) = |H(f)|**2 = VAR{v[n]} / |W(f)|**2
    w, hw = freq_response(sigma_v ** 0.5, a=np.r_[1, -ak],
                          n_freqs=n_freqs, sides=sides)
    ar_psd = (hw * hw.conj()).real
    return (w, 2 * ar_psd) if sides == 'onesided' else (w, ar_psd)


#-----------------------------------------------------------------------------
# Granger causality analysis
#-----------------------------------------------------------------------------
def transfer_function_xy(a, n_freqs=1024):
    r"""Helper routine to compute the transfer function H(w) based
    on sequence of coefficient matrices A(i). The z transforms
    follow from this definition:

    X[t] + sum_{k=1}^P a[k]X[t-k] = Err[t]

    Parameters
    ----------

    a : ndarray, shape (P, 2, 2)
      sequence of coef matrices describing an mAR process
    n_freqs : int, optional
      number of frequencies to compute in range [0,PI]

    Returns
    -------

    Hw : ndarray
      The transfer function from innovations process vector to
      mAR process X

    """
    # these concatenations follow from the observation that A(0) is
    # implicitly the identity matrix
    ai = np.r_[1, a[:, 0, 0]]
    bi = np.r_[0, a[:, 0, 1]]
    ci = np.r_[0, a[:, 1, 0]]
    di = np.r_[1, a[:, 1, 1]]

    # compute A(w) such that A(w)X(w) = Err(w)
    w, aw = freq_response(ai, n_freqs=n_freqs)
    _, bw = freq_response(bi, n_freqs=n_freqs)
    _, cw = freq_response(ci, n_freqs=n_freqs)
    _, dw = freq_response(di, n_freqs=n_freqs)

    #A = np.array([ [1-aw, -bw], [-cw, 1-dw] ])
    A = np.array([[aw, bw], [cw, dw]])
    # compute the transfer function from Err to X. Since Err(w) is 1(w),
    # the transfer function H(w) = A^(-1)(w)
    # (use 2x2 matrix shortcut)
    detA = (A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])
    Hw = np.array([[dw, -bw], [-cw, aw]])
    Hw /= detA
    return w, Hw


def spectral_matrix_xy(Hw, cov):
    r"""Compute the spectral matrix S(w), from the convention:

    X[t] + sum_{k=1}^P a[k]X[t-k] = Err[t]

    The formulation follows from Ding, Chen, Bressler 2008,
    pg 6 eqs (11) to (15)

    The transfer function H(w) should be computed first from
    transfer_function_xy()

    Parameters
    ----------

    Hw : ndarray (2, 2, n_freqs)
      Pre-computed transfer function from transfer_function_xy()

    cov : ndarray (2, 2)
      The covariance between innovations processes in Err[t]

    Returns
    -------

    Sw : ndarrays
      matrix of spectral density functions
    """

    nw = Hw.shape[-1]
    # now compute specral density function estimate
    # S(w) = H(w)SigH*(w)
    Sw = np.empty((2, 2, nw), 'D')

    # do a shortcut for 2x2:
    # compute T(w) = SigH*(w)
    # t00 = Sig[0,0] * H*_00(w) + Sig[0,1] * H*_10(w)
    t00 = cov[0, 0] * Hw[0, 0].conj() + cov[0, 1] * Hw[0, 1].conj()
    # t01 = Sig[0,0] * H*_01(w) + Sig[0,1] * H*_11(w)
    t01 = cov[0, 0] * Hw[1, 0].conj() + cov[0, 1] * Hw[1, 1].conj()
    # t10 = Sig[1,0] * H*_00(w) + Sig[1,1] * H*_10(w)
    t10 = cov[1, 0] * Hw[0, 0].conj() + cov[1, 1] * Hw[0, 1].conj()
    # t11 = Sig[1,0] * H*_01(w) + Sig[1,1] * H*_11(w)
    t11 = cov[1, 0] * Hw[1, 0].conj() + cov[1, 1] * Hw[1, 1].conj()

    # now S(w) = H(w)T(w)
    Sw[0, 0] = Hw[0, 0] * t00 + Hw[0, 1] * t10
    Sw[0, 1] = Hw[0, 0] * t01 + Hw[0, 1] * t11
    Sw[1, 0] = Hw[1, 0] * t00 + Hw[1, 1] * t10
    Sw[1, 1] = Hw[1, 0] * t01 + Hw[1, 1] * t11

    return Sw


def coherence_from_spectral(Sw):
    r"""Compute the spectral coherence between processes X and Y,
    given their spectral matrix S(w)

    Parameters
    ----------

    Sw : ndarray
      spectral matrix
    """

    Sxx = Sw[0, 0].real
    Syy = Sw[1, 1].real

    Sxy_mod_sq = (Sw[0, 1] * Sw[1, 0]).real
    Sxy_mod_sq /= Sxx
    Sxy_mod_sq /= Syy
    return Sxy_mod_sq


def interdependence_xy(Sw):
    r"""Compute the 'total interdependence' between processes X and Y,
    given their spectral matrix S(w)

    Parameters
    ----------

    Sw : ndarray
      spectral matrix

    Returns
    -------

    fxy(w)
      interdependence function of frequency
    """

    Cw = coherence_from_spectral(Sw)
    return -np.log(1 - Cw)


def granger_causality_xy(a, cov, n_freqs=1024):
    r"""Compute the Granger causality between processes X and Y, which
    are linked in a multivariate autoregressive (mAR) model parameterized
    by coefficient matrices a(i) and the innovations covariance matrix

    X[t] + sum_{k=1}^P a[k]X[t-k] = Err[t]

    Parameters
    ----------

    a : ndarray, (P,2,2)
      coefficient matrices characterizing the autoregressive mixing
    cov : ndarray, (2,2)
      covariance matrix characterizing the innovations vector
    n_freqs: int
      number of frequencies to compute in the fourier transform

    Returns
    -------

    w, f_x_on_y, f_y_on_x, f_xy, Sw
      1) vector of frequencies
      2) function of the Granger causality of X on Y
      3) function of the Granger causality of Y on X
      4) function of the 'instantaneous causality' between X and Y
      5) spectral density matrix
    """

    w, Hw = transfer_function_xy(a, n_freqs=n_freqs)

    sigma = cov[0, 0]
    upsilon = cov[0, 1]
    gamma = cov[1, 1]

    # this transformation of the transfer functions computes the
    # Granger causality of Y on X
    gamma2 = gamma - upsilon ** 2 / sigma

    Hxy = Hw[0, 1]
    Hxx_hat = Hw[0, 0] + (upsilon / sigma) * Hxy

    xx_auto_component = (sigma * Hxx_hat * Hxx_hat.conj()).real
    cross_component = gamma2 * Hxy * Hxy.conj()
    Sxx = xx_auto_component + cross_component
    f_y_on_x = np.log(Sxx.real / xx_auto_component)

    # this transformation computes the Granger causality of X on Y
    sigma2 = sigma - upsilon ** 2 / gamma

    Hyx = Hw[1, 0]
    Hyy_hat = Hw[1, 1] + (upsilon / gamma) * Hyx
    yy_auto_component = (gamma * Hyy_hat * Hyy_hat.conj()).real
    cross_component = sigma2 * Hyx * Hyx.conj()
    Syy = yy_auto_component + cross_component
    f_x_on_y = np.log(Syy.real / yy_auto_component)

    # now compute cross densities, using the latest transformation
    Hxx = Hw[0, 0]
    Hyx = Hw[1, 0]
    Hxy_hat = Hw[0, 1] + (upsilon / gamma) * Hxx
    Sxy = sigma2 * Hxx * Hyx.conj() + gamma * Hxy_hat * Hyy_hat.conj()
    Syx = sigma2 * Hyx * Hxx.conj() + gamma * Hyy_hat * Hxy_hat.conj()

    # can safely throw away imaginary part
    # since Sxx and Syy are real, and Sxy == Syx*
    detS = (Sxx * Syy - Sxy * Syx).real
    f_xy = xx_auto_component * yy_auto_component
    f_xy /= detS
    f_xy = np.log(f_xy)

    return w, f_x_on_y, f_y_on_x, f_xy, np.array([[Sxx, Sxy], [Syx, Syy]])
