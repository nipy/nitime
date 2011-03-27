"""

Autoregressive processes are processes of the form:


This module contains functions for the estimation of 
"""


import numpy as np
from scipy import linalg

import nitime.utils as utils
from spectral import freq_response

def AR_est_YW(s, order, sxx=None):
    """Finds the parameters for an autoregressive model of order norder
    of the process s. Using these parameters, an estimate of the PSD
    is calculated from [-PI,PI) in Nfreqs, or [0,PI] in {N/2+1}freqs.
    Uses the basic Yule Walker system of equations, and a baised estimate
    of sxx (unless sxx is provided).

    The model for the autoregressive process takes this convention:
    s[n] = a1*s[n-1] + a2*s[n-2] + ... aP*s[n-P] + v[n]

    where v[n] is a zero-mean white noise process with variance=sigma_v

    Parameters
    ----------
    s : ndarray
        The sampled autoregressive random process

    order : int
        The order P of the AR system

    sxx : ndarray (optional)
        An optional, possibly unbiased estimate of the autocovariance of s

    Returns
    -------
    a, ecov: The system coefficients and the estimated covariance

    """
    if sxx is not None and type(sxx) == np.ndarray:
        sxx_m = sxx[:order+1]
    else:
        sxx_m = utils.autocov(s)[:order+1]

    R = linalg.toeplitz(sxx_m[:order].conj())
    y = sxx_m[1:].conj()
    ak = linalg.solve(R,y)
    sigma_v = sxx_m[0] - np.dot(sxx_m[1:], ak)
    return ak, sigma_v

def AR_est_LD(s, order, sxx=None):
    """Finds the parameters for an autoregressive model of order norder
    of the process s. Using these parameters, an estimate of the PSD
    is calculated from [-PI,PI) in Nfreqs, or [0,PI] in {N/2+1}freqs.
    Uses the Levinson-Durbin recursion method, and a baised estimate
    of sxx (unless sxx is provided).

    The model for the autoregressive process takes this convention:
    s[n] = a1*s[n-1] + a2*s[n-2] + ... aP*s[n-P] + v[n]

    where v[n] is a zero-mean white noise process with variance=sigma_v

    Parameters
    ----------
    s : ndarray
        The sampled autoregressive random process

    order : int
        The order P of the AR system

    sxx : ndarray (optional)
        An optional, possibly unbiased estimate of the autocovariance of s

    
    Returns
    -------
    a, ecov: The system coefficients and the estimated covariance

    """
    if sxx is not None and type(sxx) == np.ndarray:
        sxx_m = sxx[:order+1]
    else:
        sxx_m = utils.autocov(s)[:order+1]
    
    phi = np.zeros((order+1, order+1), 'd')
    sig = np.zeros(order+1)
    # initial points for the recursion
    phi[1,1] = sxx_m[1]/sxx_m[0]
    sig[1] = sxx_m[0] - phi[1,1]*sxx_m[1]
    for k in xrange(2,order+1):
        phi[k,k] = (sxx_m[k]-np.dot(phi[1:k,k-1], sxx_m[1:k][::-1]))/sig[k-1]
        for j in xrange(1,k):
            phi[j,k] = phi[j,k-1] - phi[k,k]*phi[k-j,k-1]
        sig[k] = sig[k-1]*(1 - phi[k,k]**2)

    sigma_v = sig[-1]; ak = phi[1:,-1]
    return ak, sigma_v

def MAR_est_LWR(s, order, sxx=None):

    """
    MAR estimation, using the LWR algorithm, as in Morf et al.


    Parameters
    ----------
    s : ndarray
        The sampled autoregressive random process

    order : int
        The order P of the AR system

    sxx : ndarray (optional)
        An optional, possibly unbiased estimate of the autocovariance of s

    Returns
    -------
    a, ecov: The system coefficients and the estimated covariance
    """
    Rxx = utils.autocov_vector(s, nlags=order)
    a, ecov = utils.lwr(Rxx.transpose(2,0,1))
    return a,ecov

def AR_psd(ak, sigma_v, Nfreqs=1024, sides='onesided'):
    """What does this do?

    Nfreqs : int
        The number of spacings on the frequency grid from [-PI,PI).
        If sides=='onesided', Nfreqs/2+1 frequencies are computed from [0,PI]

    sides : str (optional)
        Indicates whether to return a one-sided or two-sided PSD

    system : bool (optional)
        If True, return the AR system parameters, sigma_v and a{k}


    Returns
    -------
    (w, ar_psd)
    w : Array of normalized frequences from [-.5, .5) or [0,.5]
    ar_psd : A PSD estimate computed by sigma_v / |1-a(f)|**2 , where
             a(f) = DTFT(ak)


    """
    # compute the psd as |h(f)|**2, where h(f) is the transfer function..
    # for this model s[n] = a1*s[n-1] + a2*s[n-2] + ... aP*s[n-P] + v[n]
    # Taken as a FIR system from s[n] to v[n],
    # v[n] = w0*s[n] + w1*s[n-1] + w2*s[n-2] + ... + wP*s[n-P],
    # where w0 = 1, and wk = -ak for k>0
    # the transfer function here is H(f) = DTFT(w)
    # leading to Sxx(f) = Vxx(f) / |H(f)|**2 = sigma_v / |H(f)|**2
    w, hw = freq_response(sigma_v**0.5, a=np.concatenate(([1], -ak)),
                     Nfreqs=Nfreqs, sides=sides)
    ar_psd = (hw*hw.conj()).real
    return (w,2*ar_psd) if sides=='onesided' else (w,ar_psd)




#-----------------------------------------------------------------------------
# Granger causality analysis
#-----------------------------------------------------------------------------

def transfer_function_xy(a, Nfreqs=1024):
    """Helper routine to compute the transfer function H(w) based
    on sequence of coefficient matrices A(i). The z transforms
    follow from this definition:

    X[t] + sum_{k=1}^P a[k]X[t-k] = Err[t]

    Parameters
    ----------

    a : ndarray, shape (P, 2, 2)
      sequence of coef matrices describing an mAR process
    Nfreqs : int, optional
      number of frequencies to compute in range [0,PI]

    Returns
    -------

    Hw : ndarray
      The transfer function from innovations process vector to
      mAR process X

    """
    # these concatenations follow from the observation that A(0) is
    # implicitly the identity matrix
    ai = np.r_[1, a[:,0,0]]
    bi = np.r_[0, a[:,0,1]]
    ci = np.r_[0, a[:,1,0]]
    di = np.r_[1, a[:,1,1]]

    # compute A(w) such that A(w)X(w) = Err(w)
    w, aw = freq_response(ai, Nfreqs=Nfreqs)
    _, bw = freq_response(bi, Nfreqs=Nfreqs)
    _, cw = freq_response(ci, Nfreqs=Nfreqs)
    _, dw = freq_response(di, Nfreqs=Nfreqs)

    #A = np.array([ [1-aw, -bw], [-cw, 1-dw] ])
    A = np.array([ [aw, bw], [cw, dw] ])
    # compute the transfer function from Err to X. Since Err(w) is 1(w),
    # the transfer function H(w) = A^(-1)(w)
    # (use 2x2 matrix shortcut)
    detA = (A[0,0]*A[1,1] - A[0,1]*A[1,0])
    Hw = np.array( [ [dw, -bw], [-cw, aw] ] )
    Hw /= detA
    return w, Hw

def spectral_matrix_xy(Hw, cov):
    """Compute the spectral matrix S(w), from the convention:

    X[t] + sum_{k=1}^P a[k]X[t-k] = Err[t]

    The formulation follows from Ding, Chen, Bressler 2008,
    pg 6 eqs (11) to (15)

    The transfer function H(w) should be computed first from
    transfer_function_xy()

    Parameters
    ----------

    Hw : ndarray (2, 2, Nfreqs)
      Pre-computed transfer function from transfer_function_xy()

    cov : ndarray (2, 2)
      The covariance between innovations processes in Err[t]

    Returns
    -------

    Sw: ndarrays
      matrix of spectral density functions
    """

    nw = Hw.shape[-1]
    # now compute specral density function estimate
    # S(w) = H(w)SigH*(w)
    Sw = np.empty( (2,2,nw), 'D')

    # do a shortcut for 2x2:
    # compute T(w) = SigH*(w)
    # t00 = Sig[0,0] * H*_00(w) + Sig[0,1] * H*_10(w)
    t00 = cov[0,0]*Hw[0,0].conj() + cov[0,1]*Hw[0,1].conj()
    # t01 = Sig[0,0] * H*_01(w) + Sig[0,1] * H*_11(w)
    t01 = cov[0,0]*Hw[1,0].conj() + cov[0,1]*Hw[1,1].conj()
    # t10 = Sig[1,0] * H*_00(w) + Sig[1,1] * H*_10(w)
    t10 = cov[1,0]*Hw[0,0].conj() + cov[1,1]*Hw[0,1].conj()
    # t11 = Sig[1,0] * H*_01(w) + Sig[1,1] * H*_11(w)
    t11 = cov[1,0]*Hw[1,0].conj() + cov[1,1]*Hw[1,1].conj()

    # now S(w) = H(w)T(w)
    Sw[0,0] = Hw[0,0]*t00 + Hw[0,1]*t10
    Sw[0,1] = Hw[0,0]*t01 + Hw[0,1]*t11
    Sw[1,0] = Hw[1,0]*t00 + Hw[1,1]*t10
    Sw[1,1] = Hw[1,0]*t01 + Hw[1,1]*t11

    return Sw

def coherence_from_spectral(Sw):
    """Compute the spectral coherence between processes X and Y,
    given their spectral matrix S(w)

    Parameters
    ----------

    Sw : ndarray
      spectral matrix
    """

    Sxx = Sw[0,0].real
    Syy = Sw[1,1].real

    Sxy_mod_sq = (Sw[0,1]*Sw[1,0]).real
    Sxy_mod_sq /= Sxx
    Sxy_mod_sq /= Syy
    return Sxy_mod_sq

def interdependence_xy(Sw):
    """Compute the 'total interdependence' between processes X and Y,
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
    return -np.log(1-Cw)

def granger_causality_xy(a, cov, Nfreqs=1024):
    """Compute the Granger causality between processes X and Y, which
    are linked in a multivariate autoregressive (mAR) model parameterized
    by coefficient matrices a(i) and the innovations covariance matrix

    X[t] + sum_{k=1}^P a[k]X[t-k] = Err[t]

    Parameters
    ----------

    a : ndarray, (P,2,2)
      coefficient matrices characterizing the autoregressive mixing
    cov : ndarray, (2,2)
      covariance matrix characterizing the innovations vector
    Nfreqs: int
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

    w, Hw = transfer_function_xy(a, Nfreqs=Nfreqs)

    sigma = cov[0,0]; upsilon = cov[0,1]; gamma = cov[1,1]

    # this transformation of the transfer functions computes the
    # Granger causality of Y on X
    gamma2 = gamma - upsilon**2/sigma

    Hxy = Hw[0,1]
    Hxx_hat = Hw[0,0] + (upsilon/sigma)*Hxy

    xx_auto_component = (sigma*Hxx_hat*Hxx_hat.conj()).real
    cross_component = gamma2*Hxy*Hxy.conj()
    Sxx = xx_auto_component + cross_component
    f_y_on_x = np.log( Sxx.real / xx_auto_component )


    # this transformation computes the Granger causality of X on Y
    sigma2 = sigma - upsilon**2/gamma

    Hyx = Hw[1,0]
    Hyy_hat = Hw[1,1] + (upsilon/gamma)*Hyx
    yy_auto_component = (gamma*Hyy_hat*Hyy_hat.conj()).real
    cross_component = sigma2*Hyx*Hyx.conj()
    Syy = yy_auto_component + cross_component
    f_x_on_y = np.log( Syy.real / yy_auto_component )

    # now compute cross densities, using the latest transformation
    Hxx = Hw[0,0]
    Hyx = Hw[1,0]
    Hxy_hat = Hw[0,1] + (upsilon/gamma)*Hxx
    Sxy = sigma2*Hxx*Hyx.conj() + gamma*Hxy_hat*Hyy_hat.conj()
    Syx = sigma2*Hyx*Hxx.conj() + gamma*Hyy_hat*Hxy_hat.conj()

    # can safely throw away imaginary part
    # since Sxx and Syy are real, and Sxy == Syx*
    detS = (Sxx*Syy - Sxy*Syx).real
    f_xy = xx_auto_component * yy_auto_component
    f_xy /= detS
    f_xy = np.log(f_xy)

    return w, f_x_on_y, f_y_on_x, f_xy, np.array([[Sxx, Sxy], [Syx, Syy]])
