"""Miscellaneous utilities for time series analysis.

"""
from __future__ import print_function
import warnings
import numpy as np
import scipy.ndimage as ndimage

from nitime.lazy import scipy_linalg as linalg
from nitime.lazy import scipy_signal as sig
from nitime.lazy import scipy_fftpack as fftpack
from nitime.lazy import scipy_signal_signaltools as signaltools
from nitime.lazy import scipy_stats_distributions as dists
from nitime.lazy import scipy_interpolate as interpolate


#-----------------------------------------------------------------------------
# Spectral estimation testing utilities
#-----------------------------------------------------------------------------
def square_window_spectrum(N, Fs):
    r"""
    Calculate the analytical spectrum of a square window

    Parameters
    ----------
    N : int
       the size of the window

    Fs : float
       The sampling rate

    Returns
    -------
    float array - the frequency bands, given N and FS
    complex array: the power in the spectrum of the square window in the
    frequency bands

    Notes
    -----
    This is equation 21c in Harris (1978):

    .. math::

      W(\theta) = exp(-j \frac{N-1}{2} \theta) \frac{sin \frac{N\theta}{2}} {sin\frac{\theta}{2}}

    F.J. Harris (1978). On the use of windows for harmonic analysis with the
    discrete Fourier transform. Proceedings of the IEEE, 66:51-83
    """
    f = get_freqs(Fs, N - 1)
    j = 0 + 1j
    a = -j * (N - 1) * f / 2
    b = np.sin(N * f / 2.0)
    c = np.sin(f / 2.0)
    make = np.exp(a) * b / c

    return f,  make[1:] / make[1]


def hanning_window_spectrum(N, Fs):
    r"""
    Calculate the analytical spectrum of a Hanning window

    Parameters
    ----------
    N : int
       The size of the window

    Fs : float
       The sampling rate

    Returns
    -------
    float array - the frequency bands, given N and FS
    complex array: the power in the spectrum of the square window in the
    frequency bands

    Notes
    -----
    This is equation 28b in Harris (1978):

    .. math::

      W(\theta) = 0.5 D(\theta) + 0.25 (D(\theta - \frac{2\pi}{N}) +
                D(\theta + \frac{2\pi}{N}) ),

    where:

    .. math::

      D(\theta) = exp(j\frac{\theta}{2})
                  \frac{sin\frac{N\theta}{2}}{sin\frac{\theta}{2}}

    F.J. Harris (1978). On the use of windows for harmonic analysis with the
    discrete Fourier transform. Proceedings of the IEEE, 66:51-83
    """
    #A helper function
    D = lambda theta, n: (
        np.exp((0 + 1j) * theta / 2) * ((np.sin(n * theta / 2)) / (theta / 2)))

    f = get_freqs(Fs, N)

    make = 0.5 * D(f, N) + 0.25 * (D((f - (2 * np.pi / N)), N) +
                                   D((f + (2 * np.pi / N)), N))
    return f, make[1:] / make[1]


def ar_generator(N=512, sigma=1., coefs=None, drop_transients=0, v=None):
    """
    This generates a signal u(n) = a1*u(n-1) + a2*u(n-2) + ... + v(n)
    where v(n) is a stationary stochastic process with zero mean
    and variance = sigma. XXX: confusing variance notation

    Parameters
    ----------

    N : float
       The number of points in the AR process generated. Default: 512
    sigma : float
       The variance of the noise in the AR process. Default: 1
    coefs : list or array of floats
       The AR model coefficients. Default: [2.7607, -3.8106, 2.6535, -0.9238],
       which is a sequence shown to be well-estimated by an order 8 AR system.
    drop_transients : float
       How many samples to drop from the beginning of the sequence (the
       transient phases of the process), so that the process can be considered
       stationary.
    v : float array
       Optionally, input a specific sequence of noise samples (this over-rides
       the sigma parameter). Default: None

    Returns
    -------

    u : ndarray
       the AR sequence
    v : ndarray
       the unit-variance innovations sequence
    coefs : ndarray
       feedback coefficients from k=1,len(coefs)

    The form of the feedback coefficients is a little different than
    the normal linear constant-coefficient difference equation. Therefore
    the transfer function implemented in this method is

    H(z) = sigma**0.5 / ( 1 - sum_k coefs(k)z**(-k) )    1 <= k <= P

    Examples
    --------

    >>> import nitime.algorithms as alg
    >>> ar_seq, nz, alpha = ar_generator()
    >>> fgrid, hz = alg.freq_response(1.0, a=np.r_[1, -alpha])
    >>> sdf_ar = (hz * hz.conj()).real

    """
    if coefs is None:
        # this sequence is shown to be estimated well by an order 8 AR system
        coefs = np.array([2.7607, -3.8106, 2.6535, -0.9238])
    else:
        coefs = np.asarray(coefs)

    # The number of terms we generate must include the dropped transients, and
    # then at the end we cut those out of the returned array.
    N += drop_transients

    # Typically uses just pass sigma in, but optionally they can provide their
    # own noise vector, case in which we use it
    if v is None:
        v = np.random.normal(size=N)
        v -= v[drop_transients:].mean()

    b = [sigma ** 0.5]
    a = np.r_[1, -coefs]
    u = sig.lfilter(b, a, v)

    # Only return the data after the drop_transients terms
    return u[drop_transients:], v[drop_transients:], coefs


def circularize(x, bottom=0, top=2 * np.pi, deg=False):
    """Maps the input into the continuous interval (bottom, top) where
    bottom defaults to 0 and top defaults to 2*pi

    Parameters
    ----------

    x : ndarray - the input array

    bottom : float, optional (defaults to 0).
        If you want to set the bottom of the interval into which you
        modulu to something else than 0.

    top : float, optional (defaults to 2*pi).
        If you want to set the top of the interval into which you
        modulu to something else than 2*pi

    Returns
    -------
    The input array, mapped into the interval (bottom,top)

    """
    x = np.asarray([x])

    if  (np.all(x[np.isfinite(x)] >= bottom) and
         np.all(x[np.isfinite(x)] <= top)):
        return np.squeeze(x)
    else:
        x[np.where(x < 0)] += top
        x[np.where(x > top)] -= top

    return np.squeeze(circularize(x, bottom=bottom, top=top))


def dB(x, power=True):
    """Convert the values in x to decibels.
    If the values in x are in 'power'-like units, then set the power
    flag accordingly

    1) dB(x) = 10log10(x)                     (if power==True)
    2) dB(x) = 10log10(|x|^2) = 20log10(|x|)  (if power==False)
    """
    if not power:
        return 20 * np.log10(np.abs(x))
    return 10 * np.log10(np.abs(x))


#-----------------------------------------------------------------------------
# Stats utils
#-----------------------------------------------------------------------------

def normalize_coherence(x, dof, copy=True):
    """
    The generally accepted choice to transform coherence measures into
    a more normal distribution

    Parameters
    ----------
    x : ndarray, real
       square-root of magnitude-square coherence measures
    dof : int
       number of degrees of freedom in the multitaper model
    copy : bool
        Copy or return inplace modified x.

    Returns
    -------
    y : ndarray, real
        The transformed array.
    """
    if copy:
        x = x.copy()
    np.arctanh(x, x)
    x *= np.sqrt(dof)
    return x


def normal_coherence_to_unit(y, dof, out=None):
    """
    The inverse transform of the above normalization
    """
    if out is None:
        x = y / np.sqrt(dof)
    else:
        y /= np.sqrt(dof)
        x = y
    np.tanh(x, x)
    return x


def expected_jk_variance(K):
    """Compute the expected value of the jackknife variance estimate
    over K windows below. This expected value formula is based on the
    asymptotic expansion of the trigamma function derived in
    [Thompson_1994]

    Parameters
    ---------

    K : int
      Number of tapers used in the multitaper method

    Returns
    -------

    evar : float
      Expected value of the jackknife variance estimator

    """

    kf = float(K)
    return ((1 / kf) * (kf - 1) / (kf - 0.5) *
            ((kf - 1) / (kf - 2)) ** 2 * (kf - 3) / (kf - 2))


def jackknifed_sdf_variance(yk, eigvals, sides='onesided', adaptive=True):
    r"""
    Returns the variance of the log-sdf estimated through jack-knifing
    a group of independent sdf estimates.

    Parameters
    ----------

    yk : ndarray (K, L)
       The K DFTs of the tapered sequences
    eigvals : ndarray (K,)
       The eigenvalues corresponding to the K DPSS tapers
    sides : str, optional
       Compute the jackknife pseudovalues over as one-sided or
       two-sided spectra
    adpative : bool, optional
       Compute the adaptive weighting for each jackknife pseudovalue

    Returns
    -------

    var : The estimate for log-sdf variance

    Notes
    -----

    The jackknifed mean estimate is distributed about the true mean as
    a Student's t-distribution with (K-1) degrees of freedom, and
    standard error equal to sqrt(var). However, Thompson and Chave [1]
    point out that this variance better describes the sample mean.


    [1] Thomson D J, Chave A D (1991) Advances in Spectrum Analysis and Array
    Processing (Prentice-Hall, Englewood Cliffs, NJ), 1, pp 58-113.
    """
    K = yk.shape[0]

    from nitime.algorithms import mtm_cross_spectrum

    # the samples {S_k} are defined, with or without weights, as
    # S_k = | x_k |**2
    # | x_k |**2 = | y_k * d_k |**2          (with adaptive weights)
    # | x_k |**2 = | y_k * sqrt(eig_k) |**2  (without adaptive weights)

    all_orders = set(range(K))
    jk_sdf = []
    # get the leave-one-out estimates -- ideally, weights are recomputed
    # for each leave-one-out. This is now the case.
    for i in range(K):
        items = list(all_orders.difference([i]))
        spectra_i = np.take(yk, items, axis=0)
        eigs_i = np.take(eigvals, items)
        if adaptive:
            # compute the weights
            weights, _ = adaptive_weights(spectra_i, eigs_i, sides=sides)
        else:
            weights = eigs_i[:, None]
        # this is the leave-one-out estimate of the sdf
        jk_sdf.append(
            mtm_cross_spectrum(
                spectra_i, spectra_i, weights, sides=sides
                )
            )
    # log-transform the leave-one-out estimates and the mean of estimates
    jk_sdf = np.log(jk_sdf)
    # jk_avg should be the mean of the log(jk_sdf(i))
    jk_avg = jk_sdf.mean(axis=0)

    K = float(K)

    jk_var = (jk_sdf - jk_avg)
    np.power(jk_var, 2, jk_var)
    jk_var = jk_var.sum(axis=0)

    # Thompson's recommended factor, eq 18
    # Jackknifing Multitaper Spectrum Estimates
    # IEEE SIGNAL PROCESSING MAGAZINE [20] JULY 2007
    f = (K - 1) ** 2 / K / (K - 0.5)
    jk_var *= f
    return jk_var


def jackknifed_coh_variance(tx, ty, eigvals, adaptive=True):
    """
    Returns the variance of the coherency between x and y, estimated
    through jack-knifing the tapered samples in {tx, ty}.

    Parameters
    ----------

    tx : ndarray, (K, L)
       The K complex spectra of tapered timeseries x
    ty : ndarray, (K, L)
       The K complex spectra of tapered timeseries y
    eigvals : ndarray (K,)
       The eigenvalues associated with the K DPSS tapers

    Returns
    -------

    jk_var : ndarray
       The variance computed in the transformed domain (see
       normalize_coherence)
    """

    K = tx.shape[0]

    # calculate leave-one-out estimates of MSC (magnitude squared coherence)
    jk_coh = []
    # coherence is symmetric (right??)
    sides = 'onesided'
    all_orders = set(range(K))

    import nitime.algorithms as alg

    # get the leave-one-out estimates
    for i in range(K):
        items = list(all_orders.difference([i]))
        tx_i = np.take(tx, items, axis=0)
        ty_i = np.take(ty, items, axis=0)
        eigs_i = np.take(eigvals, items)
        if adaptive:
            wx, _ = adaptive_weights(tx_i, eigs_i, sides=sides)
            wy, _ = adaptive_weights(ty_i, eigs_i, sides=sides)
        else:
            wx = wy = eigs_i[:, None]
        # The CSD
        sxy_i = alg.mtm_cross_spectrum(tx_i, ty_i, (wx, wy), sides=sides)
        # The PSDs
        sxx_i = alg.mtm_cross_spectrum(tx_i, tx_i, wx, sides=sides)
        syy_i = alg.mtm_cross_spectrum(ty_i, ty_i, wy, sides=sides)
        # these are the | c_i | samples
        msc = np.abs(sxy_i)
        msc /= np.sqrt(sxx_i * syy_i)
        jk_coh.append(msc)

    jk_coh = np.array(jk_coh)
    # now normalize the coherence estimates and take the mean
    normalize_coherence(jk_coh, 2 * K - 2, copy=False)  # inplace
    jk_avg = np.mean(jk_coh, axis=0)

    jk_var = (jk_coh - jk_avg)
    np.power(jk_var, 2, jk_var)
    jk_var = jk_var.sum(axis=0)

    # Do/Don't use the alternative scaling here??
    f = float(K - 1) / K

    jk_var *= f

    return jk_var


#-----------------------------------------------------------------------------
# Multitaper utils
#-----------------------------------------------------------------------------
def adaptive_weights(yk, eigvals, sides='onesided', max_iter=150):
    r"""
    Perform an iterative procedure to find the optimal weights for K
    direct spectral estimators of DPSS tapered signals.

    Parameters
    ----------

    yk : ndarray (K, N)
       The K DFTs of the tapered sequences
    eigvals : ndarray, length-K
       The eigenvalues of the DPSS tapers
    sides : str
       Whether to compute weights on a one-sided or two-sided spectrum
    max_iter : int
       Maximum number of iterations for weight computation

    Returns
    -------

    weights, nu

       The weights (array like sdfs), and the
       "equivalent degrees of freedom" (array length-L)

    Notes
    -----

    The weights to use for making the multitaper estimate, such that
    :math:`S_{mt} = \sum_{k} |w_k|^2S_k^{mt} / \sum_{k} |w_k|^2`

    If there are less than 3 tapers, then the adaptive weights are not
    found. The square root of the eigenvalues are returned as weights,
    and the degrees of freedom are 2*K

    """
    from nitime.algorithms import mtm_cross_spectrum
    K = len(eigvals)
    if len(eigvals) < 3:
        print("""
        Warning--not adaptively combining the spectral estimators
        due to a low number of tapers.
        """)
        # we'll hope this is a correct length for L
        N = yk.shape[-1]
        L = N // 2 + 1 if sides == 'onesided' else N
        return (np.multiply.outer(np.sqrt(eigvals), np.ones(L)), 2 * K)
    rt_eig = np.sqrt(eigvals)

    # combine the SDFs in the traditional way in order to estimate
    # the variance of the timeseries
    N = yk.shape[1]
    sdf = mtm_cross_spectrum(yk, yk, eigvals[:, None], sides=sides)
    L = sdf.shape[-1]
    var_est = np.sum(sdf, axis=-1) / N
    bband_sup = (1-eigvals)*var_est

    # The process is to iteratively switch solving for the following
    # two expressions:
    # (1) Adaptive Multitaper SDF:
    # S^{mt}(f) = [ sum |d_k(f)|^2 S_k(f) ]/ sum |d_k(f)|^2
    #
    # (2) Weights
    # d_k(f) = [sqrt(lam_k) S^{mt}(f)] / [lam_k S^{mt}(f) + E{B_k(f)}]
    #
    # Where lam_k are the eigenvalues corresponding to the DPSS tapers,
    # and the expected value of the broadband bias function
    # E{B_k(f)} is replaced by its full-band integration
    # (1/2pi) int_{-pi}^{pi} E{B_k(f)} = sig^2(1-lam_k)

    # start with an estimate from incomplete data--the first 2 tapers
    sdf_iter = mtm_cross_spectrum(yk[:2], yk[:2], eigvals[:2, None],
                                  sides=sides)
    err = np.zeros((K, L))
    # for numerical considerations, don't bother doing adaptive
    # weighting after 150 dB down
    min_pwr = sdf_iter.max() * 10 ** (-150/20.)
    default_weights = np.where(sdf_iter < min_pwr)[0]
    adaptiv_weights = np.where(sdf_iter >= min_pwr)[0]

    w_def = rt_eig[:,None] * sdf_iter[default_weights]
    w_def /= eigvals[:, None] * sdf_iter[default_weights] + bband_sup[:,None]

    d_sdfs = np.abs(yk[:,adaptiv_weights])**2
    if L < N:
        d_sdfs *= 2
    sdf_iter = sdf_iter[adaptiv_weights]
    yk = yk[:,adaptiv_weights]
    for n in range(max_iter):
        d_k = rt_eig[:,None] * sdf_iter[None, :]
        d_k /= eigvals[:, None]*sdf_iter[None, :] + bband_sup[:,None]
        # Test for convergence -- this is overly conservative, since
        # iteration only stops when all frequencies have converged.
        # A better approach is to iterate separately for each freq, but
        # that is a nonvectorized algorithm.
        #sdf_iter = mtm_cross_spectrum(yk, yk, d_k, sides=sides)
        sdf_iter = np.sum( d_k**2 * d_sdfs, axis=0 )
        sdf_iter /= np.sum( d_k**2, axis=0 )
        # Compute the cost function from eq 5.4 in Thomson 1982
        cfn = eigvals[:,None] * (sdf_iter[None,:] - d_sdfs)
        cfn /= (eigvals[:,None] * sdf_iter[None,:] + bband_sup[:,None])**2
        cfn = np.sum(cfn, axis=0)
        # there seem to be some pathological freqs sometimes ..
        # this should be a good heuristic
        if np.percentile(cfn**2, 95) < 1e-12:
            break
    else:  # If you have reached maximum number of iterations
        # Issue a warning and return non-converged weights:
        e_s = 'Breaking due to iterative meltdown in '
        e_s += 'nitime.utils.adaptive_weights.'
        warnings.warn(e_s, RuntimeWarning)
    weights = np.zeros( (K,L) )
    weights[:,adaptiv_weights] = d_k
    weights[:,default_weights] = w_def
    nu = 2 * (weights ** 2).sum(axis=-2)
    return weights, nu


def dpss_windows(N, NW, Kmax, interp_from=None, interp_kind='linear'):
    """
    Returns the Discrete Prolate Spheroidal Sequences of orders [0,Kmax-1]
    for a given frequency-spacing multiple NW and sequence length N.

    Parameters
    ----------
    N : int
        sequence length
    NW : float, unitless
        standardized half bandwidth corresponding to 2NW = BW/f0 = BW*N*dt
        but with dt taken as 1
    Kmax : int
        number of DPSS windows to return is Kmax (orders 0 through Kmax-1)
    interp_from : int (optional)
        The dpss can be calculated using interpolation from a set of dpss
        with the same NW and Kmax, but shorter N. This is the length of this
        shorter set of dpss windows.
    interp_kind : str (optional)
        This input variable is passed to scipy.interpolate.interp1d and
        specifies the kind of interpolation as a string ('linear', 'nearest',
        'zero', 'slinear', 'quadratic, 'cubic') or as an integer specifying the
        order of the spline interpolator to use.


    Returns
    -------
    v, e : tuple,
        v is an array of DPSS windows shaped (Kmax, N)
        e are the eigenvalues

    Notes
    -----
    Tridiagonal form of DPSS calculation from:

    Slepian, D. Prolate spheroidal wave functions, Fourier analysis, and
    uncertainty V: The discrete case. Bell System Technical Journal,
    Volume 57 (1978), 1371430
    """
    Kmax = int(Kmax)
    W = float(NW) / N
    nidx = np.arange(N, dtype='d')

    # In this case, we create the dpss windows of the smaller size
    # (interp_from) and then interpolate to the larger size (N)
    if interp_from is not None:
        if interp_from > N:
            e_s = 'In dpss_windows, interp_from is: %s ' % interp_from
            e_s += 'and N is: %s. ' % N
            e_s += 'Please enter interp_from smaller than N.'
            raise ValueError(e_s)
        dpss = []
        d, e = dpss_windows(interp_from, NW, Kmax)
        for this_d in d:
            x = np.arange(this_d.shape[-1])
            I = interpolate.interp1d(x, this_d, kind=interp_kind)
            d_temp = I(np.linspace(0, this_d.shape[-1] - 1, N, endpoint=False))

            # Rescale:
            d_temp = d_temp / np.sqrt(np.sum(d_temp ** 2))

            dpss.append(d_temp)

        dpss = np.array(dpss)

    else:
        # here we want to set up an optimization problem to find a sequence
        # whose energy is maximally concentrated within band [-W,W].
        # Thus, the measure lambda(T,W) is the ratio between the energy within
        # that band, and the total energy. This leads to the eigen-system
        # (A - (l1)I)v = 0, where the eigenvector corresponding to the largest
        # eigenvalue is the sequence with maximally concentrated energy. The
        # collection of eigenvectors of this system are called Slepian
        # sequences, or discrete prolate spheroidal sequences (DPSS). Only the
        # first K, K = 2NW/dt orders of DPSS will exhibit good spectral
        # concentration
        # [see http://en.wikipedia.org/wiki/Spectral_concentration_problem]

        # Here I set up an alternative symmetric tri-diagonal eigenvalue
        # problem such that
        # (B - (l2)I)v = 0, and v are our DPSS (but eigenvalues l2 != l1)
        # the main diagonal = ([N-1-2*t]/2)**2 cos(2PIW), t=[0,1,2,...,N-1]
        # and the first off-diagonal = t(N-t)/2, t=[1,2,...,N-1]
        # [see Percival and Walden, 1993]
        diagonal = ((N - 1 - 2 * nidx) / 2.) ** 2 * np.cos(2 * np.pi * W)
        off_diag = np.zeros_like(nidx)
        off_diag[:-1] = nidx[1:] * (N - nidx[1:]) / 2.
        # put the diagonals in LAPACK "packed" storage
        ab = np.zeros((2, N), 'd')
        ab[1] = diagonal
        ab[0, 1:] = off_diag[:-1]
        # only calculate the highest Kmax eigenvalues
        w = linalg.eigvals_banded(ab, select='i',
                                  select_range=(N - Kmax, N - 1))
        w = w[::-1]

        # find the corresponding eigenvectors via inverse iteration
        t = np.linspace(0, np.pi, N)
        dpss = np.zeros((Kmax, N), 'd')
        for k in range(Kmax):
            dpss[k] = tridi_inverse_iteration(
                diagonal, off_diag, w[k], x0=np.sin((k + 1) * t)
                )

    # By convention (Percival and Walden, 1993 pg 379)
    # * symmetric tapers (k=0,2,4,...) should have a positive average.
    # * antisymmetric tapers should begin with a positive lobe
    fix_symmetric = (dpss[0::2].sum(axis=1) < 0)
    for i, f in enumerate(fix_symmetric):
        if f:
            dpss[2 * i] *= -1
    # rather than test the sign of one point, test the sign of the
    # linear slope up to the first (largest) peak
    pk = np.argmax(np.abs(dpss[1::2, :N//2]), axis=1)
    for i, p in enumerate(pk):
        if np.sum(dpss[2 * i + 1, :p]) < 0:
            dpss[2 * i + 1] *= -1

    # Now find the eigenvalues of the original spectral concentration problem
    # Use the autocorr sequence technique from Percival and Walden, 1993 pg 390
    dpss_rxx = autocorr(dpss) * N
    r = 4 * W * np.sinc(2 * W * nidx)
    r[0] = 2 * W
    eigvals = np.dot(dpss_rxx, r)

    return dpss, eigvals


def tapered_spectra(s, tapers, NFFT=None, low_bias=True):
    """
    Compute the tapered spectra of the rows of s.

    Parameters
    ----------

    s : ndarray, (n_arr, n_pts)
        An array whose rows are timeseries.

    tapers : ndarray or container
        Either the precomputed DPSS tapers, or the pair of parameters
        (NW, K) needed to compute K tapers of length n_pts.

    NFFT : int
        Number of FFT bins to compute

    low_bias : Boolean
        If compute DPSS, automatically select tapers corresponding to
        > 90% energy concentration.

    Returns
    -------

    t_spectra : ndarray, shaped (n_arr, K, NFFT)
      The FFT of the tapered sequences in s. First dimension is squeezed
      out if n_arr is 1.
    eigvals : ndarray
      The eigenvalues are also returned if DPSS are calculated here.

    """
    N = s.shape[-1]
    # XXX: don't allow NFFT < N -- not every implementation is so restrictive!
    if NFFT is None or NFFT < N:
        NFFT = N
    rest_of_dims = s.shape[:-1]
    M = int(np.product(rest_of_dims))

    s = s.reshape(int(np.product(rest_of_dims)), N)
    # de-mean this sucker
    s = remove_bias(s, axis=-1)

    if not isinstance(tapers, np.ndarray):
        # then tapers is (NW, K)
        args = (N,) + tuple(tapers)
        dpss, eigvals = dpss_windows(*args)
        if low_bias:
            keepers = (eigvals > 0.9)
            dpss = dpss[keepers]
            eigvals = eigvals[keepers]
        tapers = dpss
    else:
        eigvals = None
    K = tapers.shape[0]
    sig_sl = [slice(None)] * len(s.shape)
    sig_sl.insert(len(s.shape) - 1, np.newaxis)

    # tapered.shape is (M, Kmax, N)
    tapered = s[tuple(sig_sl)] * tapers

    # compute the y_{i,k}(f) -- full FFT takes ~1.5x longer, but unpacking
    # results of real-valued FFT eats up memory
    t_spectra = fftpack.fft(tapered, n=NFFT, axis=-1)
    t_spectra.shape = rest_of_dims + (K, NFFT)
    if eigvals is None:
        return t_spectra
    return t_spectra, eigvals



def detect_lines(s, tapers, p=None, **taper_kws):
    """
    Detect the presence of line spectra in s using the F-test
    described in "Spectrum estimation and harmonic analysis" (Thompson 81).
    Strategies for detecting harmonics in low SNR include increasing the
    number of FFT points (NFFT keyword arg) and/or increasing the stability
    of the spectral estimate by using more tapers (higher NW parameter).

    s : ndarray
        The sequence(s) to test. If s.ndim > 1, then test sequences in
        the last axis in parallel

    tapers : ndarray or container
        Either the precomputed DPSS tapers, or the pair of parameters
        (NW, K) needed to compute K tapers of length n_pts.

    p : float
        The confidence threshold: under the null hypothesis of
        a locally white spectrum, there is a threshold such that
        there is a (1-p)% chance of a line amplitude being larger
        than that threshold. Only detect lines with amplitude greater
        than this threshold. The default is 1/NFFT, to control for false
        positives.

    taper_kws
        Options for the tapered_spectra method, if no DPSS are provided.

    Returns
    -------

    (freq, beta) : sequence
        The frequencies (normalized in [0, .5]) and coefficients of the
        complex exponentials detected in the spectrum. A pair is returned
        for each sequence tested.

        One can reconstruct the line components as such:

        sn = 2*(beta[:,None]*np.exp(i*2*np.pi*np.arange(N)*freq[:,None])).real
        sn = sn.sum(axis=0)

    """
    N = s.shape[-1]
    # Some boiler-plate --
    # 1) set up tapers
    # 2) perform FFT on all windowed series
    if not isinstance(tapers, np.ndarray):
        # then tapers is (NW, K)
        args = (N,) + tuple(tapers)
        dpss, eigvals = dpss_windows(*args)
        if taper_kws.pop('low_bias', False):
            keepers = (eigvals > 0.9)
            dpss = dpss[keepers]
        tapers = dpss
    # spectra is (n_arr, K, nfft)
    spectra = tapered_spectra(s, tapers, **taper_kws)
    nfft = spectra.shape[-1]
    spectra = spectra[..., :nfft//2 + 1]

    # Set up some data for the following calculations --
    #   get the DC component of the taper spectra
    K = tapers.shape[0]
    U0 = tapers.sum(axis=1)
    U_sq = np.sum(U0**2)
    #  first order linear regression for mu to explain spectra
    mu = np.sum( U0[:,None] * spectra, axis=-2 ) / U_sq

    # numerator of F-stat -- strength of regression
    numr = 0.5 * np.abs(mu)**2 * U_sq
    numr[...,0] = 1; # don't care about DC
    # denominator -- strength of residual
    spectra = np.rollaxis(spectra, -2, 0)
    U0.shape = (K,) + (1,) * (spectra.ndim-1)
    denomr = spectra - U0*mu
    denomr = np.sum(np.abs(denomr)**2, axis=0) / (2*K-2)
    denomr[...,0] = 1;
    f_stat = numr / denomr

    # look for lines in each F-spectrum
    if not p:
        # the number of simultaneous tests are nfft/2, so this puts
        # the expected value for false detection somewhere less than 1
        p = 1.0/nfft
    #thresh = dists.f.isf(p, 2, 2*K-2)
    thresh = dists.f.isf(p, 2, K-1)
    f_stat = np.atleast_2d(f_stat)
    mu = np.atleast_2d(mu)
    lines = ()
    for fs, m in zip(f_stat, mu):
        detected = np.where(fs > thresh)[0]
        # do a quick pass through the detected lines to reject multiple
        # hits within the 2NW resolution of the MT analysis -- approximate
        # 2NW by K
        ddiff = np.diff(detected)
        flagged_groups, last_group = ndimage.label( (ddiff < K) )
        for g in range(1,last_group+1):
            idx = np.where(flagged_groups==g)[0]
            idx = np.r_[idx, idx[-1]+1]
            # keep the super-threshold point with largest amplitude
            mx = np.argmax(np.abs(m[ detected[idx] ]))
            i_sv = detected[idx[mx]]
            detected[idx] = -1
            detected[idx[mx]] = i_sv
        detected = detected[detected>0]
        if len(detected):
            lines = lines + ( (detected/float(nfft), m[detected]), )
        else:
            lines = lines + ( (), )
    if len(lines) == 1:
        lines = lines[0]
    return lines


#-----------------------------------------------------------------------------
# Eigensystem utils
#-----------------------------------------------------------------------------

# If we can get it, we want the cythonized version
try:
    from _utils import tridisolve

# If that doesn't work, we define it here:
except ImportError:
    def tridisolve(d, e, b, overwrite_b=True):
        """
        Symmetric tridiagonal system solver,
        from Golub and Van Loan, Matrix Computations pg 157

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
        N = len(b)
        # work vectors
        dw = d.copy()
        ew = e.copy()
        if overwrite_b:
            x = b
        else:
            x = b.copy()
        for k in range(1, N):
            # e^(k-1) = e(k-1) / d(k-1)
            # d(k) = d(k) - e^(k-1)e(k-1) / d(k-1)
            t = ew[k - 1]
            ew[k - 1] = t / dw[k - 1]
            dw[k] = dw[k] - t * ew[k - 1]
        for k in range(1, N):
            x[k] = x[k] - ew[k - 1] * x[k - 1]
        x[N - 1] = x[N - 1] / dw[N - 1]
        for k in range(N - 2, -1, -1):
            x[k] = x[k] / dw[k] - ew[k] * x[k + 1]

        if not overwrite_b:
            return x


def tridi_inverse_iteration(d, e, w, x0=None, rtol=1e-8):
    """Perform an inverse iteration to find the eigenvector corresponding
    to the given eigenvalue in a symmetric tridiagonal system.

    Parameters
    ----------

    d : ndarray
      main diagonal of the tridiagonal system
    e : ndarray
      offdiagonal stored in e[:-1]
    w : float
      eigenvalue of the eigenvector
    x0 : ndarray
      initial point to start the iteration
    rtol : float
      tolerance for the norm of the difference of iterates

    Returns
    -------

    e : ndarray
      The converged eigenvector

    """
    eig_diag = d - w
    if x0 is None:
        x0 = np.random.randn(len(d))
    x_prev = np.zeros_like(x0)
    norm_x = np.linalg.norm(x0)
    # the eigenvector is unique up to sign change, so iterate
    # until || |x^(n)| - |x^(n-1)| ||^2 < rtol
    x0 /= norm_x
    while np.linalg.norm(np.abs(x0) - np.abs(x_prev)) > rtol:
        x_prev = x0.copy()
        tridisolve(eig_diag, e, x0)
        norm_x = np.linalg.norm(x0)
        x0 /= norm_x
    return x0

#-----------------------------------------------------------------------------
# Correlation/Covariance utils
#-----------------------------------------------------------------------------


def remove_bias(x, axis):
    "Subtracts an estimate of the mean from signal x at axis"
    padded_slice = [slice(d) for d in x.shape]
    padded_slice[axis] = np.newaxis
    mn = np.mean(x, axis=axis)
    return x - mn[tuple(padded_slice)]


def crosscov(x, y, axis=-1, all_lags=False, debias=True, normalize=True):
    r"""Returns the crosscovariance sequence between two ndarrays.
    This is performed by calling fftconvolve on x, y[::-1]

    Parameters
    ----------

    x : ndarray
    y : ndarray
    axis : time axis
    all_lags : {True/False}
       whether to return all nonzero lags, or to clip the length of s_xy
       to be the length of x and y. If False, then the zero lag covariance
       is at index 0. Otherwise, it is found at (len(x) + len(y) - 1)/2
    debias : {True/False}
       Always removes an estimate of the mean along the axis, unless
       told not to (eg X and Y are known zero-mean)

    Returns
    -------

    cxy : ndarray
       The crosscovariance function

    Notes
    -----

    cross covariance of processes x and y is defined as

    .. math::

    C_{xy}[k]=E\{(X(n+k)-E\{X\})(Y(n)-E\{Y\})^{*}\}

    where X and Y are discrete, stationary (or ergodic) random processes

    Also note that this routine is the workhorse for all auto/cross/cov/corr
    functions.
    """

    if x.shape[axis] != y.shape[axis]:
        raise ValueError(
            'crosscov() only works on same-length sequences for now'
            )
    if debias:
        x = remove_bias(x, axis)
        y = remove_bias(y, axis)
    slicing = [slice(d) for d in x.shape]
    slicing[axis] = slice(None, None, -1)
    cxy = fftconvolve(x, y[tuple(slicing)].conj(), axis=axis, mode='full')
    N = x.shape[axis]
    if normalize:
        cxy /= N
    if all_lags:
        return cxy
    slicing[axis] = slice(N - 1, 2 * N - 1)
    return cxy[tuple(slicing)]


def crosscorr(x, y, **kwargs):
    r"""
    Returns the crosscorrelation sequence between two ndarrays.
    This is performed by calling fftconvolve on x, y[::-1]

    Parameters
    ----------

    x : ndarray
    y : ndarray
    axis : time axis
    all_lags : {True/False}
       whether to return all nonzero lags, or to clip the length of r_xy
       to be the length of x and y. If False, then the zero lag correlation
       is at index 0. Otherwise, it is found at (len(x) + len(y) - 1)/2

    Returns
    -------

    rxy : ndarray
       The crosscorrelation function

    Notes
    -----

    cross correlation is defined as

    .. math::

    R_{xy}[k]=E\{X[n+k]Y^{*}[n]\}

    where X and Y are discrete, stationary (ergodic) random processes
    """
    # just make the same computation as the crosscovariance,
    # but without subtracting the mean
    kwargs['debias'] = False
    rxy = crosscov(x, y, **kwargs)
    return rxy


def autocov(x, **kwargs):
    r"""Returns the autocovariance of signal s at all lags.

    Parameters
    ----------

    x : ndarray
    axis : time axis
    all_lags : {True/False}
       whether to return all nonzero lags, or to clip the length of r_xy
       to be the length of x and y. If False, then the zero lag correlation
       is at index 0. Otherwise, it is found at (len(x) + len(y) - 1)/2

    Returns
    -------

    cxx : ndarray
       The autocovariance function

    Notes
    -----

    Adheres to the definition

    .. math::

    C_{xx}[k]=E\{(X[n+k]-E\{X\})(X[n]-E\{X\})^{*}\}

    where X is a discrete, stationary (ergodic) random process
    """
    # only remove the mean once, if needed
    debias = kwargs.pop('debias', True)
    axis = kwargs.get('axis', -1)
    if debias:
        x = remove_bias(x, axis)
    kwargs['debias'] = False
    return crosscov(x, x, **kwargs)


def autocorr(x, **kwargs):
    r"""Returns the autocorrelation of signal s at all lags.

    Parameters
    ----------

    x : ndarray
    axis : time axis
    all_lags : {True/False}
       whether to return all nonzero lags, or to clip the length of r_xy
       to be the length of x and y. If False, then the zero lag correlation
       is at index 0. Otherwise, it is found at (len(x) + len(y) - 1)/2

    Notes
    -----

    Adheres to the definition

    .. math::

    R_{xx}[k]=E\{X[n+k]X^{*}[n]\}

    where X is a discrete, stationary (ergodic) random process

    """
    # do same computation as autocovariance,
    # but without subtracting the mean
    kwargs['debias'] = False
    return autocov(x, **kwargs)


def fftconvolve(in1, in2, mode="full", axis=None):
    """ Convolve two N-dimensional arrays using FFT. See convolve.

    This is a fix of scipy.signal.fftconvolve, adding an axis argument.
    """
    s1 = np.array(in1.shape)
    s2 = np.array(in2.shape)
    complex_result = (np.issubdtype(in1.dtype, np.complex128) or
                      np.issubdtype(in2.dtype, np.complex128))

    if axis is None:
        size = s1 + s2 - 1
        fslice = tuple([slice(0, int(sz)) for sz in size])
    else:
        equal_shapes = s1 == s2
        # allow equal_shapes[axis] to be False
        equal_shapes[axis] = True
        assert equal_shapes.all(), 'Shape mismatch on non-convolving axes'
        size = s1[axis] + s2[axis] - 1
        fslice = [slice(l) for l in s1]
        fslice[axis] = slice(0, int(size))
        fslice = tuple(fslice)

    # Always use 2**n-sized FFT
    fsize = 2 ** int(np.ceil(np.log2(size)))
    if axis is None:
        IN1 = fftpack.fftn(in1, fsize)
        IN1 *= fftpack.fftn(in2, fsize)
        ret = fftpack.ifftn(IN1)[fslice].copy()
    else:
        IN1 = fftpack.fft(in1, fsize, axis=axis)
        IN1 *= fftpack.fft(in2, fsize, axis=axis)
        ret = fftpack.ifft(IN1, axis=axis)[fslice].copy()
    del IN1
    if not complex_result:
        ret = ret.real
    if mode == "full":
        return ret
    elif mode == "same":
        if np.product(s1, axis=0) > np.product(s2, axis=0):
            osize = s1
        else:
            osize = s2
        return signaltools._centered(ret, osize)
    elif mode == "valid":
        return signaltools._centered(ret, abs(s2 - s1) + 1)


#-----------------------------------------------------------------------------
# 'get' utils
#-----------------------------------------------------------------------------
def get_freqs(Fs, n):
    """Returns the center frequencies of the frequency decomposition of a time
    series of length n, sampled at Fs Hz"""

    return np.linspace(0, Fs / 2, int(n / 2 + 1))


def circle_to_hz(omega, Fsamp):
    """For a frequency grid spaced on the unit circle of an imaginary plane,
    return the corresponding frequency grid in Hz.
    """
    return Fsamp * omega / (2 * np.pi)


def get_bounds(f, lb=0, ub=None):
    """ Find the indices of the lower and upper bounds within an array f

    Parameters
    ----------
    f, array

    lb,ub, float

    Returns
    -------

    lb_idx, ub_idx: the indices into 'f' which correspond to values bounded
    between ub and lb in that array
    """
    lb_idx = np.searchsorted(f, lb, 'left')
    if ub == None:
        ub_idx = len(f)
    else:
        ub_idx = np.searchsorted(f, ub, 'right')

    return lb_idx, ub_idx


def unwrap_phases(a):
    """
    Changes consecutive jumps larger than pi to their 2*pi complement.
    """
    pi = np.pi

    diffs = np.diff(a)
    mod_diffs = np.mod(diffs + pi, 2 * pi) - pi
    neg_pi_idx = np.where(mod_diffs == -1 * np.pi)
    pos_idx = np.where(diffs > 0)
    this_idx = np.intersect1d(neg_pi_idx[0], pos_idx[0])
    mod_diffs[this_idx] = pi
    correction = mod_diffs - diffs
    correction[np.where(np.abs(diffs) < pi)] = 0
    a[1:] += np.cumsum(correction)

    return a


def multi_intersect(input):
    """ A function for finding the intersection of several different arrays

    Parameters
    ----------
    input is a tuple of arrays, with all the different arrays

    Returns
    -------
    array - the intersection of the inputs

    Notes
    -----
    Simply runs intersect1d iteratively on the inputs
    """
    arr  = input[0].ravel()
    for this in input[1:]:
        arr = np.intersect1d(arr, this.ravel())

    return arr

def zero_pad(time_series, NFFT):
    """
    Pad a time-series with zeros on either side, depending on its length

    Parameters
    ----------
    time_series : n-d array
       Time-series data with time as the last dimension

    NFFT : int
       The length to pad the data up to.

    """

    n_dims = len(time_series.shape)
    n_time_points = time_series.shape[-1]

    if n_dims>1:
        n_channels = time_series.shape[:-1]
        shape_out = n_channels + (NFFT,)
    else:
        shape_out = NFFT
    # zero pad if time_series is too short
    if n_time_points < NFFT:
        tmp = time_series
        time_series = np.zeros(shape_out, time_series.dtype)
        time_series[..., :n_time_points] = tmp
        del tmp

    return time_series


#-----------------------------------------------------------------------------
# Numpy utilities - Note: these have been sent into numpy itself, so eventually
# we'll be able to get rid of them here.
#-----------------------------------------------------------------------------
def fill_diagonal(a, val):
    """Fill the main diagonal of the given array of any dimensionality.

    For an array with ndim > 2, the diagonal is the list of locations with
    indices a[i,i,...,i], all identical.

    This function modifies the input array in-place, it does not return a
    value.

    This functionality can be obtained via diag_indices(), but internally this
    version uses a much faster implementation that never constructs the indices
    and uses simple slicing.

    Parameters
    ----------
    a : array, at least 2-dimensional.
      Array whose diagonal is to be filled, it gets modified in-place.

    val : scalar
      Value to be written on the diagonal, its type must be compatible with
      that of the array a.

    Examples
    --------
    >>> a = np.zeros((3,3),int)
    >>> fill_diagonal(a,5)
    >>> a
    array([[5, 0, 0],
           [0, 5, 0],
           [0, 0, 5]])

    The same function can operate on a 4-d array:
    >>> a = np.zeros((3,3,3,3),int)
    >>> fill_diagonal(a,4)

    We only show a few blocks for clarity:
    >>> a[0,0]
    array([[4, 0, 0],
           [0, 0, 0],
           [0, 0, 0]])
    >>> a[1,1]
    array([[0, 0, 0],
           [0, 4, 0],
           [0, 0, 0]])
    >>> a[2,2]
    array([[0, 0, 0],
           [0, 0, 0],
           [0, 0, 4]])

    See also
    --------
    - diag_indices: indices to access diagonals given shape information.
    - diag_indices_from: indices to access diagonals given an array.
    """
    if a.ndim < 2:
        raise ValueError("array must be at least 2-d")
    if a.ndim == 2:
        # Explicit, fast formula for the common case.  For 2-d arrays, we
        # accept rectangular ones.
        step = a.shape[1] + 1
    else:
        # For more than d=2, the strided formula is only valid for arrays with
        # all dimensions equal, so we check first.
        if not np.alltrue(np.diff(a.shape) == 0):
            raise ValueError("All dimensions of input must be of equal length")
        step = np.cumprod((1,) + a.shape[:-1]).sum()

    # Write the value out into the diagonal.
    a.flat[::step] = val


def diag_indices(n, ndim=2):
    """Return the indices to access the main diagonal of an array.

    This returns a tuple of indices that can be used to access the main
    diagonal of an array with ndim (>=2) dimensions and shape (n,n,...,n).  For
    ndim=2 this is the usual diagonal, for ndim>2 this is the set of indices
    to access A[i,i,...,i] for i=[0..n-1].

    Parameters
    ----------
    n : int
      The size, along each dimension, of the arrays for which the returned
      indices can be used.

    ndim : int, optional
      The number of dimensions

    Examples
    --------
    Create a set of indices to access the diagonal of a (4,4) array:
    >>> di = diag_indices(4)

    >>> a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
    >>> a
    array([[ 1,  2,  3,  4],
           [ 5,  6,  7,  8],
           [ 9, 10, 11, 12],
           [13, 14, 15, 16]])
    >>> a[di] = 100
    >>> a
    array([[100,   2,   3,   4],
           [  5, 100,   7,   8],
           [  9,  10, 100,  12],
           [ 13,  14,  15, 100]])

    Now, we create indices to manipulate a 3-d array:
    >>> d3 = diag_indices(2,3)

    And use it to set the diagonal of a zeros array to 1:
    >>> a = np.zeros((2,2,2),int)
    >>> a[d3] = 1
    >>> a
    array([[[1, 0],
            [0, 0]],
    <BLANKLINE>
           [[0, 0],
            [0, 1]]])

    See also
    --------
    - diag_indices_from: create the indices based on the shape of an existing
    array.
    """
    idx = np.arange(n)
    return (idx,) * ndim


def diag_indices_from(arr):
    """Return the indices to access the main diagonal of an n-dimensional
    array.

    See diag_indices() for full details.

    Parameters
    ----------
    arr : array, at least 2-d
    """
    if not arr.ndim >= 2:
        raise ValueError("input array must be at least 2-d")
    # For more than d=2, the strided formula is only valid for arrays with
    # all dimensions equal, so we check first.
    if not np.alltrue(np.diff(arr.shape) == 0):
        raise ValueError("All dimensions of input must be of equal length")

    return diag_indices(arr.shape[0], arr.ndim)


def mask_indices(n, mask_func, k=0):
    """Return the indices to access (n,n) arrays, given a masking function.

    Assume mask_func() is a function that, for a square array a of size (n,n)
    with a possible offset argument k, when called as mask_func(a,k) returns a
    new array with zeros in certain locations (functions like triu() or tril()
    do precisely this).  Then this function returns the indices where the
    non-zero values would be located.

    Parameters
    ----------
    n : int
      The returned indices will be valid to access arrays of shape (n,n).

    mask_func : callable
      A function whose api is similar to that of numpy.tri{u,l}.  That is,
      mask_func(x,k) returns a boolean array, shaped like x.  k is an optional
      argument to the function.

    k : scalar
      An optional argument which is passed through to mask_func().  Functions
      like tri{u,l} take a second argument that is interpreted as an offset.

    Returns
    -------
    indices : an n-tuple of index arrays.
      The indices corresponding to the locations where mask_func(ones((n,n)),k)
      is True.

    Examples
    --------
    These are the indices that would allow you to access the upper triangular
    part of any 3x3 array:
    >>> iu = mask_indices(3,np.triu)

    For example, if `a` is a 3x3 array:
    >>> a = np.arange(9).reshape(3,3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])

    Then:
    >>> a[iu]
    array([0, 1, 2, 4, 5, 8])

    An offset can be passed also to the masking function.  This gets us the
    indices starting on the first diagonal right of the main one:
    >>> iu1 = mask_indices(3,np.triu,1)

    with which we now extract only three elements:
    >>> a[iu1]
    array([1, 2, 5])
    """
    m = np.ones((n, n), int)
    a = mask_func(m, k)
    return np.where(a != 0)


def tril_indices(n, k=0):
    """Return the indices for the lower-triangle of an (n,n) array.

    Parameters
    ----------
    n : int
      Sets the size of the arrays for which the returned indices will be valid.

    k : int, optional
      Diagonal offset (see tril() for details).

    Examples
    --------
    Commpute two different sets of indices to access 4x4 arrays, one for the
    lower triangular part starting at the main diagonal, and one starting two
    diagonals further right:

    >>> il1 = tril_indices(4)
    >>> il2 = tril_indices(4,2)

    Here is how they can be used with a sample array:
    >>> a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
    >>> a
    array([[ 1,  2,  3,  4],
           [ 5,  6,  7,  8],
           [ 9, 10, 11, 12],
           [13, 14, 15, 16]])

    Both for indexing:
    >>> a[il1]
    array([ 1,  5,  6,  9, 10, 11, 13, 14, 15, 16])

    And for assigning values:
    >>> a[il1] = -1
    >>> a
    array([[-1,  2,  3,  4],
           [-1, -1,  7,  8],
           [-1, -1, -1, 12],
           [-1, -1, -1, -1]])

    These cover almost the whole array (two diagonals right of the main one):
    >>> a[il2] = -10
    >>> a
    array([[-10, -10, -10,   4],
           [-10, -10, -10, -10],
           [-10, -10, -10, -10],
           [-10, -10, -10, -10]])

    See also
    --------
    - triu_indices : similar function, for upper-triangular.
    - mask_indices : generic function accepting an arbitrary mask function.
    """
    return mask_indices(n, np.tril, k)


def tril_indices_from(arr, k=0):
    """Return the indices for the lower-triangle of an (n,n) array.

    See tril_indices() for full details.

    Parameters
    ----------
    n : int
      Sets the size of the arrays for which the returned indices will be valid.

    k : int, optional
      Diagonal offset (see tril() for details).

    """
    if not arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        raise ValueError("input array must be 2-d and square")
    return tril_indices(arr.shape[0], k)


def triu_indices(n, k=0):
    """Return the indices for the upper-triangle of an (n,n) array.

    Parameters
    ----------
    n : int
      Sets the size of the arrays for which the returned indices will be valid.

    k : int, optional
      Diagonal offset (see triu() for details).

    Examples
    --------
    Commpute two different sets of indices to access 4x4 arrays, one for the
    upper triangular part starting at the main diagonal, and one starting two
    diagonals further right:

    >>> iu1 = triu_indices(4)
    >>> iu2 = triu_indices(4,2)

    Here is how they can be used with a sample array:
    >>> a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
    >>> a
    array([[ 1,  2,  3,  4],
           [ 5,  6,  7,  8],
           [ 9, 10, 11, 12],
           [13, 14, 15, 16]])

    Both for indexing:
    >>> a[iu1]
    array([ 1,  2,  3,  4,  6,  7,  8, 11, 12, 16])

    And for assigning values:
    >>> a[iu1] = -1
    >>> a
    array([[-1, -1, -1, -1],
           [ 5, -1, -1, -1],
           [ 9, 10, -1, -1],
           [13, 14, 15, -1]])

    These cover almost the whole array (two diagonals right of the main one):
    >>> a[iu2] = -10
    >>> a
    array([[ -1,  -1, -10, -10],
           [  5,  -1,  -1, -10],
           [  9,  10,  -1,  -1],
           [ 13,  14,  15,  -1]])

    See also
    --------
    - tril_indices : similar function, for lower-triangular.
    - mask_indices : generic function accepting an arbitrary mask function.
    """
    return mask_indices(n, np.triu, k)


def triu_indices_from(arr, k=0):
    """Return the indices for the lower-triangle of an (n,n) array.

    See triu_indices() for full details.

    Parameters
    ----------
    n : int
      Sets the size of the arrays for which the returned indices will be valid.

    k : int, optional
      Diagonal offset (see triu() for details).

    """
    if not arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        raise ValueError("input array must be 2-d and square")
    return triu_indices(arr.shape[0], k)


def structured_rand_arr(size, sample_func=np.random.random,
                        ltfac=None, utfac=None, fill_diag=None):
    """Make a structured random 2-d array of shape (size,size).

    If no optional arguments are given, a symmetric array is returned.

    Parameters
    ----------
    size : int
      Determines the shape of the output array: (size,size).

    sample_func : function, optional.
      Must be a function which when called with a 2-tuple of ints, returns a
      2-d array of that shape.  By default, np.random.random is used, but any
      other sampling function can be used as long as matches this API.

    utfac : float, optional
      Multiplicative factor for the upper triangular part of the matrix.

    ltfac : float, optional
      Multiplicative factor for the lower triangular part of the matrix.

    fill_diag : float, optional
      If given, use this value to fill in the diagonal.  Otherwise the diagonal
      will contain random elements.

    Examples
    --------
    >>> np.random.seed(0)  # for doctesting
    >>> np.set_printoptions(precision=4)  # for doctesting
    >>> structured_rand_arr(4)
    array([[ 0.5488,  0.7152,  0.6028,  0.5449],
           [ 0.7152,  0.6459,  0.4376,  0.8918],
           [ 0.6028,  0.4376,  0.7917,  0.5289],
           [ 0.5449,  0.8918,  0.5289,  0.0871]])
    >>> structured_rand_arr(4,ltfac=-10,utfac=10,fill_diag=0.5)
    array([[ 0.5   ,  8.3262,  7.7816,  8.7001],
           [-8.3262,  0.5   ,  4.6148,  7.8053],
           [-7.7816, -4.6148,  0.5   ,  9.4467],
           [-8.7001, -7.8053, -9.4467,  0.5   ]])
    """
    # Make a random array from the given sampling function
    rmat = sample_func((size, size))
    # And the empty one we'll then fill in to return
    out = np.empty_like(rmat)
    # Extract indices for upper-triangle, lower-triangle and diagonal
    uidx = triu_indices(size, 1)
    lidx = tril_indices(size, -1)
    didx = diag_indices(size)
    # Extract each part from the original and copy it to the output, possibly
    # applying multiplicative factors.  We check the factors instead of
    # defaulting to 1.0 to avoid unnecessary floating point multiplications
    # which could be noticeable for very large sizes.
    if utfac:
        out[uidx] = utfac * rmat[uidx]
    else:
        out[uidx] = rmat[uidx]
    if ltfac:
        out[lidx] = ltfac * rmat.T[lidx]
    else:
        out[lidx] = rmat.T[lidx]
    # If fill_diag was provided, use it; otherwise take the values in the
    # diagonal from the original random array.
    if fill_diag is not None:
        out[didx] = fill_diag
    else:
        out[didx] = rmat[didx]

    return out


def symm_rand_arr(size, sample_func=np.random.random, fill_diag=None):
    """Make a symmetric random 2-d array of shape (size,size).

    Parameters
    ----------
    n : int
      Size of the output array.

    sample_func : function, optional.
      Must be a function which when called with a 2-tuple of ints, returns a
      2-d array of that shape.  By default, np.random.random is used, but any
      other sampling function can be used as long as matches this API.

    fill_diag : float, optional
      If given, use this value to fill in the diagonal.  Useful for

    Examples
    --------
    >>> np.random.seed(0)  # for doctesting
    >>> np.set_printoptions(precision=4)  # for doctesting
    >>> symm_rand_arr(4)
    array([[ 0.5488,  0.7152,  0.6028,  0.5449],
           [ 0.7152,  0.6459,  0.4376,  0.8918],
           [ 0.6028,  0.4376,  0.7917,  0.5289],
           [ 0.5449,  0.8918,  0.5289,  0.0871]])
    >>> symm_rand_arr(4,fill_diag=4)
    array([[ 4.    ,  0.8326,  0.7782,  0.87  ],
           [ 0.8326,  4.    ,  0.4615,  0.7805],
           [ 0.7782,  0.4615,  4.    ,  0.9447],
           [ 0.87  ,  0.7805,  0.9447,  4.    ]])
      """
    return structured_rand_arr(size, sample_func, fill_diag=fill_diag)


def antisymm_rand_arr(size, sample_func=np.random.random):
    """Make an anti-symmetric random 2-d array of shape (size,size).

    Parameters
    ----------

    n : int
      Size of the output array.

    sample_func : function, optional.
      Must be a function which when called with a 2-tuple of ints, returns a
      2-d array of that shape.  By default, np.random.random is used, but any
      other sampling function can be used as long as matches this API.

    Examples
    --------
    >>> np.random.seed(0)  # for doctesting
    >>> np.set_printoptions(precision=4)  # for doctesting
    >>> antisymm_rand_arr(4)
    array([[ 0.    ,  0.7152,  0.6028,  0.5449],
           [-0.7152,  0.    ,  0.4376,  0.8918],
           [-0.6028, -0.4376,  0.    ,  0.5289],
           [-0.5449, -0.8918, -0.5289,  0.    ]])
      """
    return structured_rand_arr(size, sample_func, ltfac=-1.0, fill_diag=0)


# --------brainx utils------------------------------------------------------
# These utils were copied over from brainx - needed for viz


def threshold_arr(cmat, threshold=0.0, threshold2=None):
    """Threshold values from the input array.

    Parameters
    ----------
    cmat : array

    threshold : float, optional.
      First threshold.

    threshold2 : float, optional.
      Second threshold.

    Returns
    -------
    indices, values: a tuple with ndim+1

    Examples
    --------
    >>> np.set_printoptions(precision=4)  # For doctesting
    >>> a = np.linspace(0,0.2,5)
    >>> a
    array([ 0.  ,  0.05,  0.1 ,  0.15,  0.2 ])
    >>> threshold_arr(a,0.1)
    (array([3, 4]), array([ 0.15,  0.2 ]))

    With two thresholds:
    >>> threshold_arr(a,0.1,0.2)
    (array([0, 1]), array([ 0.  ,  0.05]))
    """
    # Select thresholds
    if threshold2 is None:
        th_low = -np.inf
        th_hi = threshold
    else:
        th_low = threshold
        th_hi = threshold2

    # Mask out the values we are actually going to use
    idx = np.where((cmat < th_low) | (cmat > th_hi))
    vals = cmat[idx]

    return idx + (vals,)


def thresholded_arr(arr, threshold=0.0, threshold2=None, fill_val=np.nan):
    """Threshold values from the input matrix and return a new matrix.

    Parameters
    ----------
    arr : array

    threshold : float
      First threshold.

    threshold2 : float, optional.
      Second threshold.

    Returns
    -------
    An array shaped like the input, with the values outside the threshold
    replaced with fill_val.

    Examples
    --------
    """
    a2 = np.empty_like(arr)
    a2.fill(fill_val)
    mth = threshold_arr(arr, threshold, threshold2)
    idx, vals = mth[:-1], mth[-1]
    a2[idx] = vals

    return a2


def rescale_arr(arr, amin, amax):
    """Rescale an array to a new range.

    Return a new array whose range of values is (amin,amax).

    Parameters
    ----------
    arr : array-like

    amin : float
      new minimum value

    amax : float
      new maximum value

    Examples
    --------
    >>> a = np.arange(5)

    >>> rescale_arr(a,3,6)
    array([ 3.  ,  3.75,  4.5 ,  5.25,  6.  ])
    """

    # old bounds
    m = arr.min()
    M = arr.max()
    # scale/offset
    s = float(amax - amin) / (M - m)
    d = amin - s * m

    # Apply clip before returning to cut off possible overflows outside the
    # intended range due to roundoff error, so that we can absolutely guarantee
    # that on output, there are no values > amax or < amin.
    return np.clip(s * arr + d, amin, amax)


def minmax_norm(arr, mode='direct', folding_edges=None):
    """Minmax_norm an array to [0,1] range.

    By default, this simply rescales the input array to [0,1].  But it has a
    special 'folding' mode that allows for the normalization of an array with
    negative and positive values by mapping the negative values to their
    flipped sign

    Parameters
    ----------
    arr : 1d array

    mode : string, one of ['direct','folding']

    folding_edges : (float,float)
      Only needed for folding mode, ignored in 'direct' mode.

    Examples
    --------
    >>> np.set_printoptions(precision=4)  # for doctesting
    >>> a = np.linspace(0.3,0.8,4)
    >>> minmax_norm(a)
    array([ 0.    ,  0.3333,  0.6667,  1.    ])
    >>> b = np.concatenate([np.linspace(-0.7,-0.3,3),
    ...                             np.linspace(0.3,0.8,3)])
    >>> b
    array([-0.7 , -0.5 , -0.3 ,  0.3 ,  0.55,  0.8 ])
    >>> minmax_norm(b,'folding',[-0.3,0.3])
    array([ 0.8,  0.4,  0. ,  0. ,  0.5,  1. ])
    """
    if mode == 'direct':
        return rescale_arr(arr, 0, 1)
    else:
        fa, fb = folding_edges
        amin, amax = arr.min(), arr.max()
        ra, rb = float(fa - amin), float(amax - fb)  # in case inputs are ints
        if ra < 0 or rb < 0:
            raise ValueError("folding edges must be within array range")
        greater = arr >= fb
        upper_idx = greater.nonzero()
        lower_idx = (~greater).nonzero()
        # Two folding scenarios, we map the thresholds to zero but the upper
        # ranges must retain comparability.
        if ra > rb:
            lower = 1.0 - rescale_arr(arr[lower_idx], 0, 1.0)
            upper = rescale_arr(arr[upper_idx], 0, float(rb) / ra)
        else:
            upper = rescale_arr(arr[upper_idx], 0, 1)
            # The lower range is trickier: we need to rescale it and then flip
            # it, so the edge goes to 0.
            resc_a = float(ra) / rb
            lower = rescale_arr(arr[lower_idx], 0, resc_a)
            lower = resc_a - lower
        # Now, make output array
        out = np.empty_like(arr)
        out[lower_idx] = lower
        out[upper_idx] = upper
        return out


#---------- intersect coords ----------------------------------------------
def intersect_coords(coords1, coords2):
    """For two sets of coordinates, find the coordinates that are common to
    both, where the dimensionality is the coords1.shape[0]"""
    # Find the longer one
    if coords1.shape[-1] > coords2.shape[-1]:
        coords_long = coords1
        coords_short = coords2
    else:
        coords_long = coords2
        coords_short = coords1

    ans = np.array([[], [], []], dtype='int')  # Initialize as a 3 row variable
    # Loop over the longer of the coordinate sets
    for i in range(coords_long.shape[-1]):
        # For each coordinate:
        this_coords = coords_long[:, i]
        # Find the matches in the other set of coordinates:
        x = np.where(coords_short[0, :] == this_coords[0])[0]
        y = np.where(coords_short[1, :] == this_coords[1])[0]
        z = np.where(coords_short[2, :] == this_coords[2])[0]

        # Use intersect1d, such that there can be more than one match (and the
        # size of idx will reflect how many such matches exist):
        idx = np.intersect1d(np.intersect1d(x, y), z)
        # Append the places where there are matches in all three dimensions:
        if len(idx):
            ans = np.hstack([ans, coords_short[:, idx]])

    return ans


#---------- Time Series Stats ----------------------------------------
def zscore(time_series, axis=-1):
    """Returns the z-score of each point of the time series
    along a given axis of the array time_series.

    Parameters
    ----------
    time_series : ndarray
        an array of time series
    axis : int, optional
        the axis of time_series along which to compute means and stdevs

    Returns
    _______
    zt : ndarray
        the renormalized time series array
    """
    time_series = np.asarray(time_series)
    et = time_series.mean(axis=axis)
    st = time_series.std(axis=axis)
    sl = [slice(None)] * len(time_series.shape)
    sl[axis] = np.newaxis
    zt = time_series - et[tuple(sl)]
    zt /= st[tuple(sl)]
    return zt


def percent_change(ts, ax=-1):
    """Returns the % signal change of each point of the times series
    along a given axis of the array time_series

    Parameters
    ----------

    ts : ndarray
        an array of time series

    ax : int, optional (default to -1)
        the axis of time_series along which to compute means and stdevs

    Returns
    -------

    ndarray
        the renormalized time series array (in units of %)

    Examples
    --------

    >>> ts = np.arange(4*5).reshape(4,5)
    >>> ax = 0
    >>> percent_change(ts,ax)
    array([[-100.    ,  -88.2353,  -78.9474,  -71.4286,  -65.2174],
           [ -33.3333,  -29.4118,  -26.3158,  -23.8095,  -21.7391],
           [  33.3333,   29.4118,   26.3158,   23.8095,   21.7391],
           [ 100.    ,   88.2353,   78.9474,   71.4286,   65.2174]])
    >>> ax = 1
    >>> percent_change(ts,ax)
    array([[-100.    ,  -50.    ,    0.    ,   50.    ,  100.    ],
           [ -28.5714,  -14.2857,    0.    ,   14.2857,   28.5714],
           [ -16.6667,   -8.3333,    0.    ,    8.3333,   16.6667],
           [ -11.7647,   -5.8824,    0.    ,    5.8824,   11.7647]])
"""
    ts = np.asarray(ts)

    return (ts / np.expand_dims(np.mean(ts, ax), ax) - 1) * 100


#----------Event-related analysis utils ----------------------------------
def fir_design_matrix(events, len_hrf):
    """Create a FIR event matrix from a time-series of events.

    Parameters
    ----------

    events : 1-d int array
       Integers denoting different kinds of events, occurring at the time
       corresponding to the bin represented by each slot in the array. In
       time-bins in which no event occurred, a 0 should be entered. If negative
       event values are entered, they will be used as "negative" events, as in
       events that should be contrasted with the postitive events (typically -1
       and 1 can be used for a simple contrast of two conditions)

    len_hrf : int
       The expected length of the HRF (in the same time-units as the events are
       represented (presumably TR). The size of the block dedicated in the
       fir_matrix to each type of event

    Returns
    -------

    fir_matrix : matrix

       The design matrix for FIR estimation
    """
    event_types = np.unique(events)[np.unique(events) != 0]
    fir_matrix = np.zeros((events.shape[0], len_hrf * event_types.shape[0]))

    for t in event_types:
        idx_h_a = (np.array(np.where(event_types == t)[0]) * len_hrf)[0]
        idx_h_b = idx_h_a + len_hrf
        idx_v = np.where(events == t)[0]
        for idx_v_a in idx_v:
            idx_v_b = idx_v_a + len_hrf
            fir_matrix[idx_v_a:idx_v_b, idx_h_a:idx_h_b] += (np.eye(len_hrf) *
                                                             np.sign(t))

    return fir_matrix


#---------- MAR utilities ----------------------------------------

# These utilities are used in the computation of multivariate autoregressive
# models (used in computing Granger causality):

def crosscov_vector(x, y, nlags=None):
    r"""
    This method computes the following function

    .. math::

        R_{xy}(k) = E{ x(t)y^{*}(t-k) } = E{ x(t+k)y^{*}(t) }
        k \in {0, 1, ..., nlags-1}

    (* := conjugate transpose)

    Note: This is related to the other commonly used definition
    for vector crosscovariance

    .. math::

        R_{xy}^{(2)}(k) = E{ x(t-k)y^{*}(t) } = R_{xy}^(-k) = R_{yx}^{*}(k)

    Parameters
    ----------

    x, y : ndarray (nc, N)

    nlags : int, optional
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

    for k in range(nlags):
        # rxy(k) = E{ x(t)y*(t-k) }
        prod = x[:, None, k:] * y[None, :, :N - k].conj()
##         # rxy(k) = E{ x(t)y*(t+k) }
##         prod = x[:,None,:N-k] * y[None,:,k:].conj()
        # Do a sample mean of N-k pts? or sum and divide by N?
        rxy[..., k] = prod.mean(axis=-1)
    return rxy


def autocov_vector(x, nlags=None):
    r"""
    This method computes the following function

    .. math::

    R_{xx}(k) = E{ x(t)x^{*}(t-k) } = E{ x(t+k)x^{*}(t) }
    k \in {0, 1, ..., nlags-1}

    (* := conjugate transpose)

    Note: this is related to
    the other commonly used definition for vector autocovariance

    .. math::

    R_{xx}^{(2)}(k) = E{ x(t-k)x^{*}(t) } = R_{xx}(-k) = R_{xx}^{*}(k)

    Parameters
    ----------

    x : ndarray (nc, N)

    nlags : int, optional
       compute lags for k in {0, ..., nlags-1}

    Returns
    -------

    rxx : ndarray (nc, nc, nlags)

    """
    return crosscov_vector(x, x, nlags=nlags)


def generate_mar(a, cov, N):
    """
    Generates a multivariate autoregressive dataset given the formula:

    X(t) + sum_{i=1}^{P} a(i)X(t-i) = E(t)

    Where E(t) is a vector of samples from possibly covarying noise processes.

    Parameters
    ----------

    a : ndarray (n_order, n_c, n_c)
       An order n_order set of coefficient matrices, each shaped (n_c, n_c) for
       n_channel data
    cov : ndarray (n_c, n_c)
       The innovations process covariance
    N : int
       how many samples to generate

    Returns
    -------

    mar, nz

    mar and noise process shaped (n_c, N)
    """
    n_c = cov.shape[0]
    n_order = a.shape[0]

    nz = np.random.multivariate_normal(
        np.zeros(n_c), cov, size=(N,)
        )

    # nz is a (N x n_seq) array

    mar = nz.copy()  # np.zeros((N, n_seq), 'd')

    # this looks like a redundant loop that can be rolled into a matrix-matrix
    # multiplication at each coef matrix a(i)

    # this rearranges the equation to read:
    # X(i) = E(i) - sum_{j=1}^{P} a(j)X(i-j)
    # where X(n) n < 0 is taken to be 0
    # In terms of the code: X is mar and E is nz, P is n_order
    for i in range(N):
        for j in range(min(i, n_order)):  # j logically in set {1, 2, ..., P}
            mar[i, :] -= np.dot(a[j], mar[i - j - 1, :])

    return mar.transpose(), nz.transpose()


#----------goodness of fit utilities ----------------------------------------

def akaike_information_criterion(ecov, p, m, Ntotal, corrected=False):

    """

    A measure of the goodness of fit of an auto-regressive model based on the
    model order and the error covariance.

    Parameters
    ----------

    ecov : float array
        The error covariance of the system
    p
        the number of channels
    m : int
        the model order
    Ntotal
        the number of total time-points (across channels)
    corrected : boolean (optional)
        Whether to correct for small sample size

    Returns
    -------

    AIC : float
        The value of the AIC


    Notes
    -----
    This is an implementation of equation (50) in Ding et al. (2006):

    M Ding and Y Chen and S Bressler (2006) Granger Causality: Basic Theory and
    Application to Neuroscience. http://arxiv.org/abs/q-bio/0608035v1


    Correction for small sample size is taken from:
    http://en.wikipedia.org/wiki/Akaike_information_criterion.

    """

    AIC = (2 * (np.log(linalg.det(ecov))) +
           ((2 * (p ** 2) * m) / (Ntotal)))

    if corrected:
        return AIC + (2 * m * (m + 1)) / (Ntotal - m - 1)
    else:
        return AIC


def bayesian_information_criterion(ecov, p, m, Ntotal):
    r"""The Bayesian Information Criterion, also known as the Schwarz criterion
     is a measure of goodness of fit of a statistical model, based on the
     number of model parameters and the likelihood of the model

    Parameters
    ----------
    ecov : float array
        The error covariance of the system

    p : int
        the system size (how many variables).

    m : int
        the model order.

    corrected : boolean (optional)
        Whether to correct for small sample size


    Returns
    -------

    BIC : float
        The value of the BIC
    a
        the resulting autocovariance vector

    Notes
    -----
    This is an implementation of equation (51) in Ding et al. (2006):

    .. math ::

    BIC(m) = 2 log(|\Sigma|) + \frac{2p^2 m log(N_{total})}{N_{total}},

    where $\Sigma$ is the noise covariance matrix. In auto-regressive model
    estimation, this matrix will contain in $\Sigma_{i,j}$ the residual
    variance in estimating time-series $i$ from $j$, $p$ is the dimensionality
    of the data, $m$ is the number of parameters in the model and $N_{total}$
    is the number of time-points.

    M Ding and Y Chen and S Bressler (2006) Granger Causality: Basic Theory and
    Application to Neuroscience. http://arxiv.org/abs/q-bio/0608035v1


    See http://en.wikipedia.org/wiki/Schwarz_criterion

    """

    BIC = (2 * (np.log(linalg.det(ecov))) +
            ((2 * (p ** 2) * m * np.log(Ntotal)) / (Ntotal)))

    return BIC
