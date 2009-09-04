"""Miscellaneous utilities for time series analysis.

XXX wrie top level doc-string

"""
import numpy as np

#-----------------------------------------------------------------------------
# Spectral estimation testing utilities
#-----------------------------------------------------------------------------
def square_window_spectrum(N,Fs):
    r"""
    Calculate the analytical spectrum of a square window

    Parameters
    ----------
    N: int
    the size of the window

    Fs: float
    The sampling rate
    
    Returns
    -------
    float array - the frequency bands, given N and FS
    complex array: the power in the spectrum of the square window in the
    frequency bands 
    
    Notes
    -----
    This is equation 21c in [1]
    
    ..math::

    W(\theta) = exp(-j \frac{N-1}{2} \theta) \frac{\frac{sin \frac{N\theta}{2}}
    {sin\frac{\theta}{2}}}

    ..[1] F.J. Harris (1978). On the use of windows for harmonic analysis with
    the discrete Fourier transform. Proceedings of the IEEE, 66:51-83
"""
    f = get_freqs(Fs,N-1)
    j = 0+1j
    a = -j * (N-1) * f / 2
    b = np.sin(N*f/2.0)
    c = np.sin(f/2.0)
    make = np.exp(a) * b / c

    return f,  make[1:]/make[1]

def hanning_window_spectrum(N, Fs):
    r"""
    Calculate the analytical spectrum of a Hanning window

    Parameters
    ----------
    N: int
    the size of the window

    Fs: float
    The sampling rate
    
    Returns
    -------
    float array - the frequency bands, given N and FS
    complex array: the power in the spectrum of the square window in the
    frequency bands 
    
    Notes
    -----
    This is equation 28b in [1]
    
    ..math::

    W(\theta) = 0.5 D(\theta) + 0.25 (D(\theta - \frac{2\pi}{N}) +
    D(\theta + \frac{2\pi}{N}) ), 

    where:

    D(\theta) = exp(j \frac{\theta}{2}) \frac{\frac{sin\frac{N\theta}{2}}
    {sin\frac{\theta}{2}}}

    ..[1] F.J. Harris (1978). On the use of windows for harmonic analysis with
    the discrete Fourier transform. Proceedings of the IEEE, 66:51-83
    """
    #A helper function
    D = lambda theta, n: (
        np.exp((0+1j) * theta / 2) * ((np.sin(n*theta/2)) / (theta/2)) )

    f = get_freqs(Fs,N)

    make = 0.5*D(f,N) + 0.25*( D((f-(2*np.pi/N)),N) + D((f+(2*np.pi/N)), N) )
    return f, make[1:]/make[1] 


def ar_generator(N=512, sigma=1.):
    # this generates a signal u(n) = a1*u(n-1) + a2*u(n-2) + ... + v(n)
    # where v(n) is a stationary stochastic process with zero mean
    # and variance = sigma
    # this sequence is shown to be estimated well by an order 8 AR system
    taps = np.array([2.7607, -3.8106, 2.6535, -0.9238])
    v = np.random.normal(size=N, scale=sigma**0.5)
    u = np.zeros(N)
    P = len(taps)
    for l in xrange(P):
        u[l] = v[l] + np.dot(u[:l][::-1], taps[:l])
    for l in xrange(P,N):
        u[l] = v[l] + np.dot(u[l-P:l][::-1], taps)
    return u, v, taps

def circularize(x,bottom=0,top=2*np.pi):
    """ Like a modulu operation into the continuous interval (bottom,top) where
    bottom defaults to 0 and top defaults to 2*pi""" 

    if  np.all(x>=bottom) and np.all(x<=top):
        return x
    else:
        x[np.where(x<0)] += top
        
        x[np.where(x>top)] -= top

    return(circularize(x))


#-----------------------------------------------------------------------------
# Correlation utils
#-----------------------------------------------------------------------------

def autocorr(s, axis=-1):
    """Returns the autocorrelation of signal s at all lags. Adheres to the
    definition r(k) = E{s(n)s*(n-k)} where E{} is the expectation operator.
    """
    N = s.shape[axis]
    S = np.fft.fft(s, n=2*N, axis=axis)
    sxx = np.fft.ifft(S*S.conjugate(), axis=axis).real[:N]
    return sxx/N

def xcorr(x,y):
    """Returns the crosscorrelation between two ndarrays, by calling
    np.correlate in 'full' mode (this is what Matlab 'xcorr' does...).
    """
    return np.correlate(x,y,'full')
                     
def norm_corr(x,y,mode = 'valid'):
    """Returns the correlation between to ndarrays, by calling np.correlate in
    'same' mode and normalizing the result by the std of the arrays and by
    their lengths. This results in a correlation = 1 for an auto-correlation"""

    return ( np.correlate(x,y,mode) /
             (np.std(x)*np.std(y)*(x.shape[-1])) )

#-----------------------------------------------------------------------------
# 'get' utils
#-----------------------------------------------------------------------------   


def get_freqs(Fs,n):
    """Returns the center frequencies of the frequency decomposotion of a time
    series of length n, sampled at Fs Hz"""

    return np.linspace(0,Fs/2,n/2+1)

def circle_to_hz(omega, Fsamp):
    """For a frequency grid spaced on the unit circle of an imaginary plane,
    return the corresponding freqency grid in Hz.
    """
    return Fsamp * omega / (2*np.pi)

def get_bounds(f,lb=0,ub=None): 
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
    lb_idx = np.searchsorted(f,lb,'left')   
    if ub==None:
        ub_idx = len(f) 
    else:
        ub_idx = np.searchsorted(f,ub,'right')
        
    return lb_idx, ub_idx

def unwrap_phases(a):
    """
    Changes consecutive jumps larger than pi to their 2*pi complement. 
    """
    
    pi = np.pi

    diffs = np.diff(a) 
    mod_diffs = np.mod(diffs+pi,2*pi) - pi
    neg_pi_idx = np.where(mod_diffs==-1*np.pi)
    pos_idx = np.where(diffs>0)
    this_idx = np.intersect1d(neg_pi_idx[0],pos_idx[0])
    mod_diffs[this_idx] = pi
    correction = mod_diffs - diffs    
    correction[np.where(np.abs(diffs)<pi)] = 0
    a[1:] += np.cumsum(correction)

    return a

def multi_intersect (input):

    """ A function for finding the intersection of several different arrays

    Parameters
    ----------
    input is a tuple of arrays, with all the different arrays 

    Returns
    -------
    array - the intersection of the inputs

    Notes
    -----
    Simply runs intersect1d_nu iteratively on the inputs

    
    """
    
    output = np.intersect1d_nu(input[0], input[1])

    for i in input:

        output = np.intersect1d_nu(output,i)

    return output

def zero_pad(time_series,NFFT):
    """Pad a time-series with zeros on either side, depending on its length"""

    n_channels, n_time_points = time_series.shape
    # zero pad if time_series is too short
    if n_time_points < NFFT:
        tmp = time_series
        time_series = np.zeros( (n_channels,NFFT), time_series.dtype)
        time_series[:,:n_time_points] = tmp
        del tmp
    
    return time_series

#-----------------------------------------------------------------------------
# Numpy utilities - Note: these have been sent into numpy itself, so eventually
# we'll be able to get rid of them here.
#-----------------------------------------------------------------------------
    

def fill_diagonal(a,val):
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
        if not np.alltrue(np.diff(a.shape)==0):
            raise ValueError("All dimensions of input must be of equal length")
        step = np.cumprod((1,)+a.shape[:-1]).sum()

    # Write the value out into the diagonal.
    a.flat[::step] = val


def diag_indices(n,ndim=2):
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
    return (idx,)*ndim


def diag_indices_from(arr):
    """Return the indices to access the main diagonal of an n-dimensional array.

    See diag_indices() for full details.

    Parameters
    ----------
    arr : array, at least 2-d
    """
    if not arr.ndim >= 2:
        raise ValueError("input array must be at least 2-d")
    # For more than d=2, the strided formula is only valid for arrays with
    # all dimensions equal, so we check first.
    if not np.alltrue(np.diff(a.shape)==0):
        raise ValueError("All dimensions of input must be of equal length")

    return diag_indices(a.shape[0],a.ndim)

    
def mask_indices(n,mask_func,k=0):
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
    m = np.ones((n,n),int)
    a = mask_func(m,k)
    return np.where(a != 0)


def tril_indices(n,k=0):
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
    return mask_indices(n,np.tril,k)


def tril_indices_from(arr,k=0):
    """Return the indices for the lower-triangle of an (n,n) array.

    See tril_indices() for full details.
    
    Parameters
    ----------
    n : int
      Sets the size of the arrays for which the returned indices will be valid.

    k : int, optional
      Diagonal offset (see tril() for details).

    """
    if not arr.ndim==2 and arr.shape[0] == arr.shape[1]:
        raise ValueError("input array must be 2-d and square")
    return tril_indices(arr.shape[0],k)

    
def triu_indices(n,k=0):
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
    return mask_indices(n,np.triu,k)


def triu_indices_from(arr,k=0):
    """Return the indices for the lower-triangle of an (n,n) array.

    See triu_indices() for full details.
    
    Parameters
    ----------
    n : int
      Sets the size of the arrays for which the returned indices will be valid.

    k : int, optional
      Diagonal offset (see triu() for details).

    """
    if not arr.ndim==2 and arr.shape[0] == arr.shape[1]:
        raise ValueError("input array must be 2-d and square")
    return triu_indices(arr.shape[0],k)


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
    rmat = sample_func((size,size))
    # And the empty one we'll then fill in to return
    out = np.empty_like(rmat)
    # Extract indices for upper-triangle, lower-triangle and diagonal
    uidx = triu_indices(size,1)
    lidx = tril_indices(size,-1)
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


def symm_rand_arr(size,sample_func=np.random.random,fill_diag=None):
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
    return structured_rand_arr(size,sample_func,fill_diag=fill_diag)


def antisymm_rand_arr(size,sample_func=np.random.random):
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
    return structured_rand_arr(size,sample_func,ltfac=-1.0,fill_diag=0)


#---------- intersect coords ----------------------------------------------------

def intersect_coords(coords1,coords2):

    """For two sets of coordinates, find the coordinates that are common to
    both, where the dimensionality is the coords1.shape[0]"""

    #find the longer one
    if coords1.shape[-1]>coords2.shape[-1]:
        coords_long = coords1
        coords_short = coords2
    else:
        coords_long = coords2
        coords_short = coords1
        
    ans = np.array([[],[],[]],dtype='int') #Initialize as a 3 row variable
    #Loop over the longer of the coordinate sets
    for i in xrange(coords_long.shape[-1]):
        #For each coordinate: 
        this_coords = coords_long[:,i]
        #Find the matches in the other set of coordinates: 
        x = np.where(coords_short[0,:] == this_coords[0])[0]
        y = np.where(coords_short[1,:] == this_coords[1])[0] 
        z = np.where(coords_short[2,:] == this_coords[2])[0]

        #Use intersect1d, such that there can be more than one match (and the
        #size of idx will reflect how many such matches exist):
        
        idx = np.intersect1d(np.intersect1d(x,y),z)
        #append the places where there are matches in all three dimensions:
        if len(idx):
            ans = np.hstack([ans,coords_short[:,idx]])
                        
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
    et = time_series.mean(axis=axis)
    st = time_series.std(axis=axis)
    sl = [slice(None)] * len(t.shape)
    sl[axis] = np.newaxis
    zt = time_series - et[sl]
    zt /= st[sl]
    return zt

def percent_change(time_series, axis=-1):
    """Returns the % signal change of each point of the times series
    along a given axis of the array time_series

    Parameters
    ----------
    time_series : ndarray
        an array of time series
    axis : int, optional
        the axis of time_series along which to compute means and stdevs

    Returns
    -------
    ndarray
        the renormalized time series array (in units of %)
    """
    
    return ((time_series.T/np.mean(time_series,axis) - 1).T)*100 
    

#----------Event-related analysis utils ----------------------------------------

def fir_design_matrix(events,len_hrf):
    """Create a FIR event matrix from a time-series of events.

    Parameters
    ----------

    events: 1-d int array
       Integers denoting different kinds of events, occuring at the time
       corresponding to the bin represented by each slot in the array. In
       time-bins in which no event occured, a 0 should be entered. If negative
       event values are entered, they will be used as "negative" events, as in
       events that should be contrasted with the postitive events (typically -1
       and 1 can be used for a simple contrast of two conditions)

    len_hrf: int
       The expected length of the HRF (in the same time-units as the events are
       represented (presumably TR). The size of the block dedicated in the
       fir_matrix to each type of event
    
    Returns
    -------

    fir_matrix: matrix

       The design matrix for FIR estimation
    
    
    """ 

    event_types = np.unique(events)[np.unique(events)!=0]
    fir_matrix = np.zeros((events.shape[0],len_hrf*event_types.shape[0]))

    for t in event_types:
        idx_h_a = np.where(event_types==t)[0] * len_hrf
        idx_h_b = idx_h_a + len_hrf
        idx_v = np.where(events == t)[0]
        for idx_v_a in idx_v:
            idx_v_b = idx_v_a + len_hrf 
            fir_matrix[idx_v_a:idx_v_b,idx_h_a:idx_h_b]+=(np.eye(len_hrf)
                                                          *np.sign(t))

    return fir_matrix


def dirac_delta(i,j):
    """The dirac delta function:

    .. math::
        \delta_{i,j} = 1 for i=j
                       0 otherwise
    """

    if i==j:
        return 1
    else:
        return 0

#goodness of fit utilities: 
def residual_sum_of_squares(x,x_hat):
    """The sum of the squares of the discrepancies between a variable x and an
    estimate of that variable (x_hat) """

    return np.sum((x-x_hat)**2)
    
def akaike_information_criterion(k,x,x_hat):
    """ A measure of the goodness of fit of a statistical model based on the
    number of parameters, k and the model likelihood, calculated from the
    discrepancy between the variable x and the model estimate of that
    variable.

    Parameters
    ----------

    k: int,
       the number of model parameters

    x: 1d np array
       the true data

    x_hat: 1d np array
        the data as estimated by the model


    Returns
    -------

    AIC: float
        The value of the AIC
        
    Notes
    -----
    See http://en.wikipedia.org/wiki/Akaike_information_criterion
    """
    
    RSS = residual_sum_of_squares(x,x_hat)
    n = x.shape[-1]
    
    AIC = 2*k + float(n)*(np.log( (2*np.pi*RSS) /float(n)) + 1)

    return AIC

def akaike_information_criterion_c(k,x,x_hat):
    """ The Akaike Information Criterion, corrected for small sample size.

    Parameters
    ----------
    
    k: int,
       the number of model parameters

    x: 1d np array
       the true data

    x_hat: 1d np array
        the data as estimated by the model


    Returns
    -------

    AICc: float
        The value of the AIC, corrected for small sample size

    Notes
    -----
    See http://en.wikipedia.org/wiki/Akaike_information_criterion"""

    n = x.shape[-1]
    AIC = akaike_information_criterion(k,x,x_hat)
    AICc = AIC + (2*k*(k+1))/(n-k-1)

    return AICc

def schwarz_criterion(k,x,x_hat):
    """The Schwarz criterion also known as the Bayesian Information Criterion
    is a measure of goodness of fit of a statistical model, based on the number
    of model parameters and the likelihood of the model

    Parameters
    ----------

    k: int,
       the number of model parameters

    x: 1d np array
       the true data

    x_hat: 1d np array
        the data as estimated by the model
    
    Returns
    -------

    BIC: float
       The value of the BIC

    Notes
    -----
    See http://en.wikipedia.org/wiki/Schwarz_criterion
    """ 

    n = x.shape[-1]
    RSS = residual_sum_of_squares(x,x_hat)
    
    BIC = n * np.log(RSS/n) + k*np.log(n)

    return BIC
