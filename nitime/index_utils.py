""" Utilities for indexing into 2-d arrays, brought in from numpy 1.4, to
support use of older versions of numpy
"""

__all__ = ['tri', 'triu', 'tril', 'mask_indices', 'tril_indices',
           'tril_indices_from', 'triu_indices', 'triu_indices_from',
           ]

from numpy.core.numeric import asanyarray, subtract, arange, \
     greater_equal, multiply, ones, asarray, where

# Need to import numpy for the doctests! 
import numpy as np  

def tri(N, M=None, k=0, dtype=float):
    """
    Construct an array filled with ones at and below the given diagonal.

    Parameters
    ----------
    N : int
        Number of rows in the array.
    M : int, optional
        Number of columns in the array.
        By default, `M` is taken equal to `N`.
    k : int, optional
        The sub-diagonal below which the array is filled.
        `k` = 0 is the main diagonal, while `k` < 0 is below it,
        and `k` > 0 is above.  The default is 0.
    dtype : dtype, optional
        Data type of the returned array.  The default is float.

    Returns
    -------
    T : (N,M) ndarray
        Array with a lower triangle filled with ones, in other words
        ``T[i,j] == 1`` for ``i <= j + k``.

    Examples
    --------
    >>> np.tri(3, 5, 2, dtype=int)
    array([[1, 1, 1, 0, 0],
           [1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1]])

    >>> np.tri(3, 5, -1)
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 1.,  0.,  0.,  0.,  0.],
           [ 1.,  1.,  0.,  0.,  0.]])

    """
    if M is None: M = N
    m = greater_equal(subtract.outer(arange(N), arange(M)),-k)
    return m.astype(dtype)

def tril(m, k=0):
    """
    Lower triangle of an array.

    Return a copy of an array with elements above the `k`-th diagonal zeroed.

    Parameters
    ----------
    m : array_like, shape (M, N)
        Input array.
    k : int
        Diagonal above which to zero elements.
        `k = 0` is the main diagonal, `k < 0` is below it and `k > 0` is above.

    Returns
    -------
    L : ndarray, shape (M, N)
        Lower triangle of `m`, of same shape and data-type as `m`.

    See Also
    --------
    triu

    Examples
    --------
    >>> np.tril([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1)
    array([[ 0,  0,  0],
           [ 4,  0,  0],
           [ 7,  8,  0],
           [10, 11, 12]])

    """
    m = asanyarray(m)
    out = multiply(tri(m.shape[0], m.shape[1], k=k, dtype=int),m)
    return out

def triu(m, k=0):
    """
    Upper triangle of an array.

    Construct a copy of a matrix with elements below the k-th diagonal zeroed.

    Please refer to the documentation for `tril`.

    See Also
    --------
    tril

    Examples
    --------
    >>> np.triu([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1)
    array([[ 1,  2,  3],
           [ 4,  5,  6],
           [ 0,  8,  9],
           [ 0,  0, 12]])

    """
    m = asanyarray(m)
    out = multiply((1-tri(m.shape[0], m.shape[1], k-1, int)),m)
    return out

# borrowed from John Hunter and matplotlib
def vander(x, N=None):
    """
    Generate a Van der Monde matrix.

    The columns of the output matrix are decreasing powers of the input
    vector.  Specifically, the i-th output column is the input vector to
    the power of ``N - i - 1``. Such a matrix with a geometric progression
    in each row is named Van Der Monde, or Vandermonde matrix, from
    Alexandre-Theophile Vandermonde.

    Parameters
    ----------
    x : array_like
        1-D input array.
    N : int, optional
        Order of (number of columns in) the output. If `N` is not specified,
        a square array is returned (``N = len(x)``).

    Returns
    -------
    out : ndarray
        Van der Monde matrix of order `N`.  The first column is ``x^(N-1)``,
        the second ``x^(N-2)`` and so forth.

    References
    ----------
    .. [1] Wikipedia, "Vandermonde matrix",
           http://en.wikipedia.org/wiki/Vandermonde_matrix

    Examples
    --------
    >>> x = np.array([1, 2, 3, 5])
    >>> N = 3
    >>> np.vander(x, N)
    array([[ 1,  1,  1],
           [ 4,  2,  1],
           [ 9,  3,  1],
           [25,  5,  1]])

    >>> np.column_stack([x**(N-1-i) for i in range(N)])
    array([[ 1,  1,  1],
           [ 4,  2,  1],
           [ 9,  3,  1],
           [25,  5,  1]])

    >>> x = np.array([1, 2, 3, 5])
    >>> np.vander(x)
    array([[  1,   1,   1,   1],
           [  8,   4,   2,   1],
           [ 27,   9,   3,   1],
           [125,  25,   5,   1]])

    """
    x = asarray(x)
    if N is None: N=len(x)
    X = ones( (len(x),N), x.dtype)
    for i in range(N-1):
        X[:,i] = x**(N-i-1)
    return X


def histogram2d(x,y, bins=10, range=None, normed=False, weights=None):
    """
    Compute the bi-dimensional histogram of two data samples.

    Parameters
    ----------
    x : array_like, shape(N,)
        A sequence of values to be histogrammed along the first dimension.
    y : array_like, shape(M,)
        A sequence of values to be histogrammed along the second dimension.
    bins : int or [int, int] or array_like or [array, array], optional
        The bin specification:

          * If int, the number of bins for the two dimensions (nx=ny=bins).
          * If [int, int], the number of bins in each dimension (nx, ny = bins).
          * If array_like, the bin edges for the two dimensions (x_edges=y_edges=bins).
          * If [array, array], the bin edges in each dimension (x_edges, y_edges = bins).

    range : array_like, shape(2,2), optional
        The leftmost and rightmost edges of the bins along each dimension
        (if not specified explicitly in the `bins` parameters):
        ``[[xmin, xmax], [ymin, ymax]]``. All values outside of this range
        will be considered outliers and not tallied in the histogram.
    normed : bool, optional
        If False, returns the number of samples in each bin. If True, returns
        the bin density, i.e. the bin count divided by the bin area.
    weights : array_like, shape(N,), optional
        An array of values ``w_i`` weighing each sample ``(x_i, y_i)``. Weights
        are normalized to 1 if `normed` is True. If `normed` is False, the
        values of the returned histogram are equal to the sum of the weights
        belonging to the samples falling into each bin.

    Returns
    -------
    H : ndarray, shape(nx, ny)
        The bi-dimensional histogram of samples `x` and `y`. Values in `x`
        are histogrammed along the first dimension and values in `y` are
        histogrammed along the second dimension.
    xedges : ndarray, shape(nx,)
        The bin edges along the first dimension.
    yedges : ndarray, shape(ny,)
        The bin edges along the second dimension.

    See Also
    --------
    histogram: 1D histogram
    histogramdd: Multidimensional histogram

    Notes
    -----
    When `normed` is True, then the returned histogram is the sample density,
    defined such that:

    .. math::
      \\sum_{i=0}^{nx-1} \\sum_{j=0}^{ny-1} H_{i,j} \\Delta x_i \\Delta y_j = 1

    where `H` is the histogram array and :math:`\\Delta x_i \\Delta y_i`
    the area of bin `{i,j}`.

    Please note that the histogram does not follow the Cartesian convention
    where `x` values are on the abcissa and `y` values on the ordinate axis.
    Rather, `x` is histogrammed along the first dimension of the array
    (vertical), and `y` along the second dimension of the array (horizontal).
    This ensures compatibility with `histogramdd`.

    Examples
    --------
    >>> x, y = np.random.randn(2, 100)
    >>> H, xedges, yedges = np.histogram2d(x, y, bins=(5, 8))
    >>> H.shape, xedges.shape, yedges.shape
    ((5, 8), (6,), (9,))

    """
    from numpy import histogramdd

    try:
        N = len(bins)
    except TypeError:
        N = 1

    if N != 1 and N != 2:
        xedges = yedges = asarray(bins, float)
        bins = [xedges, yedges]
    hist, edges = histogramdd([x,y], bins, range, normed, weights)
    return hist, edges[0], edges[1]


def mask_indices(n,mask_func,k=0):
    """
    Return the indices to access (n, n) arrays, given a masking function.

    Assume `mask_func` is a function that, for a square array a of size
    ``(n, n)`` with a possible offset argument `k`, when called as
    ``mask_func(a, k)`` returns a new array with zeros in certain locations
    (functions like `triu` or `tril` do precisely this). Then this function
    returns the indices where the non-zero values would be located.

    Parameters
    ----------
    n : int
        The returned indices will be valid to access arrays of shape (n, n).
    mask_func : callable
        A function whose call signature is similar to that of `triu`, `tril`.
        That is, ``mask_func(x, k)`` returns a boolean array, shaped like `x`.
        `k` is an optional argument to the function.
    k : scalar
        An optional argument which is passed through to `mask_func`. Functions
        like `triu`, `tril` take a second argument that is interpreted as an
        offset.

    Returns
    -------
    indices : tuple of arrays.
        The `n` arrays of indices corresponding to the locations where
        ``mask_func(np.ones((n, n)), k)`` is True.

    See Also
    --------
    triu, tril, triu_indices, tril_indices

    Notes
    -----
    .. versionadded:: 1.4.0

    Examples
    --------
    These are the indices that would allow you to access the upper triangular
    part of any 3x3 array:

    >>> iu = np.mask_indices(3, np.triu)

    For example, if `a` is a 3x3 array:

    >>> a = np.arange(9).reshape(3, 3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    >>> a[iu]
    array([0, 1, 2, 4, 5, 8])

    An offset can be passed also to the masking function.  This gets us the
    indices starting on the first diagonal right of the main one:

    >>> iu1 = np.mask_indices(3, np.triu, 1)

    with which we now extract only three elements:

    >>> a[iu1]
    array([1, 2, 5])

    """
    m = ones((n,n),int)
    a = mask_func(m,k)
    return where(a != 0)


def tril_indices(n,k=0):
    """
    Return the indices for the lower-triangle of an (n, n) array.

    Parameters
    ----------
    n : int
      Sets the size of the arrays for which the returned indices will be valid.
    k : int, optional
      Diagonal offset (see `tril` for details).

    Returns
    -------
    inds : tuple of arrays
        The indices for the triangle. The returned tuple contains two arrays,
        each with the indices along one dimension of the array.

    See also
    --------
    triu_indices : similar function, for upper-triangular.
    mask_indices : generic function accepting an arbitrary mask function.
    tril, triu

    Notes
    -----
    .. versionadded:: 1.4.0

    Examples
    --------
    Compute two different sets of indices to access 4x4 arrays, one for the
    lower triangular part starting at the main diagonal, and one starting two
    diagonals further right:

    >>> il1 = np.tril_indices(4)
    >>> il2 = np.tril_indices(4, 2)

    Here is how they can be used with a sample array:

    >>> a = np.arange(16).reshape(4, 4)
    >>> a
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])

    Both for indexing:

    >>> a[il1]
    array([ 0,  4,  5,  8,  9, 10, 12, 13, 14, 15])

    And for assigning values:

    >>> a[il1] = -1
    >>> a
    array([[-1,  1,  2,  3],
           [-1, -1,  6,  7],
           [-1, -1, -1, 11],
           [-1, -1, -1, -1]])

    These cover almost the whole array (two diagonals right of the main one):

    >>> a[il2] = -10
    >>> a
    array([[-10, -10, -10,   3],
           [-10, -10, -10, -10],
           [-10, -10, -10, -10],
           [-10, -10, -10, -10]])

    """
    return mask_indices(n,tril,k)


def tril_indices_from(arr,k=0):
    """
    Return the indices for the lower-triangle of an (n, n) array.

    See `tril_indices` for full details.

    Parameters
    ----------
    n : int
      Sets the size of the arrays for which the returned indices will be valid.
    k : int, optional
      Diagonal offset (see `tril` for details).

    See Also
    --------
    tril_indices, tril

    Notes
    -----
    .. versionadded:: 1.4.0

    """
    if not arr.ndim==2 and arr.shape[0] == arr.shape[1]:
        raise ValueError("input array must be 2-d and square")
    return tril_indices(arr.shape[0],k)


def triu_indices(n,k=0):
    """
    Return the indices for the upper-triangle of an (n, n) array.

    Parameters
    ----------
    n : int
      Sets the size of the arrays for which the returned indices will be valid.
    k : int, optional
      Diagonal offset (see `triu` for details).

    Returns
    -------
    inds : tuple of arrays
        The indices for the triangle. The returned tuple contains two arrays,
        each with the indices along one dimension of the array.

    See also
    --------
    tril_indices : similar function, for lower-triangular.
    mask_indices : generic function accepting an arbitrary mask function.
    triu, tril

    Notes
    -----
    .. versionadded:: 1.4.0

    Examples
    --------
    Compute two different sets of indices to access 4x4 arrays, one for the
    upper triangular part starting at the main diagonal, and one starting two
    diagonals further right:

    >>> iu1 = np.triu_indices(4)
    >>> iu2 = np.triu_indices(4, 2)

    Here is how they can be used with a sample array:

    >>> a = np.arange(16).reshape(4, 4)
    >>> a
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])

    Both for indexing:

    >>> a[iu1]
    array([ 0,  1,  2,  3,  5,  6,  7, 10, 11, 15])

    And for assigning values:

    >>> a[iu1] = -1
    >>> a
    array([[-1, -1, -1, -1],
           [ 4, -1, -1, -1],
           [ 8,  9, -1, -1],
           [12, 13, 14, -1]])

    These cover only a small part of the whole array (two diagonals right
    of the main one):

    >>> a[iu2] = -10
    >>> a
    array([[ -1,  -1, -10, -10],
           [  4,  -1,  -1, -10],
           [  8,   9,  -1,  -1],
           [ 12,  13,  14,  -1]])

    """
    return mask_indices(n,triu,k)


def triu_indices_from(arr,k=0):
    """
    Return the indices for the lower-triangle of an (n, n) array.

    See `triu_indices` for full details.

    Parameters
    ----------
    n : int
      Sets the size of the arrays for which the returned indices will be valid.
    k : int, optional
      Diagonal offset (see `triu` for details).

    See Also
    --------
    triu_indices, triu

    Notes
    -----
    .. versionadded:: 1.4.0

    """
    if not arr.ndim==2 and arr.shape[0] == arr.shape[1]:
        raise ValueError("input array must be 2-d and square")
    return triu_indices(arr.shape[0],k)
