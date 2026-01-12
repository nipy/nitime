"""Compatibility utilities for different dependency versions."""

import numpy as np
from packaging.version import Version


def _reshape_view(arr, shape):
    """Reshape an array as a view, raising if a copy would be required.

    This function provides compatibility across NumPy versions for reshaping
    arrays as views. On NumPy >= 2.1, it uses ``reshape(copy=False)`` which
    explicitly fails if a view cannot be created. On older versions, it uses
    direct shape assignment which has the same behavior but is deprecated in
    NumPy 2.5+.

    Parameters
    ----------
    arr : ndarray
        The array to reshape.
    shape : tuple of int
        The new shape.

    Returns
    -------
    ndarray
        A reshaped view of the array.

    Raises
    ------
    AttributeError
        If a view cannot be created on NumPy < 2.1.
    ValueError
        If a view cannot be created on NumPy >= 2.1.
    """
    if Version(np.__version__) >= Version("2.1"):
        return arr.reshape(shape, copy=False)
    else:
        arr.shape = shape
        return arr


# np.trapezoid was introduced and np.trapz deprecated in numpy 2.0
if Version(np.__version__) >= Version("2.0"):
    from numpy import trapezoid
else:
    from numpy import trapz as trapezoid
