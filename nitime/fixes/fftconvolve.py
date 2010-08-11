from scipy.fftpack import fftn, fft, ifftn, ifft
from scipy.signal.signaltools import _centered
from numpy import array, product
import numpy as np


def fftconvolve(in1, in2, mode="full", axis=None):
    """Convolve two N-dimensional arrays using FFT. See convolve.

    """
    s1 = array(in1.shape)
    s2 = array(in2.shape)
    complex_result = (np.issubdtype(in1.dtype, np.complex) or
                      np.issubdtype(in2.dtype, np.complex))
    if axis is None:
        size = s1+s2-1
        fslice = tuple([slice(0, int(sz)) for sz in size])
    else:
        equal_shapes = s1==s2
        # allow equal_shapes[axis] to be False
        equal_shapes[axis] = True
        assert equal_shapes.all(), 'Shape mismatch on non-convolving axes'
        size = s1[axis]+s2[axis]-1
        fslice = [slice(l) for l in s1]
        fslice[axis] = slice(0, int(size))
        fslice = tuple(fslice)

    # Always use 2**n-sized FFT
    fsize = 2**np.ceil(np.log2(size))
    if axis is None:
        IN1 = fftn(in1,fsize)
        IN1 *= fftn(in2,fsize)
        ret = ifftn(IN1)[fslice].copy()
    else:
        IN1 = fft(in1,fsize,axis=axis)
        IN1 *= fft(in2,fsize,axis=axis)
        ret = ifft(IN1,axis=axis)[fslice].copy()
    del IN1
    if not complex_result:
        ret = ret.real
    if mode == "full":
        return ret
    elif mode == "same":
        if product(s1,axis=0) > product(s2,axis=0):
            osize = s1
        else:
            osize = s2
        return _centered(ret,osize)
    elif mode == "valid":
        return _centered(ret,abs(s2-s1)+1)
