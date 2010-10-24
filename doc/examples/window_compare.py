import numpy as np
import matplotlib.pyplot as plt
from winspect import winspect #nitime/doc/examples/winspect.py
import scipy.signal as sig

f = plt.figure()
# Window size
npts = 128

# Boxcar with zeroed out fraction
b = sig.boxcar(npts)
zfrac = 0.15
zi = int(npts*zfrac)
b[:zi] = b[-zi:] = 0
name = 'Boxcar - zero fraction=%.2f' % zfrac
winspect(b, f, name)

winspect(sig.hanning(npts), f, 'Hanning')
winspect(sig.bartlett(npts), f, 'Bartlett')
winspect(sig.barthann(npts), f, 'Modified Bartlett-Hann')
