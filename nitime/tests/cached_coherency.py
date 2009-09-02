"""
Coherence/y use-cases
"""
#!/usr/bin/env python

import numpy as np
from matplotlib.pylab import *
import nipy.timeseries as nipyts
tsa = nipyts.algorithms
tsu = nipyts.utils
ts= nipyts.timeseries

reload(tsa)
reload(tsu)
reload(ts)

pi = np.pi
fft = np.fft

Fs = 1
NFFT = 1024.0
noverlap = 0

close('all')

t = np.linspace(0,8*pi,NFFT) 

#A periodic function made out of two sin-waves:
x = np.sin(10*t) + np.sin(12*t) + 0.05*np.random.randn(t.shape[-1])
y = np.sin(10*t) + np.sin(12*t + pi/2) + 0.05*np.random.randn(t.shape[-1])

data = np.vstack([x,y])

f1, spec1 = tsa.get_spectra(data)

ij = [(0,0),(0,1),(1,0),(1,1)]

f2, cache = tsa.cache_fft(data,ij)

spec2 = tsa.cache_to_psd(cache,ij)

f,C1 = tsa.coherency(data)
C2 = tsa.cache_to_coherency(cache,ij)


