"""
Coherence/y use-cases
"""
#!/usr/bin/env python

import numpy as np
from matplotlib.pylab import *
import nitime as nipyts
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

#Various use-cases:

#A periodic function made out of two sin-waves:
x = np.sin(10*t) + np.sin(12*t) + 0.05*np.random.randn(t.shape[-1])
y = np.sin(10*t) + np.sin(12*t + pi/2) + 0.05*np.random.randn(t.shape[-1])

#Noise and a phase shifted copy of the same noise (+ independent noise):
#x = np.random.randn(t.shape[-1])
#y = np.roll(x,25) + 0.1*np.random.randn(t.shape[-1])

data = np.vstack([x,y])

tSeries = ts.UniformTimeSeries(data,sampling_rate=Fs)

figure()

method = {'this_method':'mlab','noverlap':noverlap,'NFFT':64}
C1 = ts.CoherenceAnalyzer(tSeries,method)
plot(C1.frequencies,C1.coherence[0,1])

method = {'this_method':'multi_taper_csd'}
C2 = ts.CoherenceAnalyzer(tSeries,method)
plot(C2.frequencies,C2.coherence[0,1])

figure()
plot(C1.frequencies,C1.phase[0,1])
plot(C2.frequencies,C2.phase[0,1])

figure()
plot(C1.frequencies,C1.delay[0,1])
plot(C2.frequencies,C2.delay[0,1])

