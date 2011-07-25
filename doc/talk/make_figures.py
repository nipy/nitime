"""
Coherence/y use-cases
"""
#!/usr/bin/env python

import numpy as np
from matplotlib import pylab
import matplotlib.pyplot as plt
from matplotlib import mlab
import nitime as nipyts
tsa = nipyts.algorithms
tsu = nipyts.utils
ts= nipyts.timeseries
nta = nipyts.analysis
viz = nipyts.viz

reload(tsa)
reload(tsu)
reload(ts)
reload(viz)

pi = np.pi
fft = np.fft

Fs = pi
NFFT = 1024
noverlap = 0

pylab.close('all')

t = np.linspace(0,8*pi,NFFT) 

noise = 0.5
x =  np.sin(5*t) + np.sin(1.33*t) +  noise*np.random.randn(t.shape[-1])
y =  (np.sin(5*t + pi/4) + np.sin(1.33*t-pi/2) +
      noise*np.random.randn(t.shape[-1]))

data = np.vstack([x,y])
series = ts.TimeSeries(data,sampling_rate=Fs)

pylab.figure(1)
pylab.plot(x)
pylab.plot(y)

corrcoef = np.corrcoef(x,y)[0,1]

print(['Correlation is: ' + str(corrcoef)])

XC = nta.CorrelationAnalyzer(series)
xcorr = np.correlate(series.data[0,:],
                     series.data[1,:],'same')

fig = plt.figure(2)
ax = fig.add_subplot(1,1,1)
ax.plot(xcorr/xcorr[NFFT/2]*corrcoef)
ax.annotate('local max', xy=(450, 1),  xycoords='data',
                xytext=(0.8, 0.95), textcoords='axes fraction',
                arrowprops=dict(facecolor='black', shrink=0.05),
                horizontalalignment='right', verticalalignment='top',
                )
ax.set_xlim([0,NFFT])

num_idx = 8 
tick_unit = NFFT/num_idx
ax.set_xticks([tick_unit*(i+1) for i in range(num_idx)],
             [str(tick_unit*(i+1)-NFFT/2) for i in range(num_idx)])


method = {}
method['this_method']='welch'
method['Fs']=Fs
method['NFFT']=NFFT/4
Ch = nta.CoherenceAnalyzer(series,method=method)

pylab.figure(3)
pylab.plot(Ch.frequencies,Ch.spectrum[0,0])
pylab.plot(Ch.frequencies,Ch.spectrum[1,1])

pylab.figure(4)
pylab.plot(Ch.frequencies,Ch.spectrum[0,1])

pylab.figure(5)
pylab.plot(Ch.frequencies,Ch.coherence[0,1])

pylab.figure(6)
pylab.plot(Ch.frequencies,Ch.phase[0,1] * np.abs(Ch.coherence[0,1]**2)) 

Co = nta.CorrelationAnalyzer(series)
    
