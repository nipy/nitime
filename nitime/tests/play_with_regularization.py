"""Badly commented, badly written sand box for playing with regularization of
coherence calculations. Will eventually serve as basis for tests. For now, don't take it too seriously... 
"""
import numpy as np
from matplotlib.pylab import *
import nipy.timeseries as ts
tsa = ts.algorithms 
pi = np.pi

close('all')

t = np.linspace(0,8*pi,512) 
x = np.sin(t)+ 0.0001*np.random.randn(t.shape[-1])
y = np.sin(t+pi/2)+ 0.0001*np.random.randn(t.shape[-1])


f,fxx,fyy,fxy = tsa.get_spectra(np.hstack([x,y]))

eps = 0.0001
alph= 10000
coh_reg = tsa.coherence_reqularized(fxy,fxx,fyy, eps, alph)
coh = tsa.coherence_calculate(fxy,fxx,fyy)

figure()
plot(coh_reg)
plot(coh)
show()


f1, spec1 = tsa.coherence_phase_spectrum(x,y) 
f2, spec2 = tsa.coherence_phase_spectrum(x,y,{'this_method':'periodogram_csd'}) 
f3, spec3 = tsa.coherence_phase_spectrum(x,y,{'this_method':'multi_taper_csd'}) 

figure()
plot(f1,spec1)
#plot(f2,spec2)
#plot(f3,spec3)
show()

figure()
plot(f,tsa.coherency_phase_delay(f,fxy))
show()
