import numpy as np
from numpy.testing import *
from nitime import utils as ut
import nitime.timeseries as ts

def test_CorrelationAnalyzer():

    Fs = np.pi
    t = np.arange(1024)
    x = np.sin(10*t) + np.random.rand(t.shape[-1])
    y = np.sin(10*t) + np.random.rand(t.shape[-1])

    T = ts.UniformTimeSeries(np.vstack([x,y]),sampling_rate=Fs)

    C = ts.CorrelationAnalyzer(T)

    #Test the symmetry: correlation(x,y)==correlation(y,x) 
    np.testing.assert_equal(C.correlation[0,1],C.correlation[1,0])
    #Test the self-sameness: correlation(x,x)==1
    np.testing.assert_equal(C.correlation[0,0],1)
    np.testing.assert_equal(C.correlation[1,1],1)
    #Test the cross-correlation:
    #First the symmetry:
    np.testing.assert_array_almost_equal(C.xcorr.data[0,1],C.xcorr.data[1,0])
    #Test the normalized cross-correlation
    #Should be equal to 1 at 0 time-lag:
    np.testing.assert_equal(C.xcorr_norm.data[0,0,C.xcorr_norm.time==0],1)

    

    
    

    
