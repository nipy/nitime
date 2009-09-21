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
    #The cross-correlation should be equal to the correlation at time-lag 0
    np.testing.assert_equal(C.xcorr_norm.data[0,1,C.xcorr_norm.time==0]
                            ,C.correlation[0,1])

    #And the auto-correlation should be equal to 1 at 0 time-lag:
    np.testing.assert_equal(C.xcorr_norm.data[0,0,C.xcorr_norm.time==0],1)

    #Does it depend on having an even number of time-points?
    #make another time-series with an odd number of items:
    t = np.arange(1023)
    x = np.sin(10*t) + np.random.rand(t.shape[-1])
    y = np.sin(10*t) + np.random.rand(t.shape[-1])

    T = ts.UniformTimeSeries(np.vstack([x,y]),sampling_rate=Fs)

    C = ts.CorrelationAnalyzer(T)

    
    np.testing.assert_equal(C.xcorr_norm.data[0,1,C.xcorr_norm.time==0]
                            ,C.correlation[0,1])


def test_EventRelatedAnalyzer():

    cycles = 10
    l = 1024
    t = np.linspace(0.,2*np.pi,l)
    signal = np.sin(cycles*t)
    events = np.zeros(t.shape)
    #An event occurs at the beginning of every cycle:
    events[np.arange(0,l-(l/cycles),l/cycles)]=1
    events[np.arange(l/cycles/2,l-(l/cycles),l/cycles)]=2
    T_signal = ts.UniformTimeSeries(signal,sampling_rate=1)
    T_events = ts.UniformTimeSeries(events,sampling_rate=1)
    E = ts.EventRelatedAnalyzer(T_signal,T_events,l/(cycles*2))

