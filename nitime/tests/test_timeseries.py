import numpy as np
import numpy.testing as npt
from nitime import utils as ut
import nitime.timeseries as ts
import nose.tools as nt
import decotest

@decotest.parametric
def test_EventArray():

    time1 = ts.EventArray(range(100),time_unit='ms')
    time2 = time1+time1
    yield npt.assert_equal(time2.time_unit,'ms')

@decotest.ipdoctest    
def test_EventArray_repr():
    """
>>> a = ts.EventArray([1.1,2,3])
>>> a
EventArray([ 1.1,  2. ,  3. ], time_unit='s')
>>> t = ts.EventArray(a,time_unit='ms')
>>> t
EventArray([ 1100.,  2000.,  3000.], time_unit='ms')
>>> t[0]
1100.0 ms

    """

@decotest.parametric
def test_EventArray_new():
    for unit in ['ns','ms','s',None]:
        for flag,assertion in [(True,nt.assert_not_equal),
                (False, nt.assert_equal)]:
            #default parameters (timeunits, copy flag, etc)
            #list
            time1 = ts.EventArray(range(5),time_unit=unit, copy=flag)
            #numpy array (int)
            time2 = ts.EventArray(np.arange(5), time_unit=unit, copy=flag)
            #numpy array (float)
            time2f = ts.EventArray(np.arange(5.), time_unit=unit, copy=flag)
            #EventArray
            time3 = ts.EventArray(time1, time_unit=unit, copy=flag)
            #integer
            time4 = ts.EventArray(5,time_unit=unit,copy=flag)
            #float
            time5 = ts.EventArray(5.0,time_unit=unit,copy=flag)

            yield npt.assert_equal(time1,time2)
            yield npt.assert_equal(time2,time2f)
            yield npt.assert_equal(time1,time3)
            time3[0] +=100
            yield assertion(time1[0],time3[0])
            yield npt.assert_equal(time1[1:],time3[1:])
            yield npt.assert_equal(time4,time5)

@decotest.parametric
def test_EventArray_bool():
    time1 = ts.EventArray([1,2,3],time_unit='s')
    time2 = ts.EventArray([1000,2000,3000],time_unit='ms')
    bool_arr = np.ones(time1.shape,dtype=bool)
    yield npt.assert_equal(time1,time2)
    yield npt.assert_equal(bool_arr,time1==time2)
    yield nt.assert_not_equal(type(time1==time2),ts.EventArray)

    
@decotest.parametric
def test_EventArray_index_at():

    #Is this really the behavior we want?
    time1 = ts.EventArray(range(10),time_unit='ms')
    for i in xrange(10):
        idx = time1.index_at([i])
        yield npt.assert_equal(idx,np.array(i))

#XXX Need to write these tests:

#Test the unit conversion:
#@decotest.parametric
#def test_EventArray_unit_conversion():

#Test the overloaded __getitem__ and __setitem: 
#@decotest.parametric
#def test_EventArray_getset():

@decotest.parametric
def test_UniformTime():
    for unit in ['ns','ms','s',None]:
        duration=10
        t1 = ts.UniformTime(duration,sampling_rate=1,time_unit=unit)
        t2 = ts.UniformTime(duration,sampling_rate=10,time_unit=unit)

        #The difference between the first and last item is the duration:
        yield npt.assert_equal(t1[-1]-t1[0],
                               ts.EventArray(duration,time_unit=unit))
        #Duration doesn't depend on the sampling rate:
        yield npt.assert_equal(t1[-1]-t2[0],
                               ts.EventArray(duration,time_unit=unit))

## @decotest.ipdoctest    
## def test_UniformTime_repr():
##     """
##     >>> t = ts.UniformTime(10,1)
##     >>> t
##     UniformTime([  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.], time_unit='s')
##     >>> t[1]
##     1.0 s
##     >>> t[4] = 10
##     >>> t
##     UniformTime([  0.,   1.,   2.,   3.,  10.,   5.,   6.,   7.,   8.,   9.,  10.], time_unit='s')


##     """

    
def test_CorrelationAnalyzer():

    Fs = np.pi
    t = np.arange(1024)
    x = np.sin(10*t) + np.random.rand(t.shape[-1])
    y = np.sin(10*t) + np.random.rand(t.shape[-1])

    T = ts.UniformTimeSeries(np.vstack([x,y]),sampling_rate=Fs)

    C = ts.CorrelationAnalyzer(T)

    #Test the symmetry: correlation(x,y)==correlation(y,x) 
    npt.assert_equal(C.correlation[0,1],C.correlation[1,0])
    #Test the self-sameness: correlation(x,x)==1
    npt.assert_equal(C.correlation[0,0],1)
    npt.assert_equal(C.correlation[1,1],1)

    #Test the cross-correlation:
    #First the symmetry:
    npt.assert_array_almost_equal(C.xcorr.data[0,1],C.xcorr.data[1,0])
    
    #Test the normalized cross-correlation
    #The cross-correlation should be equal to the correlation at time-lag 0
    npt.assert_equal(C.xcorr_norm.data[0,1,C.xcorr_norm.time==0]
                            ,C.correlation[0,1])

    #And the auto-correlation should be equal to 1 at 0 time-lag:
    npt.assert_equal(C.xcorr_norm.data[0,0,C.xcorr_norm.time==0],1)

    #Does it depend on having an even number of time-points?
    #make another time-series with an odd number of items:
    t = np.arange(1023)
    x = np.sin(10*t) + np.random.rand(t.shape[-1])
    y = np.sin(10*t) + np.random.rand(t.shape[-1])

    T = ts.UniformTimeSeries(np.vstack([x,y]),sampling_rate=Fs)

    C = ts.CorrelationAnalyzer(T)

    
    npt.assert_equal(C.xcorr_norm.data[0,1,C.xcorr_norm.time==0]
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

def test_CoherenceAnalyzer():

    Fs = np.pi
    t = np.arange(1024)
    x = np.sin(10*t) + np.random.rand(t.shape[-1])
    y = np.sin(10*t) + np.random.rand(t.shape[-1])

    T = ts.UniformTimeSeries(np.vstack([x,y]),sampling_rate=Fs)

    C = ts.CoherenceAnalyzer(T)

    
