import numpy as np
import numpy.testing as npt
from nitime import utils as ut
import nitime.timeseries as ts
import nose.tools as nt
import decotest

@decotest.parametric
def test_TimeArray():

    time1 = ts.TimeArray(range(100),time_unit='ms')
    time2 = time1+time1
    yield npt.assert_equal(time2.time_unit,'ms')

@decotest.ipdoctest    
def test_TimeArray_repr():
    """
>>> a = ts.TimeArray([1.1,2,3])
>>> a
TimeArray([ 1.1,  2. ,  3. ], time_unit='s')
>>> t = ts.TimeArray(a,time_unit='ms')
>>> t
TimeArray([ 1100.,  2000.,  3000.], time_unit='ms')
>>> t[0]
1100.0 ms

    """

@decotest.parametric
def test_TimeArray_new():
    for unit in ['ns','ms','s',None]:
        for flag,assertion in [(True,nt.assert_not_equal),
                (False, nt.assert_equal)]:
            #default parameters (timeunits, copy flag, etc)
            #list
            time1 = ts.TimeArray(range(5),time_unit=unit, copy=flag)
            #numpy array (int)
            time2 = ts.TimeArray(np.arange(5), time_unit=unit, copy=flag)
            #numpy array (float)
            time2f = ts.TimeArray(np.arange(5.), time_unit=unit, copy=flag)
            #TimeArray
            time3 = ts.TimeArray(time1, time_unit=unit, copy=flag)
            #integer
            time4 = ts.TimeArray(5,time_unit=unit,copy=flag)
            #float
            time5 = ts.TimeArray(5.0,time_unit=unit,copy=flag)

            yield npt.assert_equal(time1,time2)
            yield npt.assert_equal(time2,time2f)
            yield npt.assert_equal(time1,time3)
            time3[0] +=100
            yield assertion(time1[0],time3[0])
            yield npt.assert_equal(time1[1:],time3[1:])
            yield npt.assert_equal(time4,time5)

@decotest.parametric
def test_TimeArray_bool():
    time1 = ts.TimeArray([1,2,3],time_unit='s')
    time2 = ts.TimeArray([1000,2000,3000],time_unit='ms')
    bool_arr = np.ones(time1.shape,dtype=bool)
    yield npt.assert_equal(time1,time2)
    yield npt.assert_equal(bool_arr,time1==time2)
    yield nt.assert_not_equal(type(time1==time2),ts.TimeArray)

    
@decotest.parametric
def test_TimeArray_index_at():

    #Is this really the behavior we want?
    time1 = ts.TimeArray(range(10),time_unit='ms')
    for i in xrange(10):
        idx = time1.index_at([i])
        yield npt.assert_equal(idx,np.array(i))

#XXX Need to write these tests:

#Test the unit conversion:
#@decotest.parametric
#def test_TimeArray_unit_conversion():

#Test the overloaded __getitem__ and __setitem: 
#@decotest.parametric
#def test_TimeArray_getset():

@decotest.parametric
def test_UniformTime():
    for unit in ['ns','ms','s',None]:
        duration=10
        t1 = ts.UniformTime(duration=duration,sampling_rate=1,time_unit=unit)
        t2 = ts.UniformTime(duration=duration,sampling_rate=10,time_unit=unit)

        #The difference between the first and last item is the duration:
        yield npt.assert_equal(t1[-1]-t1[0],
                               ts.TimeArray(duration,time_unit=unit))
        #Duration doesn't depend on the sampling rate:
        yield npt.assert_equal(t1[-1]-t2[0],
                               ts.TimeArray(duration,time_unit=unit))

        a = ts.UniformTime(duration=10,sampling_rate=1)
        b = ts.UniformTime(a,time_unit=unit)
        yield npt.assert_equal(a.sampling_interval,b.sampling_interval)
        yield npt.assert_equal(a.sampling_rate,b.sampling_rate)

        b = ts.UniformTime(a,duration=2000000000000000,time_unit=unit)
        yield npt.assert_equal(a.sampling_interval,b.sampling_interval)
        yield npt.assert_equal(a.sampling_rate,b.sampling_rate)
            
        b = ts.UniformTime(a,length=100,time_unit=unit)
        yield npt.assert_equal(a.sampling_interval,b.sampling_interval)
        yield npt.assert_equal(a.sampling_rate,b.sampling_rate)

        b = ts.UniformTime(a,length=100,time_unit=unit)
        yield npt.assert_equal(a.sampling_interval,b.sampling_interval)
        yield npt.assert_equal(a.sampling_rate,b.sampling_rate)
        
        b = ts.UniformTime(a,length=100,duration=10,time_unit=unit)
        c = ts.UniformTime(length=100,duration=10,time_unit=unit)
        yield npt.assert_equal(c,b)

        b = ts.UniformTime(sampling_interval=1,duration=10,time_unit=unit)
        c = ts.UniformTime(sampling_rate=1,duration=10,time_unit=unit)
        yield npt.assert_equal(c,b)

        b = ts.UniformTime(sampling_interval=0.1,duration=10,time_unit=unit)
        c = ts.UniformTime(sampling_rate=10,length=100,time_unit=unit)
        yield npt.assert_equal(c,b)

        #This should raise a value error, because the duration is shorter than
        #the sampling_interval:
        npt.assert_raises(ValueError,
                          ts.UniformTime,dict(sampling_interval=10,duration=1))
        
        
@decotest.ipdoctest    
def test_UniformTime_repr():
    """
    """

    
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

    
