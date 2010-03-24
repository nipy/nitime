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
    time1 = ts.TimeArray(10**6)
    yield npt.assert_equal(time1.__repr__(),'1000000.0 s')

                                           
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
    time1 = ts.TimeArray(range(10),time_unit='ms')
    for i in xrange(10):
        idx = time1.index_at([i])
        yield npt.assert_equal(idx,np.array(i))
        idx_secs=time1.index_at(ts.TimeArray(i/1000.))
        yield npt.assert_equal(idx_secs,np.array(i))

@decotest.parametric
def test_TimeArray_at():
    time1 = ts.TimeArray(range(10),time_unit='ms')
    for i in xrange(10):
        this = time1.at(i)
        yield npt.assert_equal(this,ts.TimeArray(i,time_unit='ms'))
        this_secs=time1.at(ts.TimeArray(i/1000.))
        yield npt.assert_equal(this_secs,ts.TimeArray(i,time_unit='ms'))

#XXX Need to write these tests:

#Test the unit conversion:
#@decotest.parametric
#def test_TimeArray_unit_conversion():

#Test the overloaded __getitem__ and __setitem: 
#@decotest.parametric
#def test_TimeArray_getset():

@decotest.parametric
def test_UniformTime():
    tuc = ts.time_unit_conversion
    
    for unit,duration in zip(['ns','ms','s',None],
                             [2*10**9,2*10**6,100,20]):
        
        t1 = ts.UniformTime(duration=duration,sampling_rate=1,
                            time_unit=unit)
        t2 = ts.UniformTime(duration=duration,sampling_rate=20,
                            time_unit=unit)

        #The following two tests verify that first-last are equal to the
        #duration, but it is unclear whether that is really the behavior we
        #want, because the t_i held by a UniformTimeSeries is the left
        #(smaller) side of the time-duration defined by the bin
        
        #The difference between the first and last item is the duration:
        #yield npt.assert_equal(t1[-1]-t1[0],
        #                       ts.TimeArray(duration,time_unit=unit))
        #Duration doesn't depend on the sampling rate:
        #yield npt.assert_equal(t1[-1]-t2[0],
        #                       ts.TimeArray(duration,time_unit=unit))

        a = ts.UniformTime(duration=10,sampling_rate=1)
        b = ts.UniformTime(a,time_unit=unit)
        yield npt.assert_equal(a.sampling_interval,b.sampling_interval)
        yield npt.assert_equal(a.sampling_rate,b.sampling_rate)

        b = ts.UniformTime(a,duration=2*duration,time_unit=unit)
        yield npt.assert_equal(a.sampling_interval,b.sampling_interval)
        yield npt.assert_equal(a.sampling_rate,b.sampling_rate)
            
        b = ts.UniformTime(a,length=100,time_unit=unit)
        yield npt.assert_equal(a.sampling_interval,b.sampling_interval)
        yield npt.assert_equal(a.sampling_rate,b.sampling_rate)

        b = ts.UniformTime(a,length=100,time_unit=unit)
        yield npt.assert_equal(a.sampling_interval,b.sampling_interval)
        yield npt.assert_equal(a.sampling_rate,b.sampling_rate)
        
        b = ts.UniformTime(a,length=100,duration=duration,time_unit=unit)
        c = ts.UniformTime(length=100,duration=duration,time_unit=unit)
        yield npt.assert_equal(c,b)

        
        b = ts.UniformTime(sampling_interval=1,duration=10,time_unit=unit)
        c = ts.UniformTime(sampling_rate=tuc['s']/tuc[unit],
                           length=10,time_unit=unit)

        yield npt.assert_equal(c,b)

        #This should raise a value error, because the duration is shorter than
        #the sampling_interval:
        yield npt.assert_raises(ValueError,
                          ts.UniformTime,dict(sampling_interval=10,duration=1))

    #Time objects can be initialized with other time objects setting the
    #duration, sampling_interval and sampling_rate:
    
    a = ts.UniformTime(length=1,sampling_rate=1)
    yield npt.assert_raises(ValueError, ts.UniformTime, dict(data=a,
        sampling_rate=10, sampling_interval=.1))
    b = ts.UniformTime(duration=2*a.sampling_interval,
                       sampling_rate=2*a.sampling_rate)

    yield npt.assert_equal(ts.Frequency(b.sampling_rate),
                     ts.Frequency (2*a.sampling_rate))
    yield npt.assert_equal(b.sampling_interval,ts.TimeArray(0.5*a.sampling_rate))

    b = ts.UniformTime(duration=10,
                       sampling_interval=a.sampling_interval)

    yield npt.assert_equal(b.sampling_rate,a.sampling_rate)

    b = ts.UniformTime(duration=10,
                       sampling_rate=a.sampling_rate)

    yield npt.assert_equal(b.sampling_interval,a.sampling_interval)

@decotest.ipdoctest    
def test_UniformTime_repr():
    """
    >>> time1 = ts.UniformTime(sampling_rate=1000,time_unit='ms',length=3)
    >>> time1.sampling_rate
    1000.0 Hz
    >>> time1
    UniformTime([ 0.,  1.,  2.], time_unit='ms')

    >>> time2= ts.UniformTime(sampling_rate=1000,time_unit='s',length=3)
    >>> time2.sampling_rate
    1000.0 Hz
    >>> time2
    UniformTime([ 0.   ,  0.001,  0.002], time_unit='s')

    In [85]: a = ts.UniformTime(length=5,sampling_rate=1,time_unit='ms')

    In [86]: b = ts.UniformTime(a)

    In [87]: b
    Out[87]: UniformTime([    0.,  1000.,  2000.,  3000.,  4000.], time_unit='ms')

    In [88]: a
    Out[88]: UniformTime([    0.,  1000.,  2000.,  3000.,  4000.], time_unit='ms')

    In [89]: b = ts.UniformTime(a,time_unit='s')

    In [90]: b
    Out[90]: UniformTime([ 0.,  1.,  2.,  3.,  4.], time_unit='s')

    In [445]: a = ts.UniformTime(length=1,sampling_rate=2)

    In [446]: b = ts.UniformTime(length=10,sampling_interval=a.sampling_interval)

    In [447]: b.sampling_rate
    Out[447]: 2.0 Hz

    """

@decotest.parametric
def test_Frequency():
    """Test frequency representation object"""
    tuc=ts.time_unit_conversion
    for unit in ['ns','ms','s',None]:
        f = ts.Frequency(1,time_unit=unit)
        yield npt.assert_equal(f.to_period(),tuc[unit]) 

        f = ts.Frequency(1000,time_unit=unit)
        yield npt.assert_equal(f.to_period(),tuc[unit]/1000)       

        f = ts.Frequency(0.001,time_unit=unit)
        yield npt.assert_equal(f.to_period(),tuc[unit]*1000)       


    
@decotest.parametric
def test_UniformTimeSeries():
    """Testing the initialization of the uniform time series object """ 

    #tseries = ts.UniformTimeSeries([1,2,3,4],duration=10)
    #downsampling:
    t1 = ts.UniformTime(length=8,sampling_rate=2)
    #duration is the same, but we're downsampling to 1Hz
    tseries1 = ts.UniformTimeSeries(data=[1,2,3,4],time=t1,sampling_rate=1)
    #If you didn't explicitely provide the rate you want to downsample to, that
    #is an error:
    npt.assert_raises(ValueError,ts.UniformTimeSeries,dict(data=[1,2,3,4],
                                                           time=t1)) 

    tseries2 = ts.UniformTimeSeries(data=[1,2,3,4],sampling_rate=1)
    tseries3 = ts.UniformTimeSeries(data=[1,2,3,4],sampling_rate=1000,
                                    time_unit='ms')
    #you can specify the sampling_rate or the sampling_interval, to the same
    #effect, where specificying the sampling_interval is in the units of that
    #time-series: 
    tseries4 = ts.UniformTimeSeries(data=[1,2,3,4],sampling_interval=1,
                                        time_unit='ms')
    npt.assert_equal(tseries4.time,tseries3.time)

    #The units you use shouldn't matter - time is time:
    tseries6 = ts.UniformTimeSeries(data=[1,2,3,4],
                                    sampling_interval=0.001,
                                    time_unit='s')
    npt.assert_equal(tseries6.time,tseries3.time)

    #And this too - perverse, but should be possible: 
    tseries5 = ts.UniformTimeSeries(data=[1,2,3,4],
                                    sampling_interval=ts.TimeArray(0.001,
                                                         time_unit='s'),
                                    time_unit='ms')

    npt.assert_equal(tseries5.time,tseries3.time)

@decotest.ipdoctest    
def test_UniformTimeSeries_repr():

    """
    >>> t=ts.UniformTime(length=3,sampling_rate=3)
    >>> tseries1 = ts.UniformTimeSeries(data=[3,5,8],time=t)
    >>> t.sampling_rate
    3.0 Hz
    >>> tseries1.sampling_rate
    3.0 Hz
    >>> tseries1 = ts.UniformTimeSeries(data=[3,5,8],sampling_rate=3)
    >>> tseries1.time
    UniformTime([ 0.        ,  0.33333333,  0.66666667], time_unit='s')
    >>> tseries1.sampling_rate
    3.0 Hz
    >>> tseries1.sampling_interval
    0.33333333333300003 s
    In [435]: a = ts.UniformTime(length=1,sampling_rate=2)

    In [436]: b = ts.UniformTimeSeries(data=[1,2,3],sampling_interval=a.sampling_interval)

    In [437]: b.sampling_rate
    Out[437]: 2.0 Hz


    In [361]: a = ts.UniformTime(length=1,sampling_rate=1)

    In [362]: b = ts.UniformTimeSeries(data=[1,2,3],sampling_interval=a.sampling_interval)

    In [363]: b.sampling_rate
    Out[363]: 1.0 Hz

    """ 
    
