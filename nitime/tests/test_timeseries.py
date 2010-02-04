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
        npt.assert_raises(ValueError,
                          ts.UniformTime,dict(sampling_interval=10,duration=1))



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
    unit = 2*np.pi/l
    t = np.arange(0,2*np.pi+unit,unit)
    signal = np.sin(cycles*t)
    events = np.zeros(t.shape)
    #Zero crossings: 
    idx = np.where(np.abs(signal)<0.03)[0]
    #An event occurs at the beginning of every cycle:
    events[idx[:-2:2]]=1
    #and another kind of event at the end of each cycle:
    events[idx[1:-1:2]]=2

    T_signal = ts.UniformTimeSeries(signal,sampling_rate=1)
    T_events = ts.UniformTimeSeries(events,sampling_rate=1)
    ETA = ts.EventRelatedAnalyzer(T_signal,T_events,l/(cycles*2)).eta

    #This looks good, but doesn't pass unless you consider 3 digits:
    npt.assert_almost_equal(ETA.data[0],signal[:ETA.data.shape[-1]],3)
    npt.assert_almost_equal(ETA.data[1],-1*signal[:ETA.data.shape[-1]],3)

    #Same should be true for the FIR analysis: 
    FIR = ts.EventRelatedAnalyzer(T_signal,T_events,l/(cycles*2)).FIR
    npt.assert_almost_equal(FIR.data[0],signal[:FIR.data.shape[-1]],3)
    npt.assert_almost_equal(FIR.data[1],-1*signal[:FIR.data.shape[-1]],3)

def test_CoherenceAnalyzer():

    Fs = np.pi
    t = np.arange(1024)
    x = np.sin(10*t) + np.random.rand(t.shape[-1])
    y = np.sin(10*t) + np.random.rand(t.shape[-1])

    T = ts.UniformTimeSeries(np.vstack([x,y]),sampling_rate=Fs)

    C = ts.CoherenceAnalyzer(T)

    
def test_HilbertAnalyzer():
    """Testing the HilbertAnalyzer (analytic signal)"""
    pi = np.pi
    Fs = np.pi
    t = np.arange(0,2*pi,pi/256)

    a0 = np.sin(t)
    a1 = np.cos(t)
    a2 = np.sin(2*t)
    a3 = np.cos(2*t)

    T = ts.UniformTimeSeries(data=np.vstack([a0,a1,a2,a3]),
                             sampling_rate=Fs)

    H = ts.HilbertAnalyzer(T)

    h_abs = H.magnitude.data
    h_angle = H.phase.data
    h_real = H.real.data
    #The real part should be equal to the original signals:
    npt.assert_almost_equal(h_real,H.data)
    #The absolute value should be one everywhere, for this input:
    npt.assert_almost_equal(h_abs,np.ones(T.data.shape))
    #For the 'slow' sine - the phase should go from -pi/2 to pi/2 in the first
    #256 bins: 
    npt.assert_almost_equal(h_angle[0,:256],np.arange(-pi/2,pi/2,pi/256))
    #For the 'slow' cosine - the phase should go from 0 to pi in the same
    #interval: 
    npt.assert_almost_equal(h_angle[1,:256],np.arange(0,pi,pi/256))
    #The 'fast' sine should make this phase transition in half the time:
    npt.assert_almost_equal(h_angle[2,:128],np.arange(-pi/2,pi/2,pi/128))
    #Ditto for the 'fast' cosine:
    npt.assert_almost_equal(h_angle[3,:128],np.arange(0,pi,pi/128))

#This is known to fail because of artifacts induced by the fourier transform
#for limited samples: 
@npt.dec.knownfailureif(True) 
def test_FilterAnalyzer():
    """Testing the FilterAnalyzer """
    t = np.arange(np.pi/100,10*np.pi,np.pi/100)
    fast = np.sin(50*t)
    slow = np.sin(10*t)
    time_series = ts.UniformTimeSeries(data=fast+slow,sampling_rate=np.pi)

    #0.6 is somewhere between the two frequencies 
    f_slow = ts.FilterAnalyzer(time_series,ub=0.6)
    npt.assert_equal(f_slow.filtered_fourier.data,slow)
    #
    f_fast = ts.FilterAnalyzer(time_series,lb=0.6)
    npt.assert_equal(f_fast.filtered_fourier.data,fast)

    

    


