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
    #TimeArray can't be more than 1-d:
    yield nt.assert_raises, ValueError, ts.Events, np.zeros((2,2))
    

def test_TimeArray_init_int64():
    """Make sure that we can initialize TimeArray with an array of ints"""
    time = ts.TimeArray(np.int64(1))
    npt.assert_equal(time.__repr__(), '1.0 s')

    pass


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
def test_TimeArray_copyflag():
    """Testing the setting of the copy-flag, where that makes sense""" 

    #These two should both generate a TimeArray, with one picosecond.
    #This one holds time_unit='s'
    t1 = ts.TimeArray(np.array([1],dtype=np.int64),copy=False)
    #This one holds time_unit='ps':
    t2 = ts.TimeArray(1,time_unit='ps')
    t3 = ts.TimeArray(t2,copy=False)
    
    yield npt.assert_equal(t1,t2)
    yield npt.assert_equal(t2.ctypes.data,t3.ctypes.data)
    
@decotest.parametric
def test_TimeArray_new():
    for unit in ['ns','ms','s',None]:
        for flag,assertion in [(True,nt.assert_not_equal),
                (False, nt.assert_equal)]:
            #list -doesn't make sense to set copy=True
            time2 = ts.TimeArray(range(5), time_unit=unit, copy=True)
            #numpy array (float) - doesn't make sense to set copy=True
            time2f = ts.TimeArray(np.arange(5.), time_unit=unit, copy=True)
            #TimeArray
            time3 = ts.TimeArray(time2, time_unit=unit, copy=flag)
            #integer
            time4 = ts.TimeArray(5,time_unit=unit,copy=True)
            #float
            time5 = ts.TimeArray(5.0,time_unit=unit,copy=True)

            yield npt.assert_equal(time2,time2f)
            yield npt.assert_equal(time2,time3)
            time3[0] +=100
            yield assertion(time2[0],time3[0])
            yield npt.assert_equal(time2[1:],time3[1:])
            yield npt.assert_equal(time4,time5)

@decotest.parametric
def test_TimeArray_bool():
    time1 = ts.TimeArray([1,2,3],time_unit='s')
    time2 = ts.TimeArray([1000,2000,3000],time_unit='ms')
    bool_arr = np.ones(time1.shape,dtype=bool)
    yield npt.assert_equal(time1,time2)
    yield npt.assert_equal(bool_arr,time1==time2)
    yield nt.assert_not_equal(type(time1==time2),ts.TimeArray)


def test_TimeArray_convert_unit():
    """
    >>> a = ts.TimeArray([1,2,3,4])
    >>> a.convert_unit('ms')
    >>> a
    TimeArray([ 1000.,  2000.,  3000.,  4000.], time_unit='ms')
    >>> a.time_unit
    'ms'
    >>> b = ts.TimeArray([1,2,3,4],'s')
    >>> a==b
    array([ True,  True,  True,  True], dtype=bool)
    """
    

@decotest.parametric
def test_TimeArray_div():

    #divide singelton by singleton:
    a = 2.0
    b = 6.0
    time1 = ts.TimeArray(a,time_unit='s')
    time2 = ts.TimeArray(b,time_unit='s')
    div1 = a/b
    #This should eliminate the units and return a float, not a TimeArray:
    div2 = time1/time2
    yield npt.assert_equal(div1,div2)

    #Divide a TimeArray by a singelton:
    a = np.array([1,2,3])
    b = 6.0
    time1 = ts.TimeArray(a,time_unit='s')
    time2 = ts.TimeArray(b,time_unit='s')
    div1 = a/b
    #This should eliminate the units and return a float array, not a TimeArray:
    div2 = time1/time2
    yield npt.assert_equal(div1,div2)

    #Divide a TimeArray by another TimeArray:
    a = np.array([1,2,3])
    b = np.array([2,2,2]).astype(float) #TimeArray division is float division!
    time1 = ts.TimeArray(a,time_unit='s')
    time2 = ts.TimeArray(b,time_unit='s')
    div1 = a/b
    #This should eliminate the units and return a float array, not a TimeArray:
    div2 = time1/time2
    yield npt.assert_equal(div1,div2)
    
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
        seconds_array = ts.TimeArray(time1,time_unit='s')
        this_secs=seconds_array.at(i/1000.)
        yield npt.assert_equal(this_secs,ts.TimeArray(i,time_unit='ms'))
        all=time1.at(ts.TimeArray(i/1000.), tol=10)
        yield npt.assert_equal(all,time1)
        if i>0 and i<9:
            this_secs=time1.at(ts.TimeArray(i/1000.), tol=1)
            yield npt.assert_equal(this_secs,ts.TimeArray([i-1,i,i+1],time_unit='ms'))

@decotest.parametric
def test_UniformTime_index_at():
    time1 = ts.UniformTime(t0=1000,length=10, sampling_rate=1000, time_unit='ms')
    mask = [False]*10
    for i in xrange(10):
        idx = time1.index_at(ts.TimeArray(1000+i,time_unit='ms'))
        yield npt.assert_equal(idx,np.array(i))
        mask[i]=True
        mask_idx = time1.index_at(ts.TimeArray(1000+i,time_unit='ms'),boolean=True)
        yield npt.assert_equal(mask_idx,mask)
        if i>0 and i<9:
            mask[i-1]=True
            mask[i+1]=True

            mask_idx = time1.index_at( ts.TimeArray([999+i,1000+i, 1001+i],
                time_unit='ms'),boolean=True)
            yield npt.assert_equal(mask_idx,mask)
        
            mask[i-1]=False
            mask[i+1]=False
        mask[i]=False

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
        #want, because the t_i held by a TimeSeries is the left
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

    >>> a = ts.UniformTime(length=5,sampling_rate=1,time_unit='ms')

    >>> b = ts.UniformTime(a)

    >>> b
    UniformTime([    0.,  1000.,  2000.,  3000.,  4000.], time_unit='ms')

    >>> a
    UniformTime([    0.,  1000.,  2000.,  3000.,  4000.], time_unit='ms')

    >>> b = ts.UniformTime(a,time_unit='s')

    >>> b
    UniformTime([ 0.,  1.,  2.,  3.,  4.], time_unit='s')

    >>> a = ts.UniformTime(length=1,sampling_rate=2)

    >>> b = ts.UniformTime(length=10,sampling_interval=a.sampling_interval)

    >>> b.sampling_rate
    2.0 Hz
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
def test_TimeSeries():
    """Testing the initialization of the uniform time series object """ 

    #Test initialization with duration:
    tseries1 = ts.TimeSeries([1,2,3,4,5,6,7,8,9,10],duration=10)
    tseries2 = ts.TimeSeries([1,2,3,4,5,6,7,8,9,10],sampling_interval=1)
    npt.assert_equal(tseries1.time,tseries2.time)

    #downsampling:
    t1 = ts.UniformTime(length=8,sampling_rate=2)
    #duration is the same, but we're downsampling to 1Hz
    tseries1 = ts.TimeSeries(data=[1,2,3,4],time=t1,sampling_rate=1)
    #If you didn't explicitely provide the rate you want to downsample to, that
    #is an error:
    yield npt.assert_raises(ValueError,ts.TimeSeries,dict(data=[1,2,3,4],
                                                           time=t1)) 

    tseries2 = ts.TimeSeries(data=[1,2,3,4],sampling_rate=1)
    tseries3 = ts.TimeSeries(data=[1,2,3,4],sampling_rate=1000,
                                    time_unit='ms')
    #you can specify the sampling_rate or the sampling_interval, to the same
    #effect, where specificying the sampling_interval is in the units of that
    #time-series: 
    tseries4 = ts.TimeSeries(data=[1,2,3,4],sampling_interval=1,
                                        time_unit='ms')
    yield npt.assert_equal(tseries4.time,tseries3.time)

    #The units you use shouldn't matter - time is time:
    tseries6 = ts.TimeSeries(data=[1,2,3,4],
                                    sampling_interval=0.001,
                                    time_unit='s')
    yield npt.assert_equal(tseries6.time,tseries3.time)

    #And this too - perverse, but should be possible: 
    tseries5 = ts.TimeSeries(data=[1,2,3,4],
                                    sampling_interval=ts.TimeArray(0.001,
                                                         time_unit='s'),
                                    time_unit='ms')

    yield npt.assert_equal(tseries5.time,tseries3.time)

    #initializing with a UniformTime object:
    t=ts.UniformTime(length=3,sampling_rate=3)

    data = [1,2,3]

    tseries7 = ts.TimeSeries(data=data,time=t)

    yield npt.assert_equal(tseries7.data,data)

    data = [1,2,3,4]
    #If the data is not the right length, that should throw an error:
    yield npt.assert_raises(ValueError,
                          ts.TimeSeries,dict(data=data,time=t))
    
def test_TimeSeries_repr():

    """
    >>> t=ts.UniformTime(length=3,sampling_rate=3)
    >>> tseries1 = ts.TimeSeries(data=[3,5,8],time=t)
    >>> t.sampling_rate
    3.0 Hz
    >>> tseries1.sampling_rate
    3.0 Hz
    >>> tseries1 = ts.TimeSeries(data=[3,5,8],sampling_rate=3)
    >>> tseries1.time
    UniformTime([ 0.    ,  0.3333,  0.6667], time_unit='s')
    >>> tseries1.sampling_rate
    3.0 Hz
    >>> tseries1.sampling_interval
    0.33333333333300003 s
    >>> a = ts.UniformTime(length=1,sampling_rate=2)

    >>> b = ts.TimeSeries(data=[1,2,3],sampling_interval=a.sampling_interval)

    >>> b.sampling_rate
    2.0 Hz


    >>> a = ts.UniformTime(length=1,sampling_rate=1)

    >>> b = ts.TimeSeries(data=[1,2,3],sampling_interval=a.sampling_interval)

    >>> b.sampling_rate
    1.0 Hz
    """ 
    
@decotest.parametric
def test_Epochs():
    tms = ts.TimeArray(data=range(100), time_unit='ms')
    tmin = ts.TimeArray(data=range(100), time_unit='m')
    tsec = ts.TimeArray(data=range(100), time_unit='s')
    
    utms = ts.UniformTime(length=100, sampling_interval=1, time_unit='ms')
    utmin = ts.UniformTime(length=100, sampling_interval=1, time_unit='m')
    utsec = ts.UniformTime(length=100, sampling_interval=1, time_unit='s')

    tsms = ts.TimeSeries(data=range(100), sampling_interval=1, time_unit='ms')
    tsmin = ts.TimeSeries(data=range(100), sampling_interval=1, time_unit='m')
    tssec = ts.TimeSeries(data=range(100), sampling_interval=1, time_unit='s')

    # one millisecond epoch
    e1ms = ts.Epochs(0,1, time_unit='ms')
    msg = "Seems like a problem with copy=False in TimeArray constructor."
    yield npt.assert_equal(e1ms.duration, ts.TimeArray(1,time_unit='ms'),msg)


    # one day
    e1d = ts.Epochs(0,1, time_unit='D')
    yield npt.assert_equal(e1d.duration, ts.TimeArray(1,time_unit='D'),msg)

    e1ms_ar = ts.Epochs([0,0],[1,1], time_unit='ms')

    for t in [tms, tmin, tsec, utms, utmin, utsec]:
        # the sample time arrays are all at least 1ms long, so this should
        # return a timearray that has exactly one time point in it
        yield npt.assert_equal(len(t.during(e1ms)),1)

        # make sure, slicing doesn't change the class
        yield npt.assert_equal(type(t),type(t.during(e1ms)))
        
    for t in [tsms, tsmin, tssec]:
        # the sample time series are all at least 1ms long, so this should
        # return a timeseries that has exactly one time point in it
        yield npt.assert_equal(len(t.during(e1ms)),1)
        
        # make sure, slicing doesn't change the class
        yield npt.assert_equal(type(t),type(t.during(e1ms)))

        # same thing but now there's an array of epochs
        e2 = ts.Epochs([0,10],[10,20],time_unit=t.time_unit)
        
        # make sure, slicing doesn't change the class for array of epochs
        yield npt.assert_equal(type(t),type(t.during(e2)))

        # Indexing with an array of epochs (all of which are the same length)
        yield npt.assert_equal(t[e2].data.shape, (2,10))
        yield npt.assert_equal(len(t.during(e2)),10)
        yield npt.assert_equal(t[e2].data.ndim,2)
        # check the data at some timepoints (a dimension was added)
        yield npt.assert_equal(t[e2][0], (0,10))
        yield npt.assert_equal(t[e2][1], (1,11))
        # check the data for each epoch
        yield npt.assert_equal(t[e2].data[0], range(10))
        yield npt.assert_equal(t[e2].data[1], range(10,20))
        yield npt.assert_equal(t[e2].duration,e2[0].duration)
        
        # slice with Epochs of different length (not supported for timeseries,
        # raise error, though future jagged array implementation could go here)
        ejag = ts.Epochs([0,10],[10,40],time_unit=t.time_unit)
        # next line is the same as t[ejag]
        yield npt.assert_raises(ValueError,t.__getitem__, ejag)


        # if an epoch lies entirely between samples in the timeseries, return
        # the previous sample. We define the bin as any time between
        # [t,t+delta_t)
        eshort = ts.Epochs(2.5,2.7,time_unit=t.time_unit)
        yield npt.assert_equal(len(t[eshort].data),1)
        yield npt.assert_equal(t[eshort].data[0],t[2])

        e1ms_outofrange = ts.Epochs(200,300,time_unit=t.time_unit)
        # assert that with the epoch moved outside of the time range of our
        # data, slicing with the epoch now yields an empty array
        yield npt.assert_raises(ValueError, t.during, dict(e=e1ms_outofrange))

        # the sample timeseries are all shorter than a day, so this should
        # raise an error (instead of padding, or returning a shorter than
        # expected array.
        yield npt.assert_raises(ValueError, t.during, dict(e=e1d))


@decotest.parametric
def test_Events():

    # time has to be one-dimensional
    yield nt.assert_raises, ValueError, ts.Events, np.zeros((2,2))

    t = ts.TimeArray([1,2,3],time_unit='ms')
    x = [1,2,3]
    y = [2,4,6]
    z = [10.,20.,30.]
    i0 = [0,0,1]
    i1 = [0,1,2]
    for unit in [None, 's','ns','D']:
        # events with data
        ev1 = ts.Events(t,time_unit=unit,i=x,j=y,k=z)

        # events with indices
        ev2 = ts.Events(t,time_unit=unit,indices=[i0,i1])

        # events with indices and labels
        ev3 = ts.Events(t,time_unit=unit,labels=['trial','other'],
                        indices=[i0,i1])

        # Note that the length of indices and labels has to be identical: 
        yield nt.assert_raises, ValueError,ts.Events,dict(t=t,time_unit=unit,labels=['trial','other'],indices=[i0])#Only one of the indices!
            

        # make sure the time is retained
        yield npt.assert_equal(ev1.time,t)
        yield npt.assert_equal(ev2.time,t)

        # make sure time unit is correct
        if unit is not None:
            yield npt.assert_equal(ev1.time_unit,unit)
            yield npt.assert_equal(ev2.time_unit,unit)
        else:
            yield npt.assert_equal(ev1.time_unit,'s')
            yield npt.assert_equal(ev2.time_unit,'s')

        # make sure we can extract data
        yield npt.assert_equal(ev1.data['i'],x)
        yield npt.assert_equal(ev1.data['j'],y)
        yield npt.assert_equal(ev1.data['k'],z)

        # make sure we can get the indices by label
        yield npt.assert_equal(ev3.index.trial,i0)
        yield npt.assert_equal(ev3.index.other,i1)

        # make sure we can get the indices by position
        yield npt.assert_equal(ev2.index.i0,i0)
        yield npt.assert_equal(ev2.index.i1,i1)


    
