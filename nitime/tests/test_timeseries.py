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
    In [108]: a = ts.EventArray([1.1,2.,3.])

    In [109]: a
    Out[109]: EventArray([ 1.1,  2. ,  3. ], time_unit='s')

    n [125]: a = ts.EventArray(arange(100))

    In [126]: t = ts.EventArray(a)

    In [127]: t
    Out[127]: 
    EventArray([  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,
            11.,  12.,  13.,  14.,  15.,  16.,  17.,  18.,  19.,  20.,  21.,
            22.,  23.,  24.,  25.,  26.,  27.,  28.,  29.,  30.,  31.,  32.,
            33.,  34.,  35.,  36.,  37.,  38.,  39.,  40.,  41.,  42.,  43.,
            44.,  45.,  46.,  47.,  48.,  49.,  50.,  51.,  52.,  53.,  54.,
            55.,  56.,  57.,  58.,  59.,  60.,  61.,  62.,  63.,  64.,  65.,
            66.,  67.,  68.,  69.,  70.,  71.,  72.,  73.,  74.,  75.,  76.,
            77.,  78.,  79.,  80.,  81.,  82.,  83.,  84.,  85.,  86.,  87.,
            88.,  89.,  90.,  91.,  92.,  93.,  94.,  95.,  96.,  97.,  98.,
            99.], time_unit='s')

    In [128]: t = ts.EventArray(a,time_unit='ms')
    
    In [129]: t
    Out[129]: 
    EventArray([     0.,   1000.,   2000.,   3000.,   4000.,   5000.,   6000.,
             7000.,   8000.,   9000.,  10000.,  11000.,  12000.,  13000.,
            14000.,  15000.,  16000.,  17000.,  18000.,  19000.,  20000.,
            21000.,  22000.,  23000.,  24000.,  25000.,  26000.,  27000.,
            28000.,  29000.,  30000.,  31000.,  32000.,  33000.,  34000.,
            35000.,  36000.,  37000.,  38000.,  39000.,  40000.,  41000.,
            42000.,  43000.,  44000.,  45000.,  46000.,  47000.,  48000.,
            49000.,  50000.,  51000.,  52000.,  53000.,  54000.,  55000.,
            56000.,  57000.,  58000.,  59000.,  60000.,  61000.,  62000.,
            63000.,  64000.,  65000.,  66000.,  67000.,  68000.,  69000.,
            70000.,  71000.,  72000.,  73000.,  74000.,  75000.,  76000.,
            77000.,  78000.,  79000.,  80000.,  81000.,  82000.,  83000.,
            84000.,  85000.,  86000.,  87000.,  88000.,  89000.,  90000.,
            91000.,  92000.,  93000.,  94000.,  95000.,  96000.,  97000.,
            98000.,  99000.], time_unit='ms')


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
def test_EventArray_index_at():

    time1 = ts.EventArray(range(10),time_unit='ms')
    for i in xrange(10):
        idx = time1.index_at(i)
        yield npt.assert_equal(idx,i)
    
    
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

    
