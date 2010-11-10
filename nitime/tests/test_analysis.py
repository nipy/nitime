import numpy as np
import numpy.testing as npt
from nitime import utils as ut
import nitime.timeseries as ts
import nitime.analysis as nta
import nose.tools as nt
import decotest

def test_SpectralAnalyzer():

    Fs = np.pi
    t = np.arange(1024)
    x = np.sin(10*t) + np.random.rand(t.shape[-1])
    y = np.sin(10*t) + np.random.rand(t.shape[-1])

    T = ts.TimeSeries(np.vstack([x,y]),sampling_rate=Fs)

    C = nta.SpectralAnalyzer(T)

    f,c = C.psd

    npt.assert_equal(f.shape,(33,)) #This is the setting for this analyzer
                                    #(window-length of 64)
    npt.assert_equal(c.shape,(2,33))

    f,c = C.cpsd
    npt.assert_equal(f.shape,(33,)) #This is the setting for this analyzer
                                    #(window-length of 64)
    npt.assert_equal(c.shape,(2,2,33))

    f,c = C.cpsd
    npt.assert_equal(f.shape,(33,)) #This is the setting for this analyzer
                                    #(window-length of 64)
    npt.assert_equal(c.shape,(2,2,33))
    
    f,c = C.spectrum_fourier

    npt.assert_equal(f.shape,(t.shape[0]/2+1,))
    npt.assert_equal(c.shape,(2,t.shape[0]/2+1))

    f,c = C.spectrum_multi_taper

    npt.assert_equal(f.shape,(t.shape[0]/2+1,))
    npt.assert_equal(c.shape,(2,2,t.shape[0]/2+1))

    f,c = C.periodogram

    npt.assert_equal(f.shape,(t.shape[0]/2+1,))
    npt.assert_equal(c.shape,(2,t.shape[0]/2+1))

def test_CoherenceAnalyzer():
    Fs = np.pi
    t = np.arange(10)
    x = np.sin(10*t) + np.random.rand(t.shape[-1])
    y = np.sin(10*t) + np.random.rand(t.shape[-1])
    T = ts.TimeSeries(np.vstack([x,y]),sampling_rate=Fs)
    C = nta.CoherenceAnalyzer(T)

    npt.assert_equal(C.coherence.shape,(2,2,33)) #Default mlab_method
    #Coherence symmetry:
    npt.assert_equal(C.coherence[0,1],C.coherence[1,0])
    #Phase/delay asymmetry:
    npt.assert_equal(C.phase[0,1],-1*C.phase[1,0])
    npt.assert_equal(C.delay[0,1][1:],-1*C.delay[1,0][1:]) #The very first one
                                        #is a nan

    #Calculation of the spectrum is the same as in the default spectral
    #analyzer: 
    S = nta.SpectralAnalyzer(T)
    npt.assert_equal(S.cpsd,(C.frequencies,C.spectrum))

def test_SparseCoherenceAnalyzer():
    Fs = np.pi
    t = np.arange(256)
    x = np.sin(10*t) + np.random.rand(t.shape[-1])
    y = np.sin(10*t) + np.random.rand(t.shape[-1])
    T = ts.TimeSeries(np.vstack([x,y]),sampling_rate=Fs)
    C1 = nta.SparseCoherenceAnalyzer(T,ij=((0,1),(1,0)))

    #Coherence symmetry:
    npt.assert_equal(np.abs(C1.coherence[0,1]),np.abs(C1.coherence[1,0]))

    #Make sure you get the same answers as you would from the standard
    #CoherenceAnalyzer: 
    C2 = nta.CoherenceAnalyzer(T)

    yield npt.assert_almost_equal, C2.coherence[0,1],C1.coherence[0,1]
    yield npt.assert_almost_equal, C2.coherence[0,1],C1.coherence[0,1]

def test_CorrelationAnalyzer():

    Fs = np.pi
    t = np.arange(1024)
    x = np.sin(10*t) + np.random.rand(t.shape[-1])
    y = np.sin(10*t) + np.random.rand(t.shape[-1])

    T = ts.TimeSeries(np.vstack([x,y]),sampling_rate=Fs)

    C = nta.CorrelationAnalyzer(T)

    #Test the symmetry: correlation(x,y)==correlation(y,x) 
    npt.assert_equal(C.corrcoef[0,1],C.corrcoef[1,0])
    #Test the self-sameness: correlation(x,x)==1
    npt.assert_equal(C.corrcoef[0,0],1)
    npt.assert_equal(C.corrcoef[1,1],1)

    #Test the cross-correlation:
    #First the symmetry:
    npt.assert_array_almost_equal(C.xcorr.data[0,1],C.xcorr.data[1,0])
    
    #Test the normalized cross-correlation
    #The cross-correlation should be equal to the correlation at time-lag 0
    npt.assert_equal(C.xcorr_norm.data[0,1,C.xcorr_norm.time==0]
                            ,C.corrcoef[0,1])

    #And the auto-correlation should be equal to 1 at 0 time-lag:
    npt.assert_equal(C.xcorr_norm.data[0,0,C.xcorr_norm.time==0],1)

    #Does it depend on having an even number of time-points?
    #make another time-series with an odd number of items:
    t = np.arange(1023)
    x = np.sin(10*t) + np.random.rand(t.shape[-1])
    y = np.sin(10*t) + np.random.rand(t.shape[-1])

    T = ts.TimeSeries(np.vstack([x,y]),sampling_rate=Fs)

    C = nta.CorrelationAnalyzer(T)
    
    npt.assert_equal(C.xcorr_norm.data[0,1,C.xcorr_norm.time==0]
                            ,C.corrcoef[0,1])

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

    T_signal = ts.TimeSeries(signal,sampling_rate=1)
    T_events = ts.TimeSeries(events,sampling_rate=1)
    ETA = nta.EventRelatedAnalyzer(T_signal,T_events,l/(cycles*2)).eta

    #This looks good, but doesn't pass unless you consider 3 digits:
    npt.assert_almost_equal(ETA.data[0],signal[:ETA.data.shape[-1]],3)
    npt.assert_almost_equal(ETA.data[1],-1*signal[:ETA.data.shape[-1]],3)

    #Same should be true for the FIR analysis: 
    FIR = nta.EventRelatedAnalyzer(T_signal,T_events,l/(cycles*2)).FIR
    npt.assert_almost_equal(FIR.data[0],signal[:FIR.data.shape[-1]],3)
    npt.assert_almost_equal(FIR.data[1],-1*signal[:FIR.data.shape[-1]],3)

    #Same should be true for 
    XCORR = nta.EventRelatedAnalyzer(T_signal,T_events,l/(cycles*2)).xcorr_eta
    npt.assert_almost_equal(XCORR.data[0],signal[:XCORR.data.shape[-1]],3)
    npt.assert_almost_equal(XCORR.data[1],-1*signal[:XCORR.data.shape[-1]],3)
    
    #More dimensions: 
    T_signal = ts.TimeSeries(np.vstack([signal,signal]),sampling_rate=1)
    T_events = ts.TimeSeries(np.vstack([events,events]),sampling_rate=1)
    ETA = nta.EventRelatedAnalyzer(T_signal,T_events,l/(cycles*2)).eta

    #The events input and the time-series input have different dimensions:
    T_events = ts.TimeSeries(events,sampling_rate=1)
    ETA = nta.EventRelatedAnalyzer(T_signal,T_events,l/(cycles*2)).eta
    npt.assert_almost_equal(ETA.data[0][0],signal[:ETA.data.shape[-1]],3)
    npt.assert_almost_equal(ETA.data[1][1],-1*signal[:ETA.data.shape[-1]],3)
    
    #Input is an Events object, instead of a time-series:
    ts1 = ts.TimeSeries(np.arange(100),sampling_rate=1)
    ev = ts.Events([10,20,30])
    et = nta.EventRelatedAnalyzer(ts1,ev,5)

    #The five points comprising the average of the three sequences:
    npt.assert_equal(et.eta.data,[20.,21.,22.,23.,24.])

    ts2 = ts.TimeSeries(np.arange(200).reshape(2,100),sampling_rate=1)
    ev = ts.Events([10,20,30])
    et = nta.EventRelatedAnalyzer(ts2,ev,5)
    
    npt.assert_equal(et.eta.data,[[  20.,   21.,   22.,   23.,   24.],
                                  [ 120.,  121.,  122.,  123.,  124.]])


    #Test that providing the analyzer with an array, instead of an Events or a
    #TimeSeries object throws an error:
    npt.assert_raises(ValueError,nta.EventRelatedAnalyzer,ts2,events,10)
     
    
def test_HilbertAnalyzer():
    """Testing the HilbertAnalyzer (analytic signal)"""
    pi = np.pi
    Fs = np.pi
    t = np.arange(0,2*pi,pi/256)

    a0 = np.sin(t)
    a1 = np.cos(t)
    a2 = np.sin(2*t)
    a3 = np.cos(2*t)

    T = ts.TimeSeries(data=np.vstack([a0,a1,a2,a3]),
                             sampling_rate=Fs)

    H = nta.HilbertAnalyzer(T)

    h_abs = H.amplitude.data
    h_angle = H.phase.data
    h_real = H.real.data
    #The real part should be equal to the original signals:
    npt.assert_almost_equal(h_real,T.data)
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

def test_FilterAnalyzer():
    """Testing the FilterAnalyzer """
    t = np.arange(np.pi/100,10*np.pi,np.pi/100)
    fast = np.sin(50*t)+10
    slow = np.sin(10*t)-20
    fast_ts = ts.TimeSeries(data=fast,sampling_rate=np.pi)
    slow_ts = ts.TimeSeries(data=slow,sampling_rate=np.pi)
   
    #Make sure that the DC is preserved
    f_slow = nta.FilterAnalyzer(slow_ts,ub=0.6)
    npt.assert_almost_equal(f_slow.filtered_fourier.data.mean(),-20)
    f_fast = nta.FilterAnalyzer(fast_ts,lb=0.6)
    npt.assert_almost_equal(f_fast.data.mean(),10)

def test_NormalizationAnalyzer():
    """Testing the NormalizationAnalyzer """

    t1 = ts.TimeSeries(data=[[99,100,101],[99,100,101]],sampling_interval=1.)
    t2 = ts.TimeSeries(data=[[-1,0,1],[-1,0,1]],sampling_interval=1.)

    N1 = nta.NormalizationAnalyzer(t1)
    npt.assert_almost_equal(N1.percent_change[0],t2[0])

    t3 = ts.TimeSeries(data=[[100,102],[1,3]],sampling_interval=1.)
    t4 = ts.TimeSeries(data=[[-1,1],[-1,1]],sampling_interval=1.)
    
    N2 = nta.NormalizationAnalyzer(t3)
    npt.assert_almost_equal(N2.z_score[0],t4[0])
    
def test_MorletWaveletAnalyzer():
    """Testing the MorletWaveletAnalyzer """
    time_series = ts.TimeSeries(data=np.random.rand(100),sampling_rate=100)

    W = nta.MorletWaveletAnalyzer(time_series,freqs=20)
    WL = nta.MorletWaveletAnalyzer(time_series,freqs=20,log_morlet=True)
    H = nta.HilbertAnalyzer(W.real)
    HL = nta.HilbertAnalyzer(WL.real)

    npt.assert_almost_equal(np.sin(H.phase.data[10:-10]),np.sin(W.phase.data[10:-10]),decimal=0)
    npt.assert_almost_equal(np.sin(HL.phase.data[10:-10]),np.sin(WL.phase.data[10:-10]),decimal=0)


def test_CoherenceMTAnalyzer():

    """ Testing the multi-taper coherence analysis. See also comparison in doc/examples/multi_taper_coh.py""" 

    
    Fs = np.pi
    t = np.arange(100)
    x = np.sin(10*t) + np.random.rand(t.shape[-1])
    y = np.sin(10*t) + np.random.rand(t.shape[-1])
    T = ts.TimeSeries(np.vstack([x,y]),sampling_rate=Fs)
    C1 = nta.MTCoherenceAnalyzer(T)

    #Coherence symmetry:
    npt.assert_equal(C1.coherence[0,1],C1.coherence[1,0])

    #Test that it runs through (it will trivially be equal to itself):
    npt.assert_equal(C1.confidence_interval,C1.confidence_interval)

    #Test that it works with adaptive set to False:
    C2 = nta.MTCoherenceAnalyzer(T,adaptive=False)

    #Coherence symmetry:
    npt.assert_equal(C2.coherence[0,1],C2.coherence[1,0])

    #Test that it runs through (it will trivially be equal to itself):
    npt.assert_equal(C2.confidence_interval,C2.confidence_interval)

    #print 'woot'
    
def test_SeedCoherenceAnalyzer():
    Fs = np.pi
    t = np.arange(256)
    x = np.sin(10*t) + np.random.rand(t.shape[-1])
    y = np.sin(10*t) + np.random.rand(t.shape[-1])
    T1 = ts.TimeSeries(np.vstack([x,y]),sampling_rate=Fs)

    z = y = np.sin(10*t) + np.random.rand(t.shape[-1])

    T2 = ts.TimeSeries(z,sampling_rate=Fs)
    
    C1 = nta.SeedCoherenceAnalyzer(T2, T1)

    T3 = ts.TimeSeries(np.vstack([x,y,z]),sampling_rate=Fs)

    #Make sure you get the same answers as you would from the standard
    #CoherenceAnalyzer: 
    C2 = nta.CoherenceAnalyzer(T3)

    npt.assert_almost_equal(C2.coherency[2,0],C1.coherency[0])

