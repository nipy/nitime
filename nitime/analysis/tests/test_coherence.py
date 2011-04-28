import numpy as np
import numpy.testing as npt
import matplotlib.mlab as mlab

import nitime.timeseries as ts
import nitime.analysis as nta

def test_CoherenceAnalyzer():
    methods = (None,
           {"this_method":'welch',"NFFT":256},
           {"this_method":'multi_taper_csd'},
           {"this_method":'periodogram_csd',"NFFT":256})

    Fs = np.pi
    t = np.arange(1024)
    x = np.sin(10*t) + np.random.rand(t.shape[-1])
    y = np.sin(10*t) + np.random.rand(t.shape[-1])
    T = ts.TimeSeries(np.vstack([x,y]),sampling_rate=Fs)

    for method in methods:
        C = nta.CoherenceAnalyzer(T,method)
        if method is None:
            # This is the default behavior (NFFT is 64):
            npt.assert_equal(C.coherence.shape,(2,2,33))
        elif method['this_method']=='welch' or method['this_method']=='periodogram_csd':
            npt.assert_equal(C.coherence.shape,(2,2,method['NFFT']//2+1))
        else:
            npt.assert_equal(C.coherence.shape,(2,2,len(t)//2+1))

        # Coherence symmetry:
        npt.assert_equal(C.coherence[0,1],C.coherence[1,0])
        # Phase/delay asymmetry:
        npt.assert_equal(C.phase[0,1],-1*C.phase[1,0])
        npt.assert_equal(C.delay[0,1][1:],-1*C.delay[1,0][1:]) # The very first one
                                                               # is a nan
        if method is not None and method['this_method']=='welch':
            S = nta.SpectralAnalyzer(T,method)
            npt.assert_almost_equal(S.cpsd[0],C.frequencies)
            npt.assert_almost_equal(S.cpsd[1],C.spectrum)

def test_SparseCoherenceAnalyzer():
    Fs = np.pi
    t = np.arange(256)
    x = np.sin(10*t) + np.random.rand(t.shape[-1])
    y = np.sin(10*t) + np.random.rand(t.shape[-1])
    T = ts.TimeSeries(np.vstack([x,y]),sampling_rate=Fs)
    C1 = nta.SparseCoherenceAnalyzer(T,ij=((0,1),(1,0)))

    # Coherence symmetry:
    npt.assert_equal(np.abs(C1.coherence[0,1]),np.abs(C1.coherence[1,0]))

    # Make sure you get the same answers as you would from the standard
    # CoherenceAnalyzer:
    C2 = nta.CoherenceAnalyzer(T)

    yield npt.assert_almost_equal, C2.coherence[0,1],C1.coherence[0,1]
    yield npt.assert_almost_equal, C2.coherence[0,1],C1.coherence[0,1]
