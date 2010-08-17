import os

import numpy as np
import numpy.testing as npt
from scipy.signal import signaltools

import nitime
from nitime import algorithms as tsa
from nitime import utils as ut

def test_scipy_resample():
    """ Tests scipy signal's resample function
    """
    # create a freq list with max freq < 16 Hz
    freq_list = np.random.randint(0,high=15,size=5)
    # make a test signal with sampling freq = 64 Hz
    a = [np.sin(2*np.pi*f*np.linspace(0,1,64,endpoint=False))
         for f in freq_list]
    tst = np.array(a).sum(axis=0)
    # interpolate to 128 Hz sampling
    t_up = signaltools.resample(tst, 128)
    np.testing.assert_array_almost_equal(t_up[::2], tst)
    # downsample to 32 Hz
    t_dn = signaltools.resample(tst, 32)
    np.testing.assert_array_almost_equal(t_dn, tst[::2])

    # downsample to 48 Hz, and compute the sampling analytically for comparison
    dn_samp_ana = np.array([np.sin(2*np.pi*f*np.linspace(0,1,48,endpoint=False))
                            for f in freq_list]).sum(axis=0)
    t_dn2 = signaltools.resample(tst, 48)
    npt.assert_array_almost_equal(t_dn2, dn_samp_ana)

def test_coherency_mlab():
    """Tests that the coherency algorithm runs smoothly, using the mlab csd
    routine, that the resulting matrix is symmetric and that the frequency bands
    in the output make sense"""
    
    t = np.linspace(0,16*np.pi,1024)
    x = np.sin(t) + np.sin(2*t) + np.sin(3*t) + np.random.rand(t.shape[-1])
    y = x + np.random.rand(t.shape[-1])

    method = {"this_method":'mlab',
              "NFFT":256,
              "Fs":2*np.pi}

    f,c = tsa.coherency(np.vstack([x,y]),csd_method=method)

    npt.assert_array_almost_equal(c[0,1],c[1,0].conjugate())
    npt.assert_array_almost_equal(c[0,0],np.ones(f.shape))
    f_theoretical = ut.get_freqs(method['Fs'],method['NFFT'])
    npt.assert_array_almost_equal(f,f_theoretical)

def test_coherency_multi_taper():
    """Tests that the coherency algorithm runs smoothly, using the multi_taper
    csd routine and that the resulting matrix is symmetric"""
    
    t = np.linspace(0,16*np.pi,1024)
    x = np.sin(t) + np.sin(2*t) + np.sin(3*t) + np.random.rand(t.shape[-1])
    y = x + np.random.rand(t.shape[-1])

    method = {"this_method":'multi_taper_csd',
              "Fs":2*np.pi}

    f,c = tsa.coherency(np.vstack([x,y]),csd_method=method)

    npt.assert_array_almost_equal(c[0,1],c[1,0].conjugate())
    npt.assert_array_almost_equal(c[0,0],np.ones(f.shape))

def test_coherence_mlab():
    """Tests that the code runs and that the resulting matrix is symmetric """  

    t = np.linspace(0,16*np.pi,1024)
    x = np.sin(t) + np.sin(2*t) + np.sin(3*t) + np.random.rand(t.shape[-1])
    y = x + np.random.rand(t.shape[-1])

    method = {"this_method":'mlab',
              "NFFT":256,
              "Fs":2*np.pi}
    
    f,c = tsa.coherence(np.vstack([x,y]),csd_method=method)
    np.testing.assert_array_almost_equal(c[0,1],c[1,0])

    f_theoretical = ut.get_freqs(method['Fs'],method['NFFT'])
    npt.assert_array_almost_equal(f,f_theoretical)

def test_coherence_multi_taper():
    """Tests that the code runs and that the resulting matrix is symmetric """  

    t = np.linspace(0,16*np.pi,1024)
    x = np.sin(t) + np.sin(2*t) + np.sin(3*t) + np.random.rand(t.shape[-1])
    y = x + np.random.rand(t.shape[-1])

    method = {"this_method":'multi_taper_csd',
              "Fs":2*np.pi}
     
    f,c = tsa.coherence(np.vstack([x,y]),csd_method=method)
    npt.assert_array_almost_equal(c[0,1],c[1,0])

def test_coherence_partial():
    """ Test partial coherence"""

    t = np.linspace(0,16*np.pi,1024)
    x = np.sin(t) + np.sin(2*t) + np.sin(3*t) + np.random.rand(t.shape[-1])
    y = x + np.random.rand(t.shape[-1])
    z = x + np.random.rand(t.shape[-1])

    method = {"this_method":'mlab',
              "NFFT":256,
              "Fs":2*np.pi}
    f,c = tsa.coherence_partial(np.vstack([x,y]),z,csd_method=method)

    f_theoretical = ut.get_freqs(method['Fs'],method['NFFT'])
    npt.assert_array_almost_equal(f,f_theoretical)
    npt.assert_array_almost_equal(c[0,1],c[1,0])

    
def test_coherency_cached():
    """Tests that the cached coherency gives the same result as the standard
    coherency"""

    t = np.linspace(0,16*np.pi,1024)
    x = np.sin(t) + np.sin(2*t) + np.sin(3*t) + np.random.rand(t.shape[-1])
    y = x + np.random.rand(t.shape[-1])

    f1,c1 = tsa.coherency(np.vstack([x,y]))

    ij = [(0,1),(1,0)]
    f2,cache = tsa.cache_fft(np.vstack([x,y]),ij)

    c2 = tsa.cache_to_coherency(cache,ij)

    npt.assert_array_almost_equal(c1[1,0],c2[1,0])
    npt.assert_array_almost_equal(c1[0,1],c2[0,1])


# XXX FIXME: http://github.com/nipy/nitime/issues/issue/1
@npt.dec.skipif(True) 
def test_coherence_linear_dependence():
    """
    Tests that the coherence between two linearly dependent time-series
    behaves as expected.
    
    From William Wei's book, according to eq. 14.5.34, if two time-series are
    linearly related through:

    y(t)  = alpha*x(t+time_shift)

    then the coherence between them should be equal to:

    .. :math:
    
    C(\nu) = \frac{1}{1+\frac{fft_{noise}(\nu)}{fft_{x}(\nu) \cdot \alpha^2}}
    
    """
    t = np.linspace(0,16*np.pi,2**14)
    x = np.sin(t) + np.sin(2*t) + np.sin(3*t) + 0.1 *np.random.rand(t.shape[-1])
    N = x.shape[-1]

    alpha = 10
    m = 3
    noise = 0.1 * np.random.randn(t.shape[-1])
    y = alpha*(np.roll(x,m)) + noise

    f_noise = np.fft.fft(noise)[0:N/2]
    f_x = np.fft.fft(x)[0:N/2]

    c_t = ( 1/( 1 + ( f_noise/( f_x*(alpha**2)) ) ) )

    method = {"this_method":'mlab',
              "NFFT":2048,
              "Fs":2*np.pi}

    f,c = tsa.coherence(np.vstack([x,y]),csd_method=method)
    c_t = np.abs(signaltools.resample(c_t,c.shape[-1]))

    npt.assert_array_almost_equal(c[0,1],c_t,2)
    
@npt.dec.skipif(True)
def test_coherence_phase_spectrum ():
    assert False, "Test Not Implemented"

@npt.dec.skipif(True)
def test_coherency_bavg():
    assert False, "Test Not Implemented"

@npt.dec.skipif(True)
def test_coherence_partial():
    assert False, "Test Not Implemented"

@npt.dec.skipif(True)
def test_coherence_partial_bavg():
    assert False, "Test Not Implemented"

#XXX def test_coherency_phase ()
#XXX def test_coherence_partial_phase()

@npt.dec.skipif(True)
def test_fir():
    assert False, "Test Not Implemented"

@npt.dec.skipif(True)
def test_percent_change():
    assert False, "Test Not Implemented"

def test_DPSS_windows():
    "Are the eigenvalues representing spectral concentration near unity"
    # these values from Percival and Walden 1993
    _, l = tsa.DPSS_windows(31, 6, 4)
    unos = np.ones(4)
    yield npt.assert_array_almost_equal, l, unos 
    _, l = tsa.DPSS_windows(31, 7, 4)
    yield npt.assert_array_almost_equal, l, unos 
    _, l = tsa.DPSS_windows(31, 8, 4)
    yield npt.assert_array_almost_equal, l, unos
                
def test_yule_walker_AR():
    arsig,_,_ = ut.ar_generator(N=512)
    avg_pwr = (arsig*arsig.conjugate()).mean()
    w, psd = tsa.yule_AR_est(arsig, 8, 1024)
    # for efficiency, let's leave out the 2PI in the numerator and denominator
    # for the following integral
    dw = 1./1024
    avg_pwr_est = np.trapz(psd, dx=dw)
    npt.assert_almost_equal(avg_pwr, avg_pwr_est, decimal=0)

def test_LD_AR():
    arsig,_,_ = ut.ar_generator(N=512)
    avg_pwr = (arsig*arsig.conjugate()).mean()
    w, psd = tsa.LD_AR_est(arsig, 8, 1024)
    # for efficiency, let's leave out the 2PI in the numerator and denominator
    # for the following integral
    dw = 1./1024
    avg_pwr_est = np.trapz(psd, dx=dw)
    npt.assert_almost_equal(avg_pwr, avg_pwr_est, decimal=0)
    
def test_periodogram():
    arsig,_,_ = ut.ar_generator(N=512)
    avg_pwr = (arsig*arsig.conjugate()).mean()
    f, psd = tsa.periodogram(arsig, N=2048)
    # for efficiency, let's leave out the 2PI in the numerator and denominator
    # for the following integral
    dw = 1./2048
    avg_pwr_est = np.trapz(psd, dx=dw)
    npt.assert_almost_equal(avg_pwr, avg_pwr_est, decimal=1)
    
def permutation_system(N):
    p = np.zeros((N,N))
    targets = range(N)
    for i in xrange(N):
        popper = np.random.randint(0, high=len(targets))
        j = targets.pop(popper)
        p[i,j] = 1
    return p

def test_boxcar_filter():
    a = np.random.rand(100)
    b = tsa.boxcar_filter(a)
    npt.assert_equal(a,b)

    #Should also work for odd number of elements:
    a = np.random.rand(99)
    b = tsa.boxcar_filter(a)
    npt.assert_equal(a,b)

    b = tsa.boxcar_filter(a,ub=0.25)
    npt.assert_equal(a.shape,b.shape)

    b = tsa.boxcar_filter(a,lb=0.25)
    npt.assert_equal(a.shape,b.shape)

def test_get_spectra():
    """Testing get_spectra"""
    t = np.linspace(0,16*np.pi,2**14)
    x = np.sin(t) + np.sin(2*t) + np.sin(3*t) + 0.1 *np.random.rand(t.shape[-1])
    x = np.reshape(x,(2,x.shape[-1]/2))
    N = x.shape[-1]

    #Make sure you get back the expected shape for different spectra: 
    NFFT = 64
    f_mlab=tsa.get_spectra(x,method={'this_method':'mlab','NFFT':NFFT})
    f_periodogram=tsa.get_spectra(x,method={'this_method':'periodogram_csd'})
    f_multi_taper=tsa.get_spectra(x,method={'this_method':'multi_taper_csd'})

    npt.assert_equal(f_mlab[0].shape[0],NFFT/2+1)
    npt.assert_equal(f_periodogram[0].shape[0],N/2+1)
    npt.assert_equal(f_multi_taper[0].shape[0],N/2+1)

def test_psd_matlab():

    """ Test the results of mlab csd/psd against saved results from Matlab"""

    from matplotlib import mlab

    test_dir_path = os.path.join(nitime.__path__[0],'tests')
    
    ts = np.loadtxt(os.path.join(test_dir_path,'tseries12.txt'))
    
    #Complex signal! 
    ts0 = ts[1] + ts[0]*np.complex(0,1) 

    NFFT = 256;
    Fs = 1.0;
    noverlap = NFFT/2

    fxx, f = mlab.psd(ts0,NFFT=NFFT,Fs=Fs,noverlap=noverlap,
                      scale_by_freq=True)

    fxx_mlab = np.fft.fftshift(fxx).squeeze()

    fxx_matlab = np.loadtxt(os.path.join(test_dir_path,'fxx_matlab.txt'))

    npt.assert_almost_equal(fxx_mlab,fxx_matlab,decimal=5)

def test_coherence_matlab():

    """ Test against coherence values calculated with matlab's mscohere"""
    test_dir_path = os.path.join(nitime.__path__[0],'tests')

    ts = np.loadtxt(os.path.join(test_dir_path,'tseries12.txt'))

    ts0 = ts[1]   
    ts1 = ts[0]  

    method = {}
    method['this_method']='mlab'
    method['NFFT'] = 64;
    method['Fs'] = 1.0;
    method['noverlap'] = method['NFFT']/2

    ttt = np.vstack([ts0,ts1])
    f,cxy_mlab = tsa.coherence(ttt,csd_method=method)
    cxy_matlab = np.loadtxt(os.path.join(test_dir_path,'cxy_matlab.txt'))

    npt.assert_almost_equal(cxy_mlab[0][1],cxy_matlab,decimal=5)
