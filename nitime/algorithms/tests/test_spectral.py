"""
Tests for the algorithms.spectral submodule

""" 

import numpy as np
import scipy
import numpy.testing as npt


import nitime
import nitime.algorithms as tsa
import nitime.utils as utils


def test_get_spectra():
    """

    Testing spectral estimation

    """

    methods = (None,
           {"this_method":'welch',"NFFT":256,"Fs":2*np.pi},
           {"this_method":'welch',"NFFT":1024,"Fs":2*np.pi})


    for method in methods:
        avg_pwr1 = []
        avg_pwr2 = []
        est_pwr1 = []
        est_pwr2 = []
        arsig1,_,_ = utils.ar_generator(N=2**16) # It needs to be that long for
                                        # the answers to converge 
        arsig2,_,_ = utils.ar_generator(N=2**16)
        
        avg_pwr1.append((arsig1**2).mean())
        avg_pwr2.append((arsig2**2).mean())
    
        tseries = np.vstack([arsig1,arsig2])
        
        f,c = tsa.get_spectra(tseries,method=method)

        # \sum_{\omega} psd d\omega:
        est_pwr1.append(np.sum(c[0,0])*(f[1]-f[0]))
        est_pwr2.append(np.sum(c[1,1])*(f[1]-f[0]))

        # Get it right within the order of magnitude:
        npt.assert_array_almost_equal(est_pwr1,avg_pwr1,decimal=-1)
        npt.assert_array_almost_equal(est_pwr2,avg_pwr2,decimal=-1)
        
def test_get_spectra_complex():
    """

    Testing spectral estimation

    """

    methods = (None,
           {"this_method":'welch',"NFFT":256,"Fs":2*np.pi},
           {"this_method":'welch',"NFFT":1024,"Fs":2*np.pi})


    for method in methods:
        avg_pwr1 = []
        avg_pwr2 = []
        est_pwr1 = []
        est_pwr2 = []

        # Make complex signals:
        r,_,_ = utils.ar_generator(N=2**16) # It needs to be that long for
                                        # the answers to converge
        c,_,_ = utils.ar_generator(N=2**16)
        arsig1 = r + c * scipy.sqrt(-1)

        r,_,_ = utils.ar_generator(N=2**16) 
        c,_,_ = utils.ar_generator(N=2**16)
        
        arsig2 = r + c * scipy.sqrt(-1)
        avg_pwr1.append((arsig1*arsig1.conjugate()).mean())
        avg_pwr2.append((arsig2*arsig2.conjugate()).mean())
    
        tseries = np.vstack([arsig1,arsig2])
        
        f,c = tsa.get_spectra(tseries,method=method)

        # \sum_{\omega} psd d\omega:
        est_pwr1.append(np.sum(c[0,0])*(f[1]-f[0]))
        est_pwr2.append(np.sum(c[1,1])*(f[1]-f[0]))

        # Get it right within the order of magnitude:
        npt.assert_array_almost_equal(est_pwr1,avg_pwr1,decimal=-1)
        npt.assert_array_almost_equal(est_pwr2,avg_pwr2,decimal=-1)

def test_get_spectra_unknown_method():
    """
    Test that providing an unknown method to get_spectra rasies a ValueError

    """ 
    tseries = np.array([[1,2,3],[4,5,6]])
    npt.assert_raises(ValueError,
                            tsa.get_spectra,tseries,method=dict(this_method='foo'))

def test_periodogram():
    """Test some of the inputs to periodogram """

    arsig,_,_ = utils.ar_generator(N=1024)
    Sk = np.fft.fft(arsig)

    f1,c1 = tsa.periodogram(arsig)
    f2,c2 = tsa.periodogram(arsig,Sk=Sk)

    npt.assert_equal(c1,c2)

    # Check that providing a complex signal does the right thing
    # (i.e. two-sided spectrum): 
    N = 1024 
    r,_,_ = utils.ar_generator(N=N) 
    c,_,_ = utils.ar_generator(N=N)
    arsig = r + c * scipy.sqrt(-1)

    f,c = tsa.periodogram(arsig)
    npt.assert_equal(f.shape[0],N) # Should be N, not the one-sided N/2 + 1
    

def test_periodogram_csd():
    """Test corner cases of  periodogram_csd"""

    arsig1,_,_ = utils.ar_generator(N=1024)
    arsig2,_,_ = utils.ar_generator(N=1024)

    tseries = np.vstack([arsig1,arsig2])
    
    Sk = np.fft.fft(tseries)

    f1,c1 = tsa.periodogram_csd(tseries)
    f2,c2 = tsa.periodogram_csd(tseries,Sk=Sk)
    npt.assert_equal(c1,c2)

    # Check that providing a complex signal does the right thing
    # (i.e. two-sided spectrum): 
    N = 1024 
    r,_,_ = utils.ar_generator(N=N) 
    c,_,_ = utils.ar_generator(N=N)
    arsig1 = r + c * scipy.sqrt(-1)

    r,_,_ = utils.ar_generator(N=N) 
    c,_,_ = utils.ar_generator(N=N)
    arsig2 = r + c * scipy.sqrt(-1)

    tseries = np.vstack([arsig1,arsig2])

    f,c = tsa.periodogram_csd(tseries)
    npt.assert_equal(f.shape[0],N) # Should be N, not the one-sided N/2 + 1

def test_DPSS_windows():
    """ Test a funky corner case of DPSS_windows """  

    N = 1024
    NW = 0 # Setting NW to 0 triggers the weird corner case in which some of
           # the symmetric tapers have a negative average
    Kmax = 7

    # But that's corrected by the algorithm: 
    d,w=tsa.DPSS_windows(1024, 0, 7)
    for this_d in d[0::2]:
        npt.assert_equal(this_d.sum(axis=-1)< 0, False)

# XXX Test line 474:
#def test_mtm_cross_spectrum():
#    """ """ 

    
def test_get_spectra_bi():
    """

    Test the bi-variate get_spectra function

    """ 

    methods = (None,
           {"this_method":'welch',"NFFT":256,"Fs":2*np.pi},
           {"this_method":'welch',"NFFT":1024,"Fs":2*np.pi})

    for method in methods:
        avg_pwr1 = []
        avg_pwr2 = []
        avg_xpwr = []
        est_pwr1 = []
        est_pwr2 = []
        est_xpwr = []
        arsig1,_,_ = utils.ar_generator(N=2**16)
        arsig2,_,_ = utils.ar_generator(N=2**16)

        avg_pwr1.append((arsig1**2).mean())
        avg_pwr2.append((arsig2**2).mean())
        avg_xpwr.append((arsig1*arsig2.conjugate()).mean())

        tseries = np.vstack([arsig1,arsig2])

        f,fxx,fyy,fxy = tsa.get_spectra_bi(arsig1,arsig2,method=method)

        # \sum_{\omega} PSD(\omega) d\omega:
        est_pwr1.append(np.sum(fxx*(f[1]-f[0])))
        est_pwr2.append(np.sum(fyy*(f[1]-f[0])))
        est_xpwr.append(np.sum(fxy*(f[1]-f[0])).real)
            
        npt.assert_array_almost_equal(est_pwr1,avg_pwr1,decimal=-1)
        npt.assert_array_almost_equal(est_pwr2,avg_pwr2,decimal=-1)
        npt.assert_array_almost_equal(np.mean(est_xpwr),np.mean(avg_xpwr),decimal=-1)
