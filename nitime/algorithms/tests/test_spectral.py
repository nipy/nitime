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

    # But that's corrected by the tsaorithm: 
    d,w=tsa.DPSS_windows(1024, 0, 7)
    for this_d in d[0::2]:
        npt.assert_equal(this_d.sum(axis=-1)< 0, False)
    
def test_get_spectra_bi():
    """

    Test the bi-variate get_spectra function

    """ 

    methods = (None,
           {"this_method":'welch',"NFFT":256,"Fs":2*np.pi},
           {"this_method":'welch',"NFFT":1024,"Fs":2*np.pi})

    for method in methods:
        arsig1,_,_ = utils.ar_generator(N=2**16)
        arsig2,_,_ = utils.ar_generator(N=2**16)

        avg_pwr1=(arsig1**2).mean()
        avg_pwr2=(arsig2**2).mean()
        avg_xpwr=(arsig1*arsig2.conjugate()).mean()

        tseries = np.vstack([arsig1,arsig2])

        f,fxx,fyy,fxy = tsa.get_spectra_bi(arsig1,arsig2,method=method)

        # \sum_{\omega} PSD(\omega) d\omega:
        est_pwr1=np.sum(fxx*(f[1]-f[0]))
        est_pwr2=np.sum(fyy*(f[1]-f[0]))
        est_xpwr=np.sum(fxy*(f[1]-f[0])).real

        # Test that we have the right order of magnitude:
        npt.assert_array_almost_equal(est_pwr1,avg_pwr1,decimal=-1)
        npt.assert_array_almost_equal(est_pwr2,avg_pwr2,decimal=-1)
        npt.assert_array_almost_equal(np.mean(est_xpwr),np.mean(avg_xpwr),decimal=-1)

def test_mtm_cross_spectrum():
    """
    
    Test the multi-taper cross-spectral estimation. Based on the example in
    doc/examples/multi_taper_coh.py

    """ 
    NW = 4
    K = 2*NW-1

    N = 2**10
    n_reps = 10
    Nfreqs=N

    tapers, eigs = tsa.DPSS_windows(N, NW, 2*NW-1)

    est_psd = []
    for k in xrange(n_reps):
        data,nz,alpha = utils.ar_generator(N=N)
        fgrid, hz = tsa.freq_response(1.0, a=np.r_[1, -alpha], Nfreqs=Nfreqs)
        # 'one-sided', so multiply by 2:
        psd = 2*(hz*hz.conj()).real

        tdata = tapers * data

        tspectra = np.fft.fft(tdata)
        mag_sqr_spectra = np.abs(tspectra)
        np.power(mag_sqr_spectra, 2, mag_sqr_spectra)

        L = N/2 + 1
        w, _ = utils.adaptive_weights(mag_sqr_spectra, eigs, L)

        sxx = tsa.mtm_cross_spectrum(tspectra, tspectra, (w, w), sides='onesided').real
        est_psd.append(sxx)

    fxx = np.mean(est_psd,0)

    psd_ratio = np.mean(fxx/psd)

    # This is a rather lenient test, making sure that the average ratio is 1 to
    # within an order of magnitude. That is, that they are equal on average:
    npt.assert_array_almost_equal(psd_ratio,1,decimal=1)

    # Test raising of error in case the inputs don't make sense:
    npt.assert_raises(ValueError,tsa.mtm_cross_spectrum,tspectra,np.r_[tspectra,tspectra],
                                                        (w,w))
    
def test_multi_taper_psd_csd():
    """

    Test the multi taper psd and csd estimation functions. Based on the example in
    doc/examples/multi_taper_spectral_estimation.py 
    
    """ 

    N = 2**10
    n_reps = 10


    psd= []
    est_psd = []
    est_csd = []
    for k in xrange(n_reps):
        ar_seq, nz, alpha = utils.ar_generator(N=N, drop_transients=10)
        ar_seq -= ar_seq.mean()
        fgrid, hz = tsa.freq_response(1.0, a=np.r_[1, -alpha], Nfreqs=N)
        psd.append(2*(hz*hz.conj()).real)
        f, psd_mt, nu = tsa.multi_taper_psd(ar_seq, adaptive=True, jackknife=True)
        est_psd.append(psd_mt)
        f, csd_mt= tsa.multi_taper_csd(np.vstack([ar_seq,ar_seq]), adaptive=True)
        # Symmetrical in this case, so take one element out:
        est_csd.append(csd_mt[0][1])

    fxx = np.mean(psd,0)
    fxx_est1= np.mean(est_psd,0)
    fxx_est2= np.mean(est_csd,0)


    # Tests the psd:
    psd_ratio1 = np.mean(fxx_est1/fxx)
    npt.assert_array_almost_equal(psd_ratio1,1,decimal=1)
    # Tests the csd:
    psd_ratio2 = np.mean(fxx_est2/fxx)
    npt.assert_array_almost_equal(psd_ratio2,1,decimal=1)
