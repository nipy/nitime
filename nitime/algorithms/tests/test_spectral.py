"""
Tests for the algorithms.spectral submodule

""" 

import numpy as np
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
