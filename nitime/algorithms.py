"""
This module contains implementations of algorithms for time series
analysis. These algorithms include:

1. Spectral estimation: calculate the spectra of time-series and cross-spectra
between time-series.

:func:`get_spectra`, :func:`get_spectra_bi`, :func:`periodogram`,
:func:`periodogram_csd`, :func:`DPSS_windows`, :func:`multi_taper_psd`,
:func:`multi_taper_csd`, :func:`mtm_cross_spectrum`

2. Coherency: calculate the pairwise correlation between time-series in the
frequency domain and related quantities.

:func:`coherency`, :func:`coherence`, :func:`coherence_regularized`,
:func:`coherency_regularized`, :func:`coherency_bavg`, :func:`coherence_bavg`,
:func:`coherence_partial`, :func:`coherence_partial_bavg`, 
:func:`coherency_phase_spectrum`, :func:`coherency_phase_delay`,
:func:`coherency_phase_delay_bavg`, :func:`correlation_spectrum`

3. Event-related analysis: calculate the correlation between time-series and
external events.

:func:`freq_domain_xcorr`, :func:`freq_domain_xcorr_zscored`, :func:`fir`

4. Cached coherency: A set of special functions for quickly calculating
coherency in large data-sets, where the calculation is done over only a subset
of the adjacency matrix edges and intermediate calculations are cached, in
order to save calculation time.

:func:`cache_fft`, :func:`cache_to_psd`, :func:`cache_to_phase`,
:func:`cache_to_relative_phase`, :func:`cache_to_coherency`.

5. Wavelet transforms: Calculate wavelet transforms of time-series data.

:func:`wmorlet`, :func:`wfmorlet_fft`, :func:`wlogmorlet`,
:func:`wlogmorlet_fft`

6. Filtering: Filter a signal in the frequency domain.

:func:`boxcar_filter`

The algorithms in this library are the functional form of the algorithms, which
accept as inputs numpy array and produce numpy array outputs. Therfore, they
can be used on any type of data which can be represented in numpy arrays. See
also :mod:`nitime.analysis` for simplified analysis interfaces, using the
data containers implemented in :mod:`nitime.timeseries`

"""

#import packages:
import numpy as np
from scipy import signal 
from scipy import stats
from matplotlib import mlab
from scipy import linalg
import utils as ut
from scipy.misc import factorial
import nitime.utils as utils

#-----------------------------------------------------------------------------
#  Coherency 
#-----------------------------------------------------------------------------

def coherency(time_series,csd_method= None):
    r"""
    Compute the coherency between the spectra of n-tuple of time series.
    Input to this function is in the time domain

    Parameters
    ----------

    time_series: n*t float array
       an array of n different time series of length t each

    csd_method: dict, optional.
       See :func:`get_spectra` documentation for details

    Returns
    -------
    
    f : float array
        The central frequencies for the frequency bands for which the spectra
        are estimated 

    c : float array
        This is a symmetric matrix with the coherencys of the signals. The
        coherency of signal i and signal j is in f[i][j]. Note that f[i][j] =
        f[j][i].conj() 

    Notes
    -----
    
    This is an implementation of equation (1) of [Sun2005]_: 

    .. math::

        R_{xy} (\lambda) = \frac{f_{xy}(\lambda)}
        {\sqrt{f_{xx} (\lambda) \cdot f_{yy}(\lambda)}}

    .. [Sun2005] F.T. Sun and L.M. Miller and M. D'Esposito(2005). Measuring
        temporal dynamics of functional networks using phase spectrum of fMRI
        data. Neuroimage, 28: 227-37.

    """
    if csd_method is None:
        csd_method = {'this_method':'welch'} #The default
  
    f,fxy = get_spectra(time_series,csd_method)

    #A container for the coherencys, with the size and shape of the expected
    #output:
    c=np.zeros((time_series.shape[0],
               time_series.shape[0],
               f.shape[0]), dtype = complex) #Make sure it's complex
    
    for i in xrange(time_series.shape[0]): 
        for j in xrange(i,time_series.shape[0]):
            c[i][j] = coherency_calculate(fxy[i][j], fxy[i][i], fxy[j][j])  

    idx = ut.tril_indices(time_series.shape[0],-1)
    c[idx[0],idx[1],...] = c[idx[1],idx[0],...].conj() #Make it symmetric
    
    return f,c 

def coherency_calculate(fxy, fxx, fyy): 
    r"""
    Compute the coherency between the spectra of two time series. 

    Input to this function is in the frequency domain.
    
    Parameters
    ----------
    
    fxy : float array
         The cross-spectrum of the time series 
    
    fyy,fxx : float array
         The spectra of the signals
    
    Returns 
    -------
    
    complex array 
        the frequency-band-dependent coherency

    See also
    --------
    :func:`coherency`
    """

    return fxy / np.sqrt(fxx*fyy)

def coherence(time_series, csd_method=None):
    r"""Compute the coherence between the spectra of an n-tuple of time_series.

    Parameters of this function are in the time domain.

    Parameters
    ----------
    time_series: float array
       an array of different time series with time as the last dimension

    csd_method: dict, optional
       See :func:`get_spectra` documentation for details

    Returns
    -------
    f : float array
        The central frequencies for the frequency bands for which the spectra are
        estimated 

    c : float array
        This is a symmetric matrix with the coherencys of the signals. The
        coherency of signal i and signal j is in f[i][j].  
    
    Notes
    -----
    
    This is an implementation of equation (2) of [Sun2005]_:

    .. math::

        Coh_{xy}(\lambda) = |{R_{xy}(\lambda)}|^2 = 
        \frac{|{f_{xy}(\lambda)}|^2}{f_{xx}(\lambda) \cdot f_{yy}(\lambda)}

    .. [Sun2005] F.T. Sun and L.M. Miller and M. D'Esposito(2005). Measuring
        temporal dynamics of functional networks using phase spectrum of fMRI
        data.  Neuroimage, 28: 227-37.

    See also
    --------
    :func:`coherence_calculate`

    """
    if csd_method is None:
        csd_method = {'this_method':'welch'} #The default

    f,fxy = get_spectra(time_series,csd_method)

    #A container for the coherences, with the size and shape of the expected
    #output:
    c=np.zeros((time_series.shape[0],
               time_series.shape[0],
               f.shape[0]))

    for i in xrange(time_series.shape[0]):
        for j in xrange(i,time_series.shape[0]):
            c[i][j] = coherence_calculate(fxy[i][j], fxy[i][i], fxy[j][j])

    idx = ut.tril_indices(time_series.shape[0],-1)
    c[idx[0],idx[1],...] = c[idx[1],idx[0],...].conj() #Make it symmetric

    return f,c

def coherence_calculate(fxy, fxx, fyy):
    r"""
    Compute the coherence between the spectra of two time series. 

    Parameters of this function are in the frequency domain.

    Parameters
    ----------
    
    fxy : array
         The cross-spectrum of the time series 

    fyy,fxx : array
         The spectra of the signals 
    
    Returns 
    -------
    
    float : a frequency-band-dependent measure of the linear association
        between the two time series
         
    See also
    --------
    :func:`coherence`
         
    """

    c = (np.abs(fxy))**2 / (fxx * fyy)

    return c  

def coherency_regularized(time_series,epsilon,alpha,csd_method=None):
    r"""
    Compute a regularized measure of the coherence.

    Regularization may be needed in order to overcome numerical imprecisions

    Parameters
    ----------
    
    time_series: float array
        The time series data for which the regularized coherence is
        calculated. Time as the last dimension.  

    epsilon: float
        Small regularization parameter. Should be much smaller than any
        meaningful value of coherence you might encounter

    alpha: float
        Large regularization parameter. Should be much larger than any meaningful
        value of coherence you might encounter (preferably much larger than 1).

    csd_method: dict, optional.
        See :func:`get_spectra` documentation for details

    Returns
    -------
    f: float array
        The central frequencies for the frequency bands for which the spectra
        are estimated 

    c: float array
        This is a symmetric matrix with the coherencys of the signals. The
        coherency of signal i and signal j is in f[i][j]. Note that f[i][j] =
        f[j][i].conj() 


    Notes
    -----
    The regularization scheme is as follows:

    .. math::

        Coh_{xy}^R = \frac{(\alpha f_{xx} + \epsilon) ^2}{\alpha^{2}(f_{xx}+\epsilon)(f_{yy}+\epsilon)}


    """
    if csd_method is None:
        csd_method = {'this_method':'welch'} #The default

    f,fxy = get_spectra(time_series,csd_method)

    #A container for the coherences, with the size and shape of the expected
    #output:
    c=np.zeros((time_series.shape[0],
               time_series.shape[0],
               f.shape[0]), dtype = complex)  #Make sure it's complex
    
    for i in xrange(time_series.shape[0]): 
        for j in xrange(i,time_series.shape[0]):
            c[i][j] = coherency_reqularized_calculate(fxy[i][j], fxy[i][i],
                                                      fxy[j][j], epsilon, alpha)

    idx = ut.tril_indices(time_series.shape[0],-1)
    c[idx[0],idx[1],...] = c[idx[1],idx[0],...].conj() #Make it symmetric

    return f,c 

def coherency_reqularized_calculate(fxy, fxx, fyy, epsilon, alpha):

    r"""
    A regularized version of the calculation of coherency, which is more
    robust to numerical noise than the standard calculation

    Input to this function is in the frequency domain.

    Parameters
    ----------

    fxy, fxx, fyy: float arrays
        The cross- and power-spectral densities of the two signals x and y

    epsilon: float
        First regularization parameter. Should be much smaller than any
        meaningful value of coherence you might encounter

    alpha: float
        Second regularization parameter. Should be much larger than any
        meaningful value of coherence you might encounter (preferably much
        larger than 1). 

    Returns
    -------
    float array
        The coherence values

    """
    
    return ( ( (alpha*fxy + epsilon) ) /
         np.sqrt( ((alpha**2) * (fxx+epsilon) * (fyy + epsilon) ) ) )

def coherence_regularized(time_series,epsilon,alpha,csd_method=None):
    r"""
    Same as coherence, except regularized in order to overcome numerical
    imprecisions

    Parameters
    ----------
    
    time_series: n-d float array
       The time series data for which the regularized coherence is calculated 

    epsilon: float
       Small regularization parameter. Should be much smaller than any
       meaningful value of coherence you might encounter 

    alpha: float
       large regularization parameter. Should be much larger than any meaningful
       value of coherence you might encounter (preferably much larger than 1).

    csd_method: dict, optional.
       See :func:`get_spectra` documentation for details

    Returns
    -------
    f: float array
       The central frequencies for the frequency bands for which the spectra
       are estimated 

    c: n-d array
       This is a symmetric matrix with the coherencys of the signals. The
       coherency of signal i and signal j is in f[i][j]. 
    
    Returns
    -------
    frequencies, coherence

    Notes
    -----
    The regularization scheme is as follows:

    .. math::
    
        C_{x,y} = \frac{(\alpha f_{xx} + \epsilon)^2}{\alpha^{2}((f_{xx}+\epsilon)(f_{yy}+\epsilon))}
        
    """
    if csd_method is None:
        csd_method = {'this_method':'welch'} #The default

    f,fxy = get_spectra(time_series,csd_method)

    #A container for the coherences, with the size and shape of the expected
    #output:
    c=np.zeros((time_series.shape[0],
               time_series.shape[0],
               f.shape[0]))
    
    for i in xrange(time_series.shape[0]): 
        for j in xrange(i,time_series.shape[0]):
            c[i][j] = coherence_reqularized_calculate(fxy[i][j], fxy[i][i],
                                                      fxy[j][j], epsilon, alpha)

    idx = ut.tril_indices(time_series.shape[0],-1)
    c[idx[0],idx[1],...] = c[idx[1],idx[0],...].conj() #Make it symmetric

    return f,c 


def coherence_reqularized_calculate(fxy, fxx, fyy, epsilon, alpha):

    r"""A regularized version of the calculation of coherence, which is more
    robust to numerical noise than the standard calculation. 

    Input to this function is in the frequency domain

    Parameters
    ----------

    fxy, fxx, fyy: float arrays
        The cross- and power-spectral densities of the two signals x and y

    epsilon: float
        First regularization parameter. Should be much smaller than any
        meaningful value of coherence you might encounter

    alpha: float
        Second regularization parameter. Should be much larger than any
        meaningful value of coherence you might encounter (preferably much
        larger than 1) 

    Returns
    -------
    float array
       The coherence values

"""
    
    return ( ( (alpha*np.abs(fxy) + epsilon)**2 ) /
         ((alpha**2) * (fxx+epsilon) * (fyy + epsilon) ) )

def coherency_bavg(time_series,lb=0,ub=None,csd_method=None):
    r"""
    Compute the band-averaged coherency between the spectra of two time series. 

    Input to this function is in the time domain.

    Parameters
    ----------
    time_series: n*t float array
       an array of n different time series of length t each

    lb, ub: float, optional
       the upper and lower bound on the frequency band to be used in averaging
       defaults to 1,max(f)

    csd_method: dict, optional.
       See :func:`get_spectra` documentation for details

    Returns 
    -------
    c: float array
        This is an upper-diagonal array, where c[i][j] is the band-averaged
        coherency between time_series[i] and time_series[j]
    
    Notes
    -----
    
    This is an implementation of equation (A4) of [Sun2005]_: 

    .. math::

        \bar{Coh_{xy}} (\bar{\lambda}) =
        \frac{\left|{\sum_\lambda{\hat{f_{xy}}}}\right|^2}
        {\sum_\lambda{\hat{f_{xx}}}\cdot sum_\lambda{\hat{f_{yy}}}} 

    .. [Sun2005] F.T. Sun and L.M. Miller and M. D'Esposito(2005). Measuring
        temporal dynamics of functional networks using phase spectrum of fMRI
        data. Neuroimage, 28: 227-37.
        
    """
    if csd_method is None:
        csd_method = {'this_method':'welch'} #The default

    f,fxy = get_spectra(time_series,csd_method)

    lb_idx,ub_idx = ut.get_bounds(f,lb,ub)

    if lb==0:
        lb_idx = 1 #The lowest frequency band should be f0

    c = np.zeros((time_series.shape[0],
               time_series.shape[0]), dtype = complex)
    
    for i in xrange(time_series.shape[0]): 
        for j in xrange(i,time_series.shape[0]):
            c[i][j] = coherency_bavg_calculate(fxy[i][j][lb_idx:ub_idx],
                                               fxy[i][i][lb_idx:ub_idx],
                                               fxy[j][j][lb_idx:ub_idx])

    idx = ut.tril_indices(time_series.shape[0],-1)
    c[idx[0],idx[1],...] = c[idx[1],idx[0],...].conj() #Make it symmetric

    return c

def coherency_bavg_calculate(fxy, fxx, fyy): 
    r"""
    Compute the band-averaged coherency between the spectra of two time series. 

    Input to this function is in the frequency domain.

    Parameters
    ----------
    
    fxy : float array
         The cross-spectrum of the time series 
    
    fyy,fxx : float array
         The spectra of the signals
 
    Returns 
    -------
    
    float
        the band-averaged coherency

    Notes
    -----
    
    This is an implementation of equation (A4) of [Sun2005]_: 

    .. math::

        \bar{Coh_{xy}} (\bar{\lambda}) =
        \frac{\left|{\sum_\lambda{\hat{f_{xy}}}}\right|^2}
        {\sum_\lambda{\hat{f_{xx}}}\cdot sum_\lambda{\hat{f_{yy}}}} 

    .. [Sun2005] F.T. Sun and L.M. Miller and M. D'Esposito(2005). Measuring
        temporal dynamics of functional networks using phase spectrum of fMRI
        data. Neuroimage, 28: 227-37.
    """

    #Average the phases and the magnitudes separately and then
    #recombine:

    p = coherency_phase_spectrum_calculate(fxy) 
    p_bavg = np.mean(p)

    m = np.abs(coherency_calculate(fxy,fxx,fyy))
    m_bavg = np.mean(m)

    return  m_bavg * (np.cos(p_bavg) + np.sin(p_bavg) *1j) #recombine
                                        #according to z = r(cos(phi)+sin(phi)i)

def coherence_bavg (time_series,lb=0,ub=None,csd_method=None):
    r"""
    Compute the band-averaged coherence between the spectra of two time series. 

    Input to this function is in the time domain.

    Parameters
    ----------
    time_series : float array
       An array of time series, time as the last dimension.

    lb, ub: float, optional
       The upper and lower bound on the frequency band to be used in averaging
       defaults to 1,max(f)

    csd_method: dict, optional.
       See :func:`get_spectra` documentation for details

    Returns 
    -------
    c : float 
       This is an upper-diagonal array, where c[i][j] is the band-averaged
       coherency between time_series[i] and time_series[j]
    """

    if csd_method is None:
        csd_method = {'this_method':'welch'} #The default

    f,fxy = get_spectra(time_series,csd_method)
    
    lb_idx,ub_idx = ut.get_bounds(f,lb,ub)

    if lb==0:
        lb_idx = 1 #The lowest frequency band should be f0

    c = np.zeros((time_series.shape[0],
                time_series.shape[0]))
    
    for i in xrange(time_series.shape[0]): 
        for j in xrange(i,time_series.shape[0]):
            c[i][j] = coherence_bavg_calculate(fxy[i][j][lb_idx:ub_idx],
                                               fxy[i][i][lb_idx:ub_idx],
                                               fxy[j][j][lb_idx:ub_idx])

    idx = ut.tril_indices(time_series.shape[0],-1)
    c[idx[0],idx[1],...] = c[idx[1],idx[0],...].conj() #Make it symmetric

    return c

def coherence_bavg_calculate(fxy, fxx, fyy):
    r"""
    Compute the band-averaged coherency between the spectra of two time series. 
    input to this function is in the frequency domain

    Parameters
    ----------
    
    fxy : float array
         The cross-spectrum of the time series 
    
    fyy,fxx : float array
         The spectra of the signals
 
    Returns 
    -------
    
    float :
        the band-averaged coherence
    """

    return ( ( np.abs( fxy.sum() )**2 ) /
             ( fxx.sum() * fyy.sum() ) )

def coherence_partial(time_series,r,csd_method=None):
    r"""
    Compute the band-specific partial coherence between the spectra of
    two time series.

    The partial coherence is the part of the coherence between x and
    y, which cannot be attributed to a common cause, r. 
    
    Input to this function is in the time domain.

    Parameters
    ----------

    time_series: float array
       An array of time-series, with time as the last dimension.
    
    r: float array
        This array represents the temporal sequence of the common cause to be
        partialed out, sampled at the same rate as time_series

    csd_method: dict, optional
       See :func:`get_spectra` documentation for details


    Returns 
    -------
    f: array,
        The mid-frequencies of the frequency bands in the spectral decomposition
    c: float array
       The frequency dependent partial coherence between time_series i and
       time_series j in c[i][j] and in c[j][i], with r partialed out 
     

    Notes
    -----
    
    This is an implementation of equation (2) of [Sun2004]_: 

    .. math::

        Coh_{xy|r} = \frac{|{R_{xy}(\lambda) - R_{xr}(\lambda)
        R_{ry}(\lambda)}|^2}{(1-|{R_{xr}}|^2)(1-|{R_{ry}}|^2)}

    .. [Sun2004] F.T. Sun and L.M. Miller and M. D'Esposito(2004). Measuring
    interregional functional connectivity using coherence and partial coherence
    analyses of fMRI data Neuroimage, 21: 647-58.
    """
    
    if csd_method is None:
        csd_method = {'this_method':'welch'} #The default

    f,fxy = get_spectra(time_series,csd_method)

    #Initialize c according to the size of f:
    
    c=np.zeros((time_series.shape[0],
                time_series.shape[0],
                f.shape[0]), dtype = complex)       

    for i in xrange(time_series.shape[0]): 
        for j in xrange(i,time_series.shape[0]):
            f,fxx,frr,frx = get_spectra_bi(time_series[i],r,csd_method)
            f,fyy,frr,fry = get_spectra_bi(time_series[j],r,csd_method)
            c[i,j] = coherence_partial_calculate(fxy[i][j],fxy[i][i],fxy[j][j],
                                        frx,fry,frr)

    idx = ut.tril_indices(time_series.shape[0],-1)
    c[idx[0],idx[1],...] = c[idx[1],idx[0],...].conj() #Make it symmetric

    return f,c

def coherence_partial_calculate(fxy,fxx,fyy,fxr,fry,frr): 
    r"""
    Compute the band-specific partial coherence between the spectra of
    two time series. See :func:`partial_coherence`. 

    Input to this function is in the frequency domain.

    Parameters
    ----------
    fxy : float array
         The cross-spectrum of the time series 
    
    fyy,fxx : float array
         The spectra of the signals

    fxr,fry : float array
         The cross-spectra of the signals with the event
    
    Returns 
    -------
    float
        the band-averaged coherency

    """
    abs = np.abs
    coh = coherency_calculate
    Rxr = coh(fxr,fxx,frr)
    Rry = coh(fry,fyy,frr)
    Rxy = coh(fxy,fxx,fyy)

    return (( (np.abs(Rxy-Rxr*Rry))**2 ) /
           ( (1-((np.abs(Rxr))**2)) * (1-((np.abs(Rry))**2)) ) )

def coherence_partial_bavg(time_series,r,csd_method=None,lb=0,ub=None):
    r"""
    Compute the band-averaged partial coherence between the spectra of two time
    series. See :func:`coherence_partial`.  

    Input to this function is in the time domain.

    Parameters
    ----------
    time_series : float array
         Time series data with the time on the last dimension
         
    r : float array
         Cause to be partialed out
         
    csd_method: dict, optional
       See :func:`get_spectra` for details

    lb: float, optional
        The lower bound frequency (in Hz) of the range over which the average
        is calculated. Default: 0

    ub: float, optional
        The upper bound frequency (in Hz) of the range over which the average
        is calculated. Defaults to the Nyquist frequency  

    Returns 
    -------
    c: float
        the band-averaged coherency
   
    """ 
    if csd_method is None:
        csd_method = {'this_method':'welch'} #The default

    f,fxy = get_spectra(time_series,csd_method)

    c=np.zeros((time_series.shape[0],
                time_series.shape[0],
                f.shape[0]), dtype = complex)       

    lb_idx,ub_idx = ut.get_bounds(f,lb,ub)

    if lb==0:
        lb_idx = 1 #The lowest frequency band should be f0

    c = np.zeros((time_series.shape[0],
                time_series.shape[0]))

    for i in xrange(time_series.shape[0]): 
        for j in xrange(i,time_series.shape[0]):
            f,fxx,frr,frx = get_spectra_bi(time_series[i],r,csd_method)
            f,fyy,frr,fry = get_spectra_bi(time_series[j],r,csd_method)
            coherence_partial_bavg_calculate(f[lb_idx:ub_idx],
                                        fxy[i][j][lb_idx:ub_idx],
                                        fxy[i][i][lb_idx:ub_idx],
                                        fxy[j][j][lb_idx:ub_idx],
                                        fxr[lb_idx:ub_idx],
                                        fry[lb_idx:ub_idx],
                                        frr[lb_idx:ub_idx])

    idx = ut.tril_indices(time_series.shape[0],-1)
    c[idx[0],idx[1],...] = c[idx[1],idx[0],...].conj() #Make it symmetric

    return c

def coherence_partial_bavg_calculate(f,fxy,fxx,fyy,fxr,fry,frr):
    r"""
    Compute the band-averaged partial coherence between the spectra of
    two time series.
    
    Input to this function is in the frequency domain.

    Parameters
    ----------

    f: the frequencies
    
    fxy : float array
         The cross-spectrum of the time series 
    
    fyy,fxx : float array
         The spectra of the signals

    fxr,fry : float array
         The cross-spectra of the signals with the event
         
    Returns 
    -------
    float
        the band-averaged coherency

    See also
    --------
    coherency, coherence, coherence_partial, coherency_bavg
   
    """
    coh = coherency
    Rxy = coh(fxy,fxx,fyy)
    Rxr = coh(fxr,fxx,frr)
    Rry = coh(fry,fyy,frr)

    return (np.sum(Rxy-Rxr*Rry)/
        np.sqrt(np.sum(1-Rxr*Rxr.conjugate())*np.sum(1-Rry*Rry.conjugate())))

def coherency_phase_spectrum (time_series,csd_method=None):
    """
    Compute the phase spectrum of the cross-spectrum between two time series. 

    The parameters of this function are in the time domain.

    Parameters
    ----------
    
    time_series: n*t float array
    The time series, with t, time, as the last dimension
        
    Returns 
    -------
    
    f: mid frequencies of the bands
    
    p: an array with the pairwise phase spectrum between the time
    series, where p[i][j] is the phase spectrum between time series[i] and
    time_series[j]
    
    Notes
    -----
    
    This is an implementation of equation (3) of Sun et al. (2005) [Sun2005]_:

    .. math::

        \phi(\lambda) = arg [R_{xy} (\lambda)] = arg [f_{xy} (\lambda)]

    .. [Sun2005] F.T. Sun and L.M. Miller and M. D'Esposito(2005). Measuring
        temporal dynamics of functional networks using phase spectrum of fMRI
        data.  Neuroimage, 28: 227-37.
    """
    if csd_method is None:
        csd_method = {'this_method':'welch'} #The default
         
    f,fxy = get_spectra(time_series,csd_method)

    p=np.zeros((time_series.shape[0],
               time_series.shape[0],
               f.shape[0]))
    
    for i in xrange(time_series.shape[0]): 
      for j in xrange(i+1,time_series.shape[0]):
          p[i][j] = coherency_phase_spectrum_calculate(fxy[i][j])
          p[j][i] = coherency_phase_spectrum_calculate(fxy[i][j].conjugate())
         
    return f,p

def coherency_phase_spectrum_calculate(fxy):
    r"""
    Compute the phase spectrum of the cross-spectrum between two time series. 

    The parameters of this function are in the frequency domain.

    Parameters
    ----------
    
    fxy : float array
         The cross-spectrum of the time series 
        
    Returns 
    -------
    
    float
        a frequency-band-dependent measure of the phase between the two
        time-series 
         
    Notes
    -----
    
    This is an implementation of equation (3) of Sun et al. (2005) [Sun2005]_:

    .. math::

        \phi(\lambda) = arg [R_{xy} (\lambda)] = arg [f_{xy} (\lambda)]

    .. [Sun2005] F.T. Sun and L.M. Miller and M. D'Esposito(2005). Measuring
        temporal dynamics of functional networks using phase spectrum of fMRI
        data.  Neuroimage, 28: 227-37.
    """
    phi = np.angle(fxy)
    
    return phi

def coherency_phase_delay(time_series,lb=0,ub=None,csd_method=None):
    """
    The temporal delay calculated from the coherency phase spectrum.

    Parameters
    ----------

    time_series: float array
       The time-series data for which the delay is calculated.

    lb, ub: float
       Frequency boundaries (in Hz), for the domain over which the delays are
       calculated. Defaults to 0-max(f)

    csd_method : dict, optional.
       See :func:`get_spectra`

    Returns
    -------
    f : float array
       The mid-frequencies for the frequency bands over which the calculation
       is done. 
    p : float array
       Pairwise temporal delays between time-series (in seconds).    
    
    """
    if csd_method is None:
        csd_method = {'this_method':'welch'} #The default

    f,fxy = get_spectra(time_series,csd_method)

    lb_idx,ub_idx = ut.get_bounds(f,lb,ub)

    if lb_idx == 0:
        lb_idx = 1
    
    p = np.zeros((time_series.shape[0],time_series.shape[0],
                  f[lb_idx:ub_idx].shape[-1]))

    for i in xrange(time_series.shape[0]): 
        for j in xrange(i,time_series.shape[0]):
            p[i][j] = coherency_phase_delay_calculate(f[lb_idx:ub_idx],
                                                      fxy[i][j][lb_idx:ub_idx])
            p[j][i] = coherency_phase_delay_calculate(f[lb_idx:ub_idx],
                                           fxy[i][j][lb_idx:ub_idx].conjugate())


    return f[lb_idx:ub_idx],p

def coherency_phase_delay_calculate(f,fxy):
    r"""
    Compute the phase delay between the spectra of two signals. The input to
    this function is in the frequency domain. 

    Parameters
    ----------

    f: float array
         The frequencies 
         
    fxy : float array
         The cross-spectrum of the time series 
    
    Returns 
    -------
    
    float array
        the phase delay (in sec) for each frequency band.
        
    """

    phi = coherency_phase_spectrum_calculate(fxy)
    
    t =  (phi)  / (2*np.pi*f)
        
    return t


def coherency_phase_delay_bavg(time_series,lb=0,ub=None,csd_method=None):
    """
    Band-averaged phase delay between time-series

    Parameters
    ----------

    time_series: float array
       The time-series data

    lb,ub : float, optional
       Lower and upper bounds on the frequency range over which the phase delay
       is averaged

    Returns
    -------
    p : float array
       The pairwise band-averaged phase-delays between the time-series. 
    
    """

    if csd_method is None:
        csd_method = {'this_method':'welch'} #The default

    f,fxy = get_spectra(time_series,csd_method)

    lb_idx,ub_idx = ut.get_bounds(f,lb,ub)

    if lb_idx == 0:
        lb_idx = 1
    
    p = np.zeros((time_series.shape[0],time_series.shape[0],
                  f[lb_idx:ub_idx].shape[-1]))

    for i in xrange(time_series.shape[0]): 
        for j in xrange(i,time_series.shape[0]):
            p[i][j] = coherency_phase_delay_bavg_calculate(f[lb_idx:ub_idx],
                                                      fxy[i][j][lb_idx:ub_idx])
            p[j][i] = coherency_phase_delay_bavg_calculate(f[lb_idx:ub_idx],
                                           fxy[i][j][lb_idx:ub_idx].conjugate())

    return p

def coherency_phase_delay_bavg_calculate(f,fxy):
    r"""
    Compute the band-averaged phase delay between the spectra of two signals 

    Parameters
    ----------

    f: float array
         The frequencies 
         
    fxy : float array
         The cross-spectrum of the time series 
    
    Returns 
    -------
    
    float
        the phase delay (in sec)
           
    """
    return np.mean(coherency_phase_spectrum (fxy)/(2*np.pi*f))
    
#XXX def coherence_partial_phase()

def correlation_spectrum(x1,x2, Fs=2, norm=False):
    """Calculate the spectral decomposition of the correlation.
    
    Parameters
    ----------
    x1,x2: ndarray
       Two arrays to be correlated. Same dimensions

    Fs: float, optional
       Sampling rate in Hz. If provided, an array of
       frequencies will be returned.Defaults to 2

    norm: bool, optional
       When this is true, the spectrum is normalized to sum to 1
    
    Returns
    -------
    ccn: ndarray
       The spectral decomposition of the correlation

    f: ndarray
       ndarray with the frequencies
    
    Notes
    -----

    This method is described in full in [Cordes2000]_
    
    .. [Cordes2000] D Cordes, V M Haughton, K Arfanakis, G J Wendt, P A Turski,
    C H Moritz, M A Quigley, M E Meyerand (2000). Mapping functionally related
    regions of brain with functional connectivity MR imaging. AJNR American
    journal of neuroradiology 21:1636-44
    
    """

    x1 = x1 - np.mean(x1)
    x2 = x2 - np.mean(x2)
    x1_f = np.fft.fft(x1)
    x2_f = np.fft.fft(x2)
    D = np.sqrt( np.sum(x1**2) * np.sum(x2**2) )
    n = x1.shape[0]

    ccn =( ( np.real(x1_f) * np.real(x2_f) +
             np.imag(x1_f) * np.imag(x2_f) ) /
           (D*n) )

    if norm:
        ccn = ccn / np.sum(ccn) * 2 #Only half of the sum is sent back because
                                    #of the freq domain symmetry. XXX Does
                                    #normalization make this strictly positive? 

    f = get_freqs(Fs,n)
    return f,ccn[0:n/2]

#-----------------------------------------------------------------------------
#Event related analysis
#-----------------------------------------------------------------------------

def fir(timeseries,design):
    """
    Calculate the FIR (finite impulse response) HRF, according to [Burock2000]_
    
    
    Parameters
    ----------
    
    timeseries : float array
            timeseries data
    
    design : int array
          This is a design matrix.  It has to have shape = (number
          of TRS, number of conditions * length of HRF)
    
          The form of the matrix is: 
            
              A B C ... 
          
          where A is a (number of TRs) x (length of HRF) matrix with a unity
          matrix placed with its top left corner placed in each TR in which
          event of type A occured in the design. B is the equivalent for
          events of type B, etc.
    
    Returns 
    -------

    HRF: float array 
        HRF is a numpy array of 1X(length of HRF * number of conditions)
        with the HRFs for the different conditions concatenated. This is an
        estimate of the linear filters between the time-series and the events
        described in design.

    Notes
    -----

    Implements equation 4 in[Burock2000]_:

    .. math::

        \hat{h} = (X^T X)^{-1} X^T y
        
    .. [Burock2000] M.A. Burock and A.M.Dale (2000). Estimation and Detection of
        Event-Related fMRI Signals with Temporally Correlated Noise: A
        Statistically Efficient and Unbiased Approach. Human Brain Mapping,
        11:249-260
         
    """
    X = np.matrix(design)
    y = np.matrix(timeseries)
    h = np.array(linalg.pinv(X.T*X) * X.T*y.T)
    return h   

def freq_domain_xcorr(tseries,events,t_before,t_after,Fs=1):
    """
    Calculates the  event related timeseries, using a cross-correlation in the
    frequency domain.

    
    Parameters
    ----------
    tseries: float array
       Time series data with time as the last dimension

    events: float array
       An array with time-resolved events, at the same sampling rate as tseries

    t_before: float
       Time before the event to include

    t_after: float
       Time after the event to include 

    Fs: float
       Sampling rate of the time-series (in Hz)
    
    Returns
    -------
    xcorr: float array
        The correlation function between the tseries and the events. Can be
        interperted as a linear filter from events to responses (the time-series)
        of an LTI.  
    
    """
    
    fft = np.fft.fft
    ifft = np.fft.ifft
    fftshift = np.fft.fftshift

    xcorr = np.real(fftshift ( ifft ( fft(tseries) * fft(np.fliplr([events]) )
    ) ) )
                     
    return xcorr[0][ np.ceil(len(xcorr[0])/2)-t_before*Fs :
                    np.ceil(len(xcorr[0])/2)+t_after/2*Fs ]/np.sum(events)


def freq_domain_xcorr_zscored(tseries,events,t_before,t_after,Fs=1):
    """
    Calculates the z-scored event related timeseries, using a cross-correlation
    in the frequency domain.

    
    Parameters
    ----------
    tseries: float array
       Time series data with time as the last dimension

    events: float array
       An array with time-resolved events, at the same sampling rate as tseries

    t_before: float
       Time before the event to include

    t_after: float
       Time after the event to include 

    Fs: float
       Sampling rate of the time-series (in Hz)
    
    Returns
    -------
    xcorr: float array
    The correlation function between the tseries and the events. Can be
    interperted as a linear filter from events to responses (the time-series)
    of an LTI. Because it is normalized to its own mean and variance, it can be
    interperted as measuring statistical significance relative to all
    time-shifted versions of the events. 
    
    """
        
    fft = np.fft.fft
    ifft = np.fft.ifft
    fftshift = np.fft.fftshift

    xcorr = np.real( fftshift ( ifft ( fft(tseries) * fft(np.fliplr([events]) )
    ) ) )
    meanSurr = np.mean(xcorr)
    stdSurr = np.std(xcorr)
    
    return ( ( (xcorr[0][ np.ceil(len(xcorr[0])/2)-t_before*Fs :
                    np.ceil(len(xcorr[0])/2)+t_after*Fs ])
             - meanSurr)
             / stdSurr )

#-----------------------------------------------------------------------------
# Spectral estimation
#-----------------------------------------------------------------------------
def get_spectra(time_series,method=None):
    r"""
    Compute the spectra of an n-tuple of time series and all of
    the pairwise cross-spectra.

    Parameters
    ----------
    time_series: float array
        The time-series, where time is the last dimension

    method: dict, optional

        contains: this_method:'welch'
           indicates that :func:`mlab.psd` will be used in
           order to calculate the psd/csd, in which case, additional optional
           inputs (and default values) are:

               NFFT=64

               Fs=2pi

               detrend=mlab.detrend_none

               window=mlab.window_hanning

               n_overlap=0

        this_method:'periodogram_csd'
           indicates that :func:`periodogram` will
           be used in order to calculate the psd/csd, in which case, additional
           optional inputs (and default values) are:

               Skx=None

               Sky=None

               N=None

               sides='onesided'

               normalize=True

               Fs=2pi

        this_method:'multi_taper_csd'
           indicates that :func:`multi_taper_psd` used in order to calculate
           psd/csd, in which case additional optional inputs (and default
           values) are:

               BW=0.01

               Fs=2pi

               sides = 'onesided'

    Returns
    -------
    
    f: float array
        The central frequencies for the frequency bands for which the spectra
        are estimated 

    fxy: float array
        A semi-filled matrix with the cross-spectra of the signals. The csd of
        signal i and signal j is in f[j][i], but not in f[i][j] (which will be
        filled with zeros). For i=j fxy[i][j] is the psd of signal i. 

    """            
    if method is None:
        method = {'this_method':'welch'} #The default
    #If no choice of method was explicitely set, but other parameters were
    #passed, assume that the method is mlab:
    this_method = method.get('this_method','welch')

    if this_method == 'welch':
        NFFT = method.get('NFFT',64)
        Fs = method.get('Fs',2*np.pi)
        detrend = method.get('detrend',mlab.detrend_none)
        window = method.get('window',mlab.window_hanning)
        n_overlap = method.get('n_overlap',int(np.ceil(NFFT/2.0)))

        #The length of the spectrum depends on how many sides are taken, which
        #depends on whether or not this is a complex object:
        if np.iscomplexobj(time_series):
            fxy_len = NFFT
        else:
            fxy_len = NFFT/2.0 + 1

        #If there is only 1 channel in the time-series:
        if len(time_series.shape)==1 or time_series.shape[0] == 1:
            temp, f = mlab.csd(time_series,time_series,
                               NFFT,Fs,detrend,window,n_overlap,
                               scale_by_freq=True)

            fxy = temp.squeeze()#the output of mlab.csd has a weird
                                        #shape
        else:
            fxy = np.zeros((time_series.shape[0],
                            time_series.shape[0],
                            fxy_len), dtype = complex) #Make sure it's complex

            for i in xrange(time_series.shape[0]):
                for j in xrange(i,time_series.shape[0]):
                    #Notice funny indexing, in order to conform to the
                    #conventions of the other methods:
                    temp, f = mlab.csd(time_series[j],time_series[i],
                                       NFFT,Fs,detrend,window,n_overlap,
                                       scale_by_freq=True)

                    fxy[i][j] = temp.squeeze() #the output of mlab.csd has a
                                               #wierd shape
    elif this_method in ('multi_taper_csd','periodogram_csd'):
        # these methods should work with similar signatures
        mdict = method.copy()
        func = eval(mdict.pop('this_method'))
        freqs, fxy = func(time_series, **mdict)
        f = ut.circle_to_hz(freqs, mdict.get('Fs', 2*np.pi))

    else:
        raise ValueError("Unknown method provided")
    
    return f,fxy.squeeze()

def get_spectra_bi(x,y,method = None):
    r"""
    Computes the spectra of two timeseries and the cross-spectrum between them

    Parameters
    ----------

    x,y : float arrays
        Time-series data

    method: dict, optional
       See :func:`get_spectra` documentation for details
    
    Returns
    -------
    f: float array
        The central frequencies for the frequency
        bands for which the spectra are estimated
    fxx: float array
         The psd of the first signal
    fyy: float array
        The psd of the second signal
    fxy: float array
        The cross-spectral density of the two signals

    """
    f, fij = get_spectra(np.vstack((x,y)), method=method)
    fxx = fij[0,0].real
    fyy = fij[1,1].real
    fxy = fij[0,1]
    return f, fxx, fyy, fxy


# The following spectrum estimates are normalized to the following convention..
# By definition, Sxx(f) = DTFT{sxx(n)}, where sxx(n) is the autocovariance
# function of s(n). Therefore the integral from
# [-PI, PI] of Sxx/(2PI) is sxx(0)
# And from the definition of sxx(n),
# sxx(0) = Expected-Value{s(n)s*(n)} = Expected-Value{ Var(s) },
# which is estimated simply as (s*s.conj()).mean()

def periodogram(s, Fs=2*np.pi, Sk=None, N=None, sides='default', normalize=True):
    """Takes an N-point periodogram estimate of the PSD function. The
    number of points N, or a precomputed FFT Sk may be provided. By default,
    the PSD function returned is normalized so that the integral of the PSD
    is equal to the mean squared amplitude (mean energy) of s (see Notes).

    Parameters
    ----------
    s : ndarray
        Signal(s) for which to estimate the PSD, time dimension in the last axis

    Fs: float (optional)
       The sampling rate. Defaults to 2*pi
       
    Sk : ndarray (optional)
        Precomputed FFT of s

    N : int (optional)
        Indicates an N-point FFT where N != s.shape[-1]
        
    sides : str (optional) [ 'default' | 'onesided' | 'twosided' ]
         This determines which sides of the spectrum to return. 
         For complex-valued inputs, the default is two-sided, for real-valued
         inputs, default is one-sided Indicates whether to return a one-sided
         or two-sided
         
    PSD normalize : boolean (optional, default=True) Normalizes the PSD

    Returns
    -------
    (f, psd): tuple
       f: The central frequencies for the frequency bands  
       PSD estimate for each row of s

    Notes
    -----
    setting dw = 2*PI/N, then the integral from -PI, PI (or 0,PI) of PSD/(2PI)
    will be nearly equal to sxx(0), where sxx is the autocovariance function
    of s(n). By definition, sxx(0) = E{s(n)s*(n)} ~ (s*s.conj()).mean()
    """
    if Sk is not None:
        N = Sk.shape[-1]
    else:
        N = s.shape[-1] if not N else N
        Sk = np.fft.fft(s, n=N)
    pshape = list(Sk.shape)
    norm = float(s.shape[-1])
    
    # if the time series is a complex vector, a one sided PSD is invalid:
    if (sides == 'default' and np.iscomplexobj(s)) or sides == 'twosided':
        sides='twosided'
    elif sides in ('default', 'onesided'):
        sides='onesided'

    if sides=='onesided':
        # putative Nyquist freq
        Fn = N/2 + 1
        # last duplicate freq
        Fl = (N+1)/2
        pshape[-1] = Fn
        P = np.zeros(pshape, 'd')
        freqs = np.linspace(0, Fs/2, Fn)
        P[...,0] = (Sk[...,0]*Sk[...,0].conj())
        P[...,1:Fl] = 2 * (Sk[...,1:Fl]*Sk[...,1:Fl].conj())
        if Fn > Fl:
            P[...,Fn-1] = (Sk[...,Fn-1]*Sk[...,Fn-1].conj())
    else:
        P = (Sk*Sk.conj()).real
        freqs = np.linspace(0, Fs, N, endpoint=False)
    if normalize:
        P /= norm
    return freqs, P

def periodogram_csd(s, Sk=None, N=None, sides='default', normalize=True):
    """Takes an N-point periodogram estimate of all the cross spectral
    density functions between rows of s.

    The number of points N, or a precomputed FFT Sk may be provided. By
    default, the CSD function returned is normalized so that the integral of
    the PSD is equal to the mean squared amplitude (mean energy) of s (see
    Notes).

    Paramters
    ---------

    s : ndarray
        Signals for which to estimate the CSD, time dimension in the last axis

    Sk : ndarray (optional)
        Precomputed FFT of rows of s

    N : int (optional)
        Indicates an N-point FFT where N != s.shape[-1]

    sides : str (optional)   [ 'default' | 'onesided' | 'twosided' ]
         This determines which sides of the spectrum to return. 
         For complex-valued inputs, the default is two-sided, for real-valued
         inputs, default is one-sided Indicates whether to return a one-sided
         or two-sided
         
    normalize : boolean (optional)
        Normalizes the PSD

    Returns
    -------
    
    (freqs, csd_est) : ndarrays
        The estimatated CSD and the frequency points vector.
        The CSD{i,j}(f) are returned in a square "matrix" of vectors
        holding Sij(f). For an input array that is reshaped to (M,N),
        the output is (M,M,N)

    Notes
    -----
    setting dw = 2*PI/N, then the integral from -PI, PI (or 0,PI) of PSD/(2PI)
    will be nearly equal to sxy(0), where sxx is the crosscovariance function
    of s1(n), s2(n). By definition, sxy(0) = E{s1(n)s2*(n)} ~ (s1*s2.conj()).mean()
    """
    s_shape = s.shape
    s.shape = (np.prod(s_shape[:-1]), s_shape[-1])
    # defining an Sk_loc is a little opaque, but it avoids having to
    # reset the shape of any user-given Sk later on
    if Sk is not None:
        Sk_shape = Sk.shape
        N = Sk.shape[-1]
        Sk_loc = Sk.reshape(np.prod(Sk_shape[:-1]), N)
    else:
        N = s.shape[-1] if not N else N
        Sk_loc = np.fft.fft(s, n=N)
    # reset s.shape
    s.shape = s_shape

    M = Sk_loc.shape[0]
    norm = float(s.shape[-1])

    # if the time series is a complex vector, a one sided PSD is invalid:
    if (sides == 'default' and np.iscomplexobj(s)) or sides == 'twosided':
        sides='twosided'
    elif sides in ('default', 'onesided'):
        sides='onesided'

    if sides=='onesided':
        # putative Nyquist freq
        Fn = N/2 + 1
        # last duplicate freq
        Fl = (N+1)/2
        csd_mat = np.empty((M,M,Fn), 'D')
        freqs = np.linspace(0, np.pi, Fn)
        for i in xrange(M):
            for j in xrange(i+1):
                csd_mat[i,j,0] = Sk_loc[i,0]*Sk_loc[j,0].conj()
                csd_mat[i,j,1:Fl] = 2 * (Sk_loc[i,1:Fl]*Sk_loc[j,1:Fl].conj())
                if Fn > Fl:
                    csd_mat[i,j,Fn-1] = Sk_loc[i,Fn-1]*Sk_loc[j,Fn-1].conj()
                    
    else:
        csd_mat = np.empty((M,M,N), 'D')
        freqs = np.linspace(0, 2*np.pi, N, endpoint=False)        
        for i in xrange(M):
            for j in xrange(i+1):
                csd_mat[i,j] = Sk_loc[i]*Sk_loc[j].conj()
    if normalize:
        csd_mat /= norm

    upper_idc = ut.triu_indices(M,k=1)
    lower_idc = ut.tril_indices(M,k=-1)
    csd_mat[upper_idc] = csd_mat[lower_idc].conj()
    return freqs, csd_mat

def DPSS_windows(N, NW, Kmax):
    """Returns the Discrete Prolate Spheroidal Sequences of orders [0,Kmax-1]
    for a given frequency-spacing multiple NW and sequence length N. 

    Paramters
    ---------
    N : int
        sequence length
    NW : float, unitless
        standardized half bandwidth corresponding to 2NW = BW*f0 = BW*N/dt
        but with dt taken as 1
    Kmax : int
        number of DPSS windows to return is Kmax (orders 0 through Kmax-1)

    Returns
    -------
    v,e : tuple,
        v is an array of DPSS windows shaped (Kmax, N)
        e are the eigenvalues 

    Notes
    -----
    Tridiagonal form of DPSS calculation from:

    Slepian, D. Prolate spheroidal wave functions, Fourier analysis, and
    uncertainty V: The discrete case. Bell System Technical Journal,
    Volume 57 (1978), 1371430
    """
    # here we want to set up an optimization problem to find a sequence
    # whose energy is maximally concentrated within band [-W,W].
    # Thus, the measure lambda(T,W) is the ratio between the energy within
    # that band, and the total energy. This leads to the eigen-system
    # (A - (l1)I)v = 0, where the eigenvector corresponding to the largest
    # eigenvalue is the sequence with maximally concentrated energy. The
    # collection of eigenvectors of this system are called Slepian sequences,
    # or discrete prolate spheroidal sequences (DPSS). Only the first K,
    # K = 2NW/dt orders of DPSS will exhibit good spectral concentration
    # [see http://en.wikipedia.org/wiki/Spectral_concentration_problem]
    
    # Here I set up an alternative symmetric tri-diagonal eigenvalue problem
    # such that
    # (B - (l2)I)v = 0, and v are our DPSS (but eigenvalues l2 != l1)
    # the main diagonal = ([N-1-2*t]/2)**2 cos(2PIW), t=[0,1,2,...,N-1]
    # and the first off-diangonal = t(N-t)/2, t=[1,2,...,N-1]
    # [see Percival and Walden, 1993]
    Kmax = int(Kmax)
    W = float(NW)/N
    ab = np.zeros((2,N), 'd')
    nidx = np.arange(N)
    ab[0,1:] = nidx[1:]*(N-nidx[1:])/2.
    ab[1] = ((N-1-2*nidx)/2.)**2 * np.cos(2*np.pi*W)
    # only calculate the highest Kmax-1 eigenvectors
    l,v = linalg.eig_banded(ab, select='i', select_range=(N-Kmax, N-1))
    dpss = v.transpose()[::-1]

    # By convention (Percival and Walden, 1993 pg 379)
    # * symmetric tapers (k=0,2,4,...) should have a positive average.
    # * antisymmetric tapers should begin with a positive lobe
    fix_symmetric = (dpss[0::2].sum(axis=1) < 0)
    for i, f in enumerate(fix_symmetric):
        if f:
            dpss[2*i] *= -1
    fix_skew = (dpss[1::2,1] < 0)
    for i, f in enumerate(fix_skew):
        if f:
            dpss[2*i+1] *= -1

    # Now find the eigenvalues of the original 
    # Use the autocovariance sequence technique from Percival and Walden, 1993
    # pg 390
    # XXX : why debias false? it's all messed up o.w., even with means
    # on the order of 1e-2
    acvs = utils.autocov(dpss, debias=False) * N
    r = 4*W*np.sinc(2*W*nidx)
    r[0] = 2*W
    eigvals = np.dot(acvs, r)
    
    return dpss, eigvals

def mtm_cross_spectrum(tx, ty, weights, sides='twosided'):
    r"""

    The cross-spectrum between two tapered time-series, derived from a
    multi-taper spectral estimation.

    Parameters
    ----------

    tx, ty: ndarray (K, ..., N)
       The tapered complex spectra, with K tapers

    weights: ndarray, or 2-tuple or list
       Weights can be specified as a length-2 list of weights for spectra tx
       and ty respectively. Alternatively, if tx is ty and this function is
       computing the spectral density function of a single sequence, the
       weights can be given as an ndarray of weights for the spectrum.
       Weights may be

       * scalars, if the shape of the array is (K, ..., 1)
       * vectors, with the shape of the array being the same as tx or ty

    sides: str in {'onesided', 'twosided'}
       For the symmetric spectra of a real sequence, optionally combine half
       of the frequencies and scale the duplicate frequencies in the range
       (0, F_nyquist).

    Notes
    -----

    spectral densities are always computed as
    :math:`S_{xy}^{mt}(f) = \frac{\sum_k [d_k^x(f)y_k^x(f)][d_k^y(f)(y_k^y(f))^{*}]}{[\sum_k d_k^x(f)^2]^{\frac{1}{2}}[\sum_k d_k^y(f)^2]^{\frac{1}{2}}}`

    """
    
    N = tx.shape[-1]
    if N!=ty.shape[-1]:
        raise ValueError('shape mismatch between tx, ty')
    pshape = list(tx.shape)

    if isinstance(weights, (list, tuple)):
        weights_x = weights[0]
        weights_y = weights[1]
        denom = (weights_x**2).sum(axis=0)**0.5
        denom *= (weights_y**2).sum(axis=0)**0.5
    else:
        weights_x = weights
        weights_y = weights
        denom = (weights**2).sum(axis=0)

    if sides=='onesided':
        # where the nyq freq should be
        Fn = N/2 + 1        
        truncated_slice = [slice(None)] * len(tx.shape)
        truncated_slice[-1] = slice(0, Fn)
        tsl = tuple(truncated_slice)
        tx = tx[tsl]
        ty = ty[tsl]
        # weights may be scalars, or already truncated
        if weights_x.shape[-1] > Fn:
            weights_x = weights_x[tsl]
        if weights_y.shape[-1] > Fn:
            weights_y = weights_y[tsl]

    sf = weights_x*tx
    sf *= (weights_y * ty.conj())
    sf = sf.sum(axis=0)
    sf /= denom

    if sides=='onesided':
        # last duplicate freq
        Fl = (N+1)/2
        sub_slice = [slice(None)] * len(sf.shape)
        sub_slice[-1] = slice(1, Fl)
        sf[tuple(sub_slice)] *= 2

    return sf
    
##     if sides=='onesided':
##         # putative Nyquist freq
##         Fn = N/2 + 1
##         # last duplicate freq
##         Fl = (N+1)/2
##         pshape[-1] = Fn
##         p = np.zeros(pshape, 'D')
##         p[...,0] = tx[...,0]*ty[...,0].conj()
##         p[...,1:Fl] = 2 * tx[...,1:Fl]*ty[...,1:Fl].conj()
##         if Fn > Fl:
##             p[...,Fn-1] = tx[...,Fn-1]*ty[...,Fn-1].conj()
##     else:
##         p = tx*ty.conj()

##     # now the combination is sum( p * (wx*wy), axis=0 ) / sum( wx*wy )
##     wslice = [np.newaxis] * len(p.shape)
##     wslice[0] = slice(None)
##     p *= (weights_x[wslice] * weights_y[wslice])
##     sxy = p.sum(axis=0)
##     sxy /= (weights_x * weights_y).sum()
##     return sxy


def multi_taper_psd(s, Fs=2*np.pi, BW = None,  adaptive=False,
                    jackknife=True,low_bias=True, sides='default'):
    """Returns an estimate of the PSD function of s using the multitaper
    method. If the NW product, or the BW and Fs in Hz are not specified
    by the user, a bandwidth of 4 times the fundamental frequency,
    corresponding to NW = 4 will be used.

    Parameters
    ----------
    s : ndarray
       An array of sampled random processes, where the time axis is
       assumed to be on the last axis
    Fs: float, Sampling rate of the signal

    BW: float, The bandwidth of the windowing function will determine the number
       tapers to use. This parameters represents trade-off between frequency
       resolution (lower main lobe BW for the taper) and variance reduction
       (higher BW and number of averaged estimates).
       
    adaptive : {True/False}
       Use an adaptive weighting routine to combine the PSD estimates of
       different tapers.
    jackknife : {True/False}
       Use the jackknife method to make an estimate of the PSD variance
       at each point.
    low_bias : {True/False}
       Rather than use 2NW tapers, only use the tapers that have better than
       90% spectral concentration within the bandwidth (still using
       a maximum of 2NW tapers)
    sides : str (optional)   [ 'default' | 'onesided' | 'twosided' ]
         This determines which sides of the spectrum to return. 
         For complex-valued inputs, the default is two-sided, for real-valued
         inputs, default is one-sided Indicates whether to return a one-sided
         or two-sided

    Returns
    -------
    (freqs, psd_est, ssigma_or_nu) : ndarrays
        The first two arrays are the frequency points vector and the
        estimatated PSD. The last returned array differs depending on whether
        the jackknife was used. It is either

        * The jackknife estimated variance, OR
        * The degrees of freedom in a chi2 model of how the estimated
          log-PSD is distributed about the true log-PSD (this is either
          2*floor(2*NW), or calculated from adaptive weights)
          

    """
    # have last axis be time series for now
    N = s.shape[-1]
    rest_of = s.shape[:-1]

    s = s.reshape( int(np.product(rest_of)), N )
    # de-mean this sucker
    s = utils.remove_bias(s, axis=-1)

    #Get the number of tapers from the sampoing rate and the bandwidth:
    if BW is not None:
        NW = BW/(2*Fs) * N
    else:
        NW = 4 

    Kmax = int(2*NW)
        
    v, l = DPSS_windows(N, NW, Kmax)
    if low_bias:
        keepers = (l > 0.9)
        v = v[keepers]
        l = l[keepers]
        Kmax = len(v)

    # if the time series is a complex vector, a one sided PSD is invalid:
    if (sides == 'default' and np.iscomplexobj(s)) or sides == 'twosided':
        sides='twosided'
    elif sides in ('default', 'onesided'):
        sides='onesided'

    sig_sl = [slice(None)]*len(s.shape)
    sig_sl.insert(-1, np.newaxis)

    # tapered.shape is (..., Kmax, N)
    tapered = s[sig_sl] * v
    # Find the direct spectral estimators S_k(f) for k tapered signals..
    # don't normalize the periodograms by 1/N as normal.. since the taper
    # windows are orthonormal, they effectively scale the signal by 1/N

##     f,tapered_sdf = periodogram(tapered, sides=sides, normalize=False)

    tapered_spectra = np.fft.fft(tapered)

    last_freq = N/2+1 if sides=='onesided' else N

    # degrees of freedom at each timeseries, at each freq
    nu = np.empty( (s.shape[0], last_freq) )
    if adaptive:
        mag_sqr_spectra = np.abs(tapered_spectra)
        np.power(mag_sqr_spectra, 2, mag_sqr_spectra)
        weights = np.empty( mag_sqr_spectra.shape[:-1] + (last_freq,) )
        for i in xrange(s.shape[0]):
            weights[i], nu[i] = utils.adaptive_weights(
                mag_sqr_spectra[i], l, last_freq
                )
    else:
        # let the weights simply be the square-root of the eigenvalues
        wshape = [1] * len(tapered.shape)
        wshape[-2] = Kmax
        weights = np.sqrt(l).reshape( *wshape )
        nu.fill(2*Kmax)

    if jackknife:
        jk_var = np.empty_like(nu)
        if not adaptive:
            # compute the magnitude squared spectra, if not done already
            mag_sqr_spectra = np.abs(tapered_spectra)
            np.power(mag_sqr_spectra, 2, mag_sqr_spectra)
        for i in xrange(s.shape[0]):
            jk_var[i] = utils.jackknifed_sdf_variance(
                mag_sqr_spectra[i], weights=weights[i], last_freq=last_freq
                )
    
    # Compute the unbiased spectral estimator for S(f) as the sum of
    # the S_k(f) weighted by the function w_k(f)**2, all divided by the
    # sum of the w_k(f)**2 over k

    # 1st, roll the tapers axis forward
    tapered_spectra = np.rollaxis(tapered_spectra, 1, start=0)
    weights = np.rollaxis(weights, 1, start=0)
    sdf_est = mtm_cross_spectrum(
        tapered_spectra, tapered_spectra, weights, sides=sides
        ).real

    if sides=='onesided':
        freqs = np.linspace(0, Fs/2, N/2+1)
        if jackknife:
            # if the sdf was scaled by 2 at duplicate freqs,
            # then the variance will have to be scaled by 2**2
            jk_var[...,1:(N+1)/2] *= 4
    else:
        freqs = np.linspace(0, Fs, N, endpoint=False)

    out_shape = rest_of + ( len(freqs), )
    sdf_est.shape = out_shape
    # XXX: always return nu and jk_var
    if jackknife:
        jk_var.shape = out_shape
        return freqs, sdf_est, jk_var
    else:
        nu.shape = out_shape
        return freqs, sdf_est, nu

def multi_taper_csd(s, Fs=2*np.pi, BW=None, low_bias=True,
                    adaptive=False, sides='default'):
    """Returns an estimate of the Cross Spectral Density (CSD) function
    between all (N choose 2) pairs of timeseries in s, using the multitaper
    method. If the NW product, or the BW and Fs in Hz are not specified by
    the user, a bandwidth of 4 times the fundamental frequency, corresponding
    to NW = 4 will be used.

    Parameters
    ----------
    s : ndarray
        An array of sampled random processes, where the time axis is
        assumed to be on the last axis. If ndim > 2, the number of time
        series to compare will still be taken as prod(s.shape[:-1])

    Fs: float, Sampling rate of the signal

    BW: float, The bandwidth of the windowing function will determine the number
       tapers to use. This parameters represents trade-off between frequency
       resolution (lower main lobe BW for the taper) and variance reduction
       (higher BW and number of averaged estimates).

    adaptive : {True, False}
       Use adaptive weighting to combine spectra
    low_bias : {True, False}
       Rather than use 2NW tapers, only use the tapers that have better than
       90% spectral concentration within the bandwidth (still using
       a maximum of 2NW tapers)
    sides : str (optional)   [ 'default' | 'onesided' | 'twosided' ]
         This determines which sides of the spectrum to return. 
         For complex-valued inputs, the default is two-sided, for real-valued
         inputs, default is one-sided Indicates whether to return a one-sided
         or two-sided

    Returns
    -------
    (freqs, csd_est) : ndarrays
        The estimatated CSD and the frequency points vector.
        The CSD{i,j}(f) are returned in a square "matrix" of vectors
        holding Sij(f). For an input array of (M,N), the output is (M,M,N)
    """
    # have last axis be time series for now
    N = s.shape[-1]
    rest_of = s.shape[:-1]
    M = int(np.product(rest_of))

    s = s.reshape( M, N )
    # de-mean this sucker
    s = utils.remove_bias(s, axis=-1)

    #Get the number of tapers from the sampling rate and the bandwidth:
    if BW is not None:
        NW = BW/(2*Fs) * N
    else:
        NW = 4 

    Kmax = int(2*NW)

    v, l = DPSS_windows(N, NW, Kmax)
    if low_bias:
        keepers = (l > 0.9)
        v = v[keepers]
        l = l[keepers]
        Kmax = len(v)
    #print 'using', Kmax, 'tapers with BW=', NW * Fs/(np.pi*N)

    # if the time series is a complex vector, a one sided PSD is invalid:
    if (sides == 'default' and np.iscomplexobj(s)) or sides == 'twosided':
        sides='twosided'
    elif sides in ('default', 'onesided'):
        sides='onesided'

    sig_sl = [slice(None)]*len(s.shape)
    sig_sl.insert(len(s.shape)-1, np.newaxis)

    # tapered.shape is (M, Kmax-1, N)
    tapered = s[sig_sl] * v

    # compute the y_{i,k}(f)
    tapered_spectra = np.fft.fft(tapered)

    # compute the cross-spectral density functions
    last_freq = N/2+1 if sides=='onesided' else N

    if adaptive:
        mag_sqr_spectra = np.abs(tapered_spectra)
        np.power(mag_sqr_spectra, 2, mag_sqr_spectra)
        w = np.empty( mag_sqr_spectra.shape[:-1] + (last_freq,) )
        nu = np.empty( (M, last_freq) )
        for i in xrange(M):
            w[i], nu[i] = utils.adaptive_weights(
                mag_sqr_spectra[i], l, last_freq
                )
    else:
        weights = np.sqrt(l).reshape(Kmax, 1)

    csdfs = np.empty((M,M,last_freq), 'D')
    for i in xrange(M):
        if adaptive:
            wi = w[i]
        else:
            wi = weights
        for j in xrange(i+1):
            if adaptive:
                wj = w[j]
            else:
                wj = weights
            ti = tapered_spectra[i]
            tj = tapered_spectra[j]
            csdfs[i,j] = mtm_cross_spectrum(ti, tj, (wi, wj), sides=sides)

    upper_idc = ut.triu_indices(M,k=1)
    lower_idc = ut.tril_indices(M,k=-1)
    csdfs[upper_idc] = csdfs[lower_idc].conj()

    if sides=='onesided':
        freqs = np.linspace(0, Fs/2, N/2+1)
    else:
        freqs = np.linspace(0, Fs, N, endpoint=False)

    return freqs, csdfs 

def my_freqz(b, a=1., Nfreqs=1024, sides='onesided'):
    """
    Returns the frequency response of the IIR or FIR filter described
    by beta and alpha coefficients. 

    Parameters
    ----------

    b : beta sequence (moving average component)
    a : alpha sequence (autoregressive component)
    Nfreqs : size of frequency grid
    sides : {'onesided', 'twosided'}
       compute frequencies between [-PI,PI), or from [0, PI]

    Returns
    -------

    fgrid, H(e^jw)

    Notes
    -----
    For a description of the linear constant-coefficient difference
    equation, see http://en.wikipedia.org/wiki/Z-transform#Linear_constant-coefficient_difference_equation

    """
    if sides=='onesided':
        fgrid = np.linspace(0,np.pi,Nfreqs/2+1)
    else:
        fgrid = np.linspace(0,2*np.pi,Nfreqs,endpoint=False)
    float_type = type(1.)
    int_type = type(1)
    Nfreqs = len(fgrid)
    if isinstance(b, float_type) or isinstance(b, int_type) or len(b) == 1:
        bw = np.ones(Nfreqs, 'D')*b
    else:
        L = len(b)
        # D_mn = exp(-j*omega(m)*n)
        # (D_mn * b) computes b(omega(m)) = sum_{n=0}^L b(n)exp(-j*omega(m)*n)
        DTFT = np.exp(-1j*fgrid[:,np.newaxis]*np.arange(0,L))
        bw = np.dot(DTFT, b)
    if isinstance(a, float_type) or isinstance(a, int_type) or len(a) == 1:
        aw = np.ones(Nfreqs, 'D')*a
    else:
        L = len(a)
        DTFT = np.exp(-1j*fgrid[:,np.newaxis]*np.arange(0,L))
        aw = np.dot(DTFT, a)
    return fgrid, bw/aw
    

def yule_AR_est(s, order, Nfreqs, sxx=None, sides='onesided', system=False):
    """Finds the parameters for an autoregressive model of order norder
    of the process s. Using these parameters, an estimate of the PSD
    is calculated from [-PI,PI) in Nfreqs, or [0,PI] in {N/2+1}freqs.
    Uses the basic Yule Walker system of equations, and a baised estimate
    of sxx (unless sxx is provided).

    The model for the autoregressive process takes this convention:
    s[n] = a1*s[n-1] + a2*s[n-2] + ... aP*s[n-P] + v[n]

    where v[n] is a zero-mean white noise process with variance=sigma_v

    Parameters
    ----------
    s : ndarray
        The sampled autoregressive random process

    order : int
        The order P of the AR system

    Nfreqs : int
        The number of spacings on the frequency grid from [-PI,PI).
        If sides=='onesided', Nfreqs/2+1 frequencies are computed from [0,PI]

    sxx : ndarray (optional)
        An optional, possibly unbiased estimate of the autocovariance of s

    sides : str (optional)
        Indicates whether to return a one-sided or two-sided PSD

    system : bool (optional)
        If True, return the AR system parameters, sigma_v and a{k}
    
    Returns
    -------
    (w, ar_psd)
    w : Array of normalized frequences from [-.5, .5) or [0,.5]
    ar_psd : A PSD estimate computed by sigma_v / |1-a(f)|**2 , where
             a(f) = DTFT(ak)
    """
    if sxx is not None and type(sxx) == np.ndarray:
        sxx_m = sxx[:order+1]
    else:
        sxx_m = ut.autocov(s)[:order+1]

    R = linalg.toeplitz(sxx_m[:order].conj())
    y = sxx_m[1:].conj()
    ak = linalg.solve(R,y)
    sigma_v = sxx_m[0] - np.dot(sxx_m[1:], ak)
    if system:
        return sigma_v, ak
    # compute the psd as |h(f)|**2, where h(f) is the transfer function..
    # for this model s[n] = a1*s[n-1] + a2*s[n-2] + ... aP*s[n-P] + v[n]
    # Taken as a FIR system from s[n] to v[n],
    # v[n] = w0*s[n] + w1*s[n-1] + w2*s[n-2] + ... + wP*s[n-P],
    # where w0 = 1, and wk = -ak for k>0
    # the transfer function here is H(f) = DTFT(w)
    # leading to Sxx(f) = Vxx(f) / |H(f)|**2 = sigma_v / |H(f)|**2
    w, hw = my_freqz(sigma_v**0.5, a=np.concatenate(([1], -ak)),
                     Nfreqs=Nfreqs, sides=sides)
    ar_psd = (hw*hw.conj()).real
    return (w,2*ar_psd) if sides=='onesided' else (w,ar_psd)
    
    
def LD_AR_est(s, order, Nfreqs, sxx=None, sides='onesided', system=False):
    """Finds the parameters for an autoregressive model of order norder
    of the process s. Using these parameters, an estimate of the PSD
    is calculated from [-PI,PI) in Nfreqs, or [0,PI] in {N/2+1}freqs.
    Uses the Levinson-Durbin recursion method, and a baised estimate
    of sxx (unless sxx is provided).

    The model for the autoregressive process takes this convention:
    s[n] = a1*s[n-1] + a2*s[n-2] + ... aP*s[n-P] + v[n]

    where v[n] is a zero-mean white noise process with variance=sigma_v

    Parameters
    ----------
    s : ndarray
        The sampled autoregressive random process

    order : int
        The order P of the AR system

    Nfreqs : int
        The number of spacings on the frequency grid from [-PI,PI).
        If sides=='onesided', Nfreqs/2+1 frequencies are computed from [0,PI]

    sxx : ndarray (optional)
        An optional, possibly unbiased estimate of the autocovariance of s

    sides : str (optional)
        Indicates whether to return a one-sided or two-sided PSD

    system : bool (optional)
        If True, return the AR system parameters, sigma_v and a{k}
    
    Returns
    -------
    (w, ar_psd)
    w : Array of normalized frequences from [-.5, .5) or [0,.5]
    ar_psd : A PSD estimate computed by sigma_v / |1-a(f)|**2 , where
             a(f) = DTFT(ak)
    """
    if sxx is not None and type(sxx) == np.ndarray:
        sxx_m = sxx[:order+1]
    else:
        sxx_m = ut.autocov(s)[:order+1]
    
    phi = np.zeros((order+1, order+1), 'd')
    sig = np.zeros(order+1)
    # initial points for the recursion
    phi[1,1] = sxx_m[1]/sxx_m[0]
    sig[1] = sxx_m[0] - phi[1,1]*sxx_m[1]
    for k in xrange(2,order+1):
        phi[k,k] = (sxx_m[k]-np.dot(phi[1:k,k-1], sxx_m[1:k][::-1]))/sig[k-1]
        for j in xrange(1,k):
            phi[j,k] = phi[j,k-1] - phi[k,k]*phi[k-j,k-1]
        sig[k] = sig[k-1]*(1 - phi[k,k]**2)

    sigma_v = sig[-1]; ak = phi[1:,-1]
    if system:
        return sigma_v, ak
    w, hw = my_freqz(sigma_v**0.5, a=np.concatenate(([1], -ak)),
                     Nfreqs=Nfreqs, sides=sides)
    ar_psd = (hw*hw.conj()).real
    return (w,2*ar_psd) if sides=='onesided' else (w,ar_psd)


def boxcar_filter(time_series,lb=0,ub=1,n_iterations=2):
    """
    Filters data into a frequency range. 

    For each of the two bounds, a low-passed version is created by convolving
    with a box-car and then the low-passed version for the upper bound is added
    to the low-passed version for the lower bound subtracted from the signal,
    resulting in a band-passed version 

    Parameters
    ----------

    time_series: float array
       the signal
    ub : float, optional
      The cut-off frequency for the low-pass filtering as a proportion of the
      sampling rate. Default to 1
    lb : float, optional
      The cut-off frequency for the high-pass filtering as a proportion of the
      sampling rate. Default to 0
    n_iterations: int, optional
      how many rounds of smoothing to do. Default to 2.

    Returns
    -------
    float array:
      The signal, filtered  
    """

    n = time_series.shape[-1]
 
    box_car_ub = np.ones(np.ceil(1.0/ub))
    box_car_ub = box_car_ub/(float(len(box_car_ub))) 
    box_car_ones_ub = np.ones(len(box_car_ub))

    if lb==0:
        lb=None
    else:
        box_car_lb = np.ones(np.ceil(1.0/lb))
        box_car_lb = box_car_lb/(float(len(box_car_lb))) 
        box_car_ones_lb = np.ones(len(box_car_lb))

    #If the time_series is a 1-d, we add a dimension, so that we can iterate
    #over 2-d inputs:
    if len(time_series.shape)==1:
        time_series = np.array([time_series])
    for i in xrange(time_series.shape[0]):
        if ub:
            #Start by applying a low-pass to the signal.  Pad the signal on
            #each side with the initial and terminal signal value:
            pad_s = np.hstack((box_car_ones_ub*time_series[i,0],time_series[i]))
            pad_s = np.hstack((pad_s, box_car_ones_ub*time_series[i,-1]))

            #Filter operation is a convolution with the box-car(iterate,
            #n_iterations times over this operation):
            for iteration in xrange(n_iterations):
                conv_s = np.convolve(pad_s,box_car_ub)

            #Extract the low pass signal by excising the central
            #len(time_series) points:        
            time_series[i] = conv_s[conv_s.shape[-1]/2-np.floor(n/2.):
                                    conv_s.shape[-1]/2+np.ceil(n/2.)]
        
        #Now, if there is a high-pass, do the same, but in the end subtract out
        #the low-passed signal:
        if lb:
            pad_s = np.hstack((box_car_ones_lb*time_series[i,0],time_series[i]))
            pad_s = np.hstack((pad_s, box_car_ones_lb * time_series[i,-1])) 
            
            #Filter operation is a convolution with the box-car(iterate,
            #n_iterations times over this operation):
            for iteration in xrange(n_iterations):
                conv_s = np.convolve(pad_s,box_car_lb)

            #Extract the low pass signal by excising the central
            #len(time_series) points:
            s_lp = conv_s[conv_s.shape[-1]/2-np.floor(n/2.):
                                    conv_s.shape[-1]/2+np.ceil(n/2.)]

            #Extract the high pass signal simply by subtracting the high pass
            #signal from the original signal:
            time_series[i] = time_series[i] - s_lp + np.mean(s_lp) #add mean
            #to make sure that there are no negative values. This also seems to
            #make sure that the mean of the signal (in % signal change) is close
            #to 0 

    return time_series.squeeze()

#-------------------------------------------------------------------------------
#Coherency calculated using cached spectra
#-------------------------------------------------------------------------------
"""The idea behind this set of functions is to keep a cache of the windowed fft
calculations of each time-series in a massive collection of time-series, so
that this calculation doesn't have to be repeated each time a cross-spectrum is
calculated. The first function creates the cache and then, another function
takes the cached spectra and calculates PSDs and CSDs, which are then passed to
coherency_calculate and organized in a data structure similar to the one
created by coherence"""

def cache_fft(time_series,ij,lb=0,ub=None,
                  method=None,prefer_speed_over_memory=False,
                  scale_by_freq=True):
    """compute and cache the windowed FFTs of the time_series, in such a way
    that computing the psd and csd of any combination of them can be done
    quickly.

    Parameters
    ----------

    time_series : float array
       An ndarray with time-series, where time is the last dimension

    ij: list of tuples
      Each tuple in this variable should contain a pair of
      indices of the form (i,j). The resulting cache will contain the fft of
      time-series in the rows indexed by the unique elements of the union of i
      and j 
    
    lb,ub: float
       Define a frequency band of interest, for which the fft will be cached

    method: dict, optional
        See :func:`get_spectra` for details on how this is used. For this set
        of functions, 'this_method' has to be 'welch' 
    

    Returns
    -------
    freqs, cache

        where: cache =
             {'FFT_slices':FFT_slices,'FFT_conj_slices':FFT_conj_slices,
             'norm_val':norm_val}

    Notes
    -----

    - For these functions, only the Welch windowed periodogram ('welch') is
      available. 

    - Detrending the input is not an option here, in order to save
      time on an empty function call.
    
    """
    if method is None:
        method = {'this_method':'welch'} #The default
    
    this_method = method.get('this_method','welch')

    if this_method == 'welch':
        NFFT = method.get('NFFT',64)
        Fs = method.get('Fs',2*np.pi)
        window = method.get('window',mlab.window_hanning)
        n_overlap = method.get('n_overlap',int(np.ceil(NFFT/2.0)))
    else:
        raise ValueError("For cache_fft, spectral estimation method must be welch")
    time_series = ut.zero_pad(time_series,NFFT)
    
    #The shape of the zero-padded version:
    n_channels, n_time_points = time_series.shape

    # get all the unique channels in time_series that we are interested in by
    # checking the ij tuples
    all_channels = set()
    for i,j in ij:
        all_channels.add(i); all_channels.add(j)
    n_channels = len(all_channels)

    # for real time_series, ignore the negative frequencies
    if np.iscomplexobj(time_series): n_freqs = NFFT
    else: n_freqs = NFFT//2+1

    #Which frequencies
    freqs = ut.get_freqs(Fs,NFFT)

    #If there are bounds, limit the calculation to within that band,
    #potentially include the DC component:
    lb_idx,ub_idx = ut.get_bounds(freqs,lb,ub)

    n_freqs=ub_idx-lb_idx
    #Make the window:
    if mlab.cbook.iterable(window):
        assert(len(window) == NFFT)
        window_vals = window
    else:
        window_vals = window(np.ones(NFFT, time_series.dtype))
        
    #Each fft needs to be normalized by the square of the norm of the window
    #and, for consistency with newer versions of mlab.csd (which, in turn, are
    #consistent with Matlab), normalize also by the sampling rate:
   
    if scale_by_freq:
        #This is the normalization factor for one-sided estimation, taking into
        #account the sampling rate. This makes the PSD a density function, with
        #units of dB/Hz, so that integrating over frequencies gives you the RMS
        #(XXX this should be in the tests!).
        norm_val = (np.abs(window_vals)**2).sum()*(Fs/2)
        
    else:
        norm_val = (np.abs(window_vals)**2).sum()/2
   
    # cache the FFT of every windowed, detrended NFFT length segement
    # of every channel.  If prefer_speed_over_memory, cache the conjugate
    # as well
        
    i_times = range(0, n_time_points-NFFT+1, NFFT-n_overlap)
    n_slices = len(i_times)
    FFT_slices = {}
    FFT_conj_slices = {}
    Pxx = {}
    
    for i_channel in all_channels:
        #dbg:
        #print i_channel
        Slices = np.zeros( (n_slices,n_freqs), dtype=np.complex)
        for iSlice in xrange(n_slices):
            thisSlice = time_series[i_channel,
                                    i_times[iSlice]:i_times[iSlice]+NFFT]

            
            #Windowing: 
            thisSlice = window_vals*thisSlice #No detrending
            #Derive the fft for that slice:
            Slices[iSlice,:] = (np.fft.fft(thisSlice)[lb_idx:ub_idx])
            
        FFT_slices[i_channel] = Slices


        if prefer_speed_over_memory:
            FFT_conj_slices[i_channel] = np.conjugate(Slices)

    cache = {'FFT_slices':FFT_slices,'FFT_conj_slices':FFT_conj_slices,
             'norm_val':norm_val}

    return freqs,cache

def cache_to_psd(cache,ij):
    """
    From a set of cached windowed fft, calculate the psd

    Parameters
    ----------
    cache : dict
        Return value from :func:`cache_fft`

    ij : list
        A list of tuples of the form (i,j).

    Returns
    -------
    Pxx : dict
        The phases for the intersection of (time_series[i],time_series[j]). The
        keys are the intersection of i,j values in the parameter ij 

    """

    #This is the way it is saved by cache_spectra:
    FFT_slices=cache['FFT_slices']
    FFT_conj_slices=cache['FFT_conj_slices']
    norm_val=cache['norm_val']

    #This is where the output goes to: 
    Pxx = {}
    all_channels = set()
    for i,j in ij:
        all_channels.add(i); all_channels.add(j)
    n_channels = len(all_channels)

    for i in all_channels:
        #dbg:
        #print i
        #If we made the conjugate slices:
        if FFT_conj_slices:
            Pxx[i] = FFT_slices[i] * FFT_conj_slices[i]
        else:
            Pxx[i] = FFT_slices[i] * np.conjugate(FFT_slices[i])
        
        #If there is more than one window
        if FFT_slices[i].shape[0]>1:
            Pxx[i] = np.mean(Pxx[i],0)

        Pxx[i] /= norm_val
    
    
    return Pxx

def cache_to_phase(cache,ij):
    """ From a set of cached set of windowed fft's, calculate the
    frequency-band dependent phase for each of the channels in ij.
    Note that this returns the absolute phases of the time-series, not the
    relative phases between them. In order to get relative phases, use
    cache_to_relative_phase 

    Parameters
    ----------
    cache : dict
         The return value of  :func:`cache_fft`

    ij: list
       A list of tuples of the form (i,j) for all the indices for which to
       calculate the phases 

    Returns
    -------

    Phase : dict
         The individual phases, keys are all the i and j in ij, such that
         Phase[i] gives you the phase for the time-series i in the input to
         :func:`cache_fft` 

    """
    FFT_slices=cache['FFT_slices']
    
    Phase = {}

    all_channels = set()
    for i,j in ij:
        all_channels.add(i); all_channels.add(j)
    n_channels = len(all_channels)

    for i in all_channels:
        Phase[i] = np.angle(FFT_slices[i])
        #If there is more than one window, average over all the windows: 
        if FFT_slices[i].shape[0]>1:
            Phase[i] = np.mean(Phase[i],0)
    
    return Phase

def cache_to_relative_phase(cache,ij):
    """ From a set of cached set of windowed fft's, calculate the
    frequency-band dependent relative phase for the combinations ij. 

    Parameters
    ----------
    cache: dict
        The return value from :func:`cache_fft`

    ij: list
       A list of tuples of the form (i,j), all the pairs of indices for which to
       calculate the relative phases 

    Returns
    -------

    Phi_xy : dict
        The relative phases between the time-series i and j. Such that
        Phi_xy[i,j] is the phase from time_series[i] to time_series[j]. 

    Note
    ----

    This function will give you a different result than using
    :func:`coherency_phase_spectrum`. This is because
    :func:`coherency_phase_spectrum` calculates the angle based on the average
    psd, whereas this function calculates the average of the angles calculated
    on individual windows.   

    """
        
    #This is the way it is saved by cache_spectra:
    FFT_slices=cache['FFT_slices']
    FFT_conj_slices=cache['FFT_conj_slices']
    norm_val=cache['norm_val']

    freqs = cache['FFT_slices'][ij[0][0]].shape[-1]

    ij_array = np.array(ij)

    channels_i = max(1,max(ij_array[:,0])+1)
    channels_j = max(1,max(ij_array[:,1])+1)
    #Pre-allocate for speed:
    Phi_xy = np.empty((channels_i,channels_j,freqs),dtype=np.complex)

    #These checks take time, so do them up front, not in every iteration:
    if FFT_slices.items()[0][1].shape[0]>1:
        if FFT_conj_slices:
            for i,j in ij:
                phi = np.angle(FFT_slices[i] * FFT_conj_slices[j])
                Phi_xy[i,j] = np.mean(phi,0)
                
        else:
            for i,j in ij:
                phi = np.angle(FFT_slices[i] * np.conjugate(FFT_slices[j]))
                Phi_xy[i,j] = np.mean(phi,0)
                
    else:
        if FFT_conj_slices:
            for i,j in ij:
                Phi_xy[i,j] = np.angle(FFT_slices[i] * FFT_conj_slices[j])
                
        else:
            for i,j in ij:
              Phi_xy[i,j] = np.angle(FFT_slices[i]*np.conjugate(FFT_slices[j]))
        
    return Phi_xy


def cache_to_coherency(cache,ij):
    """From a set of cached spectra, calculate the coherency
    relationships

    Parameters
    ----------
    cache: dict
        the return value from :func:`cache_fft`

    ij: list
      a list of (i,j) tuples, the pairs of indices for which the
      cross-coherency is to be calculated 

    Returns
    -------
    Cxy: dict
       coherence values between the time-series ij. Indexing into this dict
       takes the form Cxy[i,j] in order to extract the coherency between
       time-series i and time-series j in the original input to
       :func:`cache_fft`
    
    """
        
    #This is the way it is saved by cache_spectra:
    FFT_slices=cache['FFT_slices']
    FFT_conj_slices=cache['FFT_conj_slices']
    norm_val=cache['norm_val']

    freqs = cache['FFT_slices'][ij[0][0]].shape[-1]
    
    ij_array = np.array(ij)

    channels_i = max(1,max(ij_array[:,0])+1)
    channels_j = max(1,max(ij_array[:,1])+1)
    Cxy = np.empty((channels_i,channels_j,freqs),dtype=np.complex)

    #These checks take time, so do them up front, not in every iteration:
    if FFT_slices.items()[0][1].shape[0]>1:
        if FFT_conj_slices:
            for i,j in ij:
                #dbg:
                #print i,j
                Pxy = FFT_slices[i] * FFT_conj_slices[j]
                Pxx = FFT_slices[i] * FFT_conj_slices[i]
                Pyy = FFT_slices[j] * FFT_conj_slices[j]
                Pxx = np.mean(Pxx,0)
                Pyy = np.mean(Pyy,0)
                Pxy = np.mean(Pxy,0)
                Pxy /= norm_val
                Pxx /= norm_val
                Pyy /= norm_val
                Cxy[i,j] = Pxy / np.sqrt(Pxx*Pyy)
                
        else:
            for i,j in ij:
                Pxy = FFT_slices[i] * np.conjugate(FFT_slices[j])
                Pxx = FFT_slices[i] * np.conjugate(FFT_slices[i])
                Pyy = FFT_slices[j] * np.conjugate(FFT_slices[j])
                Pxx = np.mean(Pxx,0)
                Pyy = np.mean(Pyy,0)
                Pxy = np.mean(Pxy,0)
                Pxy /= norm_val
                Pxx /= norm_val
                Pyy /= norm_val
                Cxy[i,j] =  Pxy / np.sqrt(Pxx*Pyy)
    else:
        if FFT_conj_slices:
            for i,j in ij:
                Pxy = FFT_slices[i] * FFT_conj_slices[j]
                Pxx = FFT_slices[i] * FFT_conj_slices[i]
                Pyy = FFT_slices[j] * FFT_conj_slices[j]
                Pxy /= norm_val
                Pxx /= norm_val
                Pyy /= norm_val
                Cxy[i,j] = Pxy / np.sqrt(Pxx*Pyy)
                
        else:
            for i,j in ij:
                Pxy = FFT_slices[i] * np.conjugate(FFT_slices[j])
                Pxx = FFT_slices[i] * np.conjugate(FFT_slices[i])
                Pyy = FFT_slices[j] * np.conjugate(FFT_slices[j])
                Pxy /= norm_val
                Pxx /= norm_val
                Pyy /= norm_val
                Cxy[i,j] =  Pxy / np.sqrt(Pxx*Pyy)
        

    return Cxy



#-----------------------------------------------------------------------------
# Signal generation
#-----------------------------------------------------------------------------
def gauss_white_noise(npts):
    """Gaussian white noise.

    XXX - incomplete."""

    # Amplitude - should be a parameter
    a = 1.
    # Constant, band-limited amplitudes
    # XXX - no bandlimiting yet
    amp = np.zeros(npts)
    amp.fill(a)
    
    # uniform phases
    phi = np.random.uniform(high=2*np.pi, size=npts)
    # frequency-domain signal
    c = amp*np.exp(1j*phi)
    # time-domain
    n = np.fft.ifft(c)

    # XXX No validation that output is gaussian enough yet
    return n
        
#TODO:
# * Write tests for various morlet wavelets
# * Possibly write 'full morlet wavelet' function
def wfmorlet_fft(f0,sd,samplingrate,ns=5,nt=None):
    """
    returns a complex morlet wavelet in the frequency domain

    Parameters
    ----------
        f0 : center frequency
        sd : standard deviation of center frequency
        sampling_rate : samplingrate
        ns : window length in number of stanard deviations
        nt : window length in number of sample points
    """
    if nt==None:
        st = 1./(2.*np.pi*sd)
        nt = 2*int(ns*st*sampling_rate)+1
    f = np.fft.fftfreq(nt,1./sampling_rate)
    wf = 2*np.exp(-(f-f0)**2/(2*sd**2))*np.sqrt(sampling_rate/(np.sqrt(np.pi)*sd))
    wf[f<0] = 0
    wf[f==0] /= 2
    return wf

def wmorlet(f0,sd,sampling_rate,ns=5,normed='area'):
    """
    returns a complex morlet wavelet in the time domain

    Parameters
    ----------
        f0 : center frequency
        sd : standard deviation of frequency
        sampling_rate : samplingrate
        ns : window length in number of stanard deviations
    """
    st = 1./(2.*np.pi*sd)
    w_sz = float(int(ns*st*sampling_rate)) # half time window size
    t = np.arange(-w_sz,w_sz+1,dtype=float)/sampling_rate
    if normed == 'area':
        w = np.exp(-t**2/(2.*st**2))*np.exp(
            2j*np.pi*f0*t)/np.sqrt(np.sqrt(np.pi)*st*sampling_rate)
    elif normed == 'max':
        w = np.exp(-t**2/(2.*st**2))*np.exp(
            2j*np.pi*f0*t)*2*sd*np.sqrt(2*np.pi)/sampling_rate
    else:
        assert 0, 'unknown norm %s'%normed
    return w

def wlogmorlet_fft(f0,sd,sampling_rate,ns=5,nt=None):
    """
    returns a complex log morlet wavelet in the frequency domain

    Parameters
    ----------
        f0 : center frequency
        sd : standard deviation
        sampling_rate : samplingrate
        ns : window length in number of stanard deviations
        nt : window length in number of sample points
    """
    if nt==None:
        st = 1./(2.*np.pi*sd)
        nt = 2*int(ns*st*sampling_rate)+1
    f = np.fft.fftfreq(nt,1./sampling_rate)

    sfl = np.log(1+1.*sd/f0)
    wf = 2*np.exp(-(np.log(f)-np.log(f0))**2/(2*sfl**2))*np.sqrt(sampling_rate/(np.sqrt(np.pi)*sd))
    wf[f<0] = 0
    wf[f==0] /= 2
    return wf

def wlogmorlet(f0,sd,sampling_rate,ns=5,normed='area'):
    """
    returns a complex log morlet wavelet in the time domain

    Parameters
    ----------
        f0 : center frequency
        sd : standard deviation of frequency
        sampling_rate : samplingrate
        ns : window length in number of stanard deviations
    """
    st = 1./(2.*np.pi*sd)
    w_sz = int(ns*st*sampling_rate) # half time window size
    wf = wlogmorlet_fft(f0,sd,sampling_rate=sampling_rate,nt=2*w_sz+1)
    w = np.fft.fftshift(np.fft.ifft(wf))
    if normed == 'area':
        w /= w.real.sum()
    elif normed == 'max':
        w /= w.real.max()
    elif normed == 'energy':
        w /= np.sqrt((w**2).sum())
    else:
        assert 0, 'unknown norm %s'%normed
    return w

