#This will be a cythonized implementation of the coherenc-from-cache code, which will be compiled and called if cython is available. 

import numpy as np
cimport numpy as np
cimport cython

#------------------------------------------------------------------------------
#Coherency calculated using cached spectra
#------------------------------------------------------------------------------

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

    time_series: an ndarray with time-series, where time is the last dimension

    ij: a list of tuples, each containing a pair of indices. The resulting
    cache will contain the fft of time-series in the rows indexed by the unique
    elements of the union of i and j
    
    lb,ub: defines a frequency band of interest

    method: optional, dict

    Returns
    -------
    freqs, cache

    where: cache = {'FFT_slices':FFT_slices,'FFT_conj_slices':FFT_conj_slices,
             'norm_val':norm_val}

    
    Notes
    ----
    - For now, the only method implemented is 'mlab'
    - Notice that detrending the input is not an option here, in order to save
    time on an empty function call!
    
    """
    if method is None:
        method = {'this_method':'mlab'} #The default
        
    if method['this_method'] == 'mlab':
        NFFT = method.get('NFFT',64)
        Fs = method.get('Fs',2*np.pi)
        window = method.get('window',mlab.window_hanning)
        n_overlap = method.get('n_overlap',int(np.ceil(NFFT/2.0)))
        
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
    """ From a set of cached set of windowed fft's, calculate the psd
    for all the ij"""

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
    frequency-band dependent phase for all the ij"""

    #This is the way it is saved by cache_spectra:
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

def cache_to_coherency(cache,ij):
     """From a set of cached spectra, calculate the coherency
    relationships

    Parameters
    ----------
    cache: a cache with fft's, created by :func:`cache_fft`

    ij: the pairs of indices for which the cross-coherency is to be calculated
        
    """
        
    #This is the way it is saved by cache_spectra:
    FFT_slices=cache['FFT_slices']
    FFT_conj_slices=cache['FFT_conj_slices']
    norm_val=cache['norm_val']

    Pxx = cache_to_psd(cache,ij)
    
    Cxy = {}
    Phase = {}
    for i,j in ij:
        #dbg:
        #print i,j
        #If we made the conjugate slices:
        if FFT_conj_slices:
            Pxy = FFT_slices[i] * FFT_conj_slices[j]
        else:
            Pxy = FFT_slices[i] * np.conjugate(FFT_slices[j])

        #If there is more than one window
        if FFT_slices.items()[0][1].shape[0]>1:
            Pxy = np.mean(Pxy,0)

        Pxy /= norm_val
        Cxy[i,j] = coherency_calculate(Pxy,Pxx[i],Pxx[j])
       
       
    return Cxy

