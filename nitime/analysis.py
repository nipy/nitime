#-----------------------------------------------------------------------------
# Nitime analysis 
#-----------------------------------------------------------------------------

"""These classes are used in order to bridge between the time series objects
and the algorithms provided in the algorithms library. The different analysis
objects contain methods in order to call a family of algorithms and caches
quantities related to this particular family. In general, the objects
initialize on a time series object and analytical results are then derived from
the combination of that time-series and the algorithms  """

#Imports:
import numpy as np
import scipy.signal as signal
import scipy.stats as stats
from nitime import descriptors as desc
from nitime import utils as tsu
from nitime import algorithms as tsa
from nitime import timeseries as ts

##Spectral estimation: 
class SpectralAnalyzer(desc.ResetMixin):

    """ Analyzer object for spectral analysis """
    def __init__(self,time_series,method=None):
        self.data = time_series.data
        self.sampling_rate = time_series.sampling_rate
        self.method=method
        
        if self.method is None:
            self.method = {}

    @desc.setattr_on_read
    def spectrum_fourier(self):
        """ Simply the non-normalized Fourier transform for a real signal"""

        fft = np.fft.fft
        f = tsu.get_freqs(self.sampling_rate,self.data.shape[-1])
        spectrum_fourier = fft(self.data)[...,:f.shape[0]]
        return f,spectrum_fourier 
        
    @desc.setattr_on_read
    def spectrum_mlab(self):
        """The spectrum and cross-spectra, computed using mlab csd """

        self.mlab_method = self.method
        self.mlab_method['this_method'] = 'mlab'
        self.mlab_method['Fs'] = self.sampling_rate
        f,spectrum_mlab = tsa.get_spectra(self.data,method=self.mlab_method)

        return f,spectrum_mlab
    
    @desc.setattr_on_read
    def spectrum_multi_taper(self):
        """The spectrum and cross-spectra, computed using multi-tapered csd """

        self.multi_taper_method = np.copy(self.method)
        self.multi_taper_method['this_method'] = 'multi_taper_csd'
        self.multi_taper_method['Fs'] = self.sampling_rate
        f,spectrum_multi_taper = tsa.get_spectra(self.data,
                                               method=self.multi_taper_method)
        return f,spectrum_multi_taper
    
    
##Bivariate methods:  
class CoherenceAnalyzer(desc.ResetMixin):
    """ Analyzer object for coherence/y analysis"""
    
    def __init__(self,time_series,method=None):
        #Initialize variables from the time series
        self.data = time_series.data
        self.sampling_rate = time_series.sampling_rate
        self.time = time_series.time
        
        #Set the variables for spectral estimation (can also be entered by user):
        if method is None:
            self.method = {'this_method':'mlab'}

        else:
            self.method = method
            
        self.method['Fs'] = self.method.get('Fs',self.sampling_rate)

    @desc.setattr_on_read
    def spectrum(self):
        f,spectrum = tsa.get_spectra(self.data,method=self.method)
        return spectrum

    @desc.setattr_on_read
    def frequencies(self):
        f,spectrum = tsa.get_spectra(self.data,method=self.method)
        return f
    
    @desc.setattr_on_read
    def coherence(self):

        tseries_length = self.data.shape[0]
        spectrum_length = self.spectrum.shape[-1]
        coherence=np.zeros((tseries_length,
                            tseries_length,
                            spectrum_length))
    
        for i in xrange(tseries_length): 
            for j in xrange(i,tseries_length):
                coherence[i][j] = tsa.coherence_calculate(self.spectrum[i][j],
                                                      self.spectrum[i][i],
                                                      self.spectrum[j][j])  

        idx = tsu.tril_indices(tseries_length,-1)
        coherence[idx[0],idx[1],...] = coherence[idx[1],idx[0],...].conj()
        
        return coherence

    @desc.setattr_on_read
    def coherency(self):

        tseries_length = self.data.shape[0]
        spectrum_length = self.spectrum.shape[-1]

        coherency=np.zeros((tseries_length,
                            tseries_length,
                            spectrum_length),dtype=complex)
    
        for i in xrange(tseries_length): 
            for j in xrange(i,tseries_length):
                coherency[i][j] = tsa.coherency_calculate(self.spectrum[i][j],
                                                      self.spectrum[i][i],
                                                      self.spectrum[j][j])  

        idx = tsu.tril_indices(tseries_length,-1)
        coherency[idx[0],idx[1],...] = coherency[idx[1],idx[0],...].conj()
        
        return coherency
    
    @desc.setattr_on_read
    def phase(self):
        """ The frequency-dependent phase relationship between all the pairwise
        combinations of time-series in the data"""
        tseries_length = self.data.shape[0]
        spectrum_length = self.spectrum.shape[-1]

        phase = np.zeros((tseries_length,
                            tseries_length,
                            spectrum_length))

        for i in xrange(tseries_length): 
            for j in xrange(i,tseries_length):
                phase[i][j] = tsa.coherency_phase_spectrum_calculate\
                        (self.spectrum[i][j])

                phase[j][i] = tsa.coherency_phase_spectrum_calculate\
                        (self.spectrum[i][j].conjugate())
        return phase
    
    @desc.setattr_on_read
    def delay(self):
        """ The delay in seconds between the two time series """
        p_shape = self.phase.shape[:-1]
        delay = np.zeros(self.phase.shape)
        for i in xrange(p_shape[0]):
            for j in xrange(p_shape[1]):
                #Calculate the delay, unwrapping the phases:
                this_phase = self.phase[i,j]
                this_phase = tsu.unwrap_phases(this_phase)
                delay[i,j] = this_phase / (2*np.pi*self.frequencies)
                
        return delay
    
    @desc.setattr_on_read
    def coherence_partial(self):
        """The partial coherence between data[i] and data[j], given data[k], as
        a function of frequency band"""

        tseries_length = self.data.shape[0]
        spectrum_length = self.spectrum.shape[-1]

        p_coherence=np.zeros((tseries_length,
                              tseries_length,
                              tseries_length,
                              spectrum_length),dtype=complex)
    
        for i in xrange(tseries_length): 
            for j in xrange(tseries_length):
                for k in xrange(t_series_length):
                    p_coherence[i][j][k]=tsa.coherence_partial_calculate(
                        self.spectrum[i][j],
                        self.spectrum[i][i],
                        self.spectrum[j][j],
                        self.spectrum[i][k],
                        self.spectrum[j][k],
                        self.spectrum[k][k])  

        return p_coherence        
        
class SparseCoherenceAnalyzer(desc.ResetMixin):
    """This analyzer is intended for analysis of large sets of data, in which
    possibly only a subset of combinations of time-series needs to be compared.
    The constructor for this class receives as input not only a time-series
    object, but also a list of tuples with index combinations (i,j) for the
    combinations. Importantly, this class implements only the mlab csd function
    and cannot use other methods of spectral estimation""" 

    def __init__(self,time_series,ij,method=None,lb=0,ub=None,
                 prefer_speed_over_memory=False,
                 scale_by_freq=True):
        """The constructor for the SparseCoherenceAnalyzer

        Parameters
        ----------

        time_series: a time-series object
    
        ij: a list of tuples, each containing a pair of indices.

           The resulting cache will contain the fft of time-series in the rows
           indexed by the unique elements of the union of i and j
    
        lb,ub: float,optional, default: lb=0, ub=None (max frequency)

            define a frequency band of interest

        prefer_speed_over_memory: Boolean, optional, default=False

            Does exactly what the name implies. If you have enough memory

        method: optional, dict

         The method for spectral estimation (see `func`:algorithms.get_spectra:)

        """ 
        #Initialize variables from the time series
        self.data = time_series.data
        self.sampling_rate = time_series.sampling_rate
        self.ij = ij
        #Set the variables for spectral estimation (can also be entered by user):
        if method is None:
            self.method = {'this_method':'mlab'}

        else:
            self.method = method

        if self.method['this_method']!='mlab':
            raise ValueError("For SparseCoherenceAnalyzer, spectral estimation"
            "method must be mlab")
            
        self.method['Fs'] = self.method.get('Fs',self.sampling_rate)

        #Additional parameters for the coherency estimation: 
        self.lb = lb
        self.ub = ub
        self.prefer_speed_over_memory = prefer_speed_over_memory
        self.scale_by_freq = scale_by_freq
        
    @desc.setattr_on_read
    def cache(self):
        """Caches the fft windows required by the other methods of the
        SparseCoherenceAnalyzer. Calculate only once and reuse
        """
        f,cache = tsa.cache_fft(self.data,self.ij,
                          lb=self.lb,ub=self.ub,
                          method=self.method,
                          prefer_speed_over_memory=self.prefer_speed_over_memory,
                          scale_by_freq=self.scale_by_freq)

        return cache
    
    @desc.setattr_on_read
    def coherency(self):
        coherency = tsa.cache_to_coherency(self.cache,self.ij)

        return coherency
    
    @desc.setattr_on_read
    def spectrum(self):
        """get the spectrum for the collection of time-series in this analyzer
        """ 
        spectrum = tsa.cache_to_psd(self.cache,self.ij)

        return spectrum
    
    @desc.setattr_on_read
    def phases(self):
        """The frequency-band dependent phases of the spectra of the
           time-series i,j in the analyzer"""
        
        phase= tsa.cache_to_phase(self.cache,self.ij)

        return phase

    @desc.setattr_on_read
    def frequencies(self):
        """Get the central frequencies for the frequency bands, given the
           method of estimating the spectrum """

        NFFT = self.method.get('NFFT',64)
        Fs = self.method.get('Fs')
        freqs = tsu.get_freqs(Fs,NFFT)
        lb_idx,ub_idx = tsu.get_bounds(freqs,self.lb,self.ub)
        
        return freqs[lb_idx:ub_idx]
        
class CorrelationAnalyzer(desc.ResetMixin):
    """Analyzer object for correlation analysis. Has the same API as the
    CoherenceAnalyzer"""

    def __init__(self,time_series):
        #Initialize data from the time series
        self.data = time_series.data
        self.sampling_interval=time_series.sampling_interval

    @desc.setattr_on_read
    def correlation(self):
        """The correlation coefficient between every pairwise combination of
        time-series contained in the object""" 

        return np.corrcoef(self.data)  

    @desc.setattr_on_read
    def xcorr(self):
        """The cross-correlation between every pairwise combination time-series
        in the object. Uses np.correlation('full').

        Returns
        -------

        UniformTimeSeries: the time-dependent cross-correlation, with zero-lag
        at time=0"""
        tseries_length = self.data.shape[0]
        t_points = self.data.shape[-1]
        xcorr = np.zeros((tseries_length,
                          tseries_length,
                          t_points*2-1))
         
        for i in xrange(tseries_length): 
            for j in xrange(i,tseries_length):
                xcorr[i][j] = tsu.xcorr(self.data[i],self.data[j])

        idx = tsu.tril_indices(tseries_length,-1)
        xcorr[idx[0],idx[1],...] = xcorr[idx[1],idx[0],...]

        return ts.UniformTimeSeries(xcorr,
                                    sampling_interval=self.sampling_interval,
                                    t0=-self.sampling_interval*t_points+1)
    
    @desc.setattr_on_read
    def xcorr_norm(self):
        """The cross-correlation between every pairwise combination time-series
        in the object, where the zero lag correlation is normalized to be equal
        to the correlation coefficient between the time-series

        Returns
        -------

        UniformTimeSeries: the time-dependent cross-correlation, with zero-lag
        at time=0"""

        tseries_length = self.data.shape[0]
        t_points = self.data.shape[-1]
        xcorr = np.zeros((tseries_length,
                          tseries_length,
                          t_points*2-1))
         
        for i in xrange(tseries_length): 
            for j in xrange(i,tseries_length):
                xcorr[i,j] = tsu.xcorr(self.data[i],self.data[j])
                xcorr[i,j] /= (xcorr[i,j,t_points])
                xcorr[i,j] *= self.correlation[i,j]

        idx = tsu.tril_indices(tseries_length,-1)
        xcorr[idx[0],idx[1],...] = xcorr[idx[1],idx[0],...]

        return ts.UniformTimeSeries(xcorr,
                                    sampling_interval=self.sampling_interval,
                                    t0=-self.sampling_interval*t_points)
    
##Event-related analysis:
class EventRelatedAnalyzer(desc.ResetMixin): 
    """Analyzer object for reverse-correlation/event-related analysis.

    XXX Repeated use of the term the fmri specific term 'hrf' should be removed.

    """    

    def __init__(self,time_series,events_time_series,len_hrf,zscore=False,
                 correct_baseline=False,offset=0):
        """
        Parameters
        ----------
        time_series: a time-series object
           A time-series with data on which the event-related analysis proceeds
        
        events_time_series: a time_series object

        The events which occured in tandem with the time-series in the
        EventRelatedAnalyzer. This object's data has to have the same
        dimensions as the data in the EventRelatedAnalyzer object. In each
        sample in the time-series, there is an integer, which denotes the kind
        of event which occured at that time. In time-bins in which
        no event occured, a 0 should be entered. The data in this time series
        object needs to have the same dimensionality as the data in the data
        time-series 

        len_hrf: int
        
        The expected length of the HRF (in the same time-units as
        the events are represented (presumably TR). The size of the block
        dedicated in the fir_matrix to each type of event

        zscore: a flag to return the result in zscore (where relevant)

        correct_baseline: a flag to correct the baseline according to the first
        point in the event-triggered average (where possible)
        
        """ 
        #XXX enable the possibility that the event_time_series only has one
        #dimension, corresponding to time and then all channels have the same
        #series of events (and there is no need to loop over all channels?)
        #XXX Change so that the offset and length of the eta can be given in
        #units of time 

        #Make sure that the offset and the len_hrf values can be used, by
        #padding with zeros before and after:

        s = time_series.data.shape
        zeros_before = np.zeros((s[:-1]+ (abs(offset),)))
        zeros_after = np.zeros((s[:-1]+(abs(len_hrf),)))
        time_series_data = np.hstack([zeros_before,time_series.data,
                                      zeros_after])
        events_data = np.hstack([zeros_before,events_time_series.data,
                                 zeros_after])
        
        #If the events and the time_series have more than 1-d, the analysis can
        #traverse their first dimension
        if events_time_series.data.ndim-1>0:
            self._len_h = events_time_series.data.shape[0]
            self.events = events_data
            self.data = time_series_data
        #Otherwise, in order to extract the array from the first dimension, we
        #wrap it in a list
        
        else:
            self._len_h = 1
            self.events = [events_data]
            self.data = [time_series_data]


        self.sampling_rate = time_series.sampling_rate
        self.sampling_interval = time_series.sampling_interval
        self.len_hrf=int(len_hrf)
        self._zscore=zscore
        self._correct_baseline=correct_baseline
        self._offset=offset
        self.time_unit = time_series.time_unit
        
    @desc.setattr_on_read
    def FIR(self):
        """Calculate the FIR event-related estimated of the HRFs for different
        kinds of events

       Returns
        -------

        A time-series object, shape[:-2] are dimensions corresponding to the to
        shape[:-2] of the EventRelatedAnalyzer data, shape[-2] corresponds to
        the different kinds of events used (ordered according to the sorted
        order of the unique components in the events time-series). shape[-1]
        corresponds to time, and has length = len_hrf

        XXX code needs to be changed to use flattening (see 'eta' below)
        
        """
            
        #Make a list to put the outputs in:
        h = [0] * self._len_h

        for i in xrange(self._len_h):
            #XXX Check that the offset makes sense (there can't be an event
            #happening within one offset duration of the beginning of the
            #time-series:

            #Get the design matrix (roll by the offset, in order to get the
            #right thing): 

            roll_events = np.roll(self.events[i],self._offset)
            design = tsu.fir_design_matrix(roll_events,self.len_hrf+
                                           abs(self._offset))
            #Compute the fir estimate, in linear form: 
            this_h = tsa.fir(self.data[i],design)
            #Reshape the linear fir estimate into a event_types*hrf_len array
            u = np.unique(self.events[i])
            event_types = u[np.unique(self.events[i])!=0]
            h[i] =np.reshape(this_h,(event_types.shape[0],self.len_hrf+
                                     abs(self._offset)))

        h = np.array(h).squeeze()

        return ts.UniformTimeSeries(data=h,sampling_rate=self.sampling_rate,
                                 t0=-1*self.len_hrf*self.sampling_interval,
                                 time_unit=self.time_unit)

    
    @desc.setattr_on_read
    def FIR_estimate(self):
        """Calculate back the LTI estimate of the time-series, from FIR"""
        raise NotImplementedError
    
    @desc.setattr_on_read
    def xcorr_eta(self):
        """Compute the normalized cross-correlation estimate of the HRFs for
        different kinds of events
        
        Returns
        -------

        A time-series object, shape[:-2] are dimensions corresponding to the to
        shape[:-2] of the EventRelatedAnalyzer data, shape[-2] corresponds to
        the different kinds of events used (ordered according to the sorted
        order of the unique components in the events time-series). shape[-1]
        corresponds to time, and has length = len_hrf*2 (xcorr looks both back
        and forward for this length)


        XXX code needs to be changed to use flattening (see 'eta' below)
        """
        #Make a list to put the outputs in:
        h = [0] * self._len_h

        for i in xrange(self._len_h):
            data = self.data[i]
            u = np.unique(self.events[i])
            event_types = u[np.unique(self.events[i])!=0]
            h[i] = np.empty((event_types.shape[0],self.len_hrf*2),dtype=complex)
            for e_idx in xrange(event_types.shape[0]):
                this_e = (self.events[i]==event_types[e_idx]) * 1.0
                if self._zscore:
                    this_h = tsa.event_related_zscored(data,
                                            this_e,
                                            self.len_hrf,
                                            self.len_hrf
                                            )
                else:
                    this_h = tsa.event_related(data,
                                            this_e,
                                            self.len_hrf,
                                            self.len_hrf
                                            )
                    
                h[i][e_idx] = this_h
                
        h = np.array(h).squeeze()

        ## t0 for the object returned here needs to be the central time, not the
        ## first time point, because the functions 'look' back and forth for
        ## len_hrf bins

        return ts.UniformTimeSeries(data=h,
                                 sampling_rate=self.sampling_rate,
                                 t0 = -1*self.len_hrf*self.sampling_interval,
                                 time_unit=self.time_unit)

    @desc.setattr_on_read
    def eta(self):
        """The event-triggered average activity """
        #Make a list fo the output 
        h = [0] * self._len_h

        for i in xrange(self._len_h):
            data = self.data[i]
            u = np.unique(self.events[i])
            event_types = u[np.unique(self.events[i])!=0]
            h[i] = np.empty((event_types.shape[0],self.len_hrf),dtype=complex)
            for e_idx in xrange(event_types.shape[0]):
                idx = np.where(self.events[i]==event_types[e_idx])
                idx_w_len = np.array([idx[0]+count+self._offset for count
                                      in range(self.len_hrf)])
                event_trig = data[idx_w_len]
                #Correct baseline by removing the first point in the series for
                #each channel:
                if self._correct_baseline:
                    event_trig -= event_trig[0]
                    
                h[i][e_idx] = np.mean(event_trig,-1)
                
        h = np.array(h).squeeze()

#If the events were the same for all the channels, maybe you can take an
#        alternative approach?

#        d_flat = np.ravel(self.data)
#        e_flat = np.ravel(self.events)
#        u = np.unique(e_flat)
#        event_types = u[np.unique(self.events[i])!=0]
#        for e in event_types: 
#            idx = np.where(e_flat==e)
#            idx_new = np.array([idx[0]+i for i in range(self.len_hrf)])

        return ts.UniformTimeSeries(data=h,
                                 sampling_interval=self.sampling_interval,
                                 t0=self._offset*self.sampling_interval,
                                 time_unit=self.time_unit)

    @desc.setattr_on_read
    def ets(self):
        """The event-triggered standard error of the mean """
        #Make a list fo the output 
        h = [0] * self._len_h

        for i in xrange(self._len_h):
            data = self.data[i]
            u = np.unique(self.events[i])
            event_types = u[np.unique(self.events[i])!=0]
            h[i] = np.empty((event_types.shape[0],self.len_hrf),dtype=complex)
            for e_idx in xrange(event_types.shape[0]):
                idx = np.where(self.events[i]==event_types[e_idx])
                idx_w_len = np.array([idx[0]+count+self._offset for count
                                      in range(self.len_hrf)])
                event_trig = data[idx_w_len]
                #Correct baseline by removing the first point in the series for
                #each channel:
                if self._correct_baseline:
                    event_trig -= event_trig[0]
                    
                h[i][e_idx] = stats.sem(event_trig,axis=-1)
                
        h = np.array(h).squeeze()

        return ts.UniformTimeSeries(data=h,
                                 sampling_interval=self.sampling_interval,
                                 t0=self._offset*self.sampling_interval,
                                 time_unit=self.time_unit)

            
class HilbertAnalyzer(desc.ResetMixin):

    """Analyzer class for extracting the Hilbert transform """ 

    def __init__(self,time_series,lb=0,ub=None):
        """Constructor function for the Hilbert analyzer class.

        Parameters
        ----------
        
        lb,ub: the upper and lower bounds of the frequency range for which the
        transform is done, where filtering is done using a simple curtailment
        of the Fourier domain 

        """
    
        data_in = time_series.data 

        self.sampling_rate = time_series.sampling_rate
        freqs = tsu.get_freqs(self.sampling_rate,data_in.shape[-1])

        if ub is None:
            ub = freqs[-1]
        
        power = np.fft.fft(data_in)
        idx_0 = np.hstack([np.where(freqs<lb)[0],np.where(freqs>ub)[0]])
        power[...,idx_0] = 0
        power[...,-1*idx_0] = 0 #Take care of the negative frequencies
        data_out = np.fft.ifft(power)

        self.data = np.real(data_out) #In order to make sure that you are not
                                      #left with float-precision residual
                                      #complex parts
                                      
    @desc.setattr_on_read
    def _analytic(self):
        return ts.UniformTimeSeries(data=signal.hilbert(self.data),
                                 sampling_rate=self.sampling_rate)
        
    @desc.setattr_on_read
    def magnitude(self):
        return ts.UniformTimeSeries(data=np.abs(self._analytic.data),
                                 sampling_rate=self.sampling_rate)
                                 
    @desc.setattr_on_read
    def phase(self):
        return ts.UniformTimeSeries(data=np.angle(self._analytic.data),
                                 sampling_rate=self.sampling_rate)

    @desc.setattr_on_read
    def real(self):
        return ts.UniformTimeSeries(data=np.real(self._analytic.data),
                                 sampling_rate=self.sampling_rate)
    


class FilterAnalyzer(desc.ResetMixin):

    """ A class for performing filtering operations on time-series and
    producing the filtered versions of the time-series"""

    
    def __init__(self,time_series,lb=0,ub=None,boxcar_iterations=2):
        self.data = time_series.data 
        self.sampling_rate = time_series.sampling_rate
        self.freqs = tsu.get_freqs(self.sampling_rate,self.data.shape[-1])
        self.ub=ub
        self.lb=lb
        self.time_unit=time_series.time_unit
        self._boxcar_iterations=boxcar_iterations

        
    @desc.setattr_on_read
    def filtered_fourier(self):

        """Filter the time-series by passing it to the Fourier domain and null
        out the frequency bands outside of the range [lb,ub] """
        
        if self.ub is None:
            self.ub = self.freqs[-1]
        
        power = np.fft.fft(self.data)
        idx_0 = np.hstack([np.where(self.freqs<self.lb)[0],
                           np.where(self.freqs>self.ub)[0]])
        
        power[...,idx_0] = 0
        #power[...,-1*idx_0] = 0 #Take care of the negative frequencies
        data_out = np.fft.ifft(power)

        data_out = np.real(data_out) #In order to make sure that you are not
                                      #left with float-precision residual
                                      #complex parts

        return ts.UniformTimeSeries(data=data_out,
                                 sampling_rate=self.sampling_rate,
                                 time_unit=self.time_unit) 

    @desc.setattr_on_read
    def filtered_boxcar(self):
        """ Filte the time-series by a boxcar filter. The low pass filter is
    implemented by convolving with a boxcar function of the right length and
    amplitude and the high-pass filter is implemented by subtracting a low-pass
    version (as above) from the signal"""

        if self.ub is not None:
            ub = self.ub/self.sampling_rate
        else:
            ub=1.0
            
        lb = self.lb/self.sampling_rate

        data_out = tsa.boxcar_filter(self.data,lb=lb,ub=ub,
                                     n_iterations=self._boxcar_iterations)

        return ts.UniformTimeSeries(data=data_out,
                                 sampling_rate=self.sampling_rate,
                                 time_unit=self.time_unit) 

