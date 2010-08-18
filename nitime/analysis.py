#-----------------------------------------------------------------------------
# Nitime analysis 
#-----------------------------------------------------------------------------

"""
Nitime analysis
---------------

This module implements an analysis interface between between time-series
objects implemented in the :mod:`timeseries` module and the algorithms provided
in the :mod:`algorithms` library and other algorithms.

The general pattern of use of Analyzer objects is that they an object is
initialized with a TimeSeries object as input. Depending on the analysis
methods implemented in the particular analysis object, additional inputs may
also be required.

The methods of the object are then implemented as instances of
:obj:`OneTimeProperty`, which means that they are only calculated when they are
needed and then cached for further use.

Analyzer objects are generally implemented inheriting the
:func:`desc.ResetMixin`, which means that they have a :meth:`reset`
method. This method resets the object to its initialized state, in which none
of the :obj:`OneTimeProperty` methods have been calculated. This allows to
change parameter settings of the object and recalculating the quantities in
these methods with the new parameter setting. 

"""

#Imports:
import numpy as np
import scipy
import scipy.signal as signal
import scipy.stats as stats
from nitime import descriptors as desc
from nitime import utils as tsu
from nitime import algorithms as tsa
from nitime import timeseries as ts

# XXX - this one is only used in BaseAnalyzer.parameterlist. Should it be
# imported at module level? 
from inspect import getargspec

    
class BaseAnalyzer(desc.ResetMixin):
    """Analyzer that implements the default data flow.

       All analyzers inherit from this class at least have to
       * implement a __init__ function to set parameters
       * define the 'output' property

       >>> A = BaseAnalyzer()
       >>> A
       BaseAnalyzer(sample_parameter='default value')
       >>> A('data')
       'data'
       >>> A('new data')
       'new data'
       >>> A[2]
       'w'
    """

    @desc.setattr_on_read
    def parameterlist(self):
        plist = getargspec(self.__init__).args
        plist.remove('self')
        plist.remove('input')
        return plist

    @property
    def parameters(self):
        return dict([(p,getattr(self,p,'MISSING')) for p in self.parameterlist])

    def __init__(self,input=None):
        self.input = input
      
    @desc.setattr_on_read
    def output(self):
        """This function currently does nothing and
            is meant to be overwritten by the specific
            analyzer sub-class.
        """
        return None

    def __call__(self,input=None):
        """This fuction runs the analysis on new input
           data and returns the output data.
        """
        if input is None:
            if self.input is None:
                raise ValueError('There is no data to analyze')
            else:
                return self.output
            
        
        self.reset()
        self.input = input
        return self.output

    def __getitem__(self,key):
        try:
            return self.output[key]
        except TypeError:
            raise NotImplementedError, 'This analyzer does not support getitem'

    def __repr__(self):
        params = ', '.join(['%s=%r'%(p,getattr(self,p,'MISSING'))
                                    for p in self.parameterlist])

        return '%s(%s)'%(self.__class__.__name__,params)
    
##Spectral estimation: 
class SpectralAnalyzer(BaseAnalyzer):
    """ Analyzer object for spectral analysis"""
    def __init__(self,input=None,method=None):
        """
        The initialization of the
        
        Parameters
        ----------
        input: time-series objects

        method: dict optional, see :func:`algorithms.get_spectra` for
        specification of the spectral analysis method

        Examples
        --------

        >>> t1 = ts.TimeSeries(data = np.arange(0,1024,1).reshape(2,512),sampling_rate=np.pi)
        >>> s1 = SpectralAnalyzer(t1)
        >>> s1.method['this_method']
        'mlab'
        >>> s1.method['Fs']
        3.14159265359 Hz
        >>> f,s = s1.output
        >>> f
        array([ 0.        ,  0.04908739,  0.09817477,  0.14726216,  0.19634954,
                0.24543693,  0.29452431,  0.3436117 ,  0.39269908,  0.44178647,
                0.49087385,  0.53996124,  0.58904862,  0.63813601,  0.68722339,
                0.73631078,  0.78539816,  0.83448555,  0.88357293,  0.93266032,
                0.9817477 ,  1.03083509,  1.07992247,  1.12900986,  1.17809725,
                1.22718463,  1.27627202,  1.3253594 ,  1.37444679,  1.42353417,
                1.47262156,  1.52170894,  1.57079633])
        >>> s[0,1][0]
        (2877158.0203663893+0j)

        """
        BaseAnalyzer.__init__(self,input)

        self.method=method
        
        if self.method is None:
            self.method = {'this_method':'mlab',
                           'Fs':self.input.sampling_rate}
    @desc.setattr_on_read
    def output(self):
        """
        The standard output for this analyzer is a tuple f,s, where: f is the
        frequency bands associated with the discrete spectral components
        and s is the PSD calculated using :func:`mlab.psd`.
    
        """
        data = self.input.data
        sampling_rate = self.input.sampling_rate
        
        self.mlab_method = self.method
        self.mlab_method['this_method'] = 'mlab'
        self.mlab_method['Fs'] = sampling_rate
        f,spectrum_mlab = tsa.get_spectra(data,method=self.mlab_method)

        return f,spectrum_mlab

    @desc.setattr_on_read
    def spectrum_fourier(self):
        """

        This is the spectrum estimated as the FFT of the time-series

        Returns
        -------
        (f,spectrum): f is an array with the frequencies and spectrum is the
        complex-valued FFT. 
        
        """

        data = self.input.data
        sampling_rate = self.input.sampling_rate
        
        fft = np.fft.fft
        f = tsu.get_freqs(sampling_rate,data.shape[-1])
        spectrum_fourier = fft(data)[...,:f.shape[0]]
        return f,spectrum_fourier 
    
    @desc.setattr_on_read
    def spectrum_multi_taper(self):
        """

        The spectrum and cross-spectra, computed using
        :func:`multi_taper_csd'

        XXX This method needs to be improved to include a clever way of
        figuring out how many tapers to generate and a way to extract the
        estimate of error based on the tapers. 

        """
        data = self.input.data
        sampling_rate = self.input.sampling_rate
        self.multi_taper_method = self.method
        self.multi_taper_method['this_method'] = 'multi_taper_csd'
        self.multi_taper_method['Fs'] = sampling_rate
        f,spectrum_multi_taper = tsa.get_spectra(data,
                                                 method=self.multi_taper_method)
        return f,spectrum_multi_taper
    
##Bivariate methods:  
class CoherenceAnalyzer(BaseAnalyzer):
    """Analyzer object for coherence/coherency analysis """
    
    def __init__(self,input=None,method=None):
        """

        Parameters
        ----------

        input: TimeSeries object

        method: dict, optional, see :func:`get_spectra` documentation for
        details.


        Examples
        --------

        >>> t1 = ts.TimeSeries(data = np.arange(0,1024,1).reshape(2,512),sampling_rate=np.pi)
        >>> c1 = ta.CoherenceAnalyzer(t1)
        >>> c1.method['Fs']
        3.14159265359 Hz
        >>> c1.method['this_method']
        'mlab'
        >>> c1[0,1]
        array([ 0.94993377+0.j        ,  0.94950254-0.03322532j,
                0.86963629-0.4570688j ,  0.89177679-0.3847649j ,
                0.90987709-0.31906821j,  0.92173682-0.26785455j,
                0.92944359-0.22848318j,  0.93460158-0.19774838j,
                0.93817683-0.17323391j,  0.94073760-0.15325746j,
                0.94262536-0.13665662j,  0.94405195-0.12261778j,
                0.94515318-0.11056055j,  0.94601882-0.10006254j,
                0.94670992-0.09081034j,  0.94726903-0.08256727j,
                0.94772646-0.07515169j,  0.94810425-0.06842227j,
                0.94841870-0.06226777j,  0.94868206-0.0565999j ,
                0.94890362-0.05134834j,  0.94909057-0.0464573j ,
                0.94924845-0.04188344j,  0.94938163-0.03759492j,
                0.94949349-0.033572j  ,  0.94958665-0.02980985j,
                0.94966305-0.02632586j,  0.94972394-0.02317706j,
                0.94976954-0.0205051j ,  0.94979758-0.01867258j,
                0.94979557-0.01880992j,  0.94965655-0.02664024j,  1.00000000+0.j        ])
        >>> c1.phase[0,1]
        array([ 0.        , -0.03497807, -0.4839064 , -0.40732853, -0.33727315,
               -0.28280862, -0.24104816, -0.20851049, -0.18259287, -0.16149329,
               -0.14397143, -0.12916149, -0.11644712, -0.10538042, -0.09562945,
               -0.08694374, -0.07913123, -0.07204255, -0.06556021, -0.05959097,
               -0.0540606 , -0.04891024, -0.04409413, -0.0395787 , -0.03534307,
               -0.03138214, -0.02771417, -0.02439915, -0.0215862 , -0.019657  ,
               -0.01980159, -0.02804515,  0.        ])

        """ 
        BaseAnalyzer.__init__(self,input)
        
        #Set the variables for spectral estimation (can also be entered by user):
        if method is None:
            self.method = {'this_method':'mlab'}
        else:
            self.method = method

        #If an input is provided, get the sampling rate from there, if you want
        #to over-ride that, input a method with a 'Fs' field specified: 
        self.method['Fs'] = self.method.get('Fs',self.input.sampling_rate)

    @desc.setattr_on_read
    def output(self):
        """The standard output for this kind of analyzer is the coherency """
        data = self.input.data
        tseries_length = data.shape[0]
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
    def spectrum(self):
        """

        The spectra of each of the channels and cross-spectra between
        different channles  in the input TimeSeries object

        """
        f,spectrum = tsa.get_spectra(self.input.data,method=self.method)
        return spectrum

    @desc.setattr_on_read
    def frequencies(self):
        """

        The central frequencies in the bands
        
        """

        #XXX Use NFFT in the method in order to calculate these, without having
        #to calculate the spectrum: 
        f,spectrum = tsa.get_spectra(self.input.data,method=self.method)
        return f
    
    @desc.setattr_on_read
    def coherence(self):
        """
        The coherence between the different channels in the input TimeSeries
        object

        """

        #XXX Calculate this from the standard output, instead of recalculating
        #the coherence:
        
        tseries_length = self.input.data.shape[0]
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
    def phase(self):
        """ The frequency-dependent phase relationship between all the pairwise
        combinations of time-series in the data"""

        #XXX calcluate this from the standard output, instead of recalculating:

        tseries_length = self.input.data.shape[0]
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

        tseries_length = self.input.data.shape[0]
        spectrum_length = self.spectrum.shape[-1]

        p_coherence=np.zeros((tseries_length,
                              tseries_length,
                              tseries_length,
                              spectrum_length))
    
        for i in xrange(tseries_length): 
            for j in xrange(tseries_length):
                for k in xrange(tseries_length):
                    if j==k or i==k:
                        pass
                    else: 
                        p_coherence[i][j][k]=tsa.coherence_partial_calculate(
                            self.spectrum[i][j],
                            self.spectrum[i][i],
                            self.spectrum[j][j],
                            self.spectrum[i][k],
                            self.spectrum[j][k],
                            self.spectrum[k][k])
                        
        idx = tsu.tril_indices(tseries_length,-1)
        p_coherence[idx[0],idx[1],...] = p_coherence[idx[1],idx[0],...].conj()

        return p_coherence        
        
class SparseCoherenceAnalyzer(BaseAnalyzer):
    """This analyzer is intended for analysis of large sets of data, in which
    possibly only a subset of combinations of time-series needs to be compared.
    The constructor for this class receives as input not only a time-series
    object, but also a list of tuples with index combinations (i,j) for the
    combinations. Importantly, this class implements only the mlab csd function
    and cannot use other methods of spectral estimation""" 

    def __init__(self,time_series=None,ij=(0,0),method=None,lb=0,ub=None,
                 prefer_speed_over_memory=True,scale_by_freq=True):
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

        The method for spectral estimation (see :func:`algorithms.get_spectra`)

        """
        
        BaseAnalyzer.__init__(self,time_series)
        #Initialize variables from the time series
        self.ij = ij

        #Set the variables for spectral estimation (can also be entered by
        #user): 
        if method is None:
            self.method = {'this_method':'mlab'}

        else:
            self.method = method

        if self.method['this_method']!='mlab':
            raise ValueError("For SparseCoherenceAnalyzer, spectral estimation method must be mlab")
            

        #Additional parameters for the coherency estimation: 
        self.lb = lb
        self.ub = ub
        self.prefer_speed_over_memory = prefer_speed_over_memory
        self.scale_by_freq = scale_by_freq

    @desc.setattr_on_read
    def output(self):
        """ The default behavior is to calculate the cache, extract it and then
        output the coherency""" 
        coherency = tsa.cache_to_coherency(self.cache,self.ij)

        return coherency

    @desc.setattr_on_read
    def coherence(self):
        """ The coherence values for the output"""
        coherence = np.abs(self.output)
       
        return coherence

    @desc.setattr_on_read
    def cache(self):
        """Caches the fft windows required by the other methods of the
        SparseCoherenceAnalyzer. Calculate only once and reuse
        """
        data = self.input.data 
        f,cache = tsa.cache_fft(data,self.ij,
                        lb=self.lb,ub=self.ub,
                        method=self.method,
                        prefer_speed_over_memory=self.prefer_speed_over_memory,
                        scale_by_freq=self.scale_by_freq)

        return cache
    
    @desc.setattr_on_read
    def spectrum(self):
        """get the spectrum for the collection of time-series in this analyzer
        """
        self.method['Fs'] = self.method.get('Fs',self.input.sampling_rate)
        spectrum = cache_to_psd(self.cache,self.ij)

        return spectrum
    
    @desc.setattr_on_read
    def phases(self):
        """The frequency-band dependent phases of the spectra of the
           time-series i,j in the analyzer"""
        
        phase= cache_to_phase(self.cache,self.ij)

        return phase

    @desc.setattr_on_read
    def frequencies(self):
        """Get the central frequencies for the frequency bands, given the
           method of estimating the spectrum """

        self.method['Fs'] = self.method.get('Fs',self.input.sampling_rate)
        NFFT = self.method.get('NFFT',64)
        Fs = self.method.get('Fs')
        freqs = tsu.get_freqs(Fs,NFFT)
        lb_idx,ub_idx = tsu.get_bounds(freqs,self.lb,self.ub)
        
        return freqs[lb_idx:ub_idx]
        
class CorrelationAnalyzer(BaseAnalyzer):
    """Analyzer object for correlation analysis. Has the same API as the
    CoherenceAnalyzer"""

    def __init__(self,input=None):
        BaseAnalyzer.__init__(self,input)

    @desc.setattr_on_read
    def output(self):
        """The correlation coefficient between every pairwise combination of
        time-series contained in the object""" 
        return np.corrcoef(self.input.data)  

    @desc.setattr_on_read
    def xcorr(self):
        """The cross-correlation between every pairwise combination time-series
        in the object. Uses np.correlation('full').

        Returns
        -------

        TimeSeries: the time-dependent cross-correlation, with zero-lag
        at time=0"""
        tseries_length = self.input.data.shape[0]
        t_points = self.input.data.shape[-1]
        xcorr = np.zeros((tseries_length,
                          tseries_length,
                          t_points*2-1))
         
        for i in xrange(tseries_length): 
            for j in xrange(i,tseries_length):
                xcorr[i][j] = tsu.crosscov(
                    self.input.data[i],self.input.data[j],all_lags=True
                    )

        idx = tsu.tril_indices(tseries_length,-1)
        xcorr[idx[0],idx[1],...] = xcorr[idx[1],idx[0],...]

        return ts.TimeSeries(xcorr,
                                sampling_interval=self.input.sampling_interval,
                                t0=-self.input.sampling_interval*t_points)
    
    @desc.setattr_on_read
    def xcorr_norm(self):
        """The cross-correlation between every pairwise combination time-series
        in the object, where the zero lag correlation is normalized to be equal
        to the correlation coefficient between the time-series

        Returns
        -------

        TimeSeries: the time-dependent cross-correlation, with zero-lag
        at time=0"""

        tseries_length = self.input.data.shape[0]
        t_points = self.input.data.shape[-1]
        xcorr = np.zeros((tseries_length,
                          tseries_length,
                          t_points*2-1))
         
        for i in xrange(tseries_length): 
            for j in xrange(i,tseries_length):
                xcorr[i,j] = tsu.crosscov(
                    self.input.data[i],self.input.data[j],all_lags=True
                    )
                xcorr[i,j] /= (xcorr[i,j,t_points])
                xcorr[i,j] *= self.output[i,j]

        idx = tsu.tril_indices(tseries_length,-1)
        xcorr[idx[0],idx[1],...] = xcorr[idx[1],idx[0],...]

        return ts.TimeSeries(xcorr,
                                sampling_interval=self.input.sampling_interval,
                                t0=-self.input.sampling_interval*t_points)


##Event-related analysis:
class EventRelatedAnalyzer(desc.ResetMixin): 
    """Analyzer object for reverse-correlation/event-related analysis.
    
    Note: right now, this class assumes the input time series is only
    two-dimensional.  If your input data is something like
    (nchannels,nsubjects, ...) with more dimensions, things are likely to break
    in hard to understand ways.
    """
    
    def __init__(self,time_series,events,len_et,zscore=False,
                 correct_baseline=False,offset=0):
        """
        Parameters
        ----------
        time_series: a time-series object
           A time-series with data on which the event-related analysis proceeds
        
        events_time_series: a TimeSeries object or an Events object

        The events which occured in tandem with the time-series in the
        EventRelatedAnalyzer. This object's data has to have the same
        dimensions as the data in the EventRelatedAnalyzer object. In each
        sample in the time-series, there is an integer, which denotes the kind
        of event which occured at that time. In time-bins in which
        no event occured, a 0 should be entered. The data in this time series
        object needs to have the same dimensionality as the data in the data
        time-series 

        len_et: int
        
        The expected length of the event-triggered quantity (in the same
        time-units as the events are represented (presumably number of TRs, for
        fMRI data). For example, the size of the block dedicated in the
        fir_matrix to each type of event

        zscore: a flag to return the result in zscore (where relevant)

        correct_baseline: a flag to correct the baseline according to the first
        point in the event-triggered average (where possible)
        
        """ 
        #XXX Change so that the offset and length of the eta can be given in
        #units of time 

        #Make sure that the offset and the len_et values can be used, by
        #padding with zeros before and after:

        if  isinstance(events, ts.TimeSeries):
            #Set a flag to indicate the input is a time-series object:
            self._is_ts = True
            s = time_series.data.shape
            e_data = np.copy(events.data)
            
            #If the input is a one-dimensional (instead of an n-channel
            #dimensional) time-series, we will need to broadcast to make the
            #data assume the same number of dimensions as the time-series input:
            if len(events.shape) == 1 and len(s)>1: 
                e_data = e_data + np.zeros((s[0],1))
                
            zeros_before = np.zeros((s[:-1]+ (abs(offset),)))
            zeros_after = np.zeros((s[:-1]+(abs(len_et),)))
            time_series_data = np.hstack([zeros_before,time_series.data,
                                          zeros_after])
            events_data = np.hstack([zeros_before,e_data,
                                     zeros_after])

            #If the events and the time_series have more than 1-d, the analysis
            #can traverse their first dimension
            if time_series.data.ndim-1>0:
                self._len_h = time_series.data.shape[0]
                self.events = events_data
                self.data = time_series_data
            #Otherwise, in order to extract the array from the first dimension,
            #we wrap it in a list

            else:
                self._len_h = 1
                self.events = [events_data]
                self.data = [time_series_data]
        
        elif isinstance(events,ts.Events):
            #Set 
            self._is_ts=False
            s = time_series.data.shape
            zeros_before = np.zeros((s[:-1]+ (abs(offset),)))
            zeros_after = np.zeros((s[:-1]+(abs(len_et),)))

            #If the time_series has more than 1-d, the analysis can traverse the
            #first dimension
            if time_series.data.ndim-1>0:
                self._len_h = time_series.shape[0]
                self.data = time_series
                self.events = events

            #Otherwise, in order to extract the array from the first dimension,
            #we wrap it in a list
            else:
                self._len_h = 1
                self.data = [time_series]
                #No need to do that for the Events object:
                self.events = events
        else:
            raise ValueError("Input 'events' to EventRelatedAnalyzer must be of type Events or of type TimeSeries")
   
        self.sampling_rate = time_series.sampling_rate
        self.sampling_interval = time_series.sampling_interval
        self.len_et=int(len_et)
        self._zscore=zscore
        self._correct_baseline=correct_baseline
        self.offset=offset
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
        corresponds to time, and has length = len_et

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

            roll_events = np.roll(self.events[i],self.offset)
            design = tsu.fir_design_matrix(roll_events,self.len_et)
            #Compute the fir estimate, in linear form: 
            this_h = tsa.fir(self.data[i],design)
            #Reshape the linear fir estimate into a event_types*hrf_len array
            u = np.unique(self.events[i])
            event_types = u[np.unique(self.events[i])!=0]
            h[i] =np.reshape(this_h,(event_types.shape[0],self.len_et))

        h = np.array(h).squeeze()

        return ts.TimeSeries(data=h,sampling_rate=self.sampling_rate,
                                 t0=self.offset*self.sampling_interval,
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
        corresponds to time, and has length = len_et*2 (xcorr looks both back
        and forward for this length)


        XXX code needs to be changed to use flattening (see 'eta' below)
        """
        #Make a list to put the outputs in:
        h = [0] * self._len_h

        for i in xrange(self._len_h):
            data = self.data[i]
            u = np.unique(self.events[i])
            event_types = u[np.unique(self.events[i])!=0]
            h[i] = np.empty((event_types.shape[0],self.len_et*2),dtype=complex)
            for e_idx in xrange(event_types.shape[0]):
                this_e = (self.events[i]==event_types[e_idx]) * 1.0
                if self._zscore:
                    this_h = tsa.event_related_zscored(data,
                                            this_e,
                                            self.len_et,
                                            self.len_et
                                            )
                else:
                    this_h = tsa.event_related(data,
                                            this_e,
                                            self.len_et,
                                            self.len_et
                                            )
                    
                h[i][e_idx] = this_h
                
        h = np.array(h).squeeze()

        ## t0 for the object returned here needs to be the central time, not the
        ## first time point, because the functions 'look' back and forth for
        ## len_et bins

        return ts.TimeSeries(data=h,
                                 sampling_rate=self.sampling_rate,
                                 t0 = -1*self.len_et*self.sampling_interval,
                                 time_unit=self.time_unit)


    @desc.setattr_on_read
    def et_data(self):

        """The event-triggered data (all occurences).

        This gets the time-series corresponding to the inidividual event
        occurences. Returns a list of lists of time-series. The first dimension
        is the different channels in the original time-series data and the
        second dimension is each type of event in the event time series

        The time-series itself has the first diemnsion of the data being the
        specific occurence, with time 0 locked to the that occurence
        of the event and the last dimension is time.e

        This complicated structure is so that it can deal with situations where
        each channel has different events and different events have different #
        of occurences
        """
        #Make a list for the output 
        h = [0] * self._len_h

        for i in xrange(self._len_h):
            data = self.data[i]
            u = np.unique(self.events[i])
            event_types = u[np.unique(self.events[i])!=0]
            #Make a list in here as well:
            this_list = [0] * event_types.shape[0]
            for e_idx in xrange(event_types.shape[0]):
                idx = np.where(self.events[i]==event_types[e_idx])
                
                idx_w_len = np.array([idx[0]+count+self.offset for count
                                      in range(self.len_et)])
                event_trig = data[idx_w_len].T
                this_list[e_idx] = ts.TimeSeries(data=event_trig,
                                 sampling_interval=self.sampling_interval,
                                 t0=self.offset*self.sampling_interval,
                                 time_unit=self.time_unit)

            h[i] = this_list

        return h

    @desc.setattr_on_read
    def eta(self):
        """The event-triggered average activity.
        """
        #Make a list for the output 
        h = [0] * self._len_h

        if self._is_ts: 
            # Loop over channels
            for i in xrange(self._len_h):
                data = self.data[i]
                u = np.unique(self.events[i])
                event_types = u[np.unique(self.events[i])!=0]
                h[i] = np.empty((event_types.shape[0], self.len_et),
                                dtype=complex)

                # This offset is used to pull the event indices below, but we
                # have to broadcast it so the shape of the resulting idx+offset
                # operation below gives us the (nevents, len_et) array we want,
                # per channel.
                offset = np.arange(self.offset,
                                   self.offset+self.len_et)[:,np.newaxis]
                # Loop over event types
                for e_idx in xrange(event_types.shape[0]):
                    idx = np.where(self.events[i]==event_types[e_idx])[0]
                    event_trig = data[idx + offset]
                    #Correct baseline by removing the first point in the series
                    #for each channel:
                    if self._correct_baseline:
                        event_trig -= event_trig[0]

                    h[i][e_idx] = np.mean(event_trig,-1)

        #In case the input events are an Events:            
        else:
            #Get the indices necessary for extraction of the eta:
            add_offset = np.arange(self.offset,
                                   self.offset+self.len_et)[:,np.newaxis]

            idx = (self.events.time/self.sampling_interval).astype(int)
            
            #Make a list for the output 
            h = [0] * self._len_h

            
            # Loop over channels
            for i in xrange(self._len_h):
                #If this is a list with one element:
                if self._len_h==1:
                    event_trig = self.data[0][idx + add_offset]
                #Otherwise, you need to index straight into the underlying data
                #array:
                else:
                    event_trig = self.data.data[i][idx + add_offset]
                
                h[i]= np.mean(event_trig,-1)

        h = np.array(h).squeeze()
        return ts.TimeSeries(data=h,
                                 sampling_interval=self.sampling_interval,
                                 t0=self.offset*self.sampling_interval,
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
            h[i] = np.empty((event_types.shape[0],self.len_et),dtype=complex)
            for e_idx in xrange(event_types.shape[0]):
                idx = np.where(self.events[i]==event_types[e_idx])
                idx_w_len = np.array([idx[0]+count+self.offset for count
                                      in range(self.len_et)])
                event_trig = data[idx_w_len]
                #Correct baseline by removing the first point in the series for
                #each channel:
                if self._correct_baseline:
                    event_trig -= event_trig[0]
                    
                h[i][e_idx] = stats.sem(event_trig,axis=-1)
                
        h = np.array(h).squeeze()

        return ts.TimeSeries(data=h,
                                 sampling_interval=self.sampling_interval,
                                 t0=self.offset*self.sampling_interval,
                                 time_unit=self.time_unit)

            
class HilbertAnalyzer(BaseAnalyzer):

    """Analyzer class for extracting the Hilbert transform """ 

    def __init__(self,input=None):
        """Constructor function for the Hilbert analyzer class.

        Parameters
        ----------
        
        input: TimeSeries

        """
        BaseAnalyzer.__init__(self,input)
        
    @desc.setattr_on_read
    def output(self):
        """The natural output for this analyzer is the analytic signal """ 
        data = self.input.data
        sampling_rate = self.input.sampling_rate
        #If you have scipy with the fixed scipy.signal.hilbert (r6205 and later)
        if scipy.__version__>='0.9':
            hilbert = signal.hilbert
        else:
            hilbert = tsu.hilbert_from_new_scipy

        return ts.TimeSeries(data=hilbert(data),
                                        sampling_rate=sampling_rate)
        
    @desc.setattr_on_read
    def amplitude(self):
        return ts.TimeSeries(data=np.abs(self.output.data),
                                 sampling_rate=self.output.sampling_rate)
                                 
    @desc.setattr_on_read
    def phase(self):
        return ts.TimeSeries(data=np.angle(self.output.data),
                                 sampling_rate=self.output.sampling_rate)

    @desc.setattr_on_read
    def real(self):
        return ts.TimeSeries(data=self.output.data.real,
                                    sampling_rate=self.output.sampling_rate)
    
    @desc.setattr_on_read
    def imag(self):
        return ts.TimeSeries(data=self.output.data.imag,
                                    sampling_rate=self.output.sampling_rate)


class FilterAnalyzer(desc.ResetMixin):

    """ A class for performing filtering operations on time-series and
    producing the filtered versions of the time-series"""

    
    def __init__(self,time_series,lb=0,ub=None,boxcar_iterations=2):
        self.data = time_series.data 
        self.sampling_rate = time_series.sampling_rate
        self.ub=ub
        self.lb=lb
        self.time_unit=time_series.time_unit
        self._boxcar_iterations=boxcar_iterations

        
    @desc.setattr_on_read
    def filtered_fourier(self):

        """Filter the time-series by passing it to the Fourier domain and null
        out the frequency bands outside of the range [lb,ub] """

        freqs = tsu.get_freqs(self.sampling_rate,self.data.shape[-1])

        if self.ub is None:
            self.ub = freqs[-1]
            
        power = np.fft.fft(self.data)
        idx_0 = np.hstack([np.where(freqs<self.lb)[0],
                           np.where(freqs>self.ub)[0]])

        #Make sure that you keep the DC component:
        keep_dc = np.copy(power[...,0])
        power[...,idx_0] = 0
        power[...,-1*idx_0] = 0 #Take care of the negative frequencies
        power[...,0] = keep_dc #And put the DC back in when you're done:
        
        data_out = np.fft.ifft(power)

        data_out = np.real(data_out) #In order to make sure that you are not
                                      #left with float-precision residual
                                      #complex parts

        return ts.TimeSeries(data=data_out,
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

        return ts.TimeSeries(data=data_out,
                                 sampling_rate=self.sampling_rate,
                                 time_unit=self.time_unit) 

class NormalizationAnalyzer(BaseAnalyzer):

    """ A class for performing normalization operations on time-series and
    producing the renormalized versions of the time-series"""

    def __init__(self,input=None):
        """Constructor function for the Normalization analyzer class.

        Parameters
        ----------
        
        input: TimeSeries object

        """
        BaseAnalyzer.__init__(self,input)
        
    @desc.setattr_on_read
    def percent_change(self):
        return ts.TimeSeries(tsu.percent_change(self.input.data),
                             sampling_rate=self.input.sampling_rate,
                             time_unit = self.input.time_unit)

    @desc.setattr_on_read
    def z_score(self):
        return ts.TimeSeries(tsu.zscore(self.input.data),
                             sampling_rate=self.input.sampling_rate,
                             time_unit = self.input.time_unit)

#TODO:
# * Write test for MorletWaveletAnalyzer
class MorletWaveletAnalyzer(BaseAnalyzer):

    """Analyzer class for extracting the (complex) Morlet wavelet transform """ 

    def __init__(self,input=None,freqs=None,sd_rel=.2,sd=None,f_min=None,
                 f_max=None,nfreqs=None,log_spacing=False, log_morlet=False):
        """Constructor function for the Hilbert analyzer class.

        Parameters
        ----------
        
        freqs: list or float
          List of center frequencies for the wavelet transform, or a scalar
          for a single band-passed signal.

        sd: list or float
          List of filter bandwidths, given as standard-deviation of center
          frequencies. Alternatively sd_rel can be specified.

        sd_rel: float
          Filter bandwidth, given as a fraction of the center frequencies.

        f_min: float
          Minimal frequency.

        f_max: float
          Maximal frequency.

        nfreqs: int
          Number of frequencies.

        log_spacing: bool
          If true, frequencies will be evenly spaced on a log-scale.
          Default: False

        log_morlet: bool
          If True, a log-Morlet wavelet is used, if False, a regular Morlet
          wavelet is used. Default: False
        """
        BaseAnalyzer.__init__(self,input)
        self.freqs = freqs
        self.sd_rel = sd_rel
        self.sd = sd
        self.f_min = f_min
        self.f_max = f_max
        self.nfreqs = nfreqs
        self.log_spacing = log_spacing
        self.log_morlet = log_morlet

        if log_morlet:
            self.wavelet = tsa.wlogmorlet
        else:
            self.wavelet = tsa.wmorlet

        if freqs is not None:
            self.freqs = np.array(freqs)
        elif f_min is not None and f_max is not None and nfreqs is not None:
            if log_spacing:
                self.freqs = np.logspace(np.log10(f_min), np.log10(f_max),
                                         num=nfreqs, endpoint=True)
            else:
                self.freqs = np.linspace(f_min,f_max,num=nfreqs,endpoint=True)
        else:
            raise NotImplementedError

        if sd is None:
            self.sd = self.freqs*self.sd_rel

    @desc.setattr_on_read
    def output(self):
        """The natural output for this analyzer is the analytic signal"""
        data = self.input.data
        sampling_rate = self.input.sampling_rate
        
        a_signal =\
    ts.TimeSeries(data=np.zeros(self.freqs.shape+data.shape,
                                        dtype='D'),sampling_rate=sampling_rate)
        if self.freqs.ndim == 0:
            w = self.wavelet(self.freqs,self.sd,
                             sampling_rate=sampling_rate,ns=5,
                                                     normed='area')

            nd = (w.shape[0]-1)/2
            a_signal.data[...] = (np.convolve(data,np.real(w),mode='same') +
                                  1j*np.convolve(data,np.imag(w),mode='same'))
        else:    
            for i,(f,sd) in enumerate(zip(self.freqs,self.sd)):
                w = self.wavelet(f,sd,sampling_rate=sampling_rate,
                                 ns=5,normed='area')

                nd = (w.shape[0]-1)/2
                a_signal.data[i,...] = (np.convolve(data,np.real(w),mode='same')+1j*np.convolve(data,np.imag(w),mode='same'))
                
        return a_signal

    @desc.setattr_on_read
    def amplitude(self):
        return ts.TimeSeries(data=np.abs(self.output.data),
                                    sampling_rate=self.output.sampling_rate)
                                 
    @desc.setattr_on_read
    def phase(self):
        return ts.TimeSeries(data=np.angle(self.output.data),
                                    sampling_rate=self.output.sampling_rate)

    @desc.setattr_on_read
    def real(self):
        return ts.TimeSeries(data=self.output.data.real,
                                    sampling_rate=self.output.sampling_rate)
    
    @desc.setattr_on_read
    def imag(self):
        return ts.TimeSeries(data=self.output.data.imag,
                                    sampling_rate=self.output.sampling_rate)


