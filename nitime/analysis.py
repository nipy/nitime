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
import scipy.stats.distributions as dist

from nitime import descriptors as desc
from nitime import utils as tsu
from nitime import algorithms as tsa
from nitime import timeseries as ts

# XXX - this one is only used in BaseAnalyzer.parameterlist. Should it be
# imported at module level? 
from inspect import getargspec

class BaseAnalyzer(desc.ResetMixin):
    """
    Analyzer that implements the default data flow.

    All analyzers inherit from this class at least have to
    * implement a __init__ function to set parameters
    * define the 'output' property

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

    def __init__(self, input=None):
        self.input = input

    def set_input(self, input):
        """Set the input of the analyzer, if you want to reuse the analyzer
        with a different input than the original """

        self.reset()
        self.input = input
        
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

        >>> t1 = ts.TimeSeries(data = np.arange(0,1024,1).reshape(2,512),
        ... sampling_rate=np.pi)
        >>> s1 = SpectralAnalyzer(t1)
        >>> s1.method['this_method']
        'welch'
        >>> s1.method['Fs']
        3.14159265359 Hz
        >>> f,s = s1.psd
        >>> f
        array([ 0.    ,  0.0491,  0.0982,  0.1473,  0.1963,  0.2454,  0.2945,
                0.3436,  0.3927,  0.4418,  0.4909,  0.54  ,  0.589 ,  0.6381,
                0.6872,  0.7363,  0.7854,  0.8345,  0.8836,  0.9327,  0.9817,
                1.0308,  1.0799,  1.129 ,  1.1781,  1.2272,  1.2763,  1.3254,
                1.3744,  1.4235,  1.4726,  1.5217,  1.5708])
        >>> s[0,0]   # doctest: +ELLIPSIS
        1128276.92538360...
        """
        BaseAnalyzer.__init__(self,input)

        self.method=method
        
        if self.method is None:
            self.method = {'this_method':'welch',
                           'Fs':self.input.sampling_rate}
    @desc.setattr_on_read
    def psd(self):
        """
        The standard output for this analyzer is a tuple f,s, where: f is the
        frequency bands associated with the discrete spectral components
        and s is the PSD calculated using :func:`mlab.psd`.
    
        """
        
        NFFT = self.method.get('NFFT',64)
        Fs = self.input.sampling_rate
        detrend = self.method.get('detrend',tsa.mlab.detrend_none)
        window = self.method.get('window',tsa.mlab.window_hanning)
        n_overlap = self.method.get('n_overlap',int(np.ceil(NFFT/2.0)))

        if np.iscomplexobj(self.input.data):
            psd_len = NFFT
            dt = complex
        else:
            psd_len = NFFT/2.0 + 1
            dt = float
        psd = np.empty((self.input.shape[0],
                       psd_len),dtype=dt)

        for i in xrange(self.input.data.shape[0]):
            temp,f =  tsa.mlab.psd(self.input.data[i],
                        NFFT=NFFT,
                        Fs=Fs,
                        detrend=detrend,
                        window=window,
                        noverlap=n_overlap)
            psd[i] = temp.squeeze()

        return f,psd
    @desc.setattr_on_read
    def cpsd(self):

        """
        This outputs both the PSD and the CSD calculated using
        :func:`algorithms.get_spectra`.

        Returns
        -------

        (f,s): tuple
           f: Frequency bands over which the psd/csd are calculated and
           s: the n by n by len(f) matrix of PSD (on the main diagonal) and CSD
           (off diagonal)
           
        """
            
        self.welch_method = self.method
        self.welch_method['this_method'] = 'welch'
        self.welch_method['Fs'] = self.input.sampling_rate
        f,spectrum_welch = tsa.get_spectra(self.input.data,
                                           method=self.welch_method)

        return f,spectrum_welch

    @desc.setattr_on_read
    def periodogram(self):
        """

        This is the spectrum estimated as the FFT of the time-series

        Returns
        -------
        (f,spectrum): f is an array with the frequencies and spectrum is the
        complex-valued FFT. 
        
        """

        return tsa.periodogram(self.input.data,Fs=self.input.sampling_rate)

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
    
    def __init__(self,input=None,method=None,unwrap_phases=False):
        """

        Parameters
        ----------

        input: TimeSeries object
           Containing the data to analyze.
           
        method: dict, optional,
            This is the method used for spectral analysis of the signal for the
            coherence caclulation. See :func:`algorithms.get_spectra`
            documentation for details.  

        unwrap_phases: bool, optional
           Whether to unwrap the phases. This should be True if you assume that
           the time-delay is the same for all the frequency bands. See
           _[Sun2005] for details. Default : False   

        Examples
        --------

        >>> t1 = ts.TimeSeries(data = np.arange(0,1024,1).reshape(2,512),sampling_rate=np.pi)
        >>> c1 = CoherenceAnalyzer(t1)
        >>> c1.method['Fs']
        3.14159265359 Hz
        >>> c1.method['this_method']
        'welch'
        >>> c1.coherency[0,1]
        array([ 0.9499+0.j    ,  0.9495-0.0332j,  0.8696-0.4571j,  0.8918-0.3848j,
                0.9099-0.3191j,  0.9217-0.2679j,  0.9294-0.2285j,  0.9346-0.1977j,
                0.9382-0.1732j,  0.9407-0.1533j,  0.9426-0.1367j,  0.9441-0.1226j,
                0.9452-0.1106j,  0.9460-0.1001j,  0.9467-0.0908j,  0.9473-0.0826j,
                0.9477-0.0752j,  0.9481-0.0684j,  0.9484-0.0623j,  0.9487-0.0566j,
                0.9489-0.0513j,  0.9491-0.0465j,  0.9492-0.0419j,  0.9494-0.0376j,
                0.9495-0.0336j,  0.9496-0.0298j,  0.9497-0.0263j,  0.9497-0.0232j,
                0.9498-0.0205j,  0.9498-0.0187j,  0.9498-0.0188j,  0.9497-0.0266j,
                1.0000+0.j    ])

        >>> c1.phase[0,1]
        array([ 0.    , -0.035 , -0.4839, -0.4073, -0.3373, -0.2828, -0.241 ,
               -0.2085, -0.1826, -0.1615, -0.144 , -0.1292, -0.1164, -0.1054,
               -0.0956, -0.0869, -0.0791, -0.072 , -0.0656, -0.0596, -0.0541,
               -0.0489, -0.0441, -0.0396, -0.0353, -0.0314, -0.0277, -0.0244,
               -0.0216, -0.0197, -0.0198, -0.028 ,  0.    ])

        """ 
        BaseAnalyzer.__init__(self,input)
        
        #Set the variables for spectral estimation (can also be entered by user):
        if method is None:
            self.method = {'this_method':'welch'}
        else:
            self.method = method
        
        #If an input is provided, get the sampling rate from there, if you want
        #to over-ride that, input a method with a 'Fs' field specified: 
        self.method['Fs'] = self.method.get('Fs',self.input.sampling_rate)

        self._unwrap_phases = unwrap_phases
        
    @desc.setattr_on_read
    def coherency(self):
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
                this_phase = self.phase[i,j]
                #If requested, unwrap the phases:
                if self._unwrap_phases:
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
    
class MTCoherenceAnalyzer(BaseAnalyzer):
    """ Analyzer for multi-taper coherence analysis, including jack-knife
    estimate of confidence interval """
    def __init__(self, input=None, bandwidth=None, alpha=0.05, adaptive=True):

        """
        Initializer function for the MTCoherenceAnalyzer
        
        Parameters
        ----------

        input: TimeSeries object

        bandwidth: float,
           The bandwidth of the windowing function will determine the number
           tapers to use. This parameters represents trade-off between
           frequency resolution (lower main lobe bandwidth for the taper) and
           variance reduction (higher bandwidth and number of averaged
           estimates). Per default will be set to 4 times the fundamental
           frequency, such that NW=4

        alpha: float, default =0.05
            This is the alpha used to construct a confidence interval around
            the multi-taper csd estimate, based on a jack-knife estimate of the
            variance [Thompson2007]_.

        adaptive: bool, default to True
            Whether to set the weights for the tapered spectra according to the
            adaptive algorithm [Thompson2007]_.
            
            .. [Thompson2007] Thompson, DJ Jackknifing multitaper spectrum
            estimates. IEEE Signal Processing Magazing. 24: 20-30

        """

        BaseAnalyzer.__init__(self,input)

        if input is None:
            self.NW = 4
            self.bandwidth = None
        else:
            N = input.shape[-1]
            Fs = self.input.sampling_rate
            if bandwidth is not None:
                self.NW = bandwidth/(2*Fs) * N
            else:
                self.NW = 4
                self.bandwidth = self.NW * (2*Fs) / N
            
        self.alpha = alpha
        self._L = self.input.data.shape[-1]/2 + 1
        self._adaptive = adaptive
        
    @desc.setattr_on_read
    def tapers(self):
        return tsa.DPSS_windows(self.input.shape[-1], self.NW,
                                2*self.NW-1)[0]
        
    @desc.setattr_on_read
    def eigs(self):
        return tsa.DPSS_windows(self.input.shape[-1], self.NW,
                                      2*self.NW-1)[1]
    @desc.setattr_on_read
    def df(self):
        #The degrees of freedom: 
        return 2*self.NW-1
        
    @desc.setattr_on_read
    def spectra(self):
        tdata = self.tapers[None,:,:] *self.input.data[:,None,:]
        tspectra = np.fft.fft(tdata)
        return tspectra

    @desc.setattr_on_read
    def weights(self):
        channel_n = self.input.data.shape[0]
        w = np.empty( (channel_n, self.df,self._L) )
        

        if self._adaptive:
            mag_sqr_spectra = np.abs(self.spectra)
            np.power(mag_sqr_spectra, 2, mag_sqr_spectra)

            for i in xrange(channel_n):
                w[i] = tsu.adaptive_weights(mag_sqr_spectra[i],
                                            self.eigs,
                                            self._L)[0]
                
        #Set the weights to be the square root of the eigen-values:
        else:
            wshape = [1] * len(self.spectra.shape)
            wshape[0] = channel_n 
            wshape[-2] = int(self.df)
            pre_w = np.sqrt(self.eigs) + np.zeros( (wshape[0],
                                                    self.eigs.shape[0]) )
                
            w = pre_w.reshape(*wshape)
            
        return w

    @desc.setattr_on_read
    def coherence(self):
        nrows = self.input.data.shape[0]
        psd_mat = np.zeros((2, nrows,nrows,self._L), 'd')
        coh_mat = np.zeros((nrows,nrows,self._L), 'd')

        for i in xrange(self.input.data.shape[0]):
           for j in xrange(i):
              sxy = tsa.mtm_cross_spectrum(self.spectra[i],self.spectra[j],
                                           (self.weights[i],self.weights[j]),
                                           sides='onesided')

              sxx = tsa.mtm_cross_spectrum(self.spectra[i],self.spectra[i],
                                           (self.weights[i], self.weights[i]),
                                           sides='onesided').real
              
              syy = tsa.mtm_cross_spectrum(self.spectra[j], self.spectra[j],
                                           (self.weights[i], self.weights[j]),
                                           sides='onesided').real
              psd_mat[0,i,j] = sxx
              psd_mat[1,i,j] = syy
              coh_mat[i,j] = np.abs(sxy)**2
              coh_mat[i,j] /= (sxx * syy)

        idx = tsu.triu_indices(self.input.data.shape[0],1)
        coh_mat[idx[0],idx[1],...] = coh_mat[idx[1],idx[0],...].conj()

        return coh_mat

    @desc.setattr_on_read
    def confidence_interval(self):
        """The size of the 1-alpha confidence interval"""
        coh_var = np.zeros((self.input.data.shape[0],
                            self.input.data.shape[0],
                            self._L), 'd')
        for i in xrange(self.input.data.shape[0]):
           for j in xrange(i):
               if i != j:
                   coh_var[i,j] = tsu.jackknifed_coh_variance(self.spectra[i],
                                                              self.spectra[j],
                                weights=(self.weights[i], self.weights[j]),
                                                        last_freq=self._L)
                   
        idx = tsu.triu_indices(self.input.data.shape[0],1)
        coh_var[idx[0],idx[1],...] = coh_var[idx[1],idx[0],...].conj()

        coh_mat_xform = tsu.normalize_coherence(self.coherence, 2*self.df-2)

        lb = coh_mat_xform + dist.t.ppf(self.alpha/2,
                                                self.df-1)*np.sqrt(coh_var)
        ub = coh_mat_xform + dist.t.ppf(1-self.alpha/2,
                                                self.df-1)*np.sqrt(coh_var)

        # convert this measure with the normalizing function
        tsu.normal_coherence_to_unit(lb, 2*self.df-2, lb)
        tsu.normal_coherence_to_unit(ub, 2*self.df-2, ub)

        return ub-lb

    @desc.setattr_on_read
    def frequencies(self):
        return np.linspace(0, self.input.sampling_rate/2, self._L)
        

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

        prefer_speed_over_memory: Boolean, optional, default=True

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
            self.method = {'this_method':'welch'}

        else:
            self.method = method

        if self.method['this_method']!='welch':
            raise ValueError("For SparseCoherenceAnalyzer, spectral estimation method must be welch")
            

        #Additional parameters for the coherency estimation: 
        self.lb = lb
        self.ub = ub
        self.prefer_speed_over_memory = prefer_speed_over_memory
        self.scale_by_freq = scale_by_freq

    @desc.setattr_on_read
    def coherency(self):
        """ The default behavior is to calculate the cache, extract it and then
        output the coherency""" 
        coherency = tsa.cache_to_coherency(self.cache,self.ij)

        return coherency

    @desc.setattr_on_read
    def coherence(self):
        """ The coherence values for the output"""
        coherence = np.abs(self.coherency**2)
       
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
        spectrum = tsa.cache_to_psd(self.cache,self.ij)

        return spectrum
    
    @desc.setattr_on_read
    def phases(self):
        """The frequency-band dependent phases of the spectra of each of the
           time -series i,j in the analyzer"""
        
        phase = tsa.cache_to_phase(self.cache,self.ij)

        return phase

    @desc.setattr_on_read
    def relative_phases(self):
        """The frequency-band dependent relative phase between the two
        time-series """
        return np.angle(self.coherency)
       
    @desc.setattr_on_read
    def delay(self):
        """ The delay in seconds between the two time series """
        return self.relative_phases / (2*np.pi*self.frequencies)
        
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
        
class SeedCoherenceAnalyzer(BaseAnalyzer):
    """
    This analyzer takes two time-series. The first is designated as a
    time-series of seeds. The other is designated as a time-series of targets.
    The analyzer performs a coherence analysis between each of the channels in
    the seed time-series and *all* of the channels in the target time-series.

    Note
    ----

    This is a convenience class, which provides a convenient-to-use interface
    to the SparseCoherenceAnalyzer

    """

    def __init__(self,seed_time_series=None,target_time_series=None,
                 method=None,lb=0,ub=None,prefer_speed_over_memory=True,
                 scale_by_freq=True):
        
        """

        The constructor for the SeedCoherenceAnalyzer

        Parameters
        ----------

        seed_time_series: a time-series object

        target_time_series: a time-series object
            
        lb,ub: float,optional, default: lb=0, ub=None (max frequency)

            define a frequency band of interest

        prefer_speed_over_memory: Boolean, optional, default=True

            Makes things go a bit faster, if you have enough memory


        """
        
        BaseAnalyzer.__init__(self,seed_time_series)

        self.seed = seed_time_series
        self.target = target_time_series
        
        #Set the variables for spectral estimation (can also be entered by
        #user): 
        if method is None:
            self.method = {'this_method':'welch'}

        else:
            self.method = method

        
        if self.method.has_key('this_method') and self.method['this_method']!='welch':
            raise ValueError("For SparseCoherenceAnalyzer, spectral estimation method must be welch")
            

        #Additional parameters for the coherency estimation: 
        self.lb = lb
        self.ub = ub
        self.prefer_speed_over_memory = prefer_speed_over_memory
        self.scale_by_freq = scale_by_freq

    @desc.setattr_on_read
    def coherence(self):
        """
        The coherence between each of the channels of the seed time series and
        all the channels of the target time-series. 

        """
        return np.abs(self.coherency)**2
    
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

    @desc.setattr_on_read
    def target_cache(self):
        data = self.target.data

        #Make a cache with all the fft windows for each of the channels in the
        #target.

        #This is the kind of input that cache_fft expects: 
        ij = zip(np.arange(data.shape[0]),np.arange(data.shape[0]))
        
        f,cache = tsa.cache_fft(data,ij,lb=self.lb,ub=self.ub,
                        method=self.method,
                        prefer_speed_over_memory=self.prefer_speed_over_memory,
                        scale_by_freq=self.scale_by_freq)

        return cache


    @desc.setattr_on_read
    def coherency(self):

        #Pre-allocate the final result:
        if len(self.seed.shape)>1:
            Cxy = np.empty((self.seed.data.shape[0],
                            self.target.data.shape[0],
                            self.frequencies.shape[0]),dtype=np.complex)
        else:
            Cxy = np.empty((self.target.data.shape[0],
                            self.frequencies.shape[0]),dtype=np.complex)

        #Get the fft window cache for the target time-series: 
        cache = self.target_cache

        #A list of indices for the target:
        target_chan_idx = np.arange(self.target.data.shape[0])

        #This is a list of indices into the cached fft window libraries,
        #setting the index of the seed to be -1, so that it is easily
        #distinguished from the target indices: 
        ij = zip(np.ones_like(target_chan_idx)*-1,target_chan_idx)

        #If there is more than one channel in the seed time-series:
        if len(self.seed.shape)>1:
            for seed_idx,this_seed in enumerate(self.seed.data):
                #Here ij is 0, because it is just one channel and we stack the
                #channel onto itself in order for the input to the function to
                #make sense:
                f,seed_cache = tsa.cache_fft(np.vstack([this_seed,this_seed]),
                        [(0,0)],
                        lb=self.lb,ub=self.ub,
                        method=self.method,
                        prefer_speed_over_memory=self.prefer_speed_over_memory,
                        scale_by_freq=self.scale_by_freq)

                #Insert the seed_cache into the target_cache:
                cache['FFT_slices'][-1]=seed_cache['FFT_slices'][0] 

                #If this is true, the cache contains both FFT_slices and
                #FFT_conj_slices:
                if self.prefer_speed_over_memory:
                    cache['FFT_conj_slices'][-1]=\
                                            seed_cache['FFT_conj_slices'][0] 
                
                #This performs the caclulation for this seed:    
                Cxy[seed_idx] = tsa.cache_to_coherency(cache,ij)
                
        #In the case where there is only one channel in the seed time-series:
        else:
            f,seed_cache = tsa.cache_fft(np.vstack([self.seed.data,
                                                    self.seed.data]),
                        [(0,0)],
                        lb=self.lb,ub=self.ub,
                        method=self.method,
                        prefer_speed_over_memory=self.prefer_speed_over_memory,
                        scale_by_freq=self.scale_by_freq)

            cache['FFT_slices'][-1]=seed_cache['FFT_slices'][0]

            if self.prefer_speed_over_memory:
                cache['FFT_conj_slices'][-1]=\
                                            seed_cache['FFT_conj_slices'][0] 

            Cxy=tsa.cache_to_coherency(cache,ij)

        return Cxy.squeeze()

    @desc.setattr_on_read
    def relative_phases(self):
        """The frequency-band dependent relative phase between the two
        time-series """
        return np.angle(self.coherency)
       
    @desc.setattr_on_read
    def delay(self):
        """ The delay in seconds between the two time series """
        return self.relative_phases / (2*np.pi*self.frequencies)

class CorrelationAnalyzer(BaseAnalyzer):
    """Analyzer object for correlation analysis. Has the same API as the
    CoherenceAnalyzer"""

    def __init__(self,input=None):
        """
        Parameters
        ----------

        input: TimeSeries object
           Containing the data to analyze.

        Examples
        --------
        >>> t1 = ts.TimeSeries(data = np.sin(np.arange(0,10*np.pi,10*np.pi/100)).reshape(2,50),sampling_rate=np.pi)
        >>> c1 = CorrelationAnalyzer(t1)
        >>> c1 = CorrelationAnalyzer(t1)
        >>> c1.corrcoef
        array([[ 1., -1.],
               [-1.,  1.]])
        >>> c1.xcorr.sampling_rate
        3.1415926536 Hz
        >>> c1.xcorr.t0
        -15.915494309150001 s
        
        """ 

        BaseAnalyzer.__init__(self,input)

    @desc.setattr_on_read
    def corrcoef(self):
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
        at time=0

        """
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

        TimeSeries: A TimeSeries object
            the time-dependent cross-correlation, with zero-lag at time=0

        """

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
                xcorr[i,j] *= self.corrcoef[i,j]

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

        offset: the offset of the beginning of the event-related time-series,
        relative to the event occurence 
        
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
            err = ("Input 'events' to EventRelatedAnalyzer must be of type "
                   "Events or of type TimeSeries, %r given" % events )
            raise ValueError(err)
   
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
        corresponds to time, and has length = len_et (xcorr looks both back
        and forward for half of this length)

        """
        #Make a list to put the outputs in:
        h = [0] * self._len_h

        for i in xrange(self._len_h):
            data = self.data[i]
            u = np.unique(self.events[i])
            event_types = u[np.unique(self.events[i])!=0]
            h[i] = np.empty((event_types.shape[0],self.len_et/2),dtype=complex)
            for e_idx in xrange(event_types.shape[0]):
                this_e = (self.events[i]==event_types[e_idx]) * 1.0
                if self._zscore:
                    this_h = tsa.freq_domain_xcorr_zscored(data,
                                                    this_e,
                                                    -self.offset+1,
                                                    self.len_et-self.offset-2)
                else:
                    this_h = tsa.freq_domain_xcorr(data,
                                                   this_e,
                                                   -self.offset+1,
                                                   self.len_et-self.offset-2)
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
    def analytic(self):
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
        return ts.TimeSeries(data=np.abs(self.analytic.data),
                                 sampling_rate=self.analytic.sampling_rate)
                                 
    @desc.setattr_on_read
    def phase(self):
        return ts.TimeSeries(data=np.angle(self.analytic.data),
                                 sampling_rate=self.analytic.sampling_rate)

    @desc.setattr_on_read
    def real(self):
        return ts.TimeSeries(data=self.analytic.data.real,
                                    sampling_rate=self.analytic.sampling_rate)
    
    @desc.setattr_on_read
    def imag(self):
        return ts.TimeSeries(data=self.analytic.data.imag,
                                    sampling_rate=self.analytic.sampling_rate)


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
    def analytic(self):
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
        return ts.TimeSeries(data=np.abs(self.analytic.data),
                                    sampling_rate=self.analytic.sampling_rate)
                                 
    @desc.setattr_on_read
    def phase(self):
        return ts.TimeSeries(data=np.angle(self.analytic.data),
                                    sampling_rate=self.analytic.sampling_rate)

    @desc.setattr_on_read
    def real(self):
        return ts.TimeSeries(data=self.analytic.data.real,
                                    sampling_rate=self.analytic.sampling_rate)
    
    @desc.setattr_on_read
    def imag(self):
        return ts.TimeSeries(data=self.analytic.data.imag,
                                    sampling_rate=self.analytic.sampling_rate)


def signal_noise(response):
    """
    Signal and noise as defined in Borst and Theunissen 1999, Figure 2

    Parameters
    ----------

    response: nitime TimeSeries object
       The data here are individual responses of a single unit to the same
       stimulus, with repetitions being the first dimension and time as the
       last dimension 
    
    """

    signal = np.mean(response.data,0) #The estimate of the signal is the average
                                 #response
                                 
    noise =  response.data - signal #Noise is the individual
                               #repetition's deviation from the
                               #estimate of the signal

    #Return TimeSeries objects with the sampling rate of the input: 
    return  (ts.TimeSeries(signal,sampling_rate=response.sampling_rate),
             ts.TimeSeries(noise,sampling_rate=response.sampling_rate))
    
class SNRAnalyzer(BaseAnalyzer):
    """
    Calculate SNR for a response to repetitions of the same stimulus, according
    to [Borst1999]_ (Figure 2) and [Hsu2004]_.

    .. [Hsu2004] Hsu A, Borst A and Theunissen, FE (2004) Quantifying
    variability in neural responses ans its application for the validation of
    model predictions. Network: Comput Neural Syst 15:91-109

    .. [Borst1999] Borst A and Theunissen FE (1999) Information theory and
    neural coding. Nat Neurosci 2:947-957
    
    """

    def __init__(self,input=None,bandwidth=None,adaptive=False,low_bias=False):
        """
        Initializer for the multi_taper_SNR object

        Parameters
        ----------
        input: TimeSeries object

        bandwidth: float,
           The bandwidth of the windowing function will determine the number
           tapers to use. This parameters represents trade-off between
           frequency resolution (lower main lobe bandwidth for the taper) and
           variance reduction (higher bandwidth and number of averaged
           estimates). Per default will be set to 4 times the fundamental
           frequency, such that NW=4

        adaptive: bool, default to False
            Whether to set the weights for the tapered spectra according to the
            adaptive algorithm [Thompson2007]_.

        low_bias : bool, default to False
            Rather than use 2NW tapers, only use the tapers that have better
            than 90% spectral concentration within the bandwidth (still using a
            maximum of 2NW tapers)  

            .. [Thompson2007] Thompson, DJ Jackknifing multitaper spectrum
            estimates. IEEE Signal Processing Magazing. 24: 20-30

        """
        self.input = input
        self.signal,self.noise = signal_noise(input) 
        self.bandwidth = bandwidth
        self.adaptive = adaptive
        self.low_bias = low_bias

    @desc.setattr_on_read
    def mt_frequencies(self):
        return np.linspace(0, self.input.sampling_rate/2,
                           self.input.data.shape[-1]/2 + 1)

    @desc.setattr_on_read
    def mt_signal_psd(self):
        _,p,_ = tsa.multi_taper_psd(self.signal.data,
                                    Fs=self.input.sampling_rate,
                                    BW=self.bandwidth,adaptive=self.adaptive,
                                    low_bias=self.low_bias)
        return p
    
    @desc.setattr_on_read
    def mt_noise_psd(self):
        p = np.empty((self.noise.data.shape[0],
                     self.noise.data.shape[-1]/2+1))
        
        for i in xrange(p.shape[0]):
            _,p[i],_ = tsa.multi_taper_psd(self.noise.data[i],
                                    Fs=self.input.sampling_rate,
                                    BW=self.bandwidth,adaptive=self.adaptive,
                                    low_bias=self.low_bias)
        return np.mean(p,0)
    
    @desc.setattr_on_read
    def mt_coherence(self):
        """ """
        return self.mt_signal_psd/(self.mt_signal_psd + self.mt_noise_psd)
    
    @desc.setattr_on_read
    def mt_information(self):
        return -1*np.log2(1-self.mt_coherence)
        #These two formulations should be equivalent
        #return np.log2(1+self.mt_snr)
    
    @desc.setattr_on_read
    def mt_snr(self):
        return self.mt_signal_psd/self.mt_noise_psd

    @desc.setattr_on_read
    def correlation(self):
        """
        The correlation between all combinations of trials

        Returns
        -------
        (r,e) : tuple
           r is the mean correlation and e is the mean error of the correlation
           (with df = n_trials - 1)
        
        """

        c = np.corrcoef(self.input.data)
        c = c[np.tril_indices_from(c,-1)]

        return np.mean(c), stats.sem(c)
