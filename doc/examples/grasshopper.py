"""
=====================================
 Auditory processing in grasshoppers
=====================================

Two data files are used in this example. The first contains the times of action
potentials ('spikes'), recorded intra-cellularly from primary auditory
receptors in the grasshopper *Locusta Migratoria*. The other data file contains
the stimulus that was played during the recording. Briefly, the stimulus played
was a pure-tone in the cell's preferred frequency amplitude modulated by
Gaussian white-noise, up to a cut-off frequency (200 Hz in this case, for
details on the experimental procedures and the stimulus see [Rokem2006]_). This
data is available on the `CRCNS data sharing web-site <http://crcns.org/>`_.

In the following code-snippet, we demonstrate the calculation of the
spike-triggered average (STA). This is the average of the stimulus wave-form
preceding the emission of a spike in the neuron and can be thought of as the
stimulus 'preferred' by this neuron.
"""

#Imports:
import numpy as np
#Import nitime.timeseries for representation objects:
import nitime.timeseries as ts
#Import nitime.analysis for analysis object:
import nitime.analysis as tsa
reload(tsa)
#The viz library is used for visualization:
import nitime.viz as viz

#Load the stimulus from the data file:
maxdB = 76.4286 #Taken from the spike file header
stim = np.loadtxt('data/grasshopper_stimulus1.txt')
stim = (20*1/np.log(10))*(np.log(stim[:,1]/2.0e-5))
stim = maxdB-stim.max()+stim

#Create the stimulus time-series (sampled at 20 kHz):
stim_time_series = ts.TimeSeries(t0=0,data=stim,sampling_interval=50,
                                 time_unit='us')
#Convert time-unit:
stim_time_series.time_unit='ms'
#Load the spike-times from the data file:
spike_times = np.loadtxt('data/grasshopper_spike_times1.txt')
#Initialize the Event object holding the spike-times:
spike_ev = ts.Events(spike_times,time_unit='us')

#Initialize the event-related analyzer
event_related = tsa.EventRelatedAnalyzer(stim_time_series,spike_ev,len_et=200,
                                                                  offset=-200)

#The actual STA gets calculated in this line (the call to 'event_related.eta')
#and the result gets input directly into the plotting function:
fig = viz.plot_tseries(event_related.eta,ylabel='Amplitude (dB SPL)')

#We also plot the average of the stimulus amplitude and the time of the spike,
#using dashed lines:
ax = fig.get_axes()[0]
xlim = ax.get_xlim()
ylim = ax.get_ylim()
mean_stim = np.mean(stim_time_series.data)
ax.plot([xlim[0],xlim[1]],[mean_stim,mean_stim],'k--')

"""
The code example makes use of :class:`EventRelatedAnalyzer` and, in particular,
of the :meth:`EventRelatedAnalyzer.eta` method of this object. This method gets
evaluated as an instance of the :class:`TimeSeries` class. In addition,
vizualization, using :func:`viz.plot_tseries` is demonstrated. This function
plots the values of the :attr:`TimeSeries.data`, with the appropriate
time-units and their values on the x axis. 

In the following example, a second channel has been added to both the stimulus
and the spike-train time-series. This is the response of the same cell, to a
different stimulus, in which the frequency modulation has a higher frequency
cut-off (800 Hz). The :class:`EventRelatedAnalyzer` simply calculates the STA
for both of these. Likewise, the plotting function :func:`viz.plot_tseries`
simply adds another line in another color for this second channel.
"""

#As before the stimuli get read from files:  
maxdB1 = 76.4286 #Taken from the spike file header
stim1 = np.loadtxt('data/grasshopper_stimulus1.txt')
stim1 = (20*1/np.log(10))*(np.log(stim1[:,1]/2.0e-5))
stim1 = maxdB1-stim1.max()+stim1

maxdB2 = 71.2 #Taken from the spike file header
stim2 = np.loadtxt('data/grasshopper_stimulus2.txt')
stim2 = (20*1/np.log(10))*(np.log(stim2[:,1]/2.0e-5))
stim2 = maxdB1-stim2.max()+stim2

#Get the spike-times:
spike_times1 = np.loadtxt('data/grasshopper_spike_times1.txt')
spike_times2 = np.loadtxt('data/grasshopper_spike_times2.txt')

et = []
means = []
for stim,spike in zip([stim1,stim2],[spike_times1,spike_times2]):
    stim_time_series = ts.TimeSeries(t0=0,data=stim,sampling_interval=50,
                                 time_unit='us')
    
    stim_time_series.time_unit = 'ms'

    spike_ev = ts.Events(spike,time_unit='us')
    #Initialize the event-related analyzer
    event_related = tsa.EventRelatedAnalyzer(stim_time_series,spike_ev,
                                             len_et=200,
                                             offset=-200)

    et.append(event_related.eta)
    means.append(np.mean(stim_time_series.data))

#Stack the data from both time-series, initialize a new time-series and 
fig =viz.plot_tseries(ts.TimeSeries(data=np.vstack([et[0].data,et[1].data]),
                               sampling_rate=et[0].sampling_rate,time_unit='ms'))

ax = fig.get_axes()[0]
xlim = ax.get_xlim()
# Now we plot the means for each of the stimulus time-series:
ax.plot([xlim[0],xlim[1]],[means[0],means[0]],'b--')
ax.plot([xlim[0],xlim[1]],[means[1],means[1]],'g--')

"""
.. [Rokem2006] Ariel Rokem, Sebastian Watzl, Tim Gollisch, Martin Stemmler,
  Andreas V M Herz and Ines Samengo (2006). Spike-timing precision underlies
   the coding efficiency of auditory receptor neurons. J Neurophysiol, 95:
   2541--52
"""
