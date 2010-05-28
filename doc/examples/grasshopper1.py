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





