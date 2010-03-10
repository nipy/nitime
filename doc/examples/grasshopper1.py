#Imports:
import numpy as np
#Import nitime.timeseries for representation objects:
import nitime.timeseries as ts
#Import nitime.analysis for analysis object:
import nitime.analysis as tsa
#The viz library is used for visualization:
import nitime.viz as viz

#Load the stimulus from the data file:
maxdB = 76.4286 #Taken from the spike file header
stim = np.loadtxt('data/grasshopper_stimulus1.txt')
stim = (20*1/np.log(10))*(np.log(stim[:,1]/2.0e-5))
stim = maxdB-stim.max()+stim

#Create the stimulus time-series (sampled at 20 kHz):
stim_time_series = ts.UniformTimeSeries(t0=0,
                                        data=stim,
                                        sampling_interval=0.05,
                                        time_unit='ms') 

#Load the spike-times from the data file:
spike_times = np.loadtxt('data/grasshopper_spike_times1.txt')
#Spike times are in 0.01 msec resolution, so need to be resampled:
spike_times = (spike_times/50).astype(int)
#Initialize the time-series holding the spike-times:
spike_time_series = ts.UniformTimeSeries(t0=0,sampling_interval=0.05,
                                         time_unit='ms',
                                         data=np.zeros(stim.shape))
#The position of a spike is encoded as a '1': 
spike_time_series.data[spike_times] = 1

#Initialize the event-related analyzer
event_related = tsa.EventRelatedAnalyzer(stim_time_series,
                                        spike_time_series,len_et=250,
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
ax.plot([0,0],[ylim[0],ylim[1]],'k--')





