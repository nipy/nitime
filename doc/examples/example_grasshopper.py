import numpy as np
import matplotlib.pyplot as plt
import nitime.timeseries as ts

reload(ts)

plt.close("all")

#Load the stimulus from the data file:
maxdB = 79.8387 #Taken from the spike file header
stim = np.loadtxt('data/grasshopper_stimulus.txt')
stim = (20*1/np.log(10))*(np.log(stim[:,1]/2.0e-5))
stim = maxdB-stim.max()+stim

#Create the stimulus time-series (sampled at 20 kHz):
stim_time_series = ts.UniformTimeSeries(t0=0,
                                        data=stim,
                                        sampling_interval=0.5,
                                        time_unit='us') 

#Load the spike-times from the data file:
spike_times = np.loadtxt('data/grasshopper_spike_times.txt')
spike_times = (spike_times/50).astype(int)
spike_time_series = ts.UniformTimeSeries(t0=0,sampling_interval=0.5,
                                         time_unit='us',
                                         data=np.zeros(stim.shape))
spike_time_series.data[spike_times] = 1

event_related = ts.EventRelatedAnalyzer(stim_time_series,
                                        spike_time_series,len_hrf=150,
                                        offset=-150)

plt.plot(event_related.eta.time,event_related.eta.data)






