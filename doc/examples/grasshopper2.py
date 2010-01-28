import numpy as np
import matplotlib.pyplot as plt
import nitime.timeseries as ts
import nitime.viz as viz

reload(ts)
reload(viz)

plt.close("all")

#Load the stimulus from the data file:
maxdB1 = 76.4286 #Taken from the spike file header
stim1 = np.loadtxt('data/grasshopper_stimulus1.txt')
stim1 = (20*1/np.log(10))*(np.log(stim1[:,1]/2.0e-5))
stim1 = maxdB1-stim1.max()+stim1

maxdB2 = 71.2 #Taken from the spike file header
stim2 = np.loadtxt('data/grasshopper_stimulus2.txt')
stim2 = (20*1/np.log(10))*(np.log(stim2[:,1]/2.0e-5))
stim2 = maxdB1-stim2.max()+stim2


#Create the stimulus time-series (sampled at 20 kHz):
stim_time_series = ts.UniformTimeSeries(t0=0,
                                        data=np.vstack([stim1,stim2]),
                                        sampling_interval=0.05,
                                        time_unit='ms') 

#Load the spike-times from the data file:
spike_times1 = np.loadtxt('data/grasshopper_spike_times1.txt')
spike_times2 = np.loadtxt('data/grasshopper_spike_times2.txt')

#Spike times are in 0.01 msec resolution, so need to be resampled:
spike_times1 = (spike_times1/50).astype(int)
spike_times2 = (spike_times2/50).astype(int)

spike_time_series = ts.UniformTimeSeries(t0=0,sampling_interval=0.05,
                                     time_unit='ms',
                                     data=np.zeros(stim_time_series.data.shape))

spike_time_series.data[0][spike_times1] = 1
spike_time_series.data[1][spike_times2] = 1
event_related = ts.EventRelatedAnalyzer(stim_time_series,
                                        spike_time_series,len_hrf=250,
                                        offset=-200)

fig = viz.plot_tseries(event_related.eta,ylabel='Amplitude (dB SPL)')
ax = fig.get_axes()[0]
xlim = ax.get_xlim()
ylim = ax.get_ylim()
mean_stim1 = np.mean(stim_time_series.data[0])
mean_stim2 = np.mean(stim_time_series.data[1])
ax.plot([xlim[0],xlim[1]],[mean_stim1,mean_stim1],'b--')
ax.plot([xlim[0],xlim[1]],[mean_stim2,mean_stim2],'k--')
ax.plot([0,0],[ylim[0],ylim[1]],'k--')





