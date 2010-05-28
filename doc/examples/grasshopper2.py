import numpy as np
import nitime.timeseries as ts
import nitime.analysis as tsa
reload(tsa)
import nitime.viz as viz

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
#Now we plot the means for each of the stimulus time-series:
ax.plot([xlim[0],xlim[1]],[means[0],means[0]],'b--')
ax.plot([xlim[0],xlim[1]],[means[1],means[1]],'g--')



    
    
