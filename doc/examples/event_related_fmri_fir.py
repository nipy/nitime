import nitime.timeseries as ts
import nitime.analysis as ta 
import nitime.viz as tv
from matplotlib.mlab import csv2rec

#Load the data from file:
data =csv2rec('data/event_related_fmri.csv')
#Initialize TimeSeries objects: 
t1 = ts.TimeSeries(data.bold,sampling_interval=2)
t2 = ts.TimeSeries(data.events,sampling_interval=2)
#Initialized the event-related analyzer with the two time-series:
E = ta.EventRelatedAnalyzer(t1,t2,15,offset=-5)
#Visualize the results:
tv.plot_tseries(E.FIR,ylabel='BOLD (% signal change)')

tv.plot_tseries(E.xcorr_eta,ylabel='BOLD (% signal change)')

#Now do the same, this time: with the zscore flag set to True:
E = ta.EventRelatedAnalyzer(t1,t2,15,offset=-5,zscore=True)

fig = tv.plot_tseries(E.xcorr_eta,ylabel='BOLD (% signal change)')


