#!/usr/bin/python
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import nitime.timeseries as ts
import nitime.analysis as tsa
import nitime.utils as tsu

#Load the data from the csv file:
data_rec = mlab.csv2rec('data/fmri_timeseries.csv')

#Extract information:
roi_names= data_rec.dtype.names
n_samples = data_rec.shape[0]

#This information has to be known in advance:
TR=1.89

#Make an empty container for the data
data = np.zeros((len(roi_names),n_samples))

for n_idx in range(len(roi_names)):
   data[n_idx] = data_rec[roi_names[n_idx]]

#Normalize the data:
data = tsu.percent_change(data)

#Initialize the time-series from the normalized data:
T = ts.TimeSeries(data,sampling_interval=TR)
T.metadata['roi'] = roi_names 

C = tsa.CorrelationAnalyzer(T)
xc = C.xcorr_norm

N = len(roi_names)
ind = np.arange(N)  # the evenly spaced plot indices



