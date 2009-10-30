#!/usr/bin/python

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import nitime
ts = nitime.timeseries

data_rec = mlab.csv2rec('All_timeseries.csv')

roi_names= data_rec.dtype.names
n_samples = 250
TR=1.89

data = np.zeros([len(roi_names),n_samples])

for n_idx in range(len(roi_names)):
   data[n_idx] = data_rec[roi_names[n_idx]]
   

T = ts.UniformTimeSeries(data,sampling_interval=TR)
T.metadata['roi'] = roi_names 

C = ts.CorrelationAnalyzer(T)
XC = C.xcorr_norm

plt.matshow(C.correlation)


