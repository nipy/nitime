#!/usr/bin/python

#Imports are as above:
import numpy as np
from matplotlib.pyplot import figure,legend
from matplotlib.mlab import csv2rec
from nitime.timeseries import TimeSeries 
from nitime.analysis import CorrelationAnalyzer
from nitime.utils import percent_change
import nitime.viz
reload(nitime.viz)
from nitime.viz import plot_xcorr

#This part is the same as before
TR=1.89
data_rec = csv2rec('data/fmri_timeseries.csv')
roi_names= np.array(data_rec.dtype.names)
n_samples = data_rec.shape[0]
data = np.zeros((len(roi_names),n_samples))

for n_idx, roi in enumerate(roi_names):
   data[n_idx] = data_rec[roi]

data = percent_change(data)
T = TimeSeries(data,sampling_interval=TR)
T.metadata['roi'] = roi_names 
C = CorrelationAnalyzer(T)

#Extract the cross-correlation:
xc = C.xcorr_norm

idx_lcau = np.where(roi_names=='lcau')[0]
idx_rcau = np.where(roi_names=='rcau')[0]
idx_lput = np.where(roi_names=='lput')[0]

plot_xcorr(xc,((idx_lcau,idx_rcau),(idx_lcau,idx_lput)),
               line_labels = ['rcau','lput'])



