#!/usr/bin/python

#Imports as before:
import numpy as np
from matplotlib.pyplot import figure,legend
from matplotlib.mlab import csv2rec
from nitime.timeseries import TimeSeries
from nitime.utils import percent_change
import nitime.viz
reload(nitime.viz)
from nitime.viz import drawmatrix_channels

#This time Import the coherence analyzer 
from nitime.analysis import CoherenceAnalyzer

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
C = CoherenceAnalyzer(T)

#We look only at frequencies between 0.02 and 0.15 (the physiologically
#relevant band, see http://imaging.mrc-cbu.cam.ac.uk/imaging/DesignEfficiency:
freq_idx = np.where((C.frequencies>0.02) * (C.frequencies<0.15))[0]

#Extract the coherence and average across these frequency bands: 
coh = np.mean(C.coherence[:,:,freq_idx],-1) #Averaging on the last dimension 
drawmatrix_channels(coh,roi_names,size=[10.,10.],color_anchor=0)

