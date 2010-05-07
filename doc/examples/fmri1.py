#!/usr/bin/python

#Import from other libraries:
import numpy as np
from matplotlib.pyplot import figure
from matplotlib.mlab import csv2rec

#Import the time-series objects: 
from nitime.timeseries import TimeSeries 
#Import the correlation analyis object
from nitime.analysis import CorrelationAnalyzer
#Import utility functions:
from nitime.utils import percent_change
import nitime.viz
reload(nitime.viz)
from nitime.viz import drawmatrix_channels,drawgraph_channels

#This information (the sampling rate) has to be known in advance:
TR=1.89

#Load the data from the csv file:
data_rec = csv2rec('data/fmri_timeseries.csv')

#Extract information:
roi_names= np.array(data_rec.dtype.names)
n_samples = data_rec.shape[0]


#Make an empty container for the data
data = np.zeros((len(roi_names),n_samples))

for n_idx, roi in enumerate(roi_names):
   data[n_idx] = data_rec[roi]

#Normalize the data:
data = percent_change(data)

#Initialize the time-series from the normalized data:
T = TimeSeries(data,sampling_interval=TR)
T.metadata['roi'] = roi_names 

#Initialize the correlation analyzer
C = CorrelationAnalyzer(T)

#Display the correlation matrix
drawmatrix_channels(C(),roi_names,size=[10.,10.],color_anchor=0)

