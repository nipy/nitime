#!/usr/bin/env python

"""Sand box for testing vista utilities"""
import numpy as np
from matplotlib import pylab
import nipy.timeseries as nipyts
import scipy.io as sio
tsa = nipyts.algorithms
tsu = nipyts.utils
ts = nipyts.timeseries

reload(tsa)
reload(tsu)
reload(ts)

pylab.close('all')

ROI1 = 'LV1'
ROI2 = 'RV1'

scan = 1
TR = 2.0
f_c = 0.05

#This path is idiosyncratic to the Silver lab:
sess_dir = '/Volumes/Argent1/SchizoSpread/SchizoSpreadAnalysis/SMR033109_MC/'

time_series_file = (sess_dir + 'Inplane/Original/TSeries/Analyze/Scan'
                    + str(scan) + '.img')

ROI_file1 = sess_dir + 'Inplane/ROIs/' + ROI1 + '.mat'
ROI_file2 = sess_dir + 'Inplane/ROIs/' + ROI2 + '.mat'

coords1 = tsu.vista_getROIcoords(ROI_file1)
coords2 = tsu.vista_getROIcoords(ROI_file2)

up_sample_factor = [2,2,1]

t = tsu.vista_get_time_series_inplane([coords1,coords2],time_series_file,
                                      f_c=0.01,up_sample_factor=up_sample_factor,
                                      TR=TR)

pylab.figure()
for i in xrange(t.data.shape[0]):
    pylab.plot(t.data[i])
pylab.show()
