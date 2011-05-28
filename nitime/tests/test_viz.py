"""

Smoke testing of the viz module, based on the doc/examples/resting_state_fmri.py

"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import csv2rec
import nitime
from nitime.timeseries import TimeSeries
from nitime.analysis import CorrelationAnalyzer, CoherenceAnalyzer
from nitime.utils import percent_change
from nitime.viz import drawmatrix_channels, drawgraph_channels, plot_xcorr

# Keep these as globals:
TR = 1.89
f_lb = 0.02
f_ub = 0.15

data_dir_path = os.path.join(nitime.__path__[0], '../doc/examples/data')
data_rec = csv2rec('%s/fmri_timeseries.csv'%data_dir_path)

#Extract information:
roi_names = np.array(data_rec.dtype.names)
n_samples = data_rec.shape[0]

#Make an empty container for the data
data = np.zeros((len(roi_names), n_samples))

for n_idx, roi in enumerate(roi_names):
    data[n_idx] = data_rec[roi]

#Normalize the data:
data = percent_change(data)

T = TimeSeries(data, sampling_interval=TR)
T.metadata['roi'] = roi_names


#Initialize the correlation analyzer
C = CorrelationAnalyzer(T)

def test_drawmatrix_channels():
    fig01 = drawmatrix_channels(C.corrcoef, roi_names, size=[10., 10.], color_anchor=0)

def test_plot_xcorr():
    xc = C.xcorr_norm

    idx_lcau = np.where(roi_names == 'lcau')[0]
    idx_rcau = np.where(roi_names == 'rcau')[0]
    idx_lput = np.where(roi_names == 'lput')[0]
    idx_rput = np.where(roi_names == 'rput')[0]

    fig02 = plot_xcorr(xc,
                       ((idx_lcau, idx_rcau),
                        (idx_lcau, idx_lput)),
                       line_labels=['rcau', 'lput'])


def test_drawgraph_channels():

    fig04 = drawgraph_channels(C.corrcoef, roi_names)
