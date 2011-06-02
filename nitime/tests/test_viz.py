"""

Smoke testing of the viz module.

"""

import numpy as np

from nitime.timeseries import TimeSeries
from nitime.analysis import CorrelationAnalyzer
from nitime.viz import drawmatrix_channels, drawgraph_channels, plot_xcorr


roi_names = ['a','b','c','d','e','f','g','h','i','j']
data = np.random.rand(10,1024)

T = TimeSeries(data, sampling_interval=np.pi)
T.metadata['roi'] = roi_names

#Initialize the correlation analyzer
C = CorrelationAnalyzer(T)

def test_drawmatrix_channels():
    fig01 = drawmatrix_channels(C.corrcoef, roi_names, size=[10., 10.], color_anchor=0)

def test_plot_xcorr():
    xc = C.xcorr_norm

    fig02 = plot_xcorr(xc,
                       ((0, 1),
                        (2, 3)),
                       line_labels=['a', 'b'])


def test_drawgraph_channels():

    fig04 = drawgraph_channels(C.corrcoef, roi_names)
