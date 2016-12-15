"""

Smoke testing of the viz module.

"""

import numpy as np
import numpy.testing as npt
import pytest

from nitime.timeseries import TimeSeries
from nitime.analysis import CorrelationAnalyzer
from nitime.viz import drawmatrix_channels, drawgraph_channels, plot_xcorr

try:
    import networkx
    no_networkx = False
    no_networkx_msg = ''
except ImportError as e:
    no_networkx = True
    no_networkx_msg = e.args[0]

import os
is_ci = "CI" in os.environ

roi_names = ['a','b','c','d','e','f','g','h','i','j']
data = np.random.rand(10,1024)

T = TimeSeries(data, sampling_interval=np.pi)
T.metadata['roi'] = roi_names

#Initialize the correlation analyzer
C = CorrelationAnalyzer(T)

@pytest.mark.skipif(is_ci, reason="Running on a CI server")
def test_drawmatrix_channels():
    fig01 = drawmatrix_channels(C.corrcoef, roi_names, size=[10., 10.], color_anchor=0)

@pytest.mark.skipif(is_ci, reason="Running on a CI server")
def test_plot_xcorr():
    xc = C.xcorr_norm

    fig02 = plot_xcorr(xc,
                       ((0, 1),
                        (2, 3)),
                       line_labels=['a', 'b'])


@pytest.mark.skipif(is_ci, reason="Running on a CI server")
@pytest.mark.skipif(no_networkx, reason=no_networkx_msg)
def test_drawgraph_channels():
    fig04 = drawgraph_channels(C.corrcoef, roi_names)
