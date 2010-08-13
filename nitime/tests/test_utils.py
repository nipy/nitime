import numpy as np
import numpy.testing as npt
from nitime import utils as ut
import nitime.timeseries as ts
import nose.tools as nt
import decotest

def test_zscore():

    x = np.array([[1,2,3],[4,5,6]])
    z = ut.zscore(x)

    npt.assert_equal(x.shape,z.shape)
    
def test_percent_change():
    x = np.array([[99,100,101],[4,5,6]])
    p = ut.percent_change(x)

    npt.assert_equal(x.shape,p.shape)
    npt.assert_almost_equal(p[0,2],1.0)
