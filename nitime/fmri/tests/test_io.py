"""

Test the io submodule of the fMRI module of nitime

"""
import os

import numpy as np
import numpy.testing as npt
import pytest

import nitime
import nitime.timeseries as ts

#Skip the tests if you can't import nibabel:
try:
    import nitime.fmri.io as io
    no_nibabel = False
    no_nibabel_msg=''
except ImportError as e:
    no_nibabel = True
    no_nibabel_msg=e.args[0]

data_path = os.path.join(nitime.__path__[0],'data')

@pytest.mark.skipif(no_nibabel, reason=no_nibabel_msg)
def test_time_series_from_file():

    """Testing reading of data from nifti files, using nibabel"""

    TR = 1.35
    ts_ff = io.time_series_from_file

    #File names:
    fmri_file1 = os.path.join(data_path,'fmri1.nii.gz')
    fmri_file2 = os.path.join(data_path,'fmri2.nii.gz')

    #Spatial coordinates into the volumes:
    coords1 = np.array([[5,5,5,5],[5,5,5,5],[1,2,3,4]])
    coords2 = np.array([[6,6,6,6],[6,6,6,6],[3,4,5,6]])

    #No averaging, no normalization:
    t1 = ts_ff([fmri_file1,fmri_file2],[coords1,coords2],TR)

    npt.assert_equal(t1[0].shape,(4,80))  # 4 coordinates, 80 time-points

    t2 = ts_ff([fmri_file1,fmri_file2],[coords1,coords2],TR,average=True)

    npt.assert_equal(t2[0].shape,(80,))  # collapse coordinates,80 time-points

    t3 = ts_ff(fmri_file1,coords1,TR,normalize='zscore')

    #The mean of each channel should be almost equal to 0:
    npt.assert_almost_equal(t3.data[0].mean(),0)
    #And the standard deviation should be almost equal to 1:
    npt.assert_almost_equal(t3.data[0].std(),1)

    t4 = ts_ff(fmri_file1,coords1,TR,normalize='percent')

    #In this case, the average is almost equal to 0, but no constraint on the
    #std:
    npt.assert_almost_equal(t4.data[0].mean(),0)

    #Make sure that we didn't mess up the sampling interval:
    npt.assert_equal(t4.sampling_interval,nitime.TimeArray(1.35))

    # Test the default behavior:
    data = io.load(fmri_file1).get_data()
    t5 = ts_ff(fmri_file1)
    npt.assert_equal(t5.shape, data.shape)
    npt.assert_equal(t5.sampling_interval, ts.TimeArray(1, time_unit='s'))

    # Test initializing TR with a TimeArray:
    t6= ts_ff(fmri_file1, TR=ts.TimeArray(1350, time_unit='ms'))
    npt.assert_equal(t4.sampling_interval, t6.sampling_interval)

    # Check the concatenation dimensions:
    t7 = ts_ff([fmri_file1, fmri_file2])
    npt.assert_equal([t7.shape[:3], t7.shape[-1]], [data.shape[:3], data.shape[-1]*2])

    t8 = ts_ff([fmri_file1, fmri_file2], average=True)
    npt.assert_equal(t8.shape[0], data.shape[-1]*2)

    t9 = ts_ff([fmri_file1, fmri_file2], average=True, normalize='zscore')
    npt.assert_almost_equal(t9.data.mean(), 0)
