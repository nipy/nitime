"""

Test the io submodule of the fMRI module of nitime

""" 
import os

import numpy as np
import numpy.testing as npt

import nitime
import nitime.fmri.io as io

#Skip the tests if you can't import nibabel:
try:
    import nibabel
    no_nibabel = False
except ImportError:
    no_nibabel = True

test_dir_path = os.path.join(nitime.__path__[0],'fmri/tests')

@npt.dec.skipif(no_nibabel)
def test_time_series_from_file():

    """Testing reading of data from nifti files, using nibabel"""
    
    TR = 1.35 
    ts_ff = io.time_series_from_file

    #File names:
    fmri_file1 = os.path.join(test_dir_path,'fmri1.nii.gz')
    fmri_file2 = os.path.join(test_dir_path,'fmri2.nii.gz')

    #Spatial coordinates into the volumes: 
    coords1 = np.array([[5,5,5,5],[5,5,5,5],[1,2,3,4]])
    coords2 = np.array([[6,6,6,6],[6,6,6,6],[3,4,5,6]])

    #No averaging, no normalization:
    t1 = ts_ff([fmri_file1,fmri_file2],[coords1,coords2],TR)

    yield npt.assert_equal,t1[0].shape,(4,80) #4 coordinates, 80 time-points
    
    t2 = ts_ff([fmri_file1,fmri_file2],[coords1,coords2],TR,average=True)
    
    yield npt.assert_equal,t2[0].shape,(80,) #collapse coordinates,80 time-points

    t3 = ts_ff(fmri_file1,coords1,TR,normalize='zscore')

    #The mean of each channel should be almost equal to 0:
    yield npt.assert_almost_equal,t3.data[0].mean(),0
    #And the standard deviation should be almost equal to 1:
    yield npt.assert_almost_equal,t3.data[0].std(),1
    
    t4 = ts_ff(fmri_file1,coords1,TR,normalize='percent')

    #In this case, the average is almost equal to 0, but no constraint on the
    #std:
    yield npt.assert_almost_equal,t4.data[0].mean(),0

    #Make sure that we didn't mess up the sampling interval:
    yield npt.assert_equal,t4.sampling_interval,nitime.TimeArray(1.35)


    
