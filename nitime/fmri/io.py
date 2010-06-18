""" Input and output for fmri data files"""

try:
    from nibabel import load
except ImportError: 
        print "nibabel required for fmri I/O"

from nitime import TimeSeries

def time_series_from_file(nifti_file,coords,TR,average=False):
    """ Make a time series from a Analyze file, provided coordinates into the
            file 

    Parameters
    ----------

    nifti_file: string.

           The full path to the file from which the time-series is extracted 
     
    coords: ndarray or list of ndarrays
           x,y,z (inplane,inplane,slice) coordinates of the ROI from which the
           time-series is to be derived. If the list has more than one such
           array, the t-series will have more than one row in the data, as many
           as there are coordinates in the total list. Averaging is done on
           each item in the list separately, such that if several ROIs are
           entered, averaging will be done on each one separately and the
           result will be a time-series with as many rows of data as different
           ROIs in the input 

    TR: float, optional
        TR, if different from the one which can be extracted from the nifti
        file header

    average: bool, optional
           whether to average the time-series across the voxels in the ROI. In
           which case, TS.data will be 1-d

    Returns
    -------

    time-series object

    """
    
    im = load(nifti_file)
    data = im.get_data()

    out_data = data[coords[0],coords[1],coords[2]]
    
    if average:
        out_data = np.mean(out_data,0)
        
    tseries = TimeSeries(out_data,sampling_interval=TR)

    return tseries


def nifti_from_time_series(volume,coords,time_series,nifti_path):
    """Makes a Nifti file out of a time_series object

    Parameters
    ----------

    volume: list (3-d, or 4-d)
        The total size of the nifti image to be created

    coords: 3*n_coords array
        The coords into which the time_series will be inserted. These need to
        be given in the order in which the time_series is organized

    time_series: a time-series object
       The time-series to be inserted into the file

    nifti_path: the full path to the file name which will be created
    
       """
    # XXX Implement! 
    raise NotImplementedError
