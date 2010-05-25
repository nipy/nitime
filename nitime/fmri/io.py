""" Input and output for fmri-related data files"""


try:
    from nipy.io.files import load
except ImportError: 
        print "nipy not available"



def time_series_from_file(analyze_file,coords,normalize=False,detrend=False,
                           average=False,f_c=0.01,TR=None):
    """ Make a time series from a Analyze file, provided coordinates into the
            file 

    Parameters
    ----------

    analyze_file: string.

           The full path to the file from which the time-series is extracted 
     
    coords: ndarray or list of ndarrays
           x,y,z (slice,inplane,inplane) coordinates of the ROI from which the
           time-series is to be derived. If the list has more than one such
           array, the t-series will have more than one row in the data, as many
           as there are coordinates in the total list. Averaging is done on
           each item in the list separately, such that if several ROIs are
           entered, averaging will be done on each one separately and the
           result will be a time-series with as many rows of data as different
           ROIs in the input 

    detrend: bool, optional
           whether to detrend the time-series . For now, we do box-car
           detrending, but in the future we will do real high-pass filtering

    normalize: bool, optional
           whether to convert the time-series values into % signal change (on a
           voxel-by-voxel level)

    average: bool, optional
           whether to average the time-series across the voxels in the ROI. In
           which case, self.data will be 1-d

    f_c: float, optional
        cut-off frequency for detrending

    TR: float, optional
        TR, if different from the one which can be extracted from the nifti
        file header

    Returns
    -------

    time-series object

        """
    
    im = load(analyze_file)
    data = np.asarray(im)
    #Per default read TR from file:
    if TR is None:
        TR = im.header.get_zooms()[-1]/1000.0 #in msec?
        
    #If we got a list of coord arrays, we're happy. Otherwise, we want to force
    #our input to be a list:
    try:
        coords.shape #If it is an array, it has a shape, otherwise, we 
        #assume it's a list. If it's an array, we want to
        #make it into a list:
        coords = [coords]
    except: #If it's a list already, we don't need to do anything:
        pass

    #Make a list the size of the coords-list, with place-holder 0's
    data_out = list([0]) * len(coords)

    for c in xrange(len(coords)): 
        data_out[c] = data[coords[c][0],coords[c][1],coords[c][2],:]
        
        if normalize:
            data_out[c] = tsu.percent_change(data_out[c])

        #Currently uses mrVista style box-car detrending, will eventually be
        #replaced by a filter:
    
        if detrend:
            from nitime import vista_utils as tsv
            data_out[c] = tsv.detrend_tseries(data_out[c],TR,f_c)
            
        if average:
            data_out[c] = np.mean(data_out[c],0)

    #Convert this into the array with which the time-series object is
    #initialized:
    data_out = np.array(data_out).squeeze()
        
    tseries = TimeSeries(data_out,sampling_interval=TR)

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
