""" Input and output for fmri data files"""

try:
    from nibabel import load
except ImportError: 
        raise ImportError("nibabel required for fmri I/O. See http://nipy.org/nibabel")

import nitime.timeseries as ts 
import nitime.analysis as tsa
import numpy as np

def time_series_from_file(nifti_files,coords,TR,normalize=None,average=False,
                          filter=None,verbose=False):
    """ Make a time series from a Analyze file, provided coordinates into the
            file 

    Parameters
    ----------

    nifti_files: a string or a list/tuple of strings.
        The full path(s) to the file(s) from which the time-series is (are)
        extracted
     
    coords: ndarray or list/tuple of ndarray
        x,y,z (inplane,inplane,slice) coordinates of the ROI(s) from which the
        time-series is (are) derived.
        
    TR: float, optional
        The TR of the fmri measurement
        
    normalize: bool, optional 
        Whether to normalize the activity in each voxel, defaults to
        None, in which case the original fMRI signal is used. Other options
        are: 'percent': the activity in each voxel is converted to percent
        change, relative to this scan. 'zscore': the activity is converted to a
        zscore relative to the mean and std in this voxel in this scan.

    average: bool, optional whether to average the time-series across the
        voxels in the ROI (assumed to be the first dimension). In which
        case, TS.data will be 1-d

    filter: dict, optional
       If provided with a dict of the form:

       {'lb':float or 0, 'ub':float or None, 'method':'fourier' or 'boxcar' }
       
       each voxel's data will be filtered into the frequency range [lb,ub] with
       nitime.analysis.FilterAnalyzer, using either the fourier or the boxcar
       method provided by that analyzer
       
    verbose: Whether to report on ROI and file being read.
    
    Returns
    -------

    time-series object

    Note
    ----

    Normalization occurs before averaging on a voxel-by-voxel basis, followed
    by the averaging. 

    """
    
    if normalize is not None:
        if normalize not in ('percent','zscore'):
            raise ValueError("Normalization of fMRI time-series can only be done using 'percent' or 'zscore' as input")
    #If just one string was provided:
    if isinstance(nifti_files,str):
        if verbose:
            print "Reading %s"%nifti_files
        im = load(nifti_files)
        data = im.get_data()
        #If the input is the coords of several ROIs
        if isinstance(coords,tuple) or isinstance(coords,list):
            n_roi = len(coords)
            out_data = [[]] * n_roi
            tseries = [[]] * n_roi
            for i in xrange(n_roi):
                tseries[i] = _tseries_from_nifti_helper(coords[i].astype(int),
                                                        data,TR,
                                                        filter,
                                                        normalize,
                                                        average)
        else:
            tseries = _tseries_from_nifti_helper(coords.astype(int),data,TR,
                                                 filter,normalize,average)
                
    #Otherwise loop over the files and concatenate:
    elif isinstance(nifti_files,tuple) or isinstance(nifti_files,list):
        tseries_list = []
        for f in nifti_files:
            if verbose:
                print "Reading %s"%f
            im = load(f)
            data = im.get_data()
            
            #If the input is the coords of several ROIs
            if isinstance(coords,tuple) or isinstance(coords,list):
                n_roi = len(coords)
                out_data = [[]] * n_roi
                tseries_list.append([[]] * n_roi)
                for i in xrange(n_roi):
                    tseries_list[-1][i] = _tseries_from_nifti_helper(
                                                coords[i].astype(int),
                                                data,TR,filter,normalize,average)

                
                
            else:
                tseries_list.append(_tseries_from_nifti_helper(
                                                       coords.astype(int),
                                                       data,TR,
                                                       filter,normalize,average))

        #Concatenate the time-series from the different scans:
                                    
        if isinstance(coords,tuple) or isinstance(coords,list):
            tseries = [[]] *n_roi
            #Do this per ROI
            for i in xrange(n_roi):
                tseries[i] = ts.concatenate_time_series(
                    [tseries_list[k][i] for k in xrange(len(tseries_list))])
            
        else:
            tseries = ts.concatenate_time_series(tseries_list)

    return tseries

def _tseries_from_nifti_helper(coords,data,TR,filter,normalize,average):
    """

    Helper function for the function time_series_from_nifti, which does the
    core operations of pulling out data from a data array given coords and then
    normalizing and averaging if needed 

    """ 
    out_data = np.asarray(data[coords[0],coords[1],coords[2]])
    tseries = ts.TimeSeries(out_data,sampling_interval=TR)

    if filter is not None:
        if filter['method'] not in ('boxcar','fourier'):
           raise ValueError("Filter method %s is not recognized"%filter['method'])
        if filter['method'] == 'boxcar':
           tseries=tsa.FilterAnalyzer(tseries,lb=filter['lb'],ub=filter['ub']).filtered_boxcar
        elif filter['method'] == 'fourier':
           tseries = tsa.FilterAnalyzer(tseries,lb=filter['lb'],ub=filter['ub']).filtered_fourier

    if normalize=='percent':
            tseries = tsa.NormalizationAnalyzer(tseries).percent_change
    elif normalize=='zscore':
            tseries = tsa.NormalizationAnalyzer(tseries).z_score
    if average:
            tseries.data = np.mean(tseries.data,0)

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
