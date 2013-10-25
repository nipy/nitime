""" Input and output for fmri data files"""
from __future__ import print_function

try:
    from nibabel import load
except ImportError:
    e_s = "nibabel required for fmri I/O. See http://nipy.org/nibabel"
    raise ImportError(e_s)

import nitime.timeseries as ts
import nitime.analysis as tsa
import numpy as np


def time_series_from_file(nifti_files, coords=None, TR=None, normalize=None,
                          average=False, filter=None, verbose=False):
    """ Make a time series from a Analyze file, provided coordinates into the
            file

    Parameters
    ----------

    nifti_files: a string or a list/tuple of strings.
        The full path(s) to the file(s) from which the time-series is (are)
        extracted

    coords: ndarray or list/tuple of ndarray, optional.
        x,y,z (inplane,inplane,slice) coordinates of the ROI(s) from which the
        time-series is (are) derived. If coordinates are provided, the
        resulting time-series object will have 2 dimentsions. The first is the
        coordinate dimension, in order of the provided coordinates and the
        second is time. If set to None, all the coords in the volume will be
        used and the coordinate system will be preserved - the output will be 4
        dimensional, with time as the last dimension.

    TR: float or TimeArray, optional
        The TR of the fmri measurement. The units are seconds, if provided as a float
        argument. Otherwise, in the units of the TimeArray object
        provided. Default: 1 second.

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

       {'lb':float or 0, 'ub':float or None, 'method':'fourier','boxcar' 'fir'
       or 'iir' }

       each voxel's data will be filtered into the frequency range [lb,ub] with
       nitime.analysis.FilterAnalyzer, using the method chosen here (defaults
       to 'fir')

    verbose: Whether to report on ROI and file being read.

    Returns
    -------

    time-series object

    Note
    ----

    Normalization occurs before averaging on a voxel-by-voxel basis, followed
    by the averaging.

    """

    # The default behavior is to assume that the TR is one second:
    if TR is None:
        TR = 1.0

    if normalize is not None:
        if normalize not in ('percent', 'zscore'):
            e_s = "Normalization of fMRI time-series can only be done"
            e_s += " using 'percent' or 'zscore' as input"
            raise ValueError(e_s)
    #If just one string was provided:
    if isinstance(nifti_files, str):
        if verbose:
            print("Reading %s" % nifti_files)
        im = load(nifti_files)
        data = im.get_data()
        # If coordinates are provided as input, read data only from these coordinates:
        if coords is not None:
            #If the input is the coords of several ROIs
            if isinstance(coords, tuple) or isinstance(coords, list):
                n_roi = len(coords)
                tseries = [[]] * n_roi
                for i in range(n_roi):
                    tseries[i] = _tseries_from_nifti_helper(
                        np.array(coords[i]).astype(int),
                        data,
                        TR,
                        filter,
                        normalize,
                        average)
            else:
                tseries = _tseries_from_nifti_helper(coords.astype(int), data, TR,
                                                     filter, normalize, average)

        # The default behavior reads in all the coordinates in the volume:
        else:
            tseries = _tseries_from_nifti_helper(coords, data, TR,
                                                     filter, normalize, average)
    #Otherwise loop over the files and concatenate:
    elif isinstance(nifti_files, tuple) or isinstance(nifti_files, list):
        tseries_list = []
        for f in nifti_files:
            if verbose:
                print("Reading %s" % f)
            im = load(f)
            data = im.get_data()
            if coords is not None:
                #If the input is the coords of several ROIs
                if isinstance(coords, tuple) or isinstance(coords, list):
                    n_roi = len(coords)
                    tseries_list.append([[]] * n_roi)
                    for i in range(n_roi):
                        tseries_list[-1][i] = _tseries_from_nifti_helper(
                            np.array(coords[i]).astype(int),
                            data,
                            TR,
                            filter,
                            normalize,
                            average)

                else:
                    tseries_list.append(_tseries_from_nifti_helper(
                        np.array(coords).astype(int),
                        data,
                        TR,
                        filter,
                        normalize,
                        average))

            # The default behavior reads in all the coordinates in the volume:
            else:
                tseries_list.append(_tseries_from_nifti_helper(coords, data, TR,
                                                               filter, normalize, average))


        #Concatenate the time-series from the different scans:
        if isinstance(coords, tuple) or isinstance(coords, list):
            tseries = [[]] * n_roi
            #Do this per ROI
            for i in range(n_roi):
                tseries[i] = ts.concatenate_time_series(
                    [tseries_list[k][i] for k in range(len(tseries_list))])

        else:
            tseries = ts.concatenate_time_series(tseries_list)

    return tseries


def _tseries_from_nifti_helper(coords, data, TR, filter, normalize, average):
    """

    Helper function for the function time_series_from_nifti, which does the
    core operations of pulling out data from a data array given coords and then
    normalizing and averaging if needed

    """
    if coords is not None:
        out_data = np.asarray(data[coords[0], coords[1], coords[2]])
    else:
        out_data = data

    tseries = ts.TimeSeries(out_data, sampling_interval=TR)

    if filter is not None:
        if filter['method'] not in ('boxcar', 'fourier', 'fir', 'iir'):
            e_s = "Filter method %s is not recognized" % filter['method']
            raise ValueError(e_s)
        else:
            #Construct the key-word arguments to FilterAnalyzer:
            kwargs = dict(lb=filter.get('lb', 0),
                          ub=filter.get('ub', None),
                          boxcar_iterations=filter.get('boxcar_iterations', 2),
                          filt_order=filter.get('filt_order', 64),
                          gpass=filter.get('gpass', 1),
                          gstop=filter.get('gstop', 60),
                          iir_ftype=filter.get('iir_ftype', 'ellip'),
                          fir_win=filter.get('fir_win', 'hamming'))

            F = tsa.FilterAnalyzer(tseries, **kwargs)

        if filter['method'] == 'boxcar':
            tseries = F.filtered_boxcar
        elif filter['method'] == 'fourier':
            tseries = F.filtered_fourier
        elif filter['method'] == 'fir':
            tseries = F.fir
        elif filter['method'] == 'iir':
            tseries = F.iir

    if normalize == 'percent':
        tseries = tsa.NormalizationAnalyzer(tseries).percent_change
    elif normalize == 'zscore':
        tseries = tsa.NormalizationAnalyzer(tseries).z_score

    if average:
        if coords is None:
            tseries.data = np.mean(np.reshape(tseries.data,
                                              (np.array(tseries.shape[:-1]).prod(),
                                               tseries.shape[-1])),0)
        else:
            tseries.data = np.mean(tseries.data, 0)

    return tseries


def nifti_from_time_series(volume, coords, time_series, nifti_path):
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
