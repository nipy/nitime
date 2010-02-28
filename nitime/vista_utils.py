#-----------------------------------------------------------------------------
# mrVista utils
# For the analysis of data created by the mrVista package
#-----------------------------------------------------------------------------   
"""These utilities can be used for extracting and processing fMRI data analyzed
using the Matlab toolbox mrVista (http://white.stanford.edu/mrvista)
""" 
import numpy as np
import scipy.io as sio
import timeseries as ts
import utils as tsu

##---- getROIcoords: -----------------------------------------------
def getROIcoords(ROI_file):
    """Get the ROI coordinates for a given ROI and scan in the Gray
    
    Parameters 
    ----------
    
    ROI_file : string, full path to the ROI file 
    
    Output
    ------

    coords: int array. The x,y,z coordinates of the ROI.

    Notes
    -----
    The order of x,y and z in the output may be slightly idiosyncratic and
    depends on the data type in question
    
    """

    ROI_mat_file = sio.loadmat(ROI_file,squeeze_me=True)
    
    return ROI_mat_file['ROI'].coords
    

##---- getTseries: -----------------------------------------------
def get_time_series_inplane(coords,scan_file,
                                  f_c=0.01,up_sample_factor=[1,1,1],
                                  detrend=True,normalize=True,average=True,
                                  TR=None):
    
    """vista_get_time_series: Acquire a time series for a particular scan/ROI.
    
    Parameters 
    ---------- 
    coords: a list of arrays
        each array holds the X,Y,Z locations of an ROI
        (as represented in the Inplane)

    scan_file: string, full path to the analyze file of the scan

    TR: float the repetition time in the experiment
    
    up_sample_factor: float
       the ratio between the size of the inplane and the size of the gray
       (taking into account FOV and number of voxels in each
       dimension). Defaults to [1,1,1] - no difference 
      
    detrend: bool, optional
      whether to detrend the signal. Default to 'True'
      
    normalize: bool, optional
      whether to transform the signal into % signal change. Default to 'True'

    average: bool, optional
      whether to average the resulting signal

    Returns
    -------
    time_series: array, the resulting time_series
    Depending on the averaging flag, can have the dimensions 1*time-points or
    number-voxels*time-points.
    
    Notes
    -----

    The order of the operations on the time-series is:
    detrend(on a voxel-by-voxel basis) => normalize (on a voxel-by-voxel basis)
    => average (across voxels, on a time-point-by-time-point basis)

    """
    from nipy.io.imageformats import load
    
    #Get the nifti image object

    print 'Reading data from %s' %scan_file 
    data = load(scan_file).get_data() #if using nipy.io.imageformats.load

    #Adjusted the coordinates according to the ratio between the
    #sampling in the gray and the sampling in the inplane, move the
    #slice dimension to be the first one and change the indexing from
    #1-based to 0-based. The coord order is as it is in the input, so need to
    #make sure that it is correct on the input side. 
    
    this_data = data[np.round(coords[0]/up_sample_factor[0]).astype(int)-1,
                         np.round(coords[1]/up_sample_factor[1]).astype(int)-1,
                         np.round(coords[2]/up_sample_factor[2]).astype(int)-1]

    if normalize:
        this_data = tsu.percent_change(this_data)

    if average:
        this_data = np.mean(this_data,0)
        
    time_series = ts.UniformTimeSeries(data=this_data,sampling_interval=TR)

    if detrend:
        F = ta.FilterAnalyzer(this_bold,lb=f_c)
        time_series = F.filtered_boxcar
        
    return time_series

#---detrend_tseries--------------------------------------------------------------
def detrend_tseries(time_series,TR,f_c,n_iterations=2):
    """ vista_detrend_tseries: detrending a-la DBR&DJH. A low-passed version is
    created by convolving with a box-car and then the low-passed version is
    subtracted from the signal, resulting in a high-passed version

    Parameters
    ----------

    time_series: float array
       the signal

    TR: float
      the sampling interval (inverse of the sampling rate)

    f_c: float
      the cut-off frequency for the high-/low-pass filtering. Default to 0.01 Hz

    n_iterations: int, optional
      how many rounds of smoothing to do (defaults to 2, based on DBR&DJH)

    Returns
    -------
    float array: the signal, filtered  
    """
    #Box-car filter
    box_car = np.ones(np.ceil(1.0/(f_c/TR)))
    box_car = box_car/(float(len(box_car))) 
    box_car_ones = np.ones(len(box_car))

    #Input can be 1-d (for a single time-series), or 2-d (for a stack of
    #time-series). Only in the latter case do we want to iterate over the
    #length of time_series: 
    
    if len(time_series.shape) > 1:        
        for i in xrange(time_series.shape[0]):    

            #Detrending: Start by applying a low-pass to the signal.
            #Pad the signal on each side with the initial and terminal
            #signal value:

            pad_s = np.append(box_car_ones * time_series[i][0],
                              time_series[i][:])
            pad_s = np.append(pad_s, box_car_ones * time_series[i][-1]) 

            #Filter operation is a convolution with the box-car(iterate,
            #n_iterations times over this operation):
            for i in xrange(n_iterations):
                conv_s = np.convolve(pad_s,box_car)

            #Extract the low pass signal by excising the central
            #len(time_series) points:
            #s_lp = conv_s[len(box_car):-1*len(box_car)]

            #does the same as this?

            s_lp= (conv_s[len(conv_s)/2-np.ceil(len(time_series[i][:])/2.0):
                         len(conv_s)/2+len(time_series[i][:])/2]) #ceil(/2.0)
            #for cases where the time_series has an odd number of points

            #Extract the high pass signal simply by subtracting the high pass
            #signal from the original signal:

            time_series[i] = time_series[i][:] - s_lp + np.mean(s_lp) #add mean
            #to make sure that there are no negative values. This also seems to
            #make sure that the mean of the signal (in % signal change) is close
            #to 0 

            
    else: #Same exact thing, but with one less index: 
        pad_s = np.append(box_car_ones * time_series[0],time_series[:])
        pad_s = np.append(pad_s, box_car_ones * time_series[-1]) 
        for i in xrange(n_iterations):
            conv_s = np.convolve(pad_s,box_car)
        s_lp= (conv_s[len(conv_s)/2-np.ceil(len(time_series[:])/2.0):
                         len(conv_s)/2+len(time_series[:])/2])
        time_series = time_series[:] - s_lp + np.mean(s_lp)
        

    #Handle memory: 
    time_series_out = np.copy(time_series)

    return time_series_out

##---- vista_filter_coords: -----------------------------------------------

def filter_coords(coords,filt,filt_thresh,up_sample_factor):
    
    """Filter the coords in an ROI, by the value in some other image (for
    example, the coherence in each of the voxels in the ROI)

    Params
    ------
    filt: an array with the values to filter on

    coords: the set of coordinates to filter

    filt_thresh: only coordinates with filter>filter_thresh will be kep
    Returns
    -------
    coords_out: array
       a new set of coords, in the same space as the input
           
    """
    coords_temp = np.where(filt>filt_thresh)
    coords_filt = np.vstack([coords_temp[0],coords_temp[1],coords_temp[2]])
        
    newCoords = np.empty(coords.shape,dtype='int')
    newCoords[0,:] = coords[0,:] / up_sample_factor[0] - 1 #Inplane 
    newCoords[1,:] = coords[1,:] / up_sample_factor[1] - 1 #Inplane
    newCoords[2,:] = coords[2,:] / up_sample_factor[2] - 1 #Slices

    coords_out = tsu.intersect_coords(newCoords,coords_filt)
        
    return coords_out
