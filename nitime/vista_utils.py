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

##---- getROIcoords: -----------------------------------------------
def vista_getROIcoords(ROI_file):
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
    
    ROIcoords = ROI_mat_file['ROI'].coords

    ROI_out = np.copy(ROIcoords)
    
    return ROI_out


##---- getTseries: -----------------------------------------------
def vista_get_time_series_inplane(coords,time_series_file,
                                  f_c=0.01,up_sample_factor=[1,1,1],
                                  detrend=True,normalize=True,average=True,
                                  TR=None):
    
    """vista_get_time_series: Acquire a time series for a particular scan/ROI.
    
    Parameters 
    ---------- 
    coords: a list of arrays
        each array holds the X,Y,Z locations of an ROI
        (as represented in the Inplane)

    time_series_file: string, full path to the analyze file of the scan

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

    #If we got a list of coord arrays, we're happy. Otherwise, we want to force
    #our input to be a list:
    try:
        coords.shape #If it is an array, it has a shape, otherwise, we 
        #assume it's a list. If it's an array, we want to
        #make it into a list:
        coords = [coords]
    except: #If it's a list already, we don't need to do anything:
        pass
    
    #Make a list the size of the coords-list, with place holder 0's: 
    newCoords = list([0]) * len(coords) 

    for i in xrange(len(coords)):            
        newCoords[i] = np.empty(coords[i].shape,dtype='int')
        #Adjusted the coordinates according to the ratio between the sampling
        #in the gray and the sampling in the inplane, move the slice dimension
        #to be the first one and change the indexing from 1-based to 0-based:
        newCoords[i][0,:] = coords[i][2,:] / up_sample_factor[2] - 1 #Slices 
        newCoords[i][1,:] = coords[i][0,:] / up_sample_factor[0] - 1 #Inplane
        newCoords[i][2,:] = coords[i][1,:] / up_sample_factor[1] - 1 #Inplane
    
    #Get the nifti image object
    print('Reading file: ' + time_series_file)

    #Initialize the time_series object and perform processing:
    time_series =  ts.time_series_from_nifti(time_series_file,
                                             newCoords,
                                             normalize=normalize,
                                             detrend=detrend,
                                             average=average,
                                             f_c=f_c,TR=TR)
        
    return time_series

#---detrend_tseries--------------------------------------------------------------
def vista_detrend_tseries(time_series,TR,f_c,n_iterations=2):
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

def vista_filter_coords(coords,filt,filt_thresh,up_sample_factor):
    
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

    coords_out = intersect_coords(newCoords,coords_filt)
        
    return coords_out
