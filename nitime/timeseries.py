"""Base classes for generic time series analysis.

The classes implemented here are meant to provide fairly basic objects for
managing time series data.  They should serve mainly as data containers, with
only minimal algorithmic functionality.

In the timeseries subpackage, there is a separate library of algorithms, and
the classes defined here mostly delegate any computational facilitites they may
have to that library.

Over time, it is OK to add increasingly functionally rich classes, but only
after their design is well proven in real-world use.

"""
#-----------------------------------------------------------------------------
# Public interface
#-----------------------------------------------------------------------------
__all__ = ['time_unit_conversion',
           'TimeSeriesInterface',
           'UniformTimeSeries',
           'TimeInterface',
           'UniformTime',
           'TimeArray',
           ]
#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

import warnings
import numpy as np

# Our own
from nitime import descriptors as desc

#-----------------------------------------------------------------------------
# Module globals
#-----------------------------------------------------------------------------

# These are the valid names for time units, taken from the Numpy date/time
# types specification document.  They conform to SI nomenclature where
# applicable.

# Most uses of this are membership checks, so we make a set for fast
# validation.  But we create them first as a list so we can print an ordered
# and easy to read error message.

time_unit_conversion = {
                        'ps':1, #picosecond 
                        'ns':10**3,  # nanosecond
                        'us':10**6,  # microsecond
                        'ms':10**9,  # millisecond
                        's':10**12,   # second
                        None:10**12, #The default is seconds (when constructor
                                     #doesn't get any input, it defaults to
                                     #None)
                        'm':60*10**12,   # minute
                        'h':3600*10**12,   # hour
                        'D':24*3600*10**12,   # day
                        'W':7*24*3600*10**12,   # week #This is not an SI unit
                        }

# The basic resolution: 
base_unit = 'ps'


#-----------------------------------------------------------------------------
# Class declarations
#-----------------------------------------------------------------------------

##Time: 
class TimeInterface(object):
    """ The minimal object interface for time representations

    This should be thought of as an abstract base class. """

    time_unit = None
    
class TimeArray(np.ndarray,TimeInterface):
    """Base-class for time representations, implementing the TimeInterface"""  
    def __new__(cls, data, time_unit=None, copy=False):
        """XXX Write a doc-string - in particular, mention the the default
        time-units to be used are seconds (which is why it is set to None) """ 

        # Check that the time units provided are sensible: 
        if time_unit not in time_unit_conversion:
             raise ValueError('Invalid time unit %s, must be one of %s' %
                             (time_unit,time_unit_conversion.keys()))         

        conv_fac = time_unit_conversion[time_unit]

        # We can only honor the copy flag in a very narrow set of cases
        # if data is already an TimeArray or if data is an ndarray with
        # dtype=int64
        if copy==False and getattr(data, 'dtype', None) == np.int64:
            time = np.asarray(data)
        else:
            # XXX: do we mean isinstance(data,TimeInterface) - it could also be
            # NonUniformTime or UniformTime, it doesn't have to be an
            # TimeArray
            if isinstance(data, TimeArray):
                time = data.copy()
            else:
                data_arr = np.asarray(data)
                if issubclass(data_arr.dtype.type,np.integer):
                    # If this is an array of integers, cast to 64 bit integer
                    # and convert to the base_unit.
                    #XXX This will fail when even 64 bit is not large enough to
                    # avoid wrap-around (When you try to make more than 10**6
                    # seconds). XXX this should be mentioned in the docstring
                    time = data_arr.astype(np.int64)*conv_fac
                else:
                    # Otherwise: first convert, round and then cast to 64 
                    time=(data_arr*conv_fac).round().astype(np.int64)

        # Make sure you have an array on your hands (for example, if you input
        # an integer, you might have reverted to an integer when multiplying
        # with the conversion factor:            
        time = np.asarray(time).view(cls)

        if time_unit is None and isinstance(data, TimeArray):
            time_unit = data.time_unit

        if time_unit is None:
            time_unit = 's'

        time.time_unit = time_unit
        time._conversion_factor = conv_fac
        return time
    
    def __array_wrap__(self, out_arr, context=None):
        # When doing comparisons between TimeArrays, make sure that you return a
        # boolean array, not a time array: 
        if out_arr.dtype==bool:
            return np.asarray(out_arr)
        else:
            return np.ndarray.__array_wrap__(self, out_arr, context)

    def __array_finalize__(self,obj):
        """XXX """
        # Make sure that the TimeArray has the time units set (and not equal to
        # None): 
        if not hasattr(self, 'time_unit') or self.time_unit is None:
            if hasattr(obj, 'time_unit'): # looks like view cast
                self.time_unit = obj.time_unit
            else:
                self.time_unit = 's'

        # Make sure that the conversion factor is set properly: 
        if not hasattr(self,'_conversion_factor'):
            if hasattr(obj,'_conversion_factor'):
                self._conversion_factor = obj._conversion_factor
            else:
                self._conversion_factor=time_unit_conversion[self.time_unit]

    def __repr__(self):
       """Pass it through the conversion factor"""

       # If the input is a single int/float (with no shape) return a 'scalar'
       # time-point: 
       if self.shape == ():
           return "%r %s"%(int(self)/float(self._conversion_factor),
                           self.time_unit)
       
       # Otherwise, return the TimeArray representation:
       else:
           return np.ndarray.__repr__(self/float(self._conversion_factor)
            )[:-1] + ", time_unit='%s')" % self.time_unit

    def __getitem__(self,key):
        # return scalar TimeArray in case key is integer
        if isinstance(key,int):
            return self[[key]].reshape(())
        elif isinstance(key,float):
            return self.at(key)
        else:
            return np.ndarray.__getitem__(self,key)

    def __setitem__(self,key,val):
        
       # look at the units - convert the values to what they need to be (in the
       # base_unit) and then delegate to the ndarray.__setitem__
       if not hasattr(val,'_conversion_factor'):
           val *= self._conversion_factor
       return np.ndarray.__setitem__(self,key,val)
    
    def index_at(self,t,tol=None):
        """ Find the integer indices that corresponds to the time t"""
        t_e = TimeArray(t,time_unit=self.time_unit)
        d = np.abs(self-t_e)
        if tol is None:
            idx=np.where(d==d.min())
        else:
            # tolerance is converted into a time-array, so that it does the
            # right thing:
            tol = TimeArray(tol,time_unit=self.time_unit)
            idx=np.where(d<=tol)            

        return idx

    def at(self,t,tol=None):
        """ Returns the values of the TimeArray object at time t"""
        return self[self.index_at(t,tol=tol)]

    def min(self,axis=None,out=None):
        """Returns the minimal time"""
        # this is a quick fix to return a time and will
        # be obsolete once we use proper time dtypes
        if axis is not None:
            raise NotImplementedError, 'axis argument not implemented'
        if out is not None:
            raise NotImplementedError, 'out argument not implemented'
        if self.ndim:
            return self[self.argmin()]
        else:
            return self

    def max(self,axis=None,out=None):
        """Returns the maximal time"""
        # this is a quick fix to return a time and will
        # be obsolete once we use proper time dtypes
        if axis is not None:
            raise NotImplementedError, 'axis argument not implemented'
        if out is not None:
            raise NotImplementedError, 'out argument not implemented'
        if self.ndim:
            return self[self.argmax()]
        else:
            return self

    def convert_unit(self,time_unit):
        """Convert from one time unit to another in place"""
        #XXX Implement
        pass
     
class UniformTime(np.ndarray,TimeInterface):
    """ A representation of time sampled uniformly

    Parameters
    ----------
    
    length: int, the number of items in the time-array

    duration: float, the duration to be represented (given in the time-unit) of
    the array. If this item is an TimeArray, the units of the UniformTime
    array resulting will 'inherit' the units of the duration. Otherwise, the
    unit of the UniformTime will be set by that kwarg

    sampling_rate: float, the sampling rate (in 1/time-unit)

    sampling_interval: float, the inverse of the sampling_interval     

    t0: the value of the first time-point in the array (in time-unit)

    time_unit:

    copy: whether to make a copy of not. Needs to be set to False 

    
    

    XXX continue writing this
    """

    def __new__(cls,data=None,length=None,duration=None,sampling_rate=None,
                sampling_interval=None,t0=0,time_unit=None, copy=False):
        """Create a new UniformTime """

        # Sanity checks. There are different valid combinations of inputs
        tspec = tuple(x is not None for x in
                      [sampling_interval,sampling_rate,length,duration])

        # Used in converting tspecs to human readable form
        tspec_arg_names = ['sampling_interval','sampling_rate','length','duration']

        # The valid configurations 
        valid_tspecs=[
            # interval, length:
            (True,False,True,False),
            # interval, duration:
            (True,False,False,True),
            # rate, length:
            (False,True,True,False),
            # rate, duration:
            (False,True,False,True),
            # length, duration:
            (False,False,True,True)
            ]
        
        if isinstance(data,UniformTime):
            # Assuming data was given, some other tspecs become valid:
            tspecs_w_data=dict(
                    nothing=(False,False,False,False),
                    sampling_interval= (True,False,False,False),
                    sampling_rate= (False,True,False,False),
                    length= (False,False,True,False),
                    duration=(False,False,False,True))
            # preserve the order of the keys
            valid_tspecs.append( tspecs_w_data['nothing'])
            for name in tspec_arg_names:
                valid_tspecs.append( tspecs_w_data[name])

        if (tspec not in valid_tspecs):
            l = ['sampling_interval','sampling_rate','length','duration']
            #args = [arg for t,arg in zip(tspec,l) if t]
            raise ValueError("Invalid time specification.\n" +
                "You provided: %s \n"
                "%s \nsee docstring for more info." 
                %(str_tspec(tspec,tspec_arg_names), str_valid_tspecs(valid_tspecs,tspec_arg_names)))
            
        if isinstance(data,UniformTime):
            # Get attributes from the UniformTime object and transfer those over:
            if tspec==tspecs_w_data['nothing']:
                sampling_rate=data.sampling_rate
                duration = data.duration
            elif tspec==tspecs_w_data['sampling_interval']:
                duration==data.duration
            elif tspec==tspecs_w_data['sampling_rate']:
                if isinstance(sampling_rate,Frequency):
                    sampling_interval=sampling_rate.to_period()
                else:
                    sampling_interval = 1.0/sampling_rate
                duration=data.duration
            elif tspec==tspecs_w_data['length']:
                duration=length*data.sampling_interval
                sampling_rate=data.sampling_rate
            elif tspec==tspecs_w_data['duration']:
                sampling_rate=data.sampling_rate
            if time_unit is None:
                # If the user didn't ask to change the time-unit, use the
                # time-unit from the object you got:
                time_unit = data.time_unit
        
        # Check that the time units provided are sensible: 
        if time_unit not in time_unit_conversion:
            raise ValueError('Invalid time unit %s, must be one of %s' %
                         (time_unit,time_unit_conversion.keys()))         

        # Calculate the sampling_interval or sampling_rate:
        if sampling_interval is None:
            if isinstance(sampling_rate,Frequency):
                sampling_interval=sampling_rate.to_period()
            elif sampling_rate is None:
                sampling_interval = float(duration)/length
                sampling_rate = Frequency(1.0/sampling_interval,
                                             time_unit=time_unit)
            else:
                sampling_rate = Frequency(sampling_rate,time_unit='s')
                sampling_interval = sampling_rate.to_period()
        else:
            if isinstance(sampling_interval,TimeInterface):
                c_f = time_unit_conversion[sampling_interval.time_unit]
                sampling_rate = Frequency(1.0/(float(sampling_interval)/c_f),
                                          time_unit=sampling_interval.time_unit)
            else:
                sampling_rate = Frequency(1.0/sampling_interval,
                                          time_unit=time_unit)

        # Calculate the duration, if that is not defined:
        if duration is None:
            duration=length*sampling_interval

        # Make sure you have a time unit:
        if time_unit is None:
            #If you gave us a duration with time_unit attached 
            if isinstance(duration,TimeInterface):
                time_unit = duration.time_unit
            #Otherwise, you might have given us a sampling_interval with a
            #time_unit attached:
            elif isinstance(sampling_interval,TimeInterface):
                time_unit = sampling_interval.time_unit
            else:
                time_unit = 's'

        # 'cast' the time inputs as TimeArray
        duration=TimeArray(duration,time_unit=time_unit)
        #XXX If data is given - the t0 should be taken from there:
        t0=TimeArray(t0,time_unit=time_unit)
        sampling_interval=TimeArray(sampling_interval,time_unit=time_unit)

        # Check that the inputs are consistent, before making the array
        # itself:
        if duration<sampling_interval:
            raise ValueError('length/duration too short for the sampling_interval/sampling_rate')
        
        # in order for time[-1]-time[0]==duration to be true (which it should)
        # add the samling_interval to the stop value: 
        # time = np.arange(np.int64(t0),np.int64(t0+duration+sampling_interval),
        #                  np.int64(sampling_interval),dtype=np.int64)

        # But it's unclear whether that's really the behavior we want?
        time = np.arange(np.int64(t0),np.int64(t0+duration),
                         np.int64(sampling_interval),dtype=np.int64)

        time = np.asarray(time).view(cls)
        time.time_unit=time_unit
        time._conversion_factor=time_unit_conversion[time_unit]
        time.duration = duration
        time.sampling_rate=Frequency(sampling_rate)
        time.sampling_interval=sampling_interval
        time.t0 = t0
        
        return time

    def __array_wrap__(self, out_arr, context=None):
        # When doing comparisons between UniformTime, make sure that you retun a
        # boolean array, not a time array: 
        if out_arr.dtype==bool:
            return np.asarray(out_arr)
        else:
            return np.ndarray.__array_wrap__(self, out_arr, context)

    def __array_finalize__(self,obj):
        """XXX """
        # Make sure that the UniformTime has the time units set (and not equal to
        # None): 
        if not hasattr(self, 'time_unit') or self.time_unit is None:
            if hasattr(obj, 'time_unit'): # looks like view cast
                self.time_unit = obj.time_unit
            else:
                self.time_unit = 's'

        # Make sure that the conversion factor is set properly: 
        if not hasattr(self,'_conversion_factor'):
            if hasattr(obj,'_conversion_factor'):
                self._conversion_factor = obj._conversion_factor
            else:
                self._conversion_factor=time_unit_conversion[self.time_unit]

    def __repr__(self):
       """Pass it through the conversion factor"""

       #If the input is a single int/float (with no shape) return a 'scalar'
       #time-point: 
       if self.shape == ():
           return "%r %s"%(int(self)/float(self._conversion_factor),
                           self.time_unit)
       
       #Otherwise, return the UniformTime representation:
       else:
           return np.ndarray.__repr__(self/float(self._conversion_factor)
            )[:-1] + ", time_unit='%s')" % self.time_unit

    def __getitem__(self,key):
        # return scalar TimeArray in case key is integer
        if isinstance(key,int):
            return self[[key]].reshape(()).view(TimeArray)
        elif isinstance(key,float):
            return self.at(key)
        else:
            return np.ndarray.__getitem__(self,key)

    def __setitem__(self,key,val):
       # look at the units - convert the values to what they need to be (in the
       # base_unit) and then delegate to the ndarray.__setitem__    
       if not hasattr(val,'_conversion_factor'):
           val *= self._conversion_factor
       return np.ndarray.__setitem__(self,key,val)

    def index_at(self,t,boolean=False):
        """Find the index that corresponds to the time bin containing t

           Returns boolean indices if boolean=True and integer indeces otherwise.
        """

        # cast t into time
        ta = TimeArray(t,time_unit=self.time_unit)

        # check that index is within range
        if ta.min() < self.t0 or ta.max() >= self.t0 + self.duration:
            raise ValueError, 'index out of range'
        
        idx = ta.view(np.ndarray)//int(self.sampling_interval)
        if boolean:
            bool_idx = np.zeros(len(self),dtype=bool)
            bool_idx[idx] = True
            return bool_idx
        elif ta.ndim == 0:
            return idx[()]
        else:
            return idx

    def at(self,t,tol=None):
        """ Returns the values of the UniformTime object at time t"""
        return TimeArray(self[self.index_at(t)],time_unit=self.time_unit)

    def min(self,axis=None,out=None):
        """Returns the minimal time"""
        # this is a quick fix to return a time and will
        # be obsolete once we use proper time dtypes
        if axis is not None:
            raise NotImplementedError, 'axis argument not implemented'
        if out is not None:
            raise NotImplementedError, 'out argument not implemented'
        if self.ndim:
            return self[self.argmin()]
        else:
            return self

    def max(self,axis=None,out=None):
        """Returns the maximal time"""
        # this is a quick fix to return a time and will
        # be obsolete once we use proper time dtypes
        if axis is not None:
            raise NotImplementedError, 'axis argument not implemented'
        if out is not None:
            raise NotImplementedError, 'out argument not implemented'
        if self.ndim:
            return self[self.argmax()]
        else:
            return self

##Frequency:

class Frequency(float):
    """A class for representation of the frequency (in Hz) """

    def __new__(cls,f,time_unit='s'):
        """Initialize a frequency object """

        tuc = time_unit_conversion
        scale_factor = (float(tuc['s'])/tuc[time_unit])
        #If the input is a Frequency object, it is already in Hz: 
        if isinstance(f,Frequency)==False:
            #But otherwise convert to Hz:
            f = f*scale_factor

        freq = super(Frequency,cls).__new__(cls,f)
        freq._time_unit = time_unit

        return freq
    
    def __repr__(self):
        
        return str(self) + ' Hz'

    def to_period(self,time_unit=base_unit):
        """Convert the value of a frequency to the corresponding period
        (defaulting to a representation in the base_unit)

        """
        tuc = time_unit_conversion
        scale_factor = (float(tuc['s'])/tuc[time_unit])
        
        return np.int64((1/self)*scale_factor)
        
##Time-series: 
class TimeSeriesInterface(TimeInterface):
    """The minimally agreed upon interface for all time series.

    This should be thought of as an abstract base class.
    """
    time = None
    data = None
    metadata = None
    

class TimeSeriesBase(object):
    """Base class for time series, implementing the TimeSeriesInterface."""

    def __init__(self,data,time_unit):
        """Common constructor shared by all TimeSeries classes."""
        # Check that sensible time units were given
        if time_unit not in time_unit_conversion:
            raise ValueError('Invalid time unit %s, must be one of %s' %
                             (time_unit,time_unit_conversion.keys()))
        
        #: the data is an arbitrary numpy array
        self.data = np.asarray(data)
        self.time_unit = time_unit

        # Every instance carries an empty metadata dict, which we promise never
        # to touch.  This reserves this name as a user area for extra
        # information without the danger of name clashes in the future.
        self.metadata = {}


    def __len__(self):
        """Return the length of the time series."""
        return self.data.shape[-1]


    def _validate_dimensionality(self):
        """Check that the data and time have the proper dimensions.
        """

        if self.time.ndim != 1:
            raise ValueError("time array must be one-dimensional")
        npoints = self.data.shape[-1]
        if npoints != len(self.time):
            raise ValueError("mismatch of time and data dimensions")


    def __getitem__(self,key):
        """use fancy time-indexing (at() method).""" 
        if isinstance(key,TimeInterface):
            return self.at(key)
        else:
            return self.data[key]

        
class UniformTimeSeries(TimeSeriesBase):
    """Represent data collected at uniform intervals.
    
    Examples 
    --------

    The minimal specication of data and sampling interval:

    >>> ts = UniformTimeSeries([1,2,3],sampling_interval=0.25)
    >>> ts.time
    UniformTime([ 0.  ,  0.25,  0.5 ], time_unit='s')
    >>> ts.t0
    0.0 s
    >>> ts.sampling_rate
    4.0 Hz

    Or data and sampling rate:
    >>> ts = UniformTimeSeries([1,2,3],sampling_rate=2)
    >>> ts.time
    UniformTime([ 0. ,  0.5,  1. ], time_unit='s')
    >>> ts.t0
    0.0 s
    >>> ts.sampling_interval
    0.5 s

    A time series where we specify the start time and sampling interval:
    >>> ts = UniformTimeSeries([1,2,3],t0=4.25,sampling_interval=0.5)
    >>> ts.data
    array([1, 2, 3])
    >>> ts.time
    UniformTime([ 4.25,  4.75,  5.25], time_unit='s')
    >>> ts.t0
    4.25 s
    >>> ts.sampling_interval
    0.5 s
    >>> ts.sampling_rate
    2.0 Hz

    >>> ts = UniformTimeSeries([1,2,3],t0=4.25,sampling_rate=2.0)
    >>> ts.data
    array([1, 2, 3])
    >>> ts.time
    UniformTime([ 4.25,  4.75,  5.25], time_unit='s')
    >>> ts.t0
    4.25 s
    >>> ts.sampl
    ts.sampling_interval  ts.sampling_rate      
    >>> ts.sampling_interval
    0.5 s
    >>> ts.sampling_rate
    2.0 Hz

    """

    @desc.setattr_on_read
    def time(self):
        """Construct time array for the time-series object. This holds a
    UniformTime object, with properties derived from the UniformTimeSeries
    object"""
        return UniformTime(length=self.__len__(),t0=self.t0,
                           sampling_interval=self.sampling_interval,
                           time_unit=self.time_unit)

    #XXX This should call the constructor in an appropriate way, when provided
    #with a UniformTime object and data, so that you don't need to deal with
    #the constructor itself:  
    @staticmethod
    def from_time_and_data(time, data):
        return UniformTimeSeries.__init__(data, time=time)
        
    
    
    def __init__(self, data, t0=None, sampling_interval=None,
                 sampling_rate=None, duration=None, time=None, time_unit='s'):
        """Create a new UniformTimeSeries.

        This class assumes that data is uniformly sampled, but you can specify
        the sampling in one of three (mutually exclusive) ways:

        - sampling_interval [, t0]: data sampled starting at t0, equal
          intervals of sampling_interval.

        - sampling_rate [, t0]: data sampled starting at t0, equal intervals of
          width 1/sampling_rate.

        - time: a UniformTime object, in which case the UniformTimeSeries can
          'inherit' the properties of this object.  
        
        Parameters
        ----------
        data : array_like
          Data array, interpreted as having its last dimension being time.
        sampling_interval : float
          Interval between successive time points.
        sampling_rate : float
          Inverse of the interval between successive time points.
        t0 : float
          If you provide a sampling rate, you can optionally also provide a
          starting time.
        time
          Instead of sampling rate, you can explicitly provide an object of
        class UniformTime. Note that you can still also provide a different
        sampling_rate/sampling_interval/duration to take the place of the one
        in this object, but only as long as the changes are consistent with the
        length of the data. 
        
        time_unit :  string
          The unit of time.
        """

        #If a UniformTime object was provided as input: 
        if isinstance(time,UniformTime):
            c_fac = time._conversion_factor
            #If the user did not provide an alternative t0, get that from the
            #input: 
            if t0 is None:
                t0=time.t0
            #If the user did not provide an alternative sampling interval/rate:
            if sampling_interval is None and sampling_rate is None:
                sampling_interval = time.sampling_interval
                sampling_rate = time.sampling_rate
            #The duration can be read either from the length of the data, or
            #from the duration specified by the time-series: 
            if duration is None:
                duration=time.duration
                length = time.shape[-1]
                #If changing the duration requires a change to the
                #sampling_rate, make sure that this was explicitely required by
                #the user - if the user did not explicitely set the
                #sampling_rate, or it is inconsistent, throw an error: 
                if (length != len(data) and
                    sampling_rate != float(len(data)*c_fac)/time.duration):
                    e_s = "Length of the data (%s) " %str(len(data))  
                    e_s += "specified sampling_rate (%s) " %str(sampling_rate)
                    e_s +="do not match."
                    raise ValueError(e_s)
            #If user does not provide a     
            if time_unit is None:
                time_unit = time.time_unit

        else:    
            ##If the input was not a UniformTime, we need to check that there
            ##is enough information in the input to generate the UniformTime
            ##array.

            #There are different valid combinations of inputs
            tspec = tuple(x is not None for x in
                      [sampling_interval,sampling_rate,duration])

            tspec_arg_names = ["sampling_interval", "sampling_rate", "duration"] 

            #The valid configurations 
            valid_tspecs=[
                      #interval,length:
                      (True,False,False),
                      #interval,duration:
                      (True,False,True),
                      #rate,length:
                      (False,True,False),
                      #rate, duration:
                      (False,True,True),
                      #length,duration:
                      (False,False,True)
                      ]

            if tspec not in valid_tspecs:
                raise ValueError("Invalid time specification. \n"
                        "You provided: %s\n %s see docstring for more info." % (
                            str_tspec(tspec, tspec_arg_names),
                            str_valid_tspecs(valid_tspecs,tspec_arg_names)))
        
        #Calculate the sampling_interval or sampling_rate from each other and
        #assign t0, if it is not already assigned:
        if sampling_interval is None:
            if isinstance(sampling_rate,Frequency):
                sampling_interval=sampling_rate.to_period()
            elif sampling_rate is None:
                sampling_interval = float(duration)/self.__len__()
                sampling_rate = Frequency(1.0/sampling_interval,
                                             time_unit=time_unit)
            else:
                sampling_rate = Frequency(sampling_rate,time_unit='s')
                sampling_interval = sampling_rate.to_period()
        else:
            if sampling_rate is None: #Only if you didn't already 'inherit'
                                      #this property from another time object
                                      #above:
                if isinstance(sampling_interval,TimeInterface):
                   c_f = time_unit_conversion[sampling_interval.time_unit]
                   sampling_rate = Frequency(1.0/(float(sampling_interval)/c_f),
                                          time_unit=sampling_interval.time_unit)
                else:
                   sampling_rate = Frequency(1.0/sampling_interval,
                                          time_unit=time_unit)

            
        #Calculate the duration, if that is not defined:
        if duration is None:
            duration=np.asarray(data).shape[-1]*sampling_interval

        if t0 is None:
           t0=0
           
        # Make sure to grab the time unit from the inputs, if it is provided: 
        if time_unit is None:
            #If you gave us a duration with time_unit attached 
            if isinstance(duration,TimeInterface):
                time_unit = duration.time_unit
            #Otherwise, you might have given us a sampling_interval with a
            #time_unit attached:
            elif isinstance(sampling_interval,TimeInterface):
                time_unit = sampling_interval.time_unit

        #Otherwise, you can still call the common constructor to get the real
        #object initialized, with time_unit set to None and that will generate
        #the object with time_unit set to 's':  
        TimeSeriesBase.__init__(self,data,time_unit)
    
        self.time_unit = time_unit
        self.sampling_interval = TimeArray(sampling_interval,
                                           time_unit=self.time_unit) 
        self.t0 = TimeArray(t0,time_unit=self.time_unit)
        self.sampling_rate = sampling_rate
        self.duration = TimeArray(duration,time_unit=self.time_unit)

    def at(self,t,tol=None):
        """ Returns the values of the TimeArray object at time t"""
        return self.data[...,self.time.index_at(t)]

            
        #Calculate the duration, if that is not defined:
        if duration is None:
            duration=np.asarray(data).shape[-1]*sampling_interval

        if t0 is None:
           t0=0
           
        # Make sure to grab the time unit from the inputs, if it is provided: 
        if time_unit is None:
            #If you gave us a duration with time_unit attached 
            if isinstance(duration,TimeInterface):
                time_unit = duration.time_unit
            #Otherwise, you might have given us a sampling_interval with a
            #time_unit attached:
            elif isinstance(sampling_interval,TimeInterface):
                time_unit = sampling_interval.time_unit

        #Otherwise, you can still call the common constructor to get the real
        #object initialized, with time_unit set to None and that will generate
        #the object with time_unit set to 's':  
        TimeSeriesBase.__init__(self,data,time_unit)
    
        self.time_unit = time_unit
        self.sampling_interval = TimeArray(sampling_interval,
                                           time_unit=self.time_unit) 
        self.t0 = TimeArray(t0,time_unit=self.time_unit)
        self.sampling_rate = sampling_rate
        self.duration = TimeArray(duration,time_unit=self.time_unit)


def str_tspec(tspec, arg_names):
    """ Turn a single tspec into human readable form"""
    # an all "False" will convert to an empty string unless we do the following
    # where we create an all False tuple of the appropriate length
    if tspec==tuple([False]*len(arg_names)):
        return "(nothing)"
    return ", ".join([arg for t,arg in zip(tspec,arg_names) if t])

def str_valid_tspecs(valid_tspecs, arg_names):
    """Given a set of valid_tspecs, return a string that turns them into
    human-readable form"""
    vargs = []
    for tsp in valid_tspecs: 
        vargs.append(str_tspec(tsp, arg_names))
    return "\n Valid time specifications are:\n\t%s" %("\n\t".join(vargs))


class NonUniformTimeSeries(TimeSeriesBase):
    """Represent data collected at arbitrary time points.

    This class combines a one dimensional array of time values (assumed, but
    not verified, to be monotonically increasing) with an n-dimensional array
    of data values.

    Examples
    --------
    >>> t = np.array([0.3, 0.5, 1, 1.9])
    >>> y = np.array([4.7, 8.4, 9.1, 10.4])
    >>> uts = NonUniformTimeSeries(t,y)
    >>> uts.time
    array([  4.7,   8.4,   9.1,  10.4])
    >>> uts.data
    array([ 0.3,  0.5,  1. ,  1.9])
    >>> uts.time_unit
    's'
    """

    def __init__(self,data,time,time_unit='s'):
        """Construct a new NonUniformTimeSeries from data and time.

        Parameters
        ----------
        data : ndarray
          An n-dimensional dataset whose last axis runs along the time
          direction.
        time : 1-d array
          A sorted array of time values, with as many points as the last
          dimension of the dataset.
        time_unit :  string
          The unit of time.
        """
        # Call the common constructor to get the real object initialized
        TimeSeriesBase.__init__(self,data,time_unit)

        self.time = np.asarray(time)


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
    try:
        from nipy.io.files import load
    except ImportError: 
        print "nipy not available"
    
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
        
    tseries = UniformTimeSeries(data_out,sampling_interval=TR)

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
    
def concatenate_uniform_time_series(time_series_list):
    """Concatenates a list of time-series objects in time, according to their
    order in the input list.

    This one requires that all the time-series in the list have the same
    sampling rate and that all the data have the same number of items in all
    dimensions, except the time dimension"""

    total_len = 0
    for i in xrange(len(time_series_list)):
        total_len += time_series_list[i].data.shape[-1]

    #The data in the output object has the size of the input time-series,
    #except in the last dimension (time), where it has the sum of all the
    #lengths of the time-series:
    
    data_out = np.empty(time_series_list[0].data.shape[0:-1]
                        + (total_len,)) #this variable is an int, so needs to
                                        #be cast into a tuple, so that it can
                                        #be used to initialize the empty variable
    
    idx_start = 0
    for i in xrange(len(time_series_list)):
        idx_end = idx_start+time_series_list[i].data.shape[-1]
        data_out[...,idx_start:idx_end] = time_series_list[i].data
        idx_start = idx_end


    tseries = UniformTimeSeries(data_out,
                    sampling_interval=time_series_list[0].sampling_interval)

    return tseries

    
def concatenate_time_series(time_series_list):
    """Concatenates a list of time series objects in time, according to their
    order in the input list.

    This one doesn't require that the time-series all have the same sampling
    rate. Requires that the data all have the same number of rows""" 

    # XXX Implement! Probably as generalization of above
    # (concatenate_uniform_time_series)
    raise NotImplementedError

