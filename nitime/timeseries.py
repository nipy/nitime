"""Base classes for generic time series analysis.

The classes implemented here are meant to provide fairly basic objects for
managing time series data.  They should serve mainly as data containers, with
only minimal algorithmic functionality.

In the timeseries subpackage, there is a separate library of algorithms, and
the classes defined here mostly delegate any computational facilities they may
have to that library.

Over time, it is OK to add increasingly functionally rich classes, but only
after their design is well proven in real-world use.

"""
#-----------------------------------------------------------------------------
# Public interface
#-----------------------------------------------------------------------------
__all__ = ['time_unit_conversion',
           'TimeSeriesInterface',
           'TimeSeries',
           'TimeInterface',
           'UniformTime',
           'TimeArray',
           'Epochs',
           'Events'
           ]
#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

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
                        'ps': 1,  # picosecond
                        'ns': 10 ** 3,  # nanosecond
                        'us': 10 ** 6,  # microsecond
                        'ms': 10 ** 9,  # millisecond
                        's': 10 ** 12,   # second
                         None: 10 ** 12,  # The default is seconds (when
                                        # constructor doesn't get any
                                        # input, it defaults to None)
                        'm': 60 * 10 ** 12,   # minute
                        'h': 3600 * 10 ** 12,   # hour
                        'D': 24 * 3600 * 10 ** 12,   # day
                        'W': 7 * 24 * 3600 * 10 ** 12,  # week
                                                        # (not an SI unit)
                        }

# The basic resolution:
base_unit = 'ps'


#-----------------------------------------------------------------------------
# Class declarations
#-----------------------------------------------------------------------------

# Time:
class TimeInterface(object):
    """ The minimal object interface for time representations

    This should be thought of as an abstract base class. """

    time_unit = None


def get_time_unit(obj):
    """
    Extract the time unit of the object. If it is an iterable, get the time
    unit of the first element.
    """

    # If this is a Time object, no problem:
    if isinstance(obj, TimeInterface):
        return obj.time_unit

    # Otherwise, if it is iterable, we recurse on it:
    try:
        it = iter(obj)
    except TypeError:
        return None
    else:
        return get_time_unit(next(it))


class TimeArray(np.ndarray, TimeInterface):
    """Base-class for time representations, implementing the TimeInterface"""
    def __new__(cls, data, time_unit=None, copy=True):
        """
        Parameters
        ----------
        data : 1-d array or `TimeArray` class instance
            Time points

        time_unit : str, optional
            The time-unit to use. This should be one of the keys of the
            `time_unit_conversion` dict from the :mod:`timeseries` module,
            which are SI units of time. Default: 's'

        copy : bool, optional
            Whether to create this instance by  copy of a

        Note
        ----
        If the 'copy' input is set to False, input must be either a `TimeArray`
        class instance, or an int64 array in the base unit of the module
        (which, unless you change it, is picoseconds)


        """

        # Check that the time units provided are sensible:
        if time_unit not in time_unit_conversion:
            raise ValueError('Invalid time unit %s, must be one of %s' %
                             (time_unit, time_unit_conversion.keys()))

        # Get the conversion factor from the input:
        conv_fac = time_unit_conversion[time_unit]

        # Call get_time_unit to pull the time_unit out from inside:
        data_time_unit = get_time_unit(data)
        # If it has a time unit, you should not convert the values to
        # base_unit, because they are already in that:
        if data_time_unit is not None:
            conv_fac = 1

        # We check whether the data has a time-unit somewhere inside (for
        # example, if it is a list of TimeArray objects):
        if time_unit is None:
            time_unit = data_time_unit

        # We can only honor the copy flag in a very narrow set of cases
        # if data is already a TimeArray or if data is an ndarray with
        # dtype=int64
        if copy == False:
            if not getattr(data, 'dtype', None) == np.int64:
                e_s = 'When copy flag is set to False, must provide a'
                e_s += 'TimeArray in object, or int64 times, in %s' % base_unit
                raise ValueError(e_s)

            time = np.array(data, copy=False)
        else:
            if isinstance(data, TimeInterface):
                time = data.copy()
            else:
                data_arr = np.asarray(data)
                if issubclass(data_arr.dtype.type, np.integer):
                    # If this is an array of integers, cast to 64 bit integer
                    # and convert to the base_unit.
                    #XXX This will fail when even 64 bit is not large enough to
                    # avoid wrap-around (When you try to make more than 10**6
                    # seconds). XXX this should be mentioned in the docstring
                    time = data_arr.astype(np.int64) * conv_fac
                else:
                    # Otherwise: first convert, round and then cast to 64
                    time = (data_arr * conv_fac).round().astype(np.int64)

        # Make sure you have an array on your hands (for example, if you input
        # an integer, you might have reverted to an integer when multiplying
        # with the conversion factor:
        time = np.asarray(time).view(cls)

        # Make sure time is one-dimensional or 0-d
        if time.ndim > 1:
            raise ValueError('TimeArray can only be one-dimensional or 0-d')

        if time_unit is None:
            time_unit = 's'

        time.time_unit = time_unit
        time._conversion_factor = time_unit_conversion[time_unit]
        return time

    def __array_wrap__(self, out_arr, context=None):
        # When doing comparisons between TimeArrays, make sure that you return
        # a boolean array, not a time array:
        if out_arr.dtype == bool:
            return np.asarray(out_arr)
        else:
            return np.ndarray.__array_wrap__(self, out_arr, context)

    def __array_finalize__(self, obj):
        """XXX """
        # Make sure that the TimeArray has the time units set (and not equal to
        # None):
        if not hasattr(self, 'time_unit') or self.time_unit is None:
            if hasattr(obj, 'time_unit'):  # looks like view cast
                self.time_unit = obj.time_unit
            else:
                self.time_unit = 's'

        # Make sure that the conversion factor is set properly:
        if not hasattr(self, '_conversion_factor'):
            if hasattr(obj, '_conversion_factor'):
                self._conversion_factor = obj._conversion_factor
            else:
                self._conversion_factor = time_unit_conversion[self.time_unit]

    def __repr__(self):
        """Pass it through the conversion factor"""

        # If the input is a single int/float (with no shape) return a 'scalar'
        # time-point:
        if self.shape == ():
            return "%r %s" % (int(self) / float(self._conversion_factor),
                           self.time_unit)
        # Otherwise, return the TimeArray representation:
        else:
            return np.ndarray.__repr__(self / float(self._conversion_factor)
             )[:-1] + ", time_unit='%s')" % self.time_unit

    def __str__(self):
        """Return a nice string representation of this TimeArray"""
        return self.__repr__()

    def __getitem__(self, key):
        # return scalar TimeArray in case key is integer
        if isinstance(key, (int, np.int64, np.int32)):
            return self[[key]].reshape(())
        elif isinstance(key, float):
            return self.at(key)
        elif isinstance(key, Epochs):
            return self.during(key)
        else:
            return np.ndarray.__getitem__(self, key)

    def __setitem__(self, key, val):
        # look at the units - convert the values to what they need to be (in
        # the base_unit) and then delegate to the ndarray.__setitem__
        if not hasattr(val, '_conversion_factor'):
            val *= self._conversion_factor
        return np.ndarray.__setitem__(self, key, val)

    def _convert_if_needed(self,val):
        if not hasattr(val, '_conversion_factor'):
            val = np.asarray(val)
            if getattr(val, 'dtype', None) == np.int32:
                # we'll overflow if val's dtype is np.int32
                val = np.array(val, dtype=np.int64)
            val *= self._conversion_factor
        return val

    def __add__(self, val):
        val = self._convert_if_needed(val)
        return np.ndarray.__add__(self, val)

    def __sub__(self, val):
        val = self._convert_if_needed(val)
        return np.ndarray.__sub__(self, val)

    def __radd__(self, val):
        val = self._convert_if_needed(val)
        return np.ndarray.__radd__(self, val)

    def __rsub__(self, val):
        val = self._convert_if_needed(val)
        return np.ndarray.__rsub__(self, val)

    def __lt__(self, val):
        val = self._convert_if_needed(val)
        return np.ndarray.__lt__(self, val)

    def __gt__(self, val):
        val = self._convert_if_needed(val)
        return np.ndarray.__gt__(self, val)

    def __le__(self, val):
        val = self._convert_if_needed(val)
        return np.ndarray.__le__(self, val)

    def __ge__(self, val):
        val = self._convert_if_needed(val)
        return np.ndarray.__ge__(self,val)

    def __eq__(self, val):
        val = self._convert_if_needed(val)
        return np.ndarray.__eq__(self,val)

    def min(self, *args, **kwargs):
        ret = TimeArray(np.ndarray.min(self, *args, **kwargs),
            time_unit=base_unit)
        ret.convert_unit(self.time_unit)
        return ret

    def max(self, *args,**kwargs):
        ret = TimeArray(np.ndarray.max(self, *args, **kwargs),
            time_unit=base_unit)
        ret.convert_unit(self.time_unit)
        return ret

    def mean(self, *args, **kwargs):
        ret = TimeArray(np.ndarray.mean(self, *args, **kwargs),
            time_unit=base_unit)
        ret.convert_unit(self.time_unit)
        return ret

    def ptp(self, *args, **kwargs):
        ret = TimeArray(np.ndarray.ptp(self, *args, **kwargs),
                        time_unit=base_unit)
        ret.convert_unit(self.time_unit)
        return ret

    def sum(self, *args,**kwargs):
        ret = TimeArray(np.ndarray.sum(self, *args, **kwargs),
                        time_unit=base_unit)
        ret.convert_unit(self.time_unit)
        return ret

    def prod(self, *args, **kwargs):
        e_s = "Product computation changes TimeArray units"
        raise NotImplementedError(e_s)

    def var(self, *args, **kwargs):
        e_s = "Variance computation changes TimeArray units"
        raise NotImplementedError(e_s)

    def std(self, *args, **kwargs):
        """Returns the standard deviation of this TimeArray (with time units)
        for detailed information, see numpy.std()
        """
        ret = TimeArray(np.ndarray.std(self, *args, **kwargs),
                        time_unit=base_unit)
        ret.convert_unit(self.time_unit)
        return ret

    def index_at(self, t, tol=None, mode='closest'):
        """ Returns the integer indices that corresponds to the time t

        The returned indices depend on both `tol` and `mode`.  The `tol`
        parameter specifies how close the given time must be to those present
        in the array to give a match, when `mode` is `closest`.  The default
        tolerance is 1 `base_unit` (by default, picoseconds).  If you specify
        the tolerance as 0, then only *exact* matches are allowed, be careful
        in this case of possible problems due to floating point roundoff error
        in your time specification.

        When mode is `before` or `after`, the tolerance is completely ignored.
        In this case, either the largest time equal or *before* the given `t`
        or the earliest time equal or *after* the given `t` is returned.

        Parameters
        ----------
        t : time-like
          Anything that is valid input for a TimeArray constructor.
        tol : time-like, optional
          Tolerance, specified in the time units of this TimeArray.
        mode : string
          One of ['closest', 'before', 'after'].

        Returns
        -------
        idx : The array with all the indices where the condition is met.
          """
        if not np.iterable(t):
            t = [t]
        t_e = TimeArray(t, time_unit=self.time_unit)
        if mode == 'closest':
            return self._index_closest(t_e, tol)
        elif mode == 'before':
            return self._index_before(t_e)
        elif mode == 'after':
            return self._index_after(t_e)
        else:
            raise ValueError('Invalid mode specification')

    def _index_closest(self, t, tol=None):
        d = np.abs(self - t)
        if tol is None:
            # If no tolerance is specified, use one clock tick of the
            # base_unit:
            tol = clock_tick

        # tolerance is converted into a time-array, so that it does the
        # right thing:
        ttol = TimeArray(tol, time_unit=self.time_unit)
        return np.where(d <= ttol)[0]

    def _index_before(self, t):
        # Use the standard Decorate-Sort-Undecorate (Schwartzian transform)
        # pattern to find the right index.
        cond = np.where(self <= t)[0]
        if len(cond) == 0:
            return cond
        idx_max = self[cond].argmax()
        return cond[idx_max]

    def _index_after(self, t):
        cond = np.where(t <= self)[0]
        if len(cond) == 0:
            return cond

        idx_min = self[cond].argmin()
        return cond[idx_min]

    def slice_during(self, e):
        """ Returns the slice that corresponds to Epoch e"""

        if not isinstance(e, Epochs):
            raise ValueError('e has to be of Epochs type')

        if e.data.ndim > 0:
            raise NotImplementedError('e has to be a scalar Epoch')

        if self.ndim != 1:
            e_s = 'slicing only implemented for 1-d TimeArrays'
            return NotImplementedError(e_s)

        # These two should be called with modes, such that they catch the right
        # slice
        start = self.index_at(e.start, mode='after')
        stop = self.index_at(e.stop, mode='before')

        # If *either* the start or stop index object comes back as the empty
        # array, then it means the condition is not satisfied, we return the
        # slice that does [:0],  i.e., always slices to nothing.
        if start.shape == (0,) or stop.shape == (0,):
            return slice(0)

        # Now,  we know the start/stop are not empty arrays, but they can be
        # either scalars or arrays.
        i_start = start if np.isscalar(start) else start.max()
        i_stop = stop if np.isscalar(stop) else stop.min()

        if e.start > self[i_start]:  # make sure self[i_start] is in epoch e
            i_start += 1
        if e.stop > self[i_stop]:  # make sure to include self[i_stop]
            i_stop += 1

        return slice(i_start, i_stop)

    def at(self, t, tol=None):
        """ Returns the values of the TimeArray object at time t"""
        return self[self.index_at(t, tol=tol)]

    def during(self, e):
        """ Returns the values of the TimeArray object during Epoch e"""

        if not isinstance(e, Epochs):
            raise ValueError('e has to be of Epochs type')

        if e.data.ndim > 0:
            ## TODO: Implement slicing with 1-d Epochs array,
            ## resulting in (ragged/jagged) 2-d TimeArray
            raise NotImplementedError('e has to be a scalar Epoch')

        return self[self.slice_during(e)]

##     def min(self,axis=None,out=None):
##         """Returns the minimal time"""
##         # this is a quick fix to return a time and will
##         # be obsolete once we use proper time dtypes
##         if axis is not None:
##             raise NotImplementedError, 'axis argument not implemented'
##         if out is not None:
##             raise NotImplementedError, 'out argument not implemented'
##         if self.ndim:
##             return self[self.argmin()]
##         else:
##             return self

    def max(self, axis=None, out=None):
        """Returns the maximal time"""
        # this is a quick fix to return a time and will
        # be obsolete once we use proper time dtypes
        if axis is not None:
            raise NotImplementedError('axis argument not implemented')
        if out is not None:
            raise NotImplementedError('out argument not implemented')
        if self.ndim:
            return self[self.argmax()]
        else:
            return self

    def convert_unit(self, time_unit):
        """Convert from one time unit to another in place"""

        self.time_unit = time_unit
        self._conversion_factor = time_unit_conversion[time_unit]

    def __div__(self, d):
        """Division by another time object eliminates units """
        if isinstance(d, TimeInterface):
            return np.divide(np.array(self), np.array(d).astype(float))
        else:
            return np.divide(self, d)

    __truediv__ = __div__ # called by python3

# Globally define a single tick of the base unit:
clock_tick = TimeArray(1, time_unit=base_unit)


class UniformTime(np.ndarray, TimeInterface):
    """ A representation of time sampled uniformly
    """

    def __new__(cls, data=None, length=None, duration=None, sampling_rate=None,
                sampling_interval=None, t0=0, time_unit=None):
        """

        Parameters
        ----------
        length : int
            The number of items in the time-array

        duration : float,
            the duration to be represented (given in the time-unit) of the
            array. If this item is an TimeArray, the units of the UniformTime
            array resulting will 'inherit' the units of the
            duration. Otherwise, the unit of the UniformTime will be set by
            that kwarg

        sampling_rate : float
            The sampling rate (in Hz)

        sampling_interval : float
            The inverse of the sampling_interval

        t0 : float, int or singleton `TimeArray`
            The value of the first time-point in the array (unless given as a
            `TimeArray`, should be in the time-unit)

        time_unit : str, optional
            The time unit to be used in the representation of time

        """

        # Sanity checks. There are different valid combinations of inputs
        tspec = tuple(x is not None for x in
                      [sampling_interval, sampling_rate, length, duration])

        # Used in converting tspecs to human readable form
        tspec_arg_names = ['sampling_interval',
                           'sampling_rate',
                           'length',
                           'duration']

        # The valid configurations
        valid_tspecs = [
            # interval, length:
            (True, False, True, False),
            # interval, duration:
            (True, False, False, True),
            # rate, length:
            (False, True, True, False),
            # rate, duration:
            (False, True, False, True),
            # length, duration:
            (False, False, True, True)
            ]

        if isinstance(data, UniformTime):
            # Assuming data was given, some other tspecs become valid:
            tspecs_w_data = dict(
                    nothing=(False, False, False, False),
                    sampling_interval=(True, False, False, False),
                    sampling_rate=(False, True, False, False),
                    length=(False, False, True, False),
                    duration=(False, False, False, True))
            # preserve the order of the keys
            valid_tspecs.append(tspecs_w_data['nothing'])
            for name in tspec_arg_names:
                valid_tspecs.append(tspecs_w_data[name])

        if (tspec not in valid_tspecs):
            # l = ['sampling_interval', 'sampling_rate', 'length', 'duration']
            # args = [arg for t,arg in zip(tspec,l) if t]
            raise ValueError("Invalid time specification.\n" +
                "You provided: %s \n"
                "%s \nsee docstring for more info."
                % (str_tspec(tspec, tspec_arg_names),
                  str_valid_tspecs(valid_tspecs,
                                   tspec_arg_names)))

        if isinstance(data, UniformTime):
            # Get attributes from the UniformTime object and transfer those
            # over:
            if tspec == tspecs_w_data['nothing']:
                sampling_rate = data.sampling_rate
                duration = data.duration
            elif tspec == tspecs_w_data['sampling_interval']:
                duration == data.duration
            elif tspec == tspecs_w_data['sampling_rate']:
                if isinstance(sampling_rate, Frequency):
                    sampling_interval = sampling_rate.to_period()
                else:
                    sampling_interval = 1.0 / sampling_rate
                duration = data.duration
            elif tspec == tspecs_w_data['length']:
                duration = length * data.sampling_interval
                sampling_rate = data.sampling_rate
            elif tspec == tspecs_w_data['duration']:
                sampling_rate = data.sampling_rate
            if time_unit is None:
                # If the user didn't ask to change the time-unit, use the
                # time-unit from the object you got:
                time_unit = data.time_unit

        # Check that the time units provided are sensible:
        if time_unit not in time_unit_conversion:
            raise ValueError('Invalid time unit %s, must be one of %s' %
                         (time_unit, time_unit_conversion.keys()))

        # Make sure you have a time unit:
        if time_unit is None:
            #If you gave us a duration with time_unit attached
            if isinstance(duration, TimeInterface):
                time_unit = duration.time_unit
            #Otherwise, you might have given us a sampling_interval with a
            #time_unit attached:
            elif isinstance(sampling_interval, TimeInterface):
                time_unit = sampling_interval.time_unit
            else:
                time_unit = 's'

        # Calculate the sampling_interval or sampling_rate:
        if sampling_interval is None:
            if isinstance(sampling_rate, Frequency):
                c_f = time_unit_conversion[time_unit]
                sampling_interval = sampling_rate.to_period() / float(c_f)
            elif sampling_rate is None:
                sampling_interval = float(duration) / length
                sampling_rate = Frequency(1.0 / sampling_interval,
                                          time_unit=time_unit)
            else:
                c_f = time_unit_conversion[time_unit]
                sampling_rate = Frequency(sampling_rate, time_unit='s')
                sampling_interval = sampling_rate.to_period() / float(c_f)
        else:
            if isinstance(sampling_interval, TimeInterface):
                c_f = time_unit_conversion[sampling_interval.time_unit]
                sampling_rate = Frequency(1.0 / (float(sampling_interval) /
                                                                       c_f),
                                     time_unit=sampling_interval.time_unit)
            else:
                sampling_rate = Frequency(1.0 / sampling_interval,
                                          time_unit=time_unit)

        # Calculate the duration, if that is not defined:
        if duration is None:
            duration = length * sampling_interval

        # 'cast' the time inputs as TimeArray
        duration = TimeArray(duration, time_unit=time_unit)
        #XXX If data is given - the t0 should be taken from there:
        t0 = TimeArray(t0, time_unit=time_unit)
        sampling_interval = TimeArray(sampling_interval, time_unit=time_unit)

        # in order for time[-1]-time[0]==duration to be true (which it should)
        # add the sampling_interval to the stop value:
        # time = np.arange(np.int64(t0),
        #                  np.int64(t0+duration+sampling_interval),
        #                  np.int64(sampling_interval),dtype=np.int64)

        # But it's unclear whether that's really the behavior we want?
        time = np.arange(np.int64(t0), np.int64(t0 + duration),
                         np.int64(sampling_interval), dtype=np.int64)

        time = np.asarray(time).view(cls)
        time.time_unit = time_unit
        time._conversion_factor = time_unit_conversion[time_unit]
        time.duration = duration
        time.sampling_rate = Frequency(sampling_rate)
        time.sampling_interval = sampling_interval
        time.t0 = t0

        return time

    def __array_wrap__(self, out_arr, context=None):
        # When doing comparisons between UniformTime, make sure that you return
        # a boolean array, not a time array:
        if out_arr.dtype == bool:
            return np.asarray(out_arr)
        else:
            return np.ndarray.__array_wrap__(self, out_arr, context)

    def __array_finalize__(self, obj):
        """XXX """
        # Make sure that the UniformTime has the time units set (and not equal
        # to None):
        if not hasattr(self, 'time_unit') or self.time_unit is None:
            if hasattr(obj, 'time_unit'):  # looks like view cast
                self.time_unit = obj.time_unit
            else:
                self.time_unit = 's'

        # Make sure that the conversion factor is set properly:
        if not hasattr(self, '_conversion_factor'):
            if hasattr(obj, '_conversion_factor'):
                self._conversion_factor = obj._conversion_factor
            else:
                self._conversion_factor = time_unit_conversion[self.time_unit]

        # Make sure that t0 attribute is set properly:
        for attr in ['t0', 'sampling_rate', 'sampling_interval', 'duration']:
            if not hasattr(self, attr) and hasattr(obj, attr):
                setattr(self, attr, getattr(obj, attr))

    def __repr__(self):
        """Pass it through the conversion factor"""

        #If the input is a single int/float (with no shape) return a 'scalar'
        #time-point:
        if self.shape == ():
            return "%r %s" % (int(self) / float(self._conversion_factor),
                            self.time_unit)

        #Otherwise, return the UniformTime representation:
        else:
            return np.ndarray.__repr__(self / float(self._conversion_factor)
             )[:-1] + ", time_unit='%s')" % self.time_unit

    def __getitem__(self, key):
        # return scalar TimeArray in case key is integer
        if isinstance(key, (int, np.int64, np.int32)):
            return self[[key]].reshape(()).view(TimeArray)
        elif isinstance(key, float) or isinstance(key, TimeInterface):
            return self.at(key)
        elif isinstance(key, Epochs):
            return self.during(key)
        else:
            return np.ndarray.__getitem__(self, key)

    def __setitem__(self, key, val):
        raise ValueError("""Setting of individual indices would break uniformity:
            You can either use += on the full array, OR
            create a new TimeArray from this UniformTime""")

    def _convert_and_check_uniformity(self, val):
        # look at the units - convert the values to what they need to be (in
        # the base_unit) and then delegate to the ndarray.__iadd__
        if not hasattr(val, '_conversion_factor'):
            val = np.asarray(val)
            if getattr(val, 'dtype', None) == np.int32:
                # we'll overflow if val's dtype is np.int32
                val = np.array(val, dtype=np.int64)
            val *= self._conversion_factor
        if hasattr(val, 'ndim') and val.ndim == 1:
            # we have to check that adding this will preserve uniformity
            dv = np.diff(val)
            uniformity_breaks, = np.where(dv!=dv[0])
            if len(uniformity_breaks) != 0:
                raise ValueError(
                    """All elements in the operand array must have a constant
                    interval between them in order to preserve uniformity.
                    Uniformity is broken at these indices: %s
                    """ %str(uniformity_breaks))
            self.sampling_interval += dv[0]
            self.sampling_rate = Frequency(1.0 / (float(self.sampling_interval) /
                                        time_unit_conversion[self.time_unit]),
                                        time_unit=self.time_unit)
        return val

    def __iadd__(self, val):
        val = self._convert_and_check_uniformity(val)
        return np.ndarray.__iadd__(self, val)

    def __isub__(self, val):
        val = self._convert_and_check_uniformity(val)
        return np.ndarray.__isub__(self, val)

    def __imul__(self, val):
        np.ndarray.__imul__(self, val)
        self.sampling_interval *= val
        self.sampling_rate = Frequency(self.sampling_rate / val)
        return self

    def __idiv__(self, val):
        np.ndarray.__idiv__(self, val)
        self.sampling_interval /= val
        self.sampling_rate = Frequency(self.sampling_rate * val)
        return self

    __itruediv__ =  __idiv__ # for py3k

    def index_at(self, t, boolean=False):
        """Find the index that corresponds to the time bin containing t

           Returns boolean mask if boolean=True and integer indices otherwise.
        """

        # cast t into time
        ta = TimeArray(t, time_unit=self.time_unit)

        # check that index is within range
        if ta.min() < self.t0 or ta.max() >= self.t0 + self.duration:
            raise ValueError('index out of range')
        idx = (ta - self.t0) // self.sampling_interval
        if boolean:
            bool_idx = np.zeros(len(self), dtype=bool)
            bool_idx[idx] = True
            return bool_idx
        elif ta.ndim == 0:
            return idx[()]
        else:
            return idx.view(np.ndarray)

    def slice_during(self, e):
        """ Returns the slice that corresponds to Epoch e"""

        if not isinstance(e, Epochs):
            raise ValueError('e has to be of Epochs type')

        if e.data.ndim > 0:
            raise NotImplementedError('e has to be a scalar Epoch')

        if self.ndim != 1:
            e_s = 'slicing only implemented for 1-d TimeArrays'
            return NotImplementedError(e_s)
        i_start = self.index_at(e.start)
        i_stop = self.index_at(e.stop)
        if e.start > self[i_start]:  # make sure self[i_start] is in epoch e
            i_start += 1
        if e.stop > self[i_stop]:  # make sure to include self[i_stop]
            i_stop += 1

        return slice(i_start, i_stop)

    def at(self, t):
        """ Returns the values of the UniformTime object at time t"""
        return TimeArray(self[self.index_at(t)], time_unit=self.time_unit)

    def during(self, e):
        """ Returns the values of the UniformTime object during Epoch e"""

        if not isinstance(e, Epochs):
            raise ValueError('e has to be of Epochs type')

        if e.data.ndim > 0:
            raise NotImplementedError('e has to be a scalar Epoch')

        return self[self.slice_during(e)]

    def min(self, axis=None, out=None):
        """Returns the minimal time"""
        # this is a quick fix to return a time and will
        # be obsolete once we use proper time dtypes
        if axis is not None:
            raise NotImplementedError('axis argument not implemented')
        if out is not None:
            raise NotImplementedError('out argument not implemented')
        if self.ndim:
            return self[self.argmin()]
        else:
            return self

    def max(self, axis=None, out=None):
        """Returns the maximal time"""
        # this is a quick fix to return a time and will
        # be obsolete once we use proper time dtypes
        if axis is not None:
            raise NotImplementedError('axis argument not implemented')
        if out is not None:
            raise NotImplementedError('out argument not implemented')
        if self.ndim:
            return self[self.argmax()]
        else:
            return self

    def __div__(self, d):
        """Division by another time object eliminates units """
        if isinstance(d, TimeInterface):
            return np.divide(np.array(self), np.array(d).astype(float))
        else:
            return np.divide(self, d)

    __truediv__ =  __div__ # for py3k

##Frequency:

class Frequency(float):
    """A class for representation of the frequency (in Hz) """

    def __new__(cls, f, time_unit='s'):
        """Initialize a frequency object """

        tuc = time_unit_conversion
        scale_factor = (float(tuc['s']) / tuc[time_unit])
        #If the input is a Frequency object, it is already in Hz:
        if isinstance(f, Frequency) == False:
            #But otherwise convert to Hz:
            f = f * scale_factor

        freq = super(Frequency, cls).__new__(cls, f)
        freq._time_unit = time_unit

        return freq

    def __repr__(self):

        return str(float(self)) + ' Hz'


    def to_period(self, time_unit=base_unit):
        """Convert the value of a frequency to the corresponding period
        (defaulting to a representation in the base_unit)

        """
        tuc = time_unit_conversion
        scale_factor = (float(tuc['s']) / tuc[time_unit])

        return np.int64((1 / self) * scale_factor)


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

    def __init__(self, data, time_unit, metadata=None):
        """Common constructor shared by all TimeSeries classes."""
        # Check that sensible time units were given
        if time_unit not in time_unit_conversion:
            raise ValueError('Invalid time unit %s, must be one of %s' %
                             (time_unit, time_unit_conversion.keys()))

        #: the data is an arbitrary numpy array
        self.data = np.asanyarray(data)
        self.time_unit = time_unit

        # Every instance carries an empty metadata dict, which we promise never
        # to touch.  This reserves this name as a user area for extra
        # information without the danger of name clashes in the future.
        if metadata is None:
            self.metadata = {}
        else:
            self.metadata = metadata

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

    def __getitem__(self, key):
        """use fancy time-indexing (at() method)."""
        if isinstance(key, TimeInterface):
            return self.at(key)
        elif isinstance(key, Epochs):
            return self.during(key)
        elif self.data.ndim == 1:
            return self.data[key]  # time is the last dimension
        else:
            return self.data[..., key]  # time is the last dimension

    def __repr__(self):
        rep = self.__class__.__name__ + ":"
        return rep + self.time.__repr__() + self.data.T.__repr__()

    # add some methods that implement arithmetic on the timeseries data
    def __add__(self, other):
        out = self.copy()
        out.data = out.data.__add__(np.asanyarray(other).T)
        return out

    def __sub__(self, other):
        out = self.copy()
        out.data = out.data.__sub__(np.asanyarray(other).T)
        return out

    def __mul__(self, other):
        out = self.copy()
        out.data = out.data.__mul__(np.asanyarray(other).T)
        return out

    def __div__(self, other):
        out = self.copy()
        out.data = out.data.__truediv__(np.asanyarray(other).T)
        return out

    __truediv__ = __div__  # for py3k

    def __iadd__(self, other):
        self.data.__iadd__(np.asanyarray(other).T)
        return self

    def __isub__(self, other):
        self.data.__isub__(np.asanyarray(other).T)
        return self

    def __imul__(self, other):
        self.data.__imul__(np.asanyarray(other).T)
        return self

    def __idiv__(self, other):
        self.data.__itruediv__(np.asanyarray(other).T)
        return self

    __itruediv__ = __idiv__  # for py3k

class TimeSeries(TimeSeriesBase):
    """Represent data collected at uniform intervals.
    """

    @desc.setattr_on_read
    def time(self):
        """Construct time array for the time-series object. This holds a
    UniformTime object, with properties derived from the TimeSeries
    object"""
        return UniformTime(length=self.__len__(), t0=self.t0,
                           sampling_interval=self.sampling_interval,
                           time_unit=self.time_unit)

    #XXX This should call the constructor in an appropriate way, when provided
    #with a UniformTime object and data, so that you don't need to deal with
    #the constructor itself:
    @staticmethod
    def from_time_and_data(time, data):
        return TimeSeries.__init__(data, time=time)

    def copy(self):
        return TimeSeries(data=self.data.copy(),
                          time=self.time.copy(),
                          time_unit=self.time_unit,
                          metadata=self.metadata.copy())

    def __init__(self, data, t0=None, sampling_interval=None,
                 sampling_rate=None, duration=None, time=None, time_unit='s',
                 metadata=None):
        """Create a new TimeSeries.

        This class assumes that data is uniformly sampled, but you can specify
        the sampling in one of three (mutually exclusive) ways:

        - sampling_interval [, t0]: data sampled starting at t0, equal
          intervals of sampling_interval.

        - sampling_rate [, t0]: data sampled starting at t0, equal intervals of
          width 1/sampling_rate.

        - time: a UniformTime object, in which case the TimeSeries can
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
          sampling_rate/sampling_interval/duration to take the place of the
          one in this object, but only as long as the changes are consistent
          with the length of the data.

        time_unit :  string
          The unit of time.

        Examples
        --------

        The minimal specification of data and sampling interval:

        >>> ts = TimeSeries([1,2,3],sampling_interval=0.25)
        >>> ts.time
        UniformTime([ 0.  ,  0.25,  0.5 ], time_unit='s')
        >>> ts.t0
        0.0 s
        >>> ts.sampling_rate
        4.0 Hz

        Or data and sampling rate:

        >>> ts = TimeSeries([1,2,3],sampling_rate=2)
        >>> ts.time
        UniformTime([ 0. ,  0.5,  1. ], time_unit='s')
        >>> ts.t0
        0.0 s
        >>> ts.sampling_interval
        0.5 s

        A time series where we specify the start time and sampling interval:

        >>> ts = TimeSeries([1,2,3],t0=4.25,sampling_interval=0.5)
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

        >>> ts = TimeSeries([1,2,3],t0=4.25,sampling_rate=2.0)
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

        """

        #If a UniformTime object was provided as input:
        if isinstance(time, UniformTime):
            c_fac = time._conversion_factor
            #If the user did not provide an alternative t0, get that from the
            #input:
            if t0 is None:
                t0 = time.t0
            #If the user did not provide an alternative sampling interval/rate:
            if sampling_interval is None and sampling_rate is None:
                sampling_interval = time.sampling_interval
                sampling_rate = time.sampling_rate
            #The duration can be read either from the length of the data, or
            #from the duration specified by the time-series:
            if duration is None:
                duration = time.duration
                length = time.shape[-1]
                #If changing the duration requires a change to the
                #sampling_rate, make sure that this was explicitly required by
                #the user - if the user did not explicitly set the
                #sampling_rate, or it is inconsistent, throw an error:
                data_len = np.array(data).shape[-1]

                if (length != data_len and
                    sampling_rate != float(data_len * c_fac) / time.duration):
                    e_s = "Length of the data (%s) " % str(len(data))
                    e_s += "specified sampling_rate (%s) " % str(sampling_rate)
                    e_s += "do not match."
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
                      [sampling_interval, sampling_rate, duration])

            tspec_arg_names = ["sampling_interval",
                               "sampling_rate",
                               "duration"]

            #The valid configurations
            valid_tspecs = [
                      #interval, length:
                      (True, False, False),
                      #interval, duration:
                      (True, False, True),
                      #rate, length:
                      (False, True, False),
                      #rate, duration:
                      (False, True, True),
                      #length, duration:
                      (False, False, True)
                      ]

            if tspec not in valid_tspecs:
                raise ValueError("Invalid time specification. \n"
                      "You provided: %s\n %s see docstring for more info." % (
                            str_tspec(tspec, tspec_arg_names),
                            str_valid_tspecs(valid_tspecs, tspec_arg_names)))

        # Make sure to grab the time unit from the inputs, if it is provided:
        if time_unit is None:
            # If you gave us a duration with time_unit attached
            if isinstance(duration, TimeInterface):
                time_unit = duration.time_unit
            # Otherwise, you might have given us a sampling_interval with a
            # time_unit attached:
            elif isinstance(sampling_interval, TimeInterface):
                time_unit = sampling_interval.time_unit

        # Calculate the sampling_interval or sampling_rate from each other and
        # assign t0, if it is not already assigned:
        if sampling_interval is None:
            if isinstance(sampling_rate, Frequency):
                c_f = time_unit_conversion[time_unit]
                sampling_interval = sampling_rate.to_period() / float(c_f)
            elif sampling_rate is None:
                data_len = np.asarray(data).shape[-1]
                sampling_interval = float(duration) / data_len
                sampling_rate = Frequency(1.0 / sampling_interval,
                                             time_unit=time_unit)
            else:
                c_f = time_unit_conversion[time_unit]
                sampling_rate = Frequency(sampling_rate, time_unit='s')
                sampling_interval = sampling_rate.to_period() / float(c_f)
        else:
            if sampling_rate is None:  # Only if you didn't already 'inherit'
                                       # this property from another time object
                                       # above:
                if isinstance(sampling_interval, TimeInterface):
                    c_f = time_unit_conversion[sampling_interval.time_unit]
                    sampling_rate = Frequency(1.0 / (float(sampling_interval) /
                                                                         c_f),
                                       time_unit=sampling_interval.time_unit)
                else:
                    sampling_rate = Frequency(1.0 / sampling_interval,
                                              time_unit=time_unit)

        #Calculate the duration, if that is not defined:
        if duration is None:
            duration = np.asarray(data).shape[-1] * sampling_interval

        if t0 is None:
            t0 = 0

        # Make sure to grab the time unit from the inputs, if it is provided:
        if time_unit is None:
            #If you gave us a duration with time_unit attached
            if isinstance(duration, TimeInterface):
                time_unit = duration.time_unit
            #Otherwise, you might have given us a sampling_interval with a
            #time_unit attached:
            elif isinstance(sampling_interval, TimeInterface):
                time_unit = sampling_interval.time_unit

        #Otherwise, you can still call the common constructor to get the real
        #object initialized, with time_unit set to None and that will generate
        #the object with time_unit set to 's':
        TimeSeriesBase.__init__(self, data, time_unit, metadata=metadata)

        self.time_unit = time_unit
        self.sampling_interval = TimeArray(sampling_interval,
                                           time_unit=self.time_unit)
        self.t0 = TimeArray(t0, time_unit=self.time_unit)
        self.sampling_rate = sampling_rate
        self.duration = TimeArray(duration, time_unit=self.time_unit)

    def at(self, t, tol=None):
        """ Returns the values of the TimeArray object at time t"""
        return self.data[..., self.time.index_at(t)]

    def during(self, e):
        """ Returns the TimeSeries slice corresponding to epoch e """

        if not isinstance(e, Epochs):
            raise ValueError('e has to be of Epochs type')

        if e.data.ndim == 0:
            return TimeSeries(data=self.data[..., self.time.slice_during(e)],
                              time_unit=self.time_unit, t0=e.offset,
                              sampling_rate=self.sampling_rate)
        else:
            # TODO: make this a more efficient implementation, naive first pass
            if (e.duration != e.duration[0]).any():
                raise ValueError("All epochs must have the same duration")

            data = np.array([self.data[..., self.time.slice_during(ep)]
                             for ep in e])

            return TimeSeries(data=data,
                              time_unit=self.time_unit, t0=e.offset,
                              sampling_rate=self.sampling_rate)

    @property
    def shape(self):
        return self.data.shape


_epochtype = np.dtype({'names': ['start', 'stop'], 'formats': [np.int64] * 2})


class Epochs(desc.ResetMixin):
    """Represents a time interval"""
    def __init__(self, t0=None, stop=None, offset=None, start=None,
                 duration=None, time_unit=None, static=None, **kwargs):
        """
        Parameters
        ----------
        t0 : 1-d array or `TimeArray`
           A time relative to which the epochs started. Per default `t0` and
          `start` are the same, but setting the `offset` parameter can adjust
           that, so that the start-times are at a fixed time, relative to t0.

        stop : 1-d array or `TimeArray`
              The times of ends of epochs

        offset : float, int or singleton `TimeArray`
            A constant offset applied to t0 to set the starts of Epochs

        start : 1-d array or `TimeArray`
              The times of beginnings of epochs

        duration : 1-d array or `TimeArray`
           The durations of intervals.

        time_unit : str, optional
              The time unit of the object and all time-related things in it.
              Default: 's'

        static : dict, optional
            For fast initialization of an `Epochs` object from another `Epochs`
            object, this dict should contain all necessary items to have an
            `Epoch` defined.

        """
        # Short-circuit path for a fast initialization. This relies on `static`
        # to be a dict that contains everything that defines an Epochs class
        # XXX: add this sort of fast __init__ to all other classes
        if static is not None:
            self.__dict__.update(static)
            # we have to reset the duration OneTimeProperty, since it refers
            # to computations performed on the former object
            self.reset()
            return

        if t0 is None and start is None:
            raise ValueError('Either start or t0 need to be specified')
        # Normal, error checking and type converting initialization logic

        if stop is None and duration is None:
            raise ValueError('Either stop or duration have to be specified')

        if stop is not None and duration is not None:
            ### TODO: check if stop and duration are consistent
            e_s = 'Only either stop or duration have to be specified'
            raise ValueError(e_s)

        if offset is None:
            offset = 0

        t_offset = TimeArray(offset, time_unit=time_unit)

        if t_offset.ndim > 0:
            raise ValueError('Only scalar offset allowed')

        if t0 is None:
            t_0 = 0
        else:
            t_0 = TimeArray(t0, time_unit=time_unit)

        if start is None:
            t_start = t_0 - t_offset
        else:
            t_start = TimeArray(start, time_unit=time_unit)

        # inherit time_unit of t_start
        self.time_unit = t_start.time_unit

        if stop is None:
            t_duration = TimeArray(duration, time_unit=time_unit)
            t_stop = t_start + t_duration
        else:
            t_stop = TimeArray(stop, time_unit=time_unit)

        if t_start.shape != t_stop.shape:
            raise ValueError('start and stop have to have same shape')

        if t_start.ndim == 0:
            # return a 'scalar' epoch
            self.data = np.empty(1, dtype=_epochtype).reshape(())
        elif t_start.ndim == 1:
            # return a 1-d epoch array
            self.data = np.empty(t_start.shape[0], dtype=_epochtype)
        else:
            e_s = 'Only 0-dim and 1-dim start and stop times allowed'
            raise ValueError(e_s)

        self.data['start'] = t_start
        self.data['stop'] = t_stop

        self.offset = t_offset

    # TODO: define setters for start, stop, offset attributes
    @property
    def start(self):
        return TimeArray(self.data['start'],
                         time_unit=self.time_unit,
                         copy=False)

    @property
    def stop(self):
        return TimeArray(self.data['stop'],
                         time_unit=self.time_unit,
                         copy=False)

    @desc.setattr_on_read
    def duration(self):
        """Duration array for the epoch"""
        return self.stop - self.start

    def __getitem__(self, key):
        # create the static dict needed for fast version of __init__
        static = self.__dict__.copy()
        static['data'] = self.data[key]
        # self.__class__ here is Epochs or a subclass of Epochs
        # and `start` is a required argument
        return self.__class__(start=None, static=static)

    def __repr__(self):
        if self.data.ndim == 0:
            z = (self.start, self.stop)
        else:
            z = list(zip(self.start, self.stop))
        rep = self.__class__.__name__ + "(" + z.__repr__()
        return rep + ", as (start,stop) tuples)"

    def __len__(self):
        return len(self.data)


def str_tspec(tspec, arg_names):
    """ Turn a single tspec into human readable form"""
    # an all "False" will convert to an empty string unless we do the following
    # where we create an all False tuple of the appropriate length
    if tspec == tuple([False] * len(arg_names)):
        return "(nothing)"
    return ", ".join([arg for t, arg in zip(tspec, arg_names) if t])


def str_valid_tspecs(valid_tspecs, arg_names):
    """Given a set of valid_tspecs, return a string that turns them into
    human-readable form"""
    vargs = []
    for tsp in valid_tspecs:
        vargs.append(str_tspec(tsp, arg_names))
    return "\n Valid time specifications are:\n\t%s" % ("\n\t".join(vargs))


def concatenate_time_series(time_series_seq):
    """Concatenates a sequence of time-series objects in time.

    The input can be any iterable of time-series objects; metadata, sampling
    rates and other attributes are kept from the last one in the sequence.

    This one requires that all the time-series in the list have the same
    sampling rate and that all the data have the same number of items in all
    dimensions, except the time dimension"""

    # Extract the data pointer for each and build a common data block
    data = []
    metadata = {}
    for ts in time_series_seq:
        data.append(ts.data)
        metadata.update(ts.metadata)

    # Sampling interval is read from the last one
    tseries = TimeSeries(np.concatenate(data,-1),
                                sampling_interval=ts.sampling_interval,
                                metadata=metadata)
    return tseries


class Events(TimeInterface):
    """Represents timestamps and associated data """

    def __init__(self, time, labels=None, indices=None,
                 time_unit=None, **data):
        """
        Parameters
        ----------
        time : array or TimeArray
            The times at which events occurred

        labels : array, optional

        indices : int array, optional


        Notes
        -----


        """
        # The time data must be at least a 1-d array, NOT a time scalar
        if not np.iterable(time):
            time = [time]

        # First initilaize the TimeArray from the time-stamps
        self.time = TimeArray(time, time_unit=time_unit)
        self.time_unit = self.time.time_unit

        # Make sure time is one-dimensional
        if self.time.ndim != 1:
            e_s = 'The TimeArray provided can only be one-dimensional'
            raise ValueError(e_s)
        # Ensure that the dict of data values has a known, uniform structure:
        # all values must be arrays, with at least one dimension.
        new_data = {}
        for k, v in data.items():
            if np.iterable(v):
                v = np.asanyarray(v)
            else:
                # For scalars, we do NOT want to create 0-d arrays, which are
                # rather tricky to work with.  So if the input value is not an
                # iterable object, we turn it into a one-element 1-d array.
                v = np.array([v])
            new_data[k] = v

        # Make sure all data has same length
        ntimepts = len(self.time)
        for check_v in new_data.values():
            if len(check_v) != ntimepts:
                e_s = 'All data in the Events must be of the same'
                e_s += 'length as the associated time'
                raise ValueError(e_s)

        # Make sure indices have same length and are integers
        if labels is not None:
            if len(labels) != len(indices):
                e_s = 'Labels and indices must have the same length'
                raise ValueError(e_s)
            dt = [(l, np.int64) for l in labels]
        else:
            dt = np.int64
            dt = [('i%d' % i, np.int64)
                  for i in range(len(indices or ()))] or np.int64

        self.index = np.array(list(zip(*(indices or ()))),
                                       dtype=dt).view(np.recarray)

        #Should data be a recarray?
##         dt = [(st,np.array(data[st]).dtype) for st in data] or None
##         self.data = np.array(zip(*data.values()),
##         dtype=dt).view(np.recarray)

        #Or a dict?
        self.data = new_data

    def __repr__(self):
        rep = self.__class__.__name__ + ":\n\t"
        rep += repr(self.time) + "\n\t"
        rep += repr(self.data)
        return rep

    def __getitem__(self, key):
        # return scalar TimeArray in case key is integer
        newdata = dict()
        newtime = self.time[key].reshape(-1)
        sl = key
        if isinstance(key, float):
            sl = self.time.index_at(key)
        elif isinstance(key, Epochs):
            sl = self.time.slice_during(key)
        for k, v in self.data.items():
            newdata[k] = v[sl]

        # XXX: I don't really understand how labels and index are supposed to
        # be used, so I'm not implementing them when slicing events - pi
        # 2010-12-04

        # self.__class__ here is Events or a subclass of Events
        return self.__class__(newtime, **newdata)

    def __len__(self):
        return len(self.time)
