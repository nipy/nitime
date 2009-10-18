=================
 Interval object
=================

Intervals are very useful! Every time-series data object has a time dimension
and we will use *Interval* objects to select (slice) subsets of the data.

I think, we would like to have a scalar and an array version of this object
and I am not sure if we should implement a new *timeinterval* dtype, or if we
just use 0-dimensional *Interval* arrays whenever we need single intervals.

A related question is the naming scheme: We could call the array *Interval*
and the scalar *timeinterval* similar to the lower case numpy dtypes. Or, we
just call both objects *Interval*. A

The *Interval* object ti has a method ti.asslice(to) which returns a slice
object that can be used to slice into the data attribute to.data of a
time-series data object to of class TC ($\in$ \{*TimeSeries*, *Event*\}:

  to[ti] = to.at(ti) = to.at(ti.indices(0)) = to.at((t_start,t_stop,t_offset))
  = TC(data=to.data[...,to.asindex(t_start):to.asindex(t_stop)],t0=-t_offset)


Attributes
----------

An *Interval* can be thought of as a slice for time-series objects and
therefore should at least implement all attributes and methods of the slice
object (sub-classing of slice doesn't seem to be possible, see below).

In particular, a time *Interval* ti has the attributes
* ti.start: the start time of the interval, t_start
* ti.stop: the end time of the interval, t_stop
* ti.step: We thought originally that this could be abused to represent a time
  offset, t_offset. This attribute is used by the indices method to convert
  the slice into a list of points.
* ti.indices(len): this method returns a tuple of time points that can be used
  for slicing. Originally, this is meant to produce a list of indices of
  length len that can be directly used to obtain a slice of the same
  length. However, when we use an *Interval* for slicing, we don't know yet,
  how long the sliced object will be (it depends on the sampling interval of
  the sliced object). If we just use len=0, the indices method just returns a
  3-tuple that still contains all necessary information and can be used for
  slicing:

  ti.indices(0) = (ti.start, ti.stop, ti.step)

* ti.mro (?): a type's method resolution order (from the slice doc string)


Initialization
--------------

There are various different ways to initialize a (scalar) time *Interval*:
* with two time points t_start and t_stop, both of dtype *datetime64*:

  Interval(t_start,t_stop)

* with a time point t_start (dtype *datetime64*) and a duration (dtype
  *timedelta64*:
  
  Interval(t_start, duration) = Interval(t_start, t_start+duration)
  
* with an optional third argument t_offset (dtype *timedelta64*) indicating a
  time offset of a time point $t0$ relative to which the time inside the
  interval should be interpreted. The relevance of this third argument will
  become relevant when the time interval is used to slice into a time-series
  object (see below).

  Interval(t_start, t_stop, t_offset) or Interval(t_start, duration, t_offset)

* with two floating point numbers, which will be interpreted as time points
  t_start and t_stop. This convention would be convenient, however, it is not
  clear what time unit would be used for conversion. Maybe, we could have a
  module-level setting of the base unit? or just decide to make this [s]?

* with three floating point numbers, which will be interpreted as t_start,
  duration, and t_offset. This is a convenient way to make initialization with
  duration accessible without having to convert floating point numbers to
  *timedelta64* values.
  

Implementation using a slice object
-----------------------------------

Sub-classing of the slice object doesn't seem to be possible:

  >>> class myslice(slice):
  ...     pass
  ... 
  ------------------------------------------------------------
  Traceback (most recent call last):
    File "<ipython console>", line 1, in <module>
  TypeError: Error when calling the metaclass bases
      type 'slice' is not an acceptable base type

However, it seems that a (scalar) *Interval* can be implemented using a slice
object, provided the time points t_start and t_end and the timedelta t_offset
implement an __index__ method:

  >>> s = slice('t_start','t_stop','t_offset')
  >>> s.start
  't_start'
  >>> s.stop
  't_stop'
  >>> s.step
  't_offset'
  >>> s.indices(1)
  ------------------------------------------------------------
  Traceback (most recent call last):
    File "<ipython console>", line 1, in <module>
  TypeError: slice indices must be integers or None or have an __index__ method

Alternatively, the *Interval* can be implemented as an original object with
the default constructor as similar as possible to the constructor of the slice
object, so that we can use slice-like operations, but still maintain slice
algebra and such.

In addition to the possibility of algebraic operations, there are other
reasons to have the *Interval* be an original class that holds a slice object
that can be returned by the method ti.asslice():


Interval arrays
---------------

In addition to scalar *Interval* objects, it also makes sense to define
arrays of *Interval* objects. These arrays can be implemented as n-dimensional
object arrays where the elements are scalar *Interval* objects. Maybe, we
should even define a new *Interval* dtype. The *timedelta64* dtype is not
sufficient since it does not contain the information about both t_start and
duration (and the t_offset).

