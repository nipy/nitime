.. _interval_class:

=================
 Interval object
=================

Intervals can carry particular special meaning in the analysis of
time-series. For example, a typical case, is when two time-series are recorded
simultaneously. One is recorded from measurement of some continuous
physilogical variable, such as fMRI BOLD (and is represented by a an object of
type :ref:`UniformTimeSeries`). The other is a series of discrete events
occuring concurrently (and can be represented by a :ref:`EventSeries` or by a
:ref:`NonUniformTimeSeries`). For example, button presses by the subject, or
trials of different kinds. If we want to analyze the progression of the
physiological time-series, locked to particular kinds of events in the
event-series, we would need a notion of an interval of time surrounding those
events.


In order to do that, we propose implementing a :class:`TimeInterval`
object.

.. _interval_attributes:

Attributes
----------

A :class:`TimeInterval` object can be thought of as a :class:`Slice` for
time-series objects and therefore should at least implement all attributes and
methods of the slice object (sub-classing of slice doesn't seem to be possible,
see :ref:`interval_from_slice`).

In particular, an object of class :class:`TimeInterval`, :attribute:`TI`, has
the attributes:

* :attribute:`TI.t_0`: the start time of the interval.
* :attribute:`TI.t_end`: the end time of the interval.
* :attribute:`TI.duration`: the duration of the interval.

Obviously, knowledge of two of these, should allow calculation of the
third. Therefore, this should be implemented in the object with a
:func:`setattr_on_read` decoration and the object should inherit
:class:`ResetMixin`.  Initialization of the object would
verify that enough information exists and that the information provided is
consistent, in the same manner that is already implemented in
:class:`UniformTimeSeries`.  

* :attribute:`TI.step`: originally, we thought that this could be abused to
  represent a time offset, relative to the attributes :attribute:`t_0` and
  :attribute:`t_end`. That is, it can tell us where relative to these two
  time-points some interesting even, which this interval surrounds, or this
  interval is close to, occurs. 

* ti.indices(len): this method returns a tuple of time points that can be used
  for slicing. Originally, this is meant to produce a list of indices of
  length len that can be directly used to obtain a slice of the same
  length. However, when we use an *Interval* for slicing, we don't know yet,
  how long the sliced object will be (it depends on the sampling interval of
  the sliced object). If we just use len=0, the indices method just returns a
  3-tuple that still contains all necessary information and can be used for
  slicing:

  ti.indices(0) = (ti.start, ti.stop, ti.step)

.. _interval_initialization:

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
  

.. _interval_from_slice:
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



Comment: :class:`timedelta64`
-----------------------------

The name of the dtype :class:`timedelta64` sounds like this would be a
representation of time intervals. However, this name is somewhat confusing in
this context, as this dtype does not cover this kind of functionality. In
particular, :class:`timedelta64`, is simply a representation of relative time
and is likely to be the kind of time we would want in order to represent time
(see :ref:`time_classes`).


Interval (array)
----------------


This class corresponds to a list of time intervals which don't have to be
contiguous and can even be overlapping. Each element is a *timeinterval* (see
:ref:`interval_object`) and can be used to index into all of the
time-series classes.
--





Every time-series data object has a time dimension and we will use *Interval*
objects to select (slice) subsets of the data.

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



