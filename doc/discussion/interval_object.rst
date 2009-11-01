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
the attributes/functions:

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

* :attribute:`TI.t_step`: originally, we thought that this could be abused to
  represent a time offset, relative to the attributes :attribute:`t_0` and
  :attribute:`t_end`. That is, it can tell us where relative to these two
  time-points some interesting even, which this interval surrounds, or this
  interval is close to, occurs. This can be used in order to interpert how
  time-series access is done using the :calls:`TimeInterval` object. See
  :ref:`time_series_access`. This attribute can be implemented as an optional
  input on initialization, such that it defaults to be equal to
  :attribute:`t0`.

* :func:`TI.indices(len)`: this method returns a tuple of time points that can
  be used for slicing. Originally, this is meant to produce a list of indices
  of length len that can be directly used to obtain a slice of the same
  length. However, when we use a :class:`TimeInterval` for slicing, we don't
  know yet, how long the sliced object will be (it depends on the sampling
  interval of the sliced object). If we just use len=0, the indices method just
  returns a 3-tuple that still contains all necessary information and can be
  used for slicing::
       
       >>>TI.indices(0)
       (TI.t0 TI.t_end, TI.t_step)

.. _interval_initialization:

Initialization
--------------

There are various different ways to initialize a :class:`TimeInterval`:

* With two time points t_start and t_stop, both of dtype :class:`deltatime64`::

       TimeInterval(t0=t1,t_end=t2)

* With a time point :attribute:`t0` (dtype :class:`deltatime64`) and a duration
  (dtype :class:`timedelta64`)::

       TimeInterval(t0=t1,duration=t2) 
 
* With an optional third argument :attribute:`step` (dtype
  :class:`timedelta64`) indicating a time offset of a time point $t_1=t_0-step$
  relative to which the time inside the interval should be interpreted. The
  relevance of this third argument will become relevant when the time interval
  is used to slice into a time-series object (see
  :ref:`time_series_access`). If not provided, this would default to be equal
  to t0::
  
  TimeInterval(t0=t1, t_end=t2, t_step=delta_t) or TimeInterval (t0=t1,
  duration=delta_t1, t_step=delta_t2)
  
* With two floating point numbers, which will be interpreted as time points
  :attribute:`t0` and :attribute:`t_end`. This convention would be convenient,
  however, it is not clear what time unit would be used for conversion. Maybe,
  we could have a module-level setting of the base unit? or just decide to make
  this 's'?

* With three floating point numbers, which will be interpreted as
  :attribute:`t0`, :attribute:`t_end`, and :attribute:`t_step`. This is a
  convenient way to make initialization with duration accessible without having
  to convert floating point numbers to :class:`timedelta64` values.
  

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

However, it seems that a (scalar) :class:`TimeInterval` can be implemented
using a slice object, provided the time points :attribute:`t0` and
:attribute:`t_end` and the time offset :attribute:`t_step` implement an __index__
method:

  >>> s = slice('t0','t_end','t_step')
  >>> s.start
  't0'
  >>> s.stop
  't_end'
  >>> s.step
  't_step'
  >>> s.indices(1)
  ------------------------------------------------------------
  Traceback (most recent call last):
    File "<ipython console>", line 1, in <module>
  TypeError: slice indices must be integers or None or have an __index__ method

Alternatively, the :class:`TimeInterval` can be implemented as an original
object with the default constructor as similar as possible to the constructor
of the slice object, so that we can use slice-like operations, but still
maintain slice algebra and such.

In addition to the possibility of algebraic operations, there are other reasons
to have the :class:`TimeInterval` be an original class that holds a slice
object that can be returned by the method :func:`TI.asslice()`.

.. _interval_arrays:

Interval arrays
---------------

In addition to scalar :class:`TimeInterval` objects, it also makes sense to
define arrays of :class:`TimeInterval` objects. These arrays can be implemented
as :class:`np.ndarray`, with an :class:`object` dtype. 

.. _comment_timedelta64:

Comment: :class:`timedelta64`
-----------------------------

The name of the dtype :class:`timedelta64` sounds like this would be a
representation of time intervals. However, this name is somewhat confusing in
this context, as this dtype does not cover this kind of functionality. In
particular, :class:`timedelta64`, is simply a representation of relative time
and is likely to be the kind of time we would want in order to represent time
(see :ref:`time_classes`).
