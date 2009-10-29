.. _base_classes:

==============
 Base classes
==============

Time
====

The first set of base classes is a set of representations of time. All these
classes inherit from :class:`np.array` with the dtype limited to be
:class:`datetime64`.

These representation will all serve as the underlying machinery to index into
the :class:`TimeSeries` objects with arrays of time-points.  The additional
functionality common to all of these is described in detail in
:ref:`time_series_access`. Briefly, they will all have a :func:`at` method,
which allows indexing with arrays of :class:`datetime64`. The result of this
indexing will be to return the time-point in the the respective which is most
appropriate (see :ref:`time_series_access` for details). They will also all
have a :func:`index_at` method, which returns the integer index of this time in
the underlying array. Finally, they will all have a :func:`during` method,
which will allow indexing into these objects with an interval bject. This will
return the appropriate times corresponding to an :ref:`interval_object` and
:func:`index_during`, which will return the array of integers corresponding to
the indices of these time-points in the array.

.. _EventArray:

:class:`EventArray`
-------------------

This class has the least restrictions on it: it will be a 1d array, which
contains time-points that are not neccesarily ordered. It can also contain
several copies of the same time-point. This class will be used in
order to represent sparsely occuring events, measured at some unspecified
sampling rate and possibly collected from several different channels, where the
data is sampled in order of channel and not in order of time.

.. _NonUniformTime:

:class:`NonUniformTime`
-------------------------

This class can be used in order to represent time with a varying sampling rate,
or also represent contains time-points that 

.. _UniformTime:

:class:`UniformTime`
--------------------

This class contains ordered time-points. In addition, this class has an
explicit representation of :attribute:`t_0`, :attribute:`sampling_rate` and
:attribute:`sampling_interval` (the latter two implemented as
:method:`setattr_on_read`, which can be computed from each other).Thus, each
element in this array can be used in order to represent the entire time
interval $t$, such that: $t_i\leq t < t + \delta t$, where $t_i$ is the nominal
value held by that element of the array, and $\delta t$ is the
value of :attribute:`sampling_interval`.



We have two basic classes for time-series data, the current *TimeSeries* which
we can think of (uniformly or non-uniformly) sampled data, and *Event* (see
:ref:`event_class`). The time dimension of a *TimeSeries*
object can be thought of as (uniformly or non-uniformly sampled) continuous
stretch of time (or time interval), whereas the time dimension of an *Event* is
a list of discrete time stamps. Corresponding to these two cases, we introduce
two time classes: *TimeBin* and *TimePoint*.

Both *TimeBin* and *TimePoint* are represented by a 1-dimensional nd-array of
dtype time, but they allow indexing with time variables instead of usual integer
indices or slices. Both *TimeBin* and *TimePoint*  can be implemented as either
a thin wrapper around nd-array or even as a nd-array sub-class.

We have the following principles:

* All data classes have a time dimension represented by the *time* attribute.
* All time classes can be used to index into each data class.

We want to limit ourselves to as few as possible classes (but not fewer :-). In
particular, the time classes have to be specific enough to be able to do
specify all common use cases to index into data by time, and the data classes
have to be specific enough so that we know what to expect when we index into
them.


Time classes
============

There are two fundamental time classes, *TimeBin* corresponding to contiguous
time intervals, broken up into smaller bins, and *TimePoint* corresponding
to discrete time points. The corresponding elementary scalar types are just
dtype *datetime64* scalars. Additionally, we will need time-intervals which are
not necessarily contiguous, the *Interval* (array) class with scalars of a
new dtype *timeinterval*.


+-------------+------------+-------------------+-----------------+-----------------+------------------+
|             | class      | subclass          | scalar type     | contiguous data | uniform sampling |
+=============+============+===================+=================+=================+==================+
|             | *TimeBin*  | *UniformTimeBin*  | *datetime64*    |      X          |        X         |
|             |            +-------------------+-----------------+-----------------+------------------+
| time        |            |                   | *datetime64*    |      X          |                  |
| intervals   +------------+-------------------+-----------------+-----------------+------------------+
|             | *Interval*                     | *timeinterval*  |                 |                  |
+-------------+--------------------------------+-----------------+-----------------+------------------+
| time points | *TimePoint*                    | *datetime64*    |                 |                  |
+-------------+--------------------------------+-----------------+-----------------+------------------+


time-point (scalar)
-------------------

A time-point is a scalar number of dtype *datetime64*. It can be used to either
represent a time bin (in the *TimeBin* array class) or to represent a
time-point in the *TimePoint* class.


time-interval (scalar)
----------------------

Do we want to implement an new dtype for this?

Even though there is a dtype *timedelta64*, this does not all the information
we would like to associate with a time interval. In particular, a time interval
should at least specify a start time t_start and a stop time t_stop, and
potentially an additional attribute t_offset (see
:ref:`interval_object`).


TimeBin (array)
---------------

This class represents the time axis of *TimeSeries* data. It is essentially a
one-dimensional nd-array of dtype *datetime64* and each element corresponds to
the left edge of a time bin.  *TimeBin* allows special indexing using time
objects (see :ref:`time_series_access`).

TimePoint (array)
-----------------

Maybe *TimeStamp* would be a better name?

This class is essentially a nd-array of dtype *datetime64* representing
discrete time points (Think of spike trains, for example) and each element
corresponds to the left edge of a time bin.

This class represents the time axis of *Event* data.  *TimePoint* allows
special indexing using time objects (see
:ref:`time_series_access`).


Interval (array)
----------------

This class corresponds to a list of time intervals which don't have to be
contiguous and can even be overlapping. Each element is a *timeinterval* (see
:ref:`interval_object`) and can be used to index into all of the
time-series classes.


Data classes
============

+---------------+---------------------+------------------+-----------------+------------------+
| class         | Subclass            | contains         | contiguous data | uniform sampling |
+===============+=====================+==================+=================+==================+
| *TimeSeries*  | *UniformTimeSeries* | *UniformTimeBin* |      X          |        X         |
|               +---------------------+------------------+-----------------+------------------+
|               |                     | *TimeBin*        |      X          |                  |
+---------------+---------------------+------------------+-----------------+------------------+
| *Event*       |                     | *TimePoint*      |                 |                  |
+---------------+---------------------+------------------+-----------------+------------------+

