.. _base_classes:

==============
 Base classes
==============

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

