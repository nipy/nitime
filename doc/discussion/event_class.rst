.. _event_class:

===============
 Events object
===============

We can think of a time-series as an n-dimensional array in which one dimension
can naturally interpreted as time.

There are two fundamentally different types of time-series: *sampled data* and
*events*. The first case corresponds to the current *TimeSeries* object whereas
the the second case would be better represented by a new *events* object.


Sampled data
============

Think of a movie: The time attribute is a list of refresh times, which really
represent the whole frame interval, and the data attribute is the
two-dimensional array of pixels that is present on the screen during the frame
interval.

time
----

In the case of sampled data, the time dimension can be thought of as a list of
time bins, represented by its starting time and of a certain length. In the
case of a uniformly sampled data, all time bins have the same length called
the sampling interval (see *UniformTimeSeries*), whereas in the general case,
the time bins can have varying length. However, in both cases should the time
axis be thought of representing a *continuous* stretch of time, broken up in
short time intervals (*time-bins*).

data
----

Correspondingly, the data of a *TimeSeries* object, can be thought of
n-dimensional data recorded during a continuous stretch of time and represented
by n-dimensional data points, one for each time bin. As discussed in more
detail in :ref:`time_series_access`, when we index sampled data with a single
time point $t$, we would like to obtain the data corresponding to the time bin
that contains this time point (which is represented by the largest $t_i$ with
$t_i<=t$.



Events
======

The second type of time-series data is the case of individual events happening
at discrete points in time. Think of a series of saccadic eye movements
recorded during the movie. These events can also have data associated to them,
such as the location point at the beginning (or the end) of the eye movement,
or the movie frame present at the screen during the eye movement.

time
----

The time dimension of events is a list of time points (*TimePoint*) rather
than a continuous time axis.

data
----

Correspondingly, the data can be thought of as a list of n-dimensional data
points recorded at the given time-points. When we index events by a single
time point $t$, we expect to obtain the data that was recorded at that point
in time, or at a point in time closest to $t$ (see
:ref:`time_series_access`)


Time points
===========

*TimePoint* can be thought of as a special case of the *Event* class, namely
the case in which the data array is empty. On the other hand, there are good
arguments to introduce a separate base class *TimePoint* and have the time
attribute of the *Event* class be an instance of *TimePoint*.

Consider the following use case: During a movie, time points are recorded at
which the eyes moved or at which a button was pressed. Subsequently, we might
want to assign data to these time points, such as the movie frame present at
the screen during the eye movement or during the button press. We can do this
by indexing with the *TimePoint* objects into the movie *TimeSeries* (see
:ref:`time_series_access`).

The distinction between *TimePoint* and *Event* is important, because it makes
sense to use *TimePoint* to index into other time-series, but it does not
really make sense to index using *Event* (since it is not clear what to do
with the associated data). In general, we have to distinguish between time
base classes and data base classes which have a time dimension, see
:ref:`base_class__discussion.rst`.
