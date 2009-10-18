====================
 Time-series access
====================

Since one dimension of time-series data is associated with time, there is a
natural way to index into time-series data using time objects. The result of
an indexing operation depends on what kind of time objects is used: We have
scalar and array-like time objects, and we have time-objects representing time
points and time intervals (see `ref`:base_class_discussion.rst:). Furthermore,
we have two fundamentally different data types: *TimeSeries*, which is data
sampled during a continuous stretch of time, and *Events*, which is data
sampled at discrete time points (see `ref`:events_class_discussion.rst:).

We implement the indexing operation using the method `func`:ts.at: with the
plan to later map the method `func`:ts.__getitem__: to the function
`func`:ts.at:. However, using the function `func`:ts.at: directly is more
flexible since it allows to use additional keyword arguments.

Before we discuss all the different possible indexing operations, we first
establish as set of guiding principles for time-series access:

* Every time-series data object has a *data* attribute which is an instance of
  an nd-array array and a *time* attribute which is an instance of a time
  class. (Scalar time-series objects don't exist, since the data attribute is
  always an nd-array).
* Indexing using integer indices and ordinary slices (with integer start,
  stop, and step) has the same effect as indexing the underlying nd-array using
  the same indices and slices.

  to.at(i) = to[i] = to.data[...,i]
  to.time(i) = to.time[i] = to.time.asarray()[i]

  I would argue that here we have another reason to choose the time dimension
  as the first dimension (not last): this would allow to rewrite the first
  line above:

  to.at(i) = to[i] = to.data[i]

* Every time-series data object can be accessed (indexed) using any time object.
* The result of indexing into a time-series data object using a time object is
  always again either an instance of the same time-series data class or an
  instance of a vanilla nd-array. The latter case only occurs, when a single
  time point is used to index into the time-series data and is analogous to
  indexing with a single integer into an nd-array.
* There are scalar and array-like time objects. Every array-like time object
  can be accessed (indexed) using any time object (just like the data
  objects).
* The result of indexing into a array-like time object using another time
  object is always again an instance of the same array-like time object or a
  time scalar. The latter case only occurs, when a single time point is used
  for indexing.

* Every time-series data (and time) object to implements a method
  `func`:to.time2index: that given a (scalar) time point t returns an integer
  index i suitable for indexing both into the nd-array to.data and into
  to.time:

  data_point = to.at(t) = to[t] = to.data[...,to.assindex(t)]
  time_point = to.time.at(t) = to.time[t] = to.time[to.time2index(t)]

* Every time-series data (and time) object to implements a method
  `func`:to.aslice: that given a (scalar) time interval ti (see
  `ref`:interval_object_discussion.rst:) returns an integer slice slice(i,j)
  suitable for indexing both into the nd-array to.data and into to.time:

  to.interval2slice(ti) = slice(to.time2index(ti.start), to.time2index(ti.stop))

  data_slice = to.data[...,to.interval2slice(ti)]
  time_slice = to.time[to.interval2slice(ti)]

 
Indexing using a (scalar) TimePoint
-----------------------------------

In the following, $t$ is a single time point represented by either a scalar of
the *TimePoint* class, by a scalar nd-array of dtype *datetime64*, or by a
floating point number. If *t* is represented as a floating point number, its
time unit is determined by the time unit of the object it is indexing into.


TimeSeries
~~~~~~~~~~

  ts.at(t) returns an nd-array ts.data[:,i], where ts.time.at(i) is the last
  time-point *preceding* t.

* what happens when $t<ts.t0$? return None?
* what happens when t is outside the last bin? return None?


Event
~~~~~

  ev.at(t) returns an nd-array ev.data[:,i], where ev.time.at(i) is the
  time-point *closest* to t.

* should there be an optional argument specifying a maximal time difference
  between the index time and the returned time?


TimeBin
~~~~~~~

  tb.at(t) returns the last time point preceding t.

* what happens when $t<tb.t0$? return None?
* what happens when t is outside the last bin? return None?


TimePoint
~~~~~~~~~

  tp.at(t) returns the time point closest to t.

* should there be an optional argument specifying a maximal time difference
  between the index time and the returned time?


Interval
~~~~~~~~

Here, ti is an array-like Interval object, not a scalar Interval (see
`ref`:interval_object_discussion.rst:). This is the only case, I am not quite
sure what to expect and if it even makes sense to implement this. One thing
that would make sense is that

  ti.at(t) returns another Interval array containing all intervals that
  contain the time point t.


Indexing using a TimePoint array
--------------------------------

TimeSeries
~~~~~~~~~~

Access with a *TimePoint* array tp into a *TimeSeries* object ts returns a new
*Event* object with time attribute tp and with data attribute

  np.array([ts[i] for i in tp],dtype=datetime64)


Indexing using a (scalar) Interval
----------------------------------

Access using intervals (see `ref`:interval_object_discussion.rst: ), will give
you back a uniform time-series objects with the time being of length of
t_start-t_end and with the ts.t0 offset by the intervals t_offset.

This works for *TimeSeries*, *Event*, and *TimePoint* classes.
