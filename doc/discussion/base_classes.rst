.. _base_classes:

==============
 Base classes
==============

We have two sets of base-classes. The first is used in order to represent time
and inherits from :class:`np.ndarray`, see :ref:`time_classes`. The other are
data containers, used to represent different kinds of time-series data, see
:ref:`time_series_classes`

.. _time_classes:

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
which will allow indexing into these objects with an interval object. This will
return the appropriate times corresponding to an :ref:`interval_class` and
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
or also represent events which occur at different times in an ordered
series. Thus, the time-points in this representation are unique (should they be
unique?) and are ordered. This will be used as the time representation used in
the :ref:`NonUniformTimeSeries` class.   


.. _UniformTime:

:class:`UniformTime`
--------------------

This class contains ordered time-points. In addition, this class has an
explicit representation of :attribute:`t_0`, :attribute:`sampling_rate` and
:attribute:`sampling_interval` (the latter two implemented as
:method:`setattr_on_read`, which can be computed from each other).Thus, each
element in this array can be used in order to represent the entire time
interval $t$, such that: $t_i\leq t < t + \delta t$, where $t_i$ is the nominal
value held by that element of the array, and $\delta t$ is the value of
:attribute:`sampling_interval`.  Notice that this kind of class can be reshaped
in such a way that it gets more dimensions (see also :ref:`time_table`). The
:ref:`NonUniformTime` class will be designed such that in the future it too can
be reshaped as a ragged/jagged array (see
http://en.wikipedia.org/wiki/Array_data_structure for details).


.. _time_table:

+-------+----------------+----+---------+--------------------+------------------+
|       | class          | 1d | ordered | unique time points | uniform sampling |
+=======+================+====+=========+====================+==================+
|       | EventArray     | y  |    n    |         n          |         n	|
|       |----------------+----+---------+--------------------+------------------+
| Time  | NonUniformTime | n  |    y    |         y          |         n        |
|  	|----------------+----+---------+--------------------+------------------+  
|       | UniformTime    | n  |    y    |         y          |         y        |
+-------+----------------+----+---------+--------------------+------------------+


.. _time_series_classes:

Time-series 
===========

These are data container classes for representing different kinds of
time-series data types.

In implementing these objects, we follow the following principles:

* The time-series data representations do not inherit from
  :class:`np.ndarray`. Instead, one of their attributes is a :attribute:`data`
  attribute, which *is* a :class:`np.ndarray`. This principle should allow for
  a clean and compact implementation, which doesn't carry all manner of
  unwanted properties into a bloated object with obscure and unknown behaviors.
* In tandem, one of their attributes is one of the base classes described
  above, in :ref:`time_classes`
* Access into the object and into the object will be uniform across the
  different classes :attribute:`data` and into the object. Described in
  :ref:`time_series_access`.
* In particular, we want to enable indexing into these data-containers with
  both arrays of time-points (arrays of dtype :class:`datetime64`), with
  intervals (see :ref:`interval_class`) and also, eventually, with
  integers. This should include operations that behave like :class:`np.ndarray`
  'fancy indexing. See :ref:`time_series_access` for detail.

.. _EventSeries:
:class:`EventSeries`
--------------------

This is an object which represents a collection of events. For example, this
can represent discrete button presses occuring during an experiment. This
object contains a :ref:`EventArray` as its representation of time. This means
that the events recorded in the :attribute:`data` array can be organized
according to any organizing principle you would want, not neccesarily according
to their organization or order in time. For example, if events are read from
different devices, the order of the events in the data array can be arbitrarily
chosen to be the order of the devices from which data is read.


.. _NonUniformTimeSeries:
:class:`NonUniformTimeSeries`
-----------------------------

As in the case of the :ref:`EventSeries`, this object also represents a
collection of events. However, in contrast, these events must be ordered. This
can be used, for example, in order to represent a rare event in continuous
time, such as a spike-train. Alternatively, it could be used in order to
represent continuous time sampling, which is done not in a constant
sampling-rate (what is an example of that?). The representation of time here is
:ref:`NonUniformTime`.


.. _UniformTimeSeries:
:class:`UniformTimeSeries`
--------------------------

This represents time-series of data collected continuously and regularly. Can
be used in order to represent typical physiological data measurements, such as
measurements of BOLD responses, or of membrane-potential. The representation of
time here is :ref:`UniformTime`.


.. _time_series_table:

+--------+---------------------+---------------+-----------------+
|        | class               |    time       | example         |
+========+=====================+===============+=================+
|        | EventSeries         | EventArray    | button presses  |
|        |---------------------+---------------+-----------------+
|  Time- | NonUniformTimeSeries| NonUniformTime| spike trains    |
| Series |---------------------+---------------+-----------------+ 
|        | UniformTimeSeri     | UniformTime   | BOLD            |
+--------+---------------------+---------------+-----------------+
