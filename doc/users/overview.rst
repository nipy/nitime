.. _overview:

===================
Nitime: an overview
===================

Nitime can be used in order to represent, manipulate and analyze data in
time-series from experimental data. The main intention of the library is to
serve as a platform for analyzing data collected in neuroscientific
experiments, ranging from single-cell recordings to fMRI. However, the
object-oriented interface may match other kinds of time-series.

In the :ref:`tutorial`, we provide examples of usage of the objects in the
library and of some basic analysis.

Here, we will provide a brief overview of the guiding principles underlying the
structure and implementation of the library and the programming interface
provided by library. 

We will survey the library their attributes and central functions and some possible example
use-cases.

=================
Design Principles
=================
The main principle of the implementation of this library is a separation
between representation of time-series and the analysis of time-series. Thus,
the implementation is divided into three main elements:

- Base classes for representation of time and data: These include objects
  representing time (including support for the representation and conversion
  between time-units) and objects that serve as containers for data:
  representation of the time-series to be analyzed. These base classes will be
  surveyed in more detail in the :ref:`base_classes`
  
- Algorithms for analysis of time-series: A library containing implementations
  of algorithms for various analysis methods is provided. Importantly, this
  library is intentionally agnostic to the existence of the library
  base-classes. Thus, users can choose to use these algorithms directly,
  instead of relying on the base-classes provided by the library  
  
- Analyzer interfaces: These objects provide an interface between the algorithm
  library and the time-series objects. Each one of these objects calls an
  algorithm from the algorithms  These objects rely on the details of the
  implementation of the time-series objects. The input to these classes is
  usually a time-series object and a set of parameters, which guide the
  analysis. Some of the analyzer objects implement a thin interface (or
  'facade') to algorithms provided by scipy.signal. 

This principle is important, because it allows use of the analysis algorithms
at two different levels. The algorithms are more general-purpose, but provide
less support for the unique properties of time-series. The analyzer objects, on
the other hand, provide a richer interface, but may be less flexible in their
usage, because they assume use of the base-classes of the library.  

This structure also makes development of new algorithms and adoption of
analysis code from other sources easier, because no specialized design
properties are required in order to include an algorithm or set of algorithms
in the algorithm library. However, once algorithms are adopted into the
library, it requires that additional development of the analyzer object
specific for this set of algorithms be implemented as well.  

Another important principle of the implementation is lazy initialization. Most
attributes of both time-series and analysis objects are provided on a
need-to-know basis. That is, initializing a time-series object, or an analyzer
object does not trigger any intensive computations. Instead the computation of
the attributes of analyzer objects is delayed until the moment the user calls
these attributes. In addition, once a computation is triggered it is stored as
an attribute of the object, which assures that accessing the results of an
analysis will trigger the computation only on the first time the analysis resut
is required. Thereafter, the result of the analysis is stored for further use
of this result.

.. _base_classes: 

==============
 Base classes
==============

The library has several sets of classes, used for the representation of time
and of time-series, in addition to classes used for analysis.

The first kind of classes is used in order to represent time and inherits from
:class:`np.ndarray`, see :ref:`time_classes`. Another are data containers, used
to represent different kinds of time-series data, see
:ref:`time_series_classes` A third important kind are *analyzer* objects. These
objects can be used in order to apply a particular analysis to time-series
objects, see :ref:`analyzer_objects`

.. _time_classes:

Time
====
Experimental data is usually represented with regard to *relative* time. That
is, the time relative to the beginning of the measurement. This is in contrast
to many other kinds of data, which are represented with regard to *absolute*
time, (one example of this kind of time is calendaric time, which includes a
reference to some common point, such as 0 CE, or Jan. 1st 1970). An example of
data which benefits from representation with absolute time is the
representation of financial time-series, which can be compared against each
other, using the common reference and for which the concept of the work-week
applies. 

However, because most often the absolute calender time of the occurence of
events in an experiment is of no importance, we can disregard it. Rather, the
comparison of the time progression of data in different experiments conducted
in different calendar times (different days, different times in the same day)
is more common.

The underlying representation of time in :mod:`nitime` is in arrays of dtype
:class:`int64`. This allows the representation to be immune to rounding errors
arising from representation of time with floating point numbers (see
[Goldberg1991]_). However, it restricts the smallest time-interval that can be
represented. In :mod:`nitime`, the smallest discrete time-points are of size
:attr:`base_unit`, and this unit is *picoseconds*. Thus, all underlying
representations of time are made in this unit. Since for most practical uses,
this representation is far too small, this might have resulted, in most cases
in representations of time too long to be useful. In order to make the
time-objects more manageable, time objects in :mod:`nitime` carry a
:attr:`time_unit` and a :attr:`_conversion_factor`, which can be used
as a convenience, in order to convert between the representation of time in the
base unit and the appearance of time in the relevant time-unit.   

The first set of base classes is a set of representations of time itself. All
these classes inherit from :class:`np.ndarray`. As mentioned above, the dtype of
these classes is :class:`int64` and the underlying representation is always at
the base unit. In addition to the methods inherited from :class:`np.ndarray`,
these time representations have an :func:`at` method which . The result of this indexing
will be to return the time-point in the the respective :class:`TimeSeries`
which is most appropriate (see :ref:`time_series_access` for details). They
have an :func:`index_at` method, which returns the integer index of this time
in the underlying array. Finally, they will all have a :func:`during` method,
which will allow indexing into these objects with an
:ref:`interval_class`. This will return the appropriate times corresponding to
an :ref:`interval_class` and :func:`index_during`, which will return the array
of integers corresponding to the indices of these time-points in the array.

For the time being, there are two types of Time classes: :ref:`TimeArray` and :ref:`UniformTime`.

.. _TimeArray:

:class:`TimeArray`
-------------------

This class has less restrictions on it: it is made of an 1-d array, which contains time-points that are not neccesarily ordered. It can also contain several copies of the same time-point. This class can be used in order to represent sparsely occuring events, measured at some unspecified sampling rate and possibly collected from several different channels, where the data is sampled in order of channel and not in order of time. As in the case of the :class:`np.ndarray`. This representation of time carries, in addition to the array itself an attribute :attr:`time_unit`, which is the unit in which we would like to present the time-points (recall that the underlying representation is always in the base-unit).

.. _UniformTime:

:class:`UniformTime`
--------------------

This class contains ordered uniformly sampled time-points. This class has an explicit representation of :attr:`t_0`, :attr:`sampling_rate` and :attr:`sampling_interval`. Thus, each element in this array can be used in order to represent the entire time interval $t$, such that: $t_i\leq t < t + \delta t$, where $t_i$ is the nominal value held by that element of the array, and $\delta t$ is the value of :attr:`sampling_interval`. 

This object contains additional attributes that are not shared by the other
time objects. In particular, an object of :class:`UniformTime`, UT, will have
the following:

* :attr:`UT.t_0`: the first time-point in the series.
* :attr:`UT.sampling_rate`: the sampling rate of the series (this is an
  instance of .
* :attr:`UT.sampling_interval`: the value of $\delta t$, mentioned above.
* :attr:`UT.duration`: the total time of the series.

Obviously, :attr:`UT.sampling_rate` and :attr:`UT.sampling_interval` are redundant, but can both be useful.


:class:`Frequency`
------------------

The :attr:`UT.sampling_rate` of :class:`UniformTime` is an object of the :class:`Frequency` class. This is a representation of the frequency in Hz. It is derived from a combination of the :attr:`sampling_interval` and the :attr:`time_unit`.

.. _time_series_classes:

Time-series 
============

These are data container classes for representing different kinds of
time-series data types.

In implementing these objects, we follow the following principles:

* The time-series data representations do not inherit from
  :class:`np.ndarray`. Instead, one of their attributes is a :attr:`data`
  attribute, which *is* a :class:`np.ndarray`. This principle should allow for
  a clean and compact implementation, which doesn't carry all manner of
  unwanted properties into a bloated object with obscure and unknown behaviors.
  We have previously decided to make *time* the last dimension in this
  object, but recently we have been considering making this a user choice (in
  order to enable indexing into the data by time in a straight-forward manner
  (using expressions such as :class:`TI.data[i]`. 
* In tandem, one of their attributes is one of the :ref:`time_classes` base
  classes described above. This is the :attr:`time` attribute of the
  time-series object. Therefore, for :class:`TimeSeries` it is implemented in
  the object with a :func:`desc.setattr_on_read` decoration, so that it is only
  generated if it is needed.

.. _TimeSeries:

:class:`TimeSeries`
--------------------------

This represents time-series of data collected continuously and regularly. Can
be used in order to represent typical physiological data measurements, such as
measurements of BOLD responses, or of membrane-potential. The representation of
time here is :ref:`UniformTime`.

XXX Write more about the different attributes of this class.

.. _Epochs:

:class:`Epochs`
---------------

This class represents intervals of time, or epochs. Each instance of this class
contains several attributes:

- :attr:`E.start`: This is an object of class :class:`TimeArray`, which
  represents a collection of starting times of epochs
- :attr:`E.stop`: This is an object of class :class:`TimeArray` which
  represents a collection of end points of the epochs. 
- :attr:`E.duration`: This is an object of class :class:`TimeArray` which
  represents the durations of the epochs.
- :attr:`E.offset`: This attribute represents the offset of the epoch 
- :attr:`E.time_unit`: This is 

.. _Events:

:class:`Events`
---------------

This is an object which represents a collection of events. For example, this
can represent discrete button presses occuring during an experiment. This
object contains a :ref:`TimeArray` as its representation of time. This means
that the events recorded in the :attr:`data` array can be organized
according to any organizing principle you would want, not neccesarily according
to their organization or order in time. For example, if events are read from
different devices, the order of the events in the data array can be arbitrarily
chosen to be the order of the devices from which data is read.



Analyzers
=========

These objects implement a particular analysis, or family of analyses. Typically, the initialization of this kind of object can happen with
a time-series object provided as input, as well as a set of parameter values  setting. However, for most analyzer objects, the inputs can be provided upong
calling the object, or by assignment to the already generated object.

Sometimes, a user may wish to revert the computation, change some of the
analysis parameters and recompute one or more of the results of the
analysis. In order to do that, the analyzer objects implement a :attr:`reset`
attribute, which reverts the computation of analysis attributes and allows to
change parameters in the analyzer and recompute the analysis results. This
structure keeps the cost of computation of quantities derived from the analysis
rather low.

.. [Goldberg1991] Goldberg D (1991). What every computer scientist should know
   about floating-point arithmetic. ACM computing surveys 23: 5-48
