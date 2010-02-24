.. _time_series_access:

====================
 Time-series access
====================

Since one dimension of time-series data is associated with time, there is a
natural way to index into time-series data using time objects as indices. For
the data classes (:ref:`time_series_classes`) an indexing operation performed
with a single time-point (a single-element :class:`TimeArray` or a
:class:`TimeArray` with more than one element) should result in returning the
data at that time-point - that is, removal of the time-dimension from the data.

The base-classes representing time (:ref:`time_classes`) serve as natural
intermediaries in this process, by providing the integer index of a particular
time-point (or returning an array of integers, as the case may be, see
below). Therefore, these classes should include a method which performs this
conversion, :func:`index_at`. This function should accept as parameter a
:class:`TimeArray` and return integers corresponding to the location of that
time-point in the different types of time classes.

Access into Time classes
------------------------

:class:`EventArray`
~~~~~~~~~~~~~~~~~~~

:func:`ev.index_at` returns the indices of the values in the array that are
*closest* to t. That is, it returns i, such that $|(t-t_i)|$ is
the minimal. 

Potentially, an optional 'tolerance' argument can be implemented, specifying a
maximal time difference between the index time and the returned time.


:class:`NonUniformTime`
~~~~~~~~~~~~~~~~~~~~~~~

As above, :func:`nut.index_at` also returns the indices in the array that
are closest to t. Since :class:`NonUniformTime` is ordered, this should give
you either the index below or the index above the time-point you provide as
input, depending on what interval ($|t-t_i|$ or $|t-t_{i+1}|$) is
smaller.


:class:`UniformTime`
~~~~~~~~~~~~~~~~~~~~

:func:`ut.index_at` returns the indices of the values in the array that are
the largest time values, smaller thatn the input values t. That is, it returns i
for which $t_i$ is the maximal one, which still fulfills: $t_i<t$.  

Questions
~~~~~~~~~
The follwing questions apply to all three cases: 

* what happens when the t is smaller than the smallest entry in the array
  return None?
* what happens when t is larget than the last entry in the time array? return
  None?

:func:`at`
~~~~~~~~~~

This function extracts the value of the time array, which corresponds to the
output of :func:`index_at` with an input t. 

That is, for an instance :class:`T` of one of the time classes, this function
will return:

.. code-block:: python

     T.time[T.index_at(t)]


Indexing into data time-series objects
--------------------------------------

Indexing with time
~~~~~~~~~~~~~~~~~~

The above function :func:`index_at` serves as the basis for the
implementation of the function :func:`at` for the time-series data objects.
This function returns the part of the data in :class:`UniformTimeSeries.data`
(or the equivalent data structure in :class:`EventSeries` and
:class:`NonUniformTimeSeries`) that corresponds to the times provided.

Importantly, the result of indexing into a time-series data object using a time
object is always again either an instance of the same time-series data class or
an instance of a vanilla nd-array. The latter case only occurs, when a single
time point is used to index into the time-series data and is analogous to
indexing with a single integer into an nd-array. Conversion between different
time-series classes can occur if the indexing time-points are non-uniform (for
conversion between :class:`UniformTimeSeries` and
:class:`NonUniformTimeSeries`) or if the time-points are not ordered (for
conversion from :class:`UniformTimeSeries` or from
:class:`NonUniformTimeSeries` to :class:`EventSeries`).  

Currently, the plan is to implement the indexing operation using the method
:func:`at` and only later to map the method :meth:`ts.__getitem__` to the
function :func:`ts.at`. For now, we not that using the function :func:`ts.at`
directly is more flexible since it allows to use additional keyword arguments,
so, for now, it is unclear what to set as the default behavior for :func:`at`,
which will be executed by :meth:`__getitem__`. 

The function :func:`during` will receive as input a :class:`TimeInterval`
objects and will return the data corresponding to the interval, while dealing
appropriately with the :attr:`TI.t_step` (see :ref:`interval_class` for
details). How is this done? For an object of class :class:`UniformTimeSeries`,
access using intervals, will give you back a uniform time-series objects with
the time being of length of :attr:`TI.t_start` - :attr:`TI.t_stop` and with
the :attr:`TS.t0` offset by the :class:`TimeInterval`'s
:attr:`TI.t_step`. 

Indexing with integers
~~~~~~~~~~~~~~~~~~~~~~

In parallel to the access with time-points, described above, we would like to
implement indexing the time-series classes directly using integer indices and
ordinary slices (with integer start, stop, and step). This should have the same
effect as indexing the underlying nd-array using the same indices and slices,
such that:

.. code-block:: python

	       T.at(T.time.index_at(i)) = T[i] = T.data[...,i]
  	       T.time.at(i) = T.time[i] = T.time.asarray()[i]

In order to make the above code more compact, would be another reason to
implement the the time dimension as the first dimension (not last, see
:ref:`time_series_classes`): this would allow to rewrite the above:

.. code-block:: python

   		T.at(i) = T[i] = T.data[i]

	       
Every time-series data (and time) object would also implements a method
:func:`T.slice_at` that given a :class:`TimeInterval` object TI (see
:ref:`interval_class`) returns an integer slice slice(i,j) suitable for
indexing both into the nd-array :attr:`T.data` and into
:attr:`T.time`:

.. code-block:: python


   T.interval2slice(TI) = slice(T.time2index(TI.t_start),
   T.time2index(TI.t_stop))

  data_slice = T.data[...,T.slice_at(TI)]
  time_slice = T.time[T.slice_at(TI)]

 



