.. _tutorial:

=========
 Tutorial
=========

In this tutorial, we will demonstrate the basic use of nitime in initializing,
manipulating and analyzing a simple time-series object. For more advanced usage
see the examples section (:ref:`examples`)

In order to get started, import :mod:`nitime.timeseries`:

.. code-block:: python

   In [1]: import nitime.timeseries as ts

Then, you can initialize a simple time-series object, by providing data and
some information about the sampling-rate or sampling-interval:

.. code-block:: python

   In [2]: t1 = ts.TimeSeries([[1,2,3],[3,6,8]],sampling_rate=0.5)

If you tab-complete, you will see that the object now has several different
attributes:

.. code-block:: python

   In [3]: t1.
   t1.at                  t1.metadata            t1.time
   t1.data                t1.sampling_interval   t1.time_unit
   t1.duration            t1.sampling_rate		
   t1.from_time_and_data  t1.t0      

Note that the sampling_interval is the inverse of the sampling_rate:

.. code-block:: python

   In [4]: t1.sampling_interval
   Out[4]: 2.0 s

In addition, the sampling rate is now represented with the units in Hz:

.. code-block:: python

   In [5]: t1.sampling_rate
   Out[5]: 0.5 Hz

Also - once this object is available to you, you have access to the underlying
representation of time:

.. code-block:: python

   In [6]: t1.time
   Out[6]: UniformTime([ 0.,  2.,  4.], time_unit='s')

Now import the analysis library:

.. code-block:: python

   In [7]: import nitime.analysis as nta

and initialize an analyzer for correlation analysis:

.. code-block:: python

   In [8]: c = nta.CorrelationAnalyzer(t1)

The simplest use of this analyzer (and also the default output) is to compute
the correlation coefficient matrix of the data in the different rows of the
time-series:

.. code-block:: python

   In [9]: c.corrcoef
   Out[9]: 
   array([[ 1.        ,  0.99339927],
          [ 0.99339927,  1.        ]])

but it can also be used in order to generate the cross-correlation function
between the channels, which is also a time-series object:

.. code-block:: python

   In [63]: x = c.xcorr

   In [64]: x.time
   Out[64]: UniformTime([-6., -4., -2.,  0.,  2.], time_unit='s')

   In [65]: x.data
   Out[65]: 
   array([[[   3.,    8.,   14.,    8.,    3.],
           [   8.,   22.,   39.,   24.,    9.]],

          [[   8.,   22.,   39.,   24.,    9.],
           [  24.,   66.,  109.,   66.,   24.]]])



   

