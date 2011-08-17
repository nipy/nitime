"""
Nitime analysis
---------------

This module implements an analysis interface between between time-series
objects implemented in the :mod:`timeseries` module and the algorithms provided
in the :mod:`algorithms` library and other algorithms.

The general pattern of use of Analyzer objects is that they an object is
initialized with a TimeSeries object as input. Depending on the analysis
methods implemented in the particular analysis object, additional inputs may
also be required.

The methods of the object are then implemented as instances of
:obj:`OneTimeProperty`, which means that they are only calculated when they are
needed and then cached for further use.

Analyzer objects are generally implemented inheriting the
:func:`desc.ResetMixin`, which means that they have a :meth:`reset`
method. This method resets the object to its initialized state, in which none
of the :obj:`OneTimeProperty` methods have been calculated. This allows to
change parameter settings of the object and recalculating the quantities in
these methods with the new parameter setting.

"""

from nitime.analysis.coherence import *
from nitime.analysis.correlation import *
from nitime.analysis.event_related import *
from nitime.analysis.normalization import *
from nitime.analysis.snr import *
from nitime.analysis.spectral import *
from nitime.analysis.granger import *
