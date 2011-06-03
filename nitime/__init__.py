"""
Nitime: Time-series analysis for neuroscience

Nitime has several sub-packages that are automatically available when ``import
nitme`` is performed:

- ``timeseries``: contains the constructors for time and time-series objects.

- ``algorithms``: Algorithms. This sub-module depends only on scipy,numpy and
  matplotlib. Contains various algorithms.

- ``analysis``: Contains *Analyzer* objects, which implement particular
  analysis methods on the time-series objects.

The following subpackages are also available in nitime, but these must be
individually imported with e.g. ``import nitime.viz``:

- ``utils``: Utility functions.

- ``viz``: Vizualization.

All of the sub-modules will be imported as part of ``__init__``, so that users
have all of these things at their fingertips.
"""

__docformat__ = 'restructuredtext'

from version import  __version__

import algorithms
import timeseries
import analysis

from timeseries import *

from .testlib import test
