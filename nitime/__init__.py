"""
Nitime: Time-series analysis for neuroscience 

The module has several sub-modules: 

- ``timeseries``: contains the constructors for time and time-series objects

- ``algorithms``: Algorithms. This sub-module depends only on scipy,numpy and
  matplotlib. Contains various algorithms. 

- ``utils``: Utility functions. 

- ``analysis``: Contains *Analyzer* objects, which implement particular
  analysis methods on the time-series objects

- ``viz``: Vizualization

All of the sub-modules will be imported as part of ``__init__``, so that users
have all of these things at their fingertips.
"""

__docformat__ = 'restructuredtext'

from version import  __version__

#The order of the following is important - make sure to import according to the
#order of dependency (for example, timeseries depends on descriptors, so
#descriptors needs to be imported before timeseries): 
import utils, algorithms, descriptors, timeseries, viz, analysis

from timeseries import *

from .testlib import test
