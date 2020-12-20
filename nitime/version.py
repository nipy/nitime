"""nitime version/release information"""

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 10
_version_micro = ''  # use '' for first of series, number for 1 and above
_version_extra = 'dev'
# _version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: BSD License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

description = "Nitime: timeseries analysis for neuroscience data"

# Note: this long_description is actually a copy/paste from the top-level
# README.txt, so that it shows up nicely on PyPI.  So please remember to edit
# it only in one place and sync it correctly.
long_description = """
===================================================
 Nitime: timeseries analysis for neuroscience data
===================================================

Nitime is library of tools and algorithms for the analysis of time-series data
from neuroscience experiments. It contains a implementation of numerical
algorithms for time-series analysis both in the time and spectral domains, a
set of container objects to represent time-series, and auxiliary objects that
expose a high level interface to the numerical machinery and make common
analysis tasks easy to express with compact and semantically clear code.

Website and mailing list
========================

Current information can always be found at the nitime `website`_. Questions and
comments can be directed to the mailing `list`_.

.. _website: http://nipy.org/nitime
.. _list: http://mail.scipy.org/mailman/listinfo/nipy-devel

Code
====

You can find our sources and single-click downloads:

* `Main repository`_ on Github.
* Documentation_ for all releases and current development tree.
* Download as a tar/zip file the `current trunk`_.
* Downloads of all `available releases`_.

.. _main repository: http://github.com/nipy/nitime
.. _Documentation: http://nipy.org/nitime
.. _current trunk: http://github.com/nipy/nitime/archives/master
.. _available releases: http://github.com/nipy/nitime/downloads


License information
===================

Nitime is licensed under the terms of the new BSD license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.

All trademarks referenced herein are property of their respective holders.

Copyright (c) 2006-2020, NIPY Developers
All rights reserved.
"""

NAME = "nitime"
MAINTAINER = "Nipy Developers"
MAINTAINER_EMAIL = "neuroimaging@python.org"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "http://nipy.org/nitime"
DOWNLOAD_URL = "http://github.com/nipy/nitime/downloads"
LICENSE = "Simplified BSD"
AUTHOR = "Nitime developers"
AUTHOR_EMAIL = "neuroimaging@python.org"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {"nitime": ["LICENSE", "tests/*.txt", "tests/*.npy",
                           "data/*.nii.gz", "data/*.txt", "data/*.csv"]}
PYTHON_REQUIRES = ">=3.5"
