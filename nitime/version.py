"""nitime version/release information"""

ISRELEASED = True

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = 0
__version__ = "%s.%s.%s" % (_version_major, _version_minor, _version_micro)


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
MISSING
"""

NAME                = "nitime"
MAINTAINER          = "Nipy Developers"
MAINTAINER_EMAIL    = "nipy-devel@neuroimaging.scipy.org"
DESCRIPTION         = description
LONG_DESCRIPTION    = long_description
URL                 = "http://nipy.org/nitime"
DOWNLOAD_URL        = "http://github.com/nipy/nitime/downloads"
LICENSE             = "Simplified BSD"
AUTHOR              = "Nitime developers"
AUTHOR_EMAIL        = "nipy-devel@neuroimaging.scipy.org"
PLATFORMS           = "OS Independent"
MAJOR               = _version_major
MINOR               = _version_minor
MICRO               = _version_micro
VERSION             = __version__
PACKAGES            = ['nitime', 'nitime.fmri', 'nitime.fmri.tests',
                       'nitime.fixes', 'nitime.tests']
PACKAGE_DATA        = {"nitime": ["LICENSE", "tests/*.txt"]}
REQUIRES            = ["numpy", "matplotlib", "scipy"]
