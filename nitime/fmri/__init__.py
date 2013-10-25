"""
Support for fMRI analysis in nitime:

This includes:

- ``io``: input and output of fMRI files to time-series objects

"""
from __future__ import print_function
try:
    from . import io
except ImportError as e:
    # allow import of fmri, since hrf does not depend on nibabel
    if 'babel' in e.args[0]:
        print(e.args[0])
    else:
        raise e

from . import hrf
from . import tests
