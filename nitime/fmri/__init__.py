"""
Support for fMRI analysis in nitime:

This includes:

- ``io``: input and output of fMRI files to time-series objects

"""
try:
    import io
except ImportError,e:
    # allow import of fmri, since hrf does not depend on nibabel
    if 'babel' in e.args[0]:
        print e.args[0]
    else:
        raise e

import hrf
import tests
