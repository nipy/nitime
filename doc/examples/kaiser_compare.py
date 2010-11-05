"""

=================
The Kaiser window
=================


"""
import numpy as np
import matplotlib.pyplot as plt
from nitime.viz import winspect
import scipy.signal as sig

#Define a helper function to inspect the kaiser windows:
def kaiser_inspect(npts, f, beta):
    name = r'Kaiser, $\beta=%1.1f$' % beta
    winspect(sig.kaiser(npts, beta), f, name)

f = plt.figure()
# Window size
npts = 128

# Various Kaiser windows
kaiser_inspect(npts,f, 0.1)
kaiser_inspect(npts,f, 1)
kaiser_inspect(npts,f, 10)
kaiser_inspect(npts,f, 100)

