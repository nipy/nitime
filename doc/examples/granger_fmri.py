"""

.. gc-fmri

================================
Granger 'causality' of fMRI data
================================

Granger 'causality' analysis relies on modeling the mutlivariate autoregressive
process

We start by importing the needed modules. First modules from the standard lib
and from 3rd parties:

"""

import os

import numpy as np
import matplotlib.pyplot as plt


"""

Notice that nibabel (http://nipy.org.nibabel) is required in order to run this
example:
"""

try:
    from nibabel import load
except ImportError:
    raise ImportError('You need nibabel (http:/nipy.org/nibabel/) in order to run this example')

"""

Import the nitime modules we will use:

"""

import nitime
import nitime.analysis as nta
import nitime.fmri.io as io

"""

We define the TR of the analysis and the frequency band of interest:

"""

TR = 1.35
f_lb = 0.02
f_ub = 0.15


"""

An fMRI data file with some actual fMRI data is shipped as part of the
distribution, the following line will find the path to this data on the
specific setup:

"""

data_file_path = test_dir_path = os.path.join(nitime.__path__[0],
                                              'fmri/tests/')

fmri_file = os.path.join(data_file_path, 'fmri1.nii.gz')


"""

Read in information about the fMRI data, using nibabel:

"""

fmri_data = load(fmri_file)

volume_shape = fmri_data.shape[:-1]

coords = list(np.ndindex(volume_shape))


"""

We choose some number of random voxels to serve as the ROIs for this analysis:

"""

n_ROI = 3

ROIs = np.random.randint(0, len(coords), n_ROI)
coords_ROIs = np.array(coords)[ROIs].T

"""

We use nitime.fmri.io in order to generate TimeSeries objects from spatial
coordinates in the data file:

"""
time_series = io.time_series_from_file(fmri_file,
                                       coords_ROIs,
                                       TR=TR,
                                       normalize='percent',
                                       filter=dict(lb=f_lb,
                                                   ub=f_ub,
                                                   method='iir'))


"""

We initialize the GrangerAnalyzer object, while specifying the order of the
autoregressive model to be 2.

"""

G = nta.GrangerAnalyzer(time_series, order=2)

"""

For comparison, we also initialize a CoherenceAnalyzer, with the same
TimeSeries object

"""

C = nta.CoherenceAnalyzer(time_series, method=dict(NFFT=20))

"""

We are only interested in the physiologically relevant frequency band:

"""

freq_idx_G = np.where((G.frequencies > f_lb) * (G.frequencies < f_ub))[0]
freq_idx_C = np.where((C.frequencies > f_lb) * (C.frequencies < f_ub))[0]


fig01 = plt.figure()
ax = fig01.add_subplot(1,1,1)
ax.plot(G.frequencies[freq_idx_G],G.causality_xy[0,1][freq_idx_G])
ax.plot(G.frequencies[freq_idx_G],G.causality_yx[0,1][freq_idx_G])
ax.plot(C.frequencies[freq_idx_C],C.coherence[0,1][freq_idx_C])
