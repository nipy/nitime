"""

=============================
Seed coherence with fMRI data
=============================

"""

import os

import numpy as np
import matplotlib.pyplot as plt
from nibabel import load

import nitime
import nitime.analysis as nta
reload(nta)
import nitime.fmri.io as io
reload(io)

TR = 1.35
f_lb = 0.02
f_ub = 0.15

data_file_path = test_dir_path = os.path.join(nitime.__path__[0],
                                              'fmri/tests/data/')

fmri_file = os.path.join(data_file_path,'fmri1.nii.gz')

fmri_data = load(fmri_file) 

#The dimensions verything but the time dimension:
volume_shape = fmri_data.shape[:-1]

coords = list(np.ndindex(volume_shape))

#How many of the coords are seed voxels:
n_seeds = 3

#Choose n_seeds random voxels to be the seed voxels
seeds = np.random.randint(0,len(coords),n_seeds)
coords_seeds = np.array(coords)[seeds].T

#The entire volume is the target:
coords_target = np.array(coords).T

#Make the seed time series:
time_series_seed = io.time_series_from_file(fmri_file,
                                           coords_seeds,
                                           TR=TR,
                                           normalize='percent',
                                           filter=dict(lb=f_lb,
                                                       ub=f_ub,
                                                       method='boxcar'))

#Make the target time series: 
time_series_target = io.time_series_from_file(fmri_file,
                                              coords_target,
                                              TR=TR,
                                              normalize='percent',
                                              filter=dict(lb=f_lb,
                                                          ub=f_ub,
                                                          method='boxcar'))

A=nta.SeedCoherenceAnalyzer(time_series_seed, time_series_target,
                            method=dict(NFFT=20))

#We look only at frequencies between 0.02 and 0.15 (the physiologically
#relevant band, see http://imaging.mrc-cbu.cam.ac.uk/imaging/DesignEfficiency:
freq_idx = np.where((A.frequencies>0.02) * (A.frequencies<0.15))[0]

coh = []

for this_coh in range(n_seeds):
    #Extract the coherence and average across these frequency bands: 
    coh.append(np.mean(A.coherence[this_coh][:,freq_idx],-1)) #Averaging on the
                                                              #last dimension

#For numpy fancy indexing into volume arrays:
coords_indices = list(coords_target)

vol = []
for this_vol in range(n_seeds):
    vol.append(np.empty(volume_shape))
    vol[-1][coords_indices] = coh[this_vol]

#Choose a random slice to display:
random_slice = np.random.randint(0,volume_shape[-1],1)

fig = plt.figure()
ax = []
for this_vox in range(n_seeds):
    ax.append(fig.add_subplot(1,n_seeds,this_vox+1))
    ax[-1].matshow(vol[this_vox][:,:,random_slice].squeeze())
    ax[-1].set_title('Seed coords: %s'%coords_seeds[:,this_vox])

fig.suptitle('Coherence between all the voxels in slice: %i and seed voxels'%random_slice)
