import os

import numpy as np
import matplotlib.pyplot as plt

import nitime
import nitime.analysis as nta
reload(nta)
import nitime.fmri.io as io
reload(io)

data_file_path = test_dir_path = os.path.join(nitime.__path__[0],
                                              'fmri/tests/data/')

fmri_file = os.path.join(data_file_path,'fmri1.nii.gz')
TR = 1.35
f_lb = 0.02
f_ub = 0.15

#Take coordinate 0,0,0 and 0,0,1 to be the seed voxels:
coords_seed = np.array([[0,0,0],[0,0,1]]).T

#The data shape is x=10,y=10,z=18,t=40: 
coords_target = []
for i in range(10):
    for j in range(10):
        for k in range(18):
            coords_target.append([i,j,k])

#Put it in the right shape, while excluding the seed coordinates:
coords_target = np.array(coords_target)[3:].T

#Make the seed time series:
time_series_seed = io.time_series_from_file(fmri_file,
                                           coords_seed,
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

A=nta.SeedCoherenceAnalyzer(time_series_seed,time_series_target,
                            method=dict(NFFT=20))

#We look only at frequencies between 0.02 and 0.15 (the physiologically
#relevant band, see http://imaging.mrc-cbu.cam.ac.uk/imaging/DesignEfficiency:
freq_idx = np.where((A.frequencies>0.02) * (A.frequencies<0.15))[0]

#Extract the coherence and average across these frequency bands: 
coh1 = np.mean(A()[0][:,freq_idx],-1) #Averaging on the last dimension 
coh2 = np.mean(A()[1][:,freq_idx],-1) #Averaging on the last dimension 

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(coh1)
ax.plot(coh2)
ax.set_xlabel('Target voxel #')
ax.set_ylabel('Coherence')
