
import nitime.analysis as nta


data_file_path = test_dir_path = os.path.join(nitime.__path__[0],
                                              'fmri/tests/data/')

#Make a time-series with dims n_vox,n_TR:

time_series_seed = ts.TimeSeries()

#Make another one (this one can have many more voxels!)
# rows are voxels, columns are timepoints
time_series_target = ts.TimeSeries()



A=nta.SeedCoherenceAnalyzer()

C = A.coherency()
