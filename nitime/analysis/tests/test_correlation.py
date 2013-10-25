import numpy as np
import numpy.testing as npt

import nitime.timeseries as ts
import nitime.analysis as nta


def test_SeedCorrelationAnalyzer():

    targ = ts.TimeSeries(np.random.rand(10, 10), sampling_interval=1)

    # Test single source case
    seed = ts.TimeSeries(np.random.rand(10), sampling_interval=1)
    corr = nta.SeedCorrelationAnalyzer(seed, targ)
    our_coef_array = corr.corrcoef
    np_coef_array = np.array([np.corrcoef(seed.data, a)[0, 1] for a in targ.data])

    npt.assert_array_almost_equal(our_coef_array, np_coef_array)

    # Test multiple sources
    seed = ts.TimeSeries(np.random.rand(2, 10), sampling_interval=1)
    corr = nta.SeedCorrelationAnalyzer(seed, targ)
    our_coef_array = corr.corrcoef
    for source in [0, 1]:
        np_coef_array = np.array(
            [np.corrcoef(seed.data[source], a)[0, 1] for a in targ.data])
        npt.assert_array_almost_equal(our_coef_array[source], np_coef_array)
