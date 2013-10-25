import numpy as np
import numpy.testing as npt

import nitime.algorithms as tsa

def test_seed_correlation():

    seed = np.random.rand(10)
    targ = np.random.rand(10, 10)

    our_coef_array = tsa.seed_corrcoef(seed, targ)
    np_coef_array = np.array([np.corrcoef(seed, a)[0, 1] for a in  targ])

    npt.assert_array_almost_equal(our_coef_array, np_coef_array)
