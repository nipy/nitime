import numpy as np
import numpy.testing as npt

import nitime.algorithms as tsa

np.random.seed(1945)

def test_entropy():
    x = np.random.randint(0, 2, size=1000)
    e1 = tsa.entropy(x)
    npt.assert_almost_equal(e1, 1, decimal=2)
    # The joint entropy of the variable with itself is the same:
    e2 = tsa.entropy(x, x)
    npt.assert_almost_equal(e1, e2)
    y = np.random.randint(0, 2, size=1000)
    # Joint entropy with another random variable is 2:
    e3 = tsa.entropy(x, y)
    npt.assert_almost_equal(e3, 2, decimal=2)


def test_conditional_entropy():
    x = np.random.randint(0, 2, size=1000)
    y = np.random.randint(0, 2, size=1000)
    e1 = tsa.conditional_entropy(x, x)
    npt.assert_almost_equal(e1, 0)
    e2 = tsa.conditional_entropy(x, y)
    npt.assert_almost_equal(e2, 1, decimal=2)


def test_mutual_information():
    x = np.random.randint(0, 2, size=1000)
    y = np.random.randint(0, 2, size=1000)
    e1 = tsa.mutual_information(x, x)
    npt.assert_almost_equal(e1, 1, decimal=2)
    e2 = tsa.mutual_information(x, y)
    npt.assert_almost_equal(e2, 0, decimal=2)


def test_entropy_cc():
    x = np.random.randint(0, 2, size=1000)
    e1 = tsa.entropy_cc(x, x)
    npt.assert_almost_equal(e1, 1, decimal=2)


def test_transfer_entropy():
    x = np.random.randint(0, 4, size=1000)
    y = np.roll(x, -1)
    e1 = tsa.transfer_entropy(x, y, lag=1)
    npt.assert_almost_equal(e1, 2, decimal=1)
