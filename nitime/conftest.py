import numpy as np
import pytest


@pytest.fixture(scope='session', autouse=True)
def legacy_printoptions():
    np.set_printoptions(legacy='1.21', precision=4)
