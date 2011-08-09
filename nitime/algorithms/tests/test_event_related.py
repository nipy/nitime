import numpy as np
import numpy.testing as npt
import nitime
import nitime.algorithms as tsa


def test_xcorr_zscored():
    """

    Test this function, which is not otherwise tested in the testing of the
    EventRelatedAnalyzer

    """

    cycles = 10
    l = 1024
    unit = 2 * np.pi / l
    t = np.arange(0, 2 * np.pi + unit, unit)
    signal = np.sin(cycles * t)
    events = np.zeros(t.shape)
    #Zero crossings:
    idx = np.where(np.abs(signal) < 0.03)[0]
    #An event occurs at the beginning of every cycle:
    events[idx[:-2:2]] = 1

    a = tsa.freq_domain_xcorr_zscored(signal, events, 1000, 1000)
    npt.assert_almost_equal(np.mean(a), 0, 1)
    npt.assert_almost_equal(np.std(a), 1, 1)
