import numpy as np
import numpy.testing as npt
import matplotlib.mlab as mlab

import nitime.timeseries as ts
import nitime.analysis as nta


def test_SNRAnalyzer():
    Fs = np.pi
    t = np.arange(1024)
    x = np.sin(10 * t) + np.random.rand(t.shape[-1])
    y = np.sin(10 * t) + np.random.rand(t.shape[-1])

    T = ts.TimeSeries(np.vstack([x, y]), sampling_rate=Fs)

    MT = nta.MTCoherenceAnalyzer(T)
    SNR = nta.SNRAnalyzer(T)
    MT_signal = nta.SpectralAnalyzer(ts.TimeSeries(np.mean(T.data, 0),
                                                   sampling_rate=Fs))

    npt.assert_equal(SNR.mt_frequencies, MT.frequencies)
    npt.assert_equal(SNR.signal, np.mean(T.data, 0))
    f, c = MT_signal.spectrum_multi_taper
    npt.assert_almost_equal(SNR.mt_signal_psd, c)
