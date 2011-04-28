import numpy as np

import nitime.timeseries as ts
from nitime import descriptors as desc
from nitime import utils as tsu
from nitime import timeseries as ts

from .base import BaseAnalyzer


class CorrelationAnalyzer(BaseAnalyzer):
    """Analyzer object for correlation analysis. Has the same API as the
    CoherenceAnalyzer"""

    def __init__(self, input=None):
        """
        Parameters
        ----------

        input: TimeSeries object
           Containing the data to analyze.

        Examples
        --------
        >>> t1 = ts.TimeSeries(data = np.sin(np.arange(0,
        ...                    10*np.pi,10*np.pi/100)).reshape(2,50),
        ...                                      sampling_rate=np.pi)
        >>> c1 = CorrelationAnalyzer(t1)
        >>> c1 = CorrelationAnalyzer(t1)
        >>> c1.corrcoef
        array([[ 1., -1.],
               [-1.,  1.]])
        >>> c1.xcorr.sampling_rate
        3.1415926536 Hz
        >>> c1.xcorr.t0
        -15.915494309150001 s

        """

        BaseAnalyzer.__init__(self, input)

    @desc.setattr_on_read
    def corrcoef(self):
        """The correlation coefficient between every pairwise combination of
        time-series contained in the object"""
        return np.corrcoef(self.input.data)

    @desc.setattr_on_read
    def xcorr(self):
        """The cross-correlation between every pairwise combination time-series
        in the object. Uses np.correlation('full').

        Returns
        -------

        TimeSeries: the time-dependent cross-correlation, with zero-lag
        at time=0

        """
        tseries_length = self.input.data.shape[0]
        t_points = self.input.data.shape[-1]
        xcorr = np.zeros((tseries_length,
                          tseries_length,
                          t_points * 2 - 1))
        data = self.input.data
        for i in xrange(tseries_length):
            data_i = data[i]
            for j in xrange(i, tseries_length):
                xcorr[i, j] = np.correlate(data_i,
                                          data[j],
                                          mode='full')

        idx = tsu.tril_indices(tseries_length, -1)
        xcorr[idx[0], idx[1], ...] = xcorr[idx[1], idx[0], ...]

        return ts.TimeSeries(xcorr,
                             sampling_interval=self.input.sampling_interval,
                             t0=-self.input.sampling_interval * t_points)

    @desc.setattr_on_read
    def xcorr_norm(self):
        """The cross-correlation between every pairwise combination time-series
        in the object, where the zero lag correlation is normalized to be equal
        to the correlation coefficient between the time-series

        Returns
        -------

        TimeSeries: A TimeSeries object
            the time-dependent cross-correlation, with zero-lag at time=0

        """

        tseries_length = self.input.data.shape[0]
        t_points = self.input.data.shape[-1]
        xcorr = np.zeros((tseries_length,
                          tseries_length,
                          t_points * 2 - 1))
        data = self.input.data
        for i in xrange(tseries_length):
            data_i = data[i]
            for j in xrange(i, tseries_length):
                xcorr[i, j] = np.correlate(data_i,
                                          data[j],
                                          mode='full')
                xcorr[i, j] /= (xcorr[i, j, t_points])
                xcorr[i, j] *= self.corrcoef[i, j]

        idx = tsu.tril_indices(tseries_length, -1)
        xcorr[idx[0], idx[1], ...] = xcorr[idx[1], idx[0], ...]

        return ts.TimeSeries(xcorr,
                             sampling_interval=self.input.sampling_interval,
                             t0=-self.input.sampling_interval * t_points)
