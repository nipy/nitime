import numpy as np

from nitime import descriptors as desc
from nitime import timeseries as ts
from nitime import algorithms as tsa

# To support older versions of numpy that don't have tril_indices:
from nitime.index_utils import tril_indices

from .base import BaseAnalyzer


class CorrelationAnalyzer(BaseAnalyzer):
    """Analyzer object for correlation analysis. Has the same API as the
    CoherenceAnalyzer"""

    def __init__(self, input=None):
        """
        Parameters
        ----------

        input : TimeSeries object
           Containing the data to analyze.

        Examples
        --------
        >>> np.set_printoptions(precision=4)  # for doctesting
        >>> t1 = ts.TimeSeries(data = np.sin(np.arange(0,
        ...                    10*np.pi,10*np.pi/100)).reshape(2,50),
        ...                                      sampling_rate=np.pi)
        >>> c1 = CorrelationAnalyzer(t1)
        >>> c1 = CorrelationAnalyzer(t1)
        >>> c1.corrcoef
        array([[ 1., -1.],
               [-1.,  1.]])
        >>> c1.xcorr.sampling_rate  # doctest: +ELLIPSIS
        3.141592653... Hz
        >>> c1.xcorr.t0  # doctest: +ELLIPSIS
        -15.91549430915... s

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

        TimeSeries : the time-dependent cross-correlation, with zero-lag
        at time=0

        """
        tseries_length = self.input.data.shape[0]
        t_points = self.input.data.shape[-1]
        xcorr = np.zeros((tseries_length,
                          tseries_length,
                          t_points * 2 - 1))
        data = self.input.data
        for i in range(tseries_length):
            data_i = data[i]
            for j in range(i, tseries_length):
                xcorr[i, j] = np.correlate(data_i,
                                          data[j],
                                          mode='full')

        idx = tril_indices(tseries_length, -1)
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

        TimeSeries : A TimeSeries object
            the time-dependent cross-correlation, with zero-lag at time=0

        """

        tseries_length = self.input.data.shape[0]
        t_points = self.input.data.shape[-1]
        xcorr = np.zeros((tseries_length,
                          tseries_length,
                          t_points * 2 - 1))
        data = self.input.data
        for i in range(tseries_length):
            data_i = data[i]
            for j in range(i, tseries_length):
                xcorr[i, j] = np.correlate(data_i,
                                          data[j],
                                          mode='full')
                xcorr[i, j] /= (xcorr[i, j, t_points])
                xcorr[i, j] *= self.corrcoef[i, j]

        idx = tril_indices(tseries_length, -1)
        xcorr[idx[0], idx[1], ...] = xcorr[idx[1], idx[0], ...]

        return ts.TimeSeries(xcorr,
                             sampling_interval=self.input.sampling_interval,
                             t0=-self.input.sampling_interval * t_points)


class SeedCorrelationAnalyzer(object):
    """
    This analyzer takes two time-series. The first is designated as a
    time-series of seeds. The other is designated as a time-series of targets.
    The analyzer performs a correlation analysis between each of the channels
    in the seed time-series and *all* of the channels in the target
    time-series.

    """
    def __init__(self, seed_time_series=None, target_time_series=None):
        """
        Parameters
        ----------

        seed_time_series : a TimeSeries object

        target_time_series : a TimeSeries object

        """
        self.seed = seed_time_series
        self.target = target_time_series

    @desc.setattr_on_read
    def corrcoef(self):

        #If there is more than one channel in the seed time-series:
        if len(self.seed.shape) > 1:

            # Preallocate results
            Cxy = np.empty((self.seed.data.shape[0],
                            self.target.data.shape[0]), dtype=np.float)

            for seed_idx, this_seed in enumerate(self.seed.data):

                Cxy[seed_idx] = tsa.seed_corrcoef(this_seed, self.target.data)

        #In the case where there is only one channel in the seed time-series:
        else:
            Cxy = tsa.seed_corrcoef(self.seed.data, self.target.data)

        return Cxy.squeeze()
