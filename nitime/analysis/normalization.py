from nitime import descriptors as desc
from nitime import utils as tsu
from nitime import timeseries as ts

from .base import BaseAnalyzer


class NormalizationAnalyzer(BaseAnalyzer):
    """ A class for performing normalization operations on time-series and
    producing the renormalized versions of the time-series"""

    def __init__(self, input=None):
        """Constructor function for the Normalization analyzer class.

        Parameters
        ----------

        input: TimeSeries object

        """
        BaseAnalyzer.__init__(self, input)

    @desc.setattr_on_read
    def percent_change(self):
        return ts.TimeSeries(tsu.percent_change(self.input.data),
                             sampling_rate=self.input.sampling_rate,
                             time_unit=self.input.time_unit)

    @desc.setattr_on_read
    def z_score(self):
        return ts.TimeSeries(tsu.zscore(self.input.data),
                             sampling_rate=self.input.sampling_rate,
                             time_unit=self.input.time_unit)
