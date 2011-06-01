"""

Analyzers for the calculation of Granger 'causality'

"""


import numpy as np
import nitime.algorithms as alg
import nitime.utils as utils

from .base import BaseAnalyzer


class GrangerAnalyzer(BaseAnalyzer):
    """Analyzer for computing all-to-all Granger 'causality' """
    def __init__(self, input=None, order=None, n_freqs=1024):
        """
        Initializer for the GrangerAnalyzer.

        Parameters
        ----------

        input: nitime TimeSeries object

        order: int (optional)
             The order of the process. If this is not known, it will be
             estimated from the data, using the AIC
        n_freqs: int (optional)
            The size of the sampling grid in the frequency domain. Defaults to 1024


        """
        self._n_process = input.shape[0]
        self._n_freqs = n_freqs
        self.order = order

    @desc.setattr_on_read
    def autocov_vector(self):
        """


        """
        Rxx = np.zeros((self._n_process,
                        self._n_process,
                        self.order), dtype=complex)

        for i in xarange(self._n_process):
            for j in xrange(i, self._n_process):
                Rxx[i, j] = autocov_vector(x, nlags=self.order)

    @desc.setattr_on_read
    def coefs(self):
        coefs = np.zeros((self._n_process,
                          self._n_process,
                          self.order), dtype=complex)

        ecov = np.zeros((self._n_process,
                         self._n_process,
                         self.order), dtype=complex)

        for i in xrange(self._n_process):
            for j in xrange(i, self._n_process):
                # Estimate the autorgressive process using LWR recursion:
                coefs[i, j], ecov[i, j] = alg.lwr_recursion(Rxx)
    def granger_caus
    w, f_x2y, f_y2x, f_xy, Sw = alg.granger_causality_xy(self.coefs,
                                                         self.ecov,
                                                         n_freqs=n_freqs)

    @desc.setattr_on_read
    def causality_xy(self):

    @desc.setattr_on_read
    def causality_yx(self):

    @desc.setattr_on_read
    def simultaneous_causality(self):

    @desc.setattr_on_read
    def order(self):
        if self.order is None:
            # Calculate the order with BIC:
