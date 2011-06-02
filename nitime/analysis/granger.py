"""

Analyzers for the calculation of Granger 'causality'

"""

import numpy as np
import nitime.algorithms as alg
import nitime.utils as utils

from .base import BaseAnalyzer


def fit_model(x1, x2, max_order=10,
              criterion=utils.bayesian_information_criterion):
    """
    Find the model order of an autoregressive

    Parameters
    ----------

    x1,x2: float arrays (n)
        x1,x2 bivariate combination.
    max_order: int
        The maximal order to fit.
    criterion: callable
       A function which defines an information criterion.

    """

    c_old = np.inf
    n_process = 2
    Ntotal = n_process * x1.shape[-1]
    autocov_vector = []

    c_x1x1 = np.correlate(x1, x1.conj(), mode='full')
    c_x2x2 = np.correlate(x2, x2.conj(), mode='full')
    c_x1x2 = np.correlate(x1, x2.conj(), mode='full')
    c_x2x1 = np.correlate(x2, x1.conj(), mode='full')

    for lag in xrange(max_order):
        idx = x1.shape[0]/2 + lag
        autocov_vector.append([[c_x1x1[idx], c_x1x2[idx]],
                               [c_x2x1[idx], c_x2x2[idx]]])
        Rxx = np.array(autocov_vector)
        coef, ecov = alg.lwr_recursion(Rxx)
        c_new = criterion(ecov, n_process, lag, Ntotal)
        if c_new > c_old:
            break
        else:
            c_old = c_new
    else:
        e_s = "Model estimation order did not converge at max_order=%s" % max_order
        raise ValueError(e_s)

    return lag, Rxx, coef, ecov

## class GrangerAnalyzer(BaseAnalyzer):
##     """Analyzer for computing all-to-all Granger 'causality' """
##     def __init__(self, input=None, order=None, n_freqs=1024):
##         """
##         Initializer for the GrangerAnalyzer.

##         Parameters
##         ----------

##         input: nitime TimeSeries object

##         order: int (optional)
##              The order of the process. If this is not known, it will be
##              estimated from the data, using the AIC
##         n_freqs: int (optional)
##             The size of the sampling grid in the frequency domain. Defaults to 1024


##         """
##         self._n_process = input.shape[0]
##         self._n_freqs = n_freqs
##         self._order = order

##     @desc.setattr_on_read
##     def order(self):
##         if self._order is None:
##             self.autocov_vector()
##             return fit_order()
##         else:
##             return self._order


##     @desc.setattr_on_read
##     def autocov_vector(self):
##         """


##         """
##         Rxx = np.zeros((self._n_process,
##                         self._n_process,
##                         self.order), dtype=complex)

##         for i in xarange(self._n_process):
##             for j in xrange(i, self._n_process):
##                 Rxx[i, j] = autocov_vector(x, nlags=self.order)

##     @desc.setattr_on_read
##     def coefs(self):
##         coefs = np.zeros((self._n_process,
##                           self._n_process,
##                           self.order), dtype=complex)

##         ecov = np.zeros((self._n_process,
##                          self._n_process,
##                          self.order), dtype=complex)

##         for i in xrange(self._n_process):
##             for j in xrange(i, self._n_process):
##                 # Estimate the autorgressive process using LWR recursion:
##                 coefs[i, j], ecov[i, j] = alg.lwr_recursion(Rxx)
##     def granger_caus
##     w, f_x2y, f_y2x, f_xy, Sw = alg.granger_causality_xy(self.coefs,
##                                                          self.ecov,
##                                                          n_freqs=n_freqs)

##     @desc.setattr_on_read
##     def causality_xy(self):

##     @desc.setattr_on_read
##     def causality_yx(self):

##     @desc.setattr_on_read
##     def simultaneous_causality(self):

##     @desc.setattr_on_read
##     def order(self):
##         if self.order is None:
##             # Calculate the order with BIC:
