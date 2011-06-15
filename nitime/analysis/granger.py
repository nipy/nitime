"""

Analyzers for the calculation of Granger 'causality'

"""

import numpy as np
import nitime.algorithms as alg
import nitime.utils as utils

from .base import BaseAnalyzer


def fit_model(x1, x2, order=None, max_order=10,
              criterion=utils.bayesian_information_criterion):
    """
    Find the model order of an autoregressive

    Parameters
    ----------

    x1,x2: float arrays (n)
        x1,x2 bivariate combination.
    order: int (optional)
        If known, the order of the autoregressive process
    max_order: int (optional)
        If the order is not known, this will be the maximal order to fit.
    criterion: callable
       A function which defines an information criterion.

    """

    c_old = np.inf
    n_process = 2
    Ntotal = n_process * x1.shape[-1]

    # If model order was provided as an input:
    if order is not None:
        lag = order + 1
        Rxx = utils.autocov_vector(np.vstack([x1,x2]), nlags=lag)
        coef, ecov = alg.lwr_recursion(np.array(Rxx).transpose(2, 0, 1))

    # If the model order is not known and provided as input:
    else:
        for lag in xrange(1, max_order):
            Rxx_new = utils.autocov_vector(np.vstack([x1,x2]), nlags=lag)
            coef_new, ecov_new = alg.lwr_recursion(np.array(Rxx_new).transpose(2, 0, 1))
            order_new = coef_new.shape[0]
            c_new = criterion(ecov_new, n_process, order_new, Ntotal)
            if c_new > c_old:
                # Keep the values you got in the last round and break out:
                break

            else:
                # Replace the output values with the new calculated values and
                # move on to the next order:
                c_old = c_new
                order = order_new
                Rxx = Rxx_new
                coef = coef_new
                ecov = ecov_new
        else:
            e_s = "Model estimation order did not converge at max_order=%s" % max_order
            raise ValueError(e_s)

    return order, Rxx, coef, ecov

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
