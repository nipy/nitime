"""

Analyzers for the calculation of Granger 'causality'

"""

import numpy as np
import nitime.algorithms as alg
import nitime.utils as utils
from nitime import descriptors as desc

from .base import BaseAnalyzer

# To support older versions of numpy that don't have tril_indices:
from nitime.index_utils import tril_indices_from

def fit_model(x1, x2, order=None, max_order=10,
              criterion=utils.bayesian_information_criterion):
    """
    Fit the auto-regressive model used in calculation of Granger 'causality'.

    Parameters
    ----------

    x1,x2: float arrays (n)
        x1,x2 bivariate combination.
    order: int (optional)
        If known, the order of the autoregressive process
    max_order: int (optional)
        If the order is not known, this will be the maximal order to fit.
    criterion: callable
       A function which defines an information criterion, used to determine the
        order of the model.

    """
    c_old = np.inf
    n_process = 2
    Ntotal = n_process * x1.shape[-1]

    # If model order was provided as an input:
    if order is not None:
        lag = order + 1
        Rxx = utils.autocov_vector(np.vstack([x1, x2]), nlags=lag)
        coef, ecov = alg.lwr_recursion(np.array(Rxx).transpose(2, 0, 1))

    # If the model order is not known and provided as input:
    else:
        for lag in range(1, max_order):
            Rxx_new = utils.autocov_vector(np.vstack([x1, x2]), nlags=lag)
            coef_new, ecov_new = alg.lwr_recursion(
                                        np.array(Rxx_new).transpose(2, 0, 1))
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
            e_s = ("Model estimation order did not converge at max_order = %s"
                                                                  % max_order)
            raise ValueError(e_s)

    return order, Rxx, coef, ecov


class GrangerAnalyzer(BaseAnalyzer):
    """Analyzer for computing all-to-all Granger 'causality' """
    def __init__(self, input=None, ij=None, order=None, max_order=10,
                 criterion=utils.bayesian_information_criterion, n_freqs=1024):
        """
        Initializer for the GrangerAnalyzer.

        Parameters
        ----------

        input: nitime TimeSeries object
        ij: List of tuples of the form: [(0, 1), (0, 2)], etc.
            These are the indices of pairs of time-series for which the
            analysis will be done. Defaults to all vs. all.
        order: int (optional)
             The order of the process. If this is not known, it will be
             estimated from the data, using the information criterion
        max_order: if the order is estimated, this is the maximal order to
             estimate for.
        n_freqs: int (optional)
            The size of the sampling grid in the frequency domain.
            Defaults to 1024
        criterion:
            XXX
        """
        self.data = input.data
        self.sampling_rate = input.sampling_rate
        self._n_process = input.shape[0]
        self._n_freqs = n_freqs
        self._order = order
        self._criterion = criterion
        self._max_order = max_order
        if ij is None:
            # The following gets the full list of combinations of
            # non-same i's and j's:
            x, y = np.meshgrid(np.arange(self._n_process),
                               np.arange(self._n_process))
            self.ij = list(zip(x[tril_indices_from(x, -1)],
                          y[tril_indices_from(y, -1)]))
        else:
            self.ij = ij

    @desc.setattr_on_read
    def _model(self):
        model = dict(order={}, autocov={}, model_coef={}, error_cov={})
        for i, j in self.ij:
            model[i, j] = dict()
            order_t, Rxx_t, coef_t, ecov_t = fit_model(self.data[i],
                                                   self.data[j],
                                                   order=self._order,
                                                   max_order=self._max_order,
                                                   criterion=self._criterion)
            model['order'][i, j] = order_t
            model['autocov'][i, j] = Rxx_t
            model['model_coef'][i, j] = coef_t
            model['error_cov'][i, j] = ecov_t

        return model

    @desc.setattr_on_read
    def order(self):
        if self._order is None:
            return self._model['order']
        else:
            order = {}
            for i, j in self.ij:
                order[i, j] = self._order
            return order

    @desc.setattr_on_read
    def autocov(self):
        return self._model['autocov']

    @desc.setattr_on_read
    def model_coef(self):
        return self._model['model_coef']

    @desc.setattr_on_read
    def error_cov(self):
        return self._model['error_cov']

    @desc.setattr_on_read
    def _granger_causality(self):
        """
        This returns a dict with the values computed by
        :func:`granger_causality_xy`, rather than arrays, so that we can delay
        the allocation of arrays as much as possible.

        """
        gc = dict(frequencies={}, gc_xy={}, gc_yx={}, gc_sim={},
                  spectral_density={})
        for i, j in self.ij:
            w, f_x2y, f_y2x, f_xy, Sw = \
               alg.granger_causality_xy(self.model_coef[i, j],
                                        self.error_cov[i, j],
                                        n_freqs=self._n_freqs)

            # All other measures are dependent on i, j:
            gc['gc_xy'][i, j] = f_x2y
            gc['gc_yx'][i, j] = f_y2x
            gc['gc_sim'][i, j] = f_xy
            gc['spectral_density'][i, j] = Sw

        return gc

    @desc.setattr_on_read
    def frequencies(self):
        return utils.get_freqs(self.sampling_rate, self._n_freqs)

    def _dict2arr(self, key):
        """
        A helper function that will generate an array with all nan's and insert
        the measure defined by 'key' into the array and return it to the
        calling function. This allows us to get matrices of the measures of
        interest, instead of a dict.
        """
        # Prepare the matrix for the output:
        arr = np.empty((self._n_process,
                        self._n_process,
                        self.frequencies.shape[0]))

        arr.fill(np.nan)

        # 'Translate' from dict form into matrix form:
        for i, j in self.ij:
            arr[i, j, :] = self._granger_causality[key][i, j]
        return arr

    @desc.setattr_on_read
    def causality_xy(self):
        return self._dict2arr('gc_xy')

    @desc.setattr_on_read
    def causality_yx(self):
        return self._dict2arr('gc_yx')

    @desc.setattr_on_read
    def simultaneous_causality(self):
        return self._dict2arr('gc_sim')

    @desc.setattr_on_read
    def spectral_matrix(self):
        return self._granger_causality['spectral_density']
