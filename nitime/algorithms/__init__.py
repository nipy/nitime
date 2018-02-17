"""
This module contains implementations of algorithms for time series
analysis. These algorithms include:

1. Spectral estimation: calculate the spectra of time-series and cross-spectra
between time-series.

:func:`get_spectra`, :func:`get_spectra_bi`, :func:`periodogram`,
:func:`periodogram_csd`, :func:`dpss_windows`, :func:`multi_taper_psd`,
:func:`multi_taper_csd`, :func:`mtm_cross_spectrum`

2. Coherency: calculate the pairwise correlation between time-series in the
frequency domain and related quantities.

:func:`coherency`, :func:`coherence`, :func:`coherence_regularized`,
:func:`coherency_regularized`, :func:`coherency_bavg`, :func:`coherence_bavg`,
:func:`coherence_partial`, :func:`coherence_partial_bavg`,
:func:`coherency_phase_spectrum`, :func:`coherency_phase_delay`,
:func:`coherency_phase_delay_bavg`, :func:`correlation_spectrum`

3. Cached coherency: A set of special functions for quickly calculating
coherency in large data-sets, where the calculation is done over only a subset
of the adjacency matrix edges and intermediate calculations are cached, in
order to save calculation time.

:func:`cache_fft`, :func:`cache_to_psd`, :func:`cache_to_phase`,
:func:`cache_to_relative_phase`, :func:`cache_to_coherency`.

4. Event-related analysis: calculate the correlation between time-series and
external events.

:func:`freq_domain_xcorr`, :func:`freq_domain_xcorr_zscored`, :func:`fir`

5. Wavelet transforms: Calculate wavelet transforms of time-series data.

:func:`wmorlet`, :func:`wfmorlet_fft`, :func:`wlogmorlet`,
:func:`wlogmorlet_fft`

6. Filtering: Filter a signal in the frequency domain.

:func:`boxcar_filter`

7. Autoregressive estimation and granger causality

:func:

8. Entropy

:func:`entropy`, :func:`conditional_entropy`, func`mutual_information`,
:func:`entropy_cc`, :func:`transfer_entropy`

The algorithms in this library are the functional form of the algorithms, which
accept as inputs numpy array and produce numpy array outputs. Therefore, they
can be used on any type of data which can be represented in numpy arrays. See
also :mod:`nitime.analysis` for simplified analysis interfaces, using the
data containers implemented in :mod:`nitime.timeseries`

"""
from nitime.algorithms.spectral import *
from nitime.algorithms.cohere import *
from nitime.algorithms.wavelet import *
from nitime.algorithms.event_related import *
from nitime.algorithms.autoregressive import *
from nitime.algorithms.filter import *
from nitime.algorithms.correlation import *
from nitime.algorithms.entropy import *
