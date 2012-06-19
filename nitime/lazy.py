"""
Commonly used nitime lazy imports are defined here, so they can be reused
throughout nitime. For an explanation of why we use lazily-loaded modules, and
how you can leverage this machinery in your code, see
:mod:`nitime.lazyimports`.

Lazily-loaded package have almost the same name as the
corresponding package's import string, except for periods are replaced with
underscores. For example, the way to lazily import ``matplotlib.mlab`` is via

    >>> from nitime.lazy import matplotlib_mlab as mlab

At this time, all lazy-loaded packages are defined manually. I (pi) made
several attempts to automate this process, such that any arbitrary package
``foo.bar.baz`` could be imported via ``from nitime.lazy import foo_bar_baz as
baz`` but had limited success.

Currently defined lazy imported packages are (remember to replace the ``.``
with ``_``) ::

    matplotlib.mlab
    scipy
    scipy.fftpack
    scipy.interpolate
    scipy.linalg
    scipy.signal
    scipy.signal.signaltools
    scipy.stats
    scipy.stats.distributions


If you want to lazily load another package in nitime, please add it to this
file, and then ``from nitime.lazy import your_new_package``.

If there's a package that you would like to lazily load in your own code that
is not listed here, use the :class:`LazyImport` class, which is in
:mod:`nitime.lazyimports`.
"""
from .lazyimports import LazyImport

# matplotlib
matplotlib_mlab = LazyImport('matplotlib.mlab')

# scipy
scipy = LazyImport('scipy')
scipy_fftpack = LazyImport('scipy.fftpack')
scipy_interpolate = LazyImport('scipy.interpolate')
scipy_linalg = LazyImport('scipy.linalg')
scipy_signal = LazyImport('scipy.signal')
scipy_signal_signaltools = LazyImport('scipy.signal.signaltools')
scipy_stats = LazyImport('scipy.stats')
scipy_stats_distributions = LazyImport('scipy.stats.distributions')

def enabled():
    "Returns ``True`` if LazyImports are globally enabled"
    import nitime.lazyimports as l
    return not l.disable_lazy_imports
