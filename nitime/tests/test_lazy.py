import sys
import numpy as np
import numpy.testing as npt

import pytest

import nitime.lazyimports as l

# The next test requires nitime.lazyimports.disable_lazy_imports to have been
# set to false, otherwise the lazy import machinery is disabled and all imports
# happen at l.LazyImport calls which become equivalent to regular import
# statements
@pytest.mark.skipif(l.disable_lazy_imports, reason="Lazy imports disabled")
def test_lazy():
    mlab = l.LazyImport('matplotlib.mlab')
    # repr for mlab should be <module 'matplotlib.mlab' will be lazily loaded>
    assert 'lazily loaded' in repr(mlab)
    # accessing mlab's attribute will cause an import of mlab
    npt.assert_(np.all(mlab.detrend_mean(np.array([1, 2, 3]))  ==
                       np.array([-1., 0., 1.])))
    # now mlab should be of class LoadedLazyImport an repr(mlab) should be
    # <module 'matplotlib.mlab' from # '.../matplotlib/mlab.pyc>
    assert 'lazily loaded' not in repr(mlab)

# A known limitation of our lazy loading implementation is that, when it it is
# enabled, reloading the module raises an ImportError, and it also does not
# actually perform a reload, as demonstrated by this test.
@pytest.mark.skipif(l.disable_lazy_imports, reason="Lazy imports disabled")
def test_lazy_noreload():
    "Reloading of lazy modules causes ImportError"
    mod = l.LazyImport('sys')
    # accessing module dictionary will trigger an import
    len(mod.__dict__)
    # do not use named tuple feature for Python 2.6 compatibility
    major, minor = sys.version_info[:2]
    if major == 2:
        with pytest.raises(ImportError) as e_info:
            reload(mod)
    elif major == 3:
        import imp
        with pytest.raises(ImportError) as e_info:
            imp.reload(mod)
