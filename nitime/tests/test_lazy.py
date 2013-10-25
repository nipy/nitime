import sys
import os
import nitime.lazyimports as l
import numpy.testing as npt
import numpy.testing.decorators as dec

# The next test requires nitime.lazyimports.disable_lazy_imports to have been
# set to false, otherwise the lazy import machinery is disabled and all imports
# happen at l.LazyImport calls which become equivalent to regular import
# statements
@dec.skipif(l.disable_lazy_imports)
def test_lazy():
    mlab = l.LazyImport('matplotlib.mlab')
    # repr for mlab should be <module 'matplotlib.mlab' will be lazily loaded>
    assert 'lazily loaded' in repr(mlab)
    # accessing mlab's attribute will cause an import of mlab
    npt.assert_equal(mlab.dist(1969,2011), 42.0)
    # now mlab should be of class LoadedLazyImport an repr(mlab) should be
    # <module 'matplotlib.mlab' from # '.../matplotlib/mlab.pyc>
    assert 'lazily loaded' not in repr(mlab)

# A known limitation of our lazy loading implementation is that, when it it is
# enabled, reloading the module raises an ImportError, and it also does not
# actually perform a reload, as demonstrated by this test.
@dec.skipif(l.disable_lazy_imports)
def test_lazy_noreload():
    "Reloading of lazy modules causes ImportError"
    mod = l.LazyImport('sys')
    # accessing module dictionary will trigger an import
    len(mod.__dict__)
    if sys.version_info.major == 2:
        npt.assert_raises(ImportError, reload, mod)
    elif sys.version_info.major == 3:
        import imp
        if sys.version_info.minor == 2:
            npt.assert_raises(ImportError, imp.reload, mod)
        elif sys.version_info.minor == 3:
            npt.assert_raises(TypeError, imp.reload, mod)
