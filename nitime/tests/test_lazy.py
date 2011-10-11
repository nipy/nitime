import sys
import os
import nitime.lazyimports as l
import numpy.testing as npt
import numpy.testing.decorators as dec

# The next test requires nitime.lazyimports.disable_lazy_imports to have been
# set to false, otherwise the lazy import machinery is disabled and all imports
# happen at l.LazyImport calls which become equivalent to regular import
# statements
@dec.knownfailureif(l.disable_lazy_imports,
    "This test fails when disable_lazy_imports is True")
def test_lazy():
    mlab = l.LazyImport('matplotlib.mlab')
    # repr for mlab should be <module 'matplotlib.mlab' will be lazily loaded>
    assert 'lazily loaded' in repr(mlab)
    # accessing mlab's attribute will cause an import of mlab
    npt.assert_equal(mlab.dist(1969,2011), 42.0)
    # now mlab should be of class LoadedLazyImport an repr(mlab) should be
    # <module 'matplotlib.mlab' from # '.../matplotlib/mlab.pyc>
    assert 'lazily loaded' not in repr(mlab)
    reload(mlab) # for lazyimports, this is a no-op, see next test

# A known limitation of our lazy loading implementation is that, when it it is
# enabled, reloading the module does not raise errors, but it also does not
# actually perform a reload, as demonstrated by this test.
@dec.knownfailureif(not l.disable_lazy_imports, "Reloading a module is a silent no-op")
def test_lazy_reload():
    f = file('baz.py', 'w')
    f.write("def foo(): return 42")
    f.close()
    b = l.LazyImport('baz')
    assert b.foo()==42
    f = file('baz.py', 'w')
    f.write("def bar(): return 0x42")
    f.flush()
    os.fsync(f)
    f.close()
    import time
    time.sleep(1)
    os.utime('baz.py', None)
    reload(b)
    os.remove('baz.py')
    assert b.bar()==0x42
