import sys
import os
import nitime.lazyimports as l
import numpy.testing as npt
import numpy.testing.decorators as dec

def test_get_time_unit():
    mlab = l.LazyImport('matplotlib.mlab')
    #reload(mlab)
    # repr for mlab should be <module 'matplotlib.mlab' will be lazily loaded>
    print repr(mlab)
    assert 'lazily loaded' in repr(mlab)
    npt.assert_equal(mlab.dist(1969,2011), 42.0)

    print mlab.dist(1969,2011)
    # now mlab should be of class LoadedLazyImport
    print repr(mlab)
    assert 'lazily loaded' not in repr(mlab)
    print [k for k in sys.modules.keys() if 'matplotlib' in k]
    print mlab.__name__ in sys.modules.keys()
    print mlab
    reload(mlab) # for lazyimports, this is a no-op, see next test

@dec.knownfailureif(True, "Reloading a module is a silent no-op")
def test_lazy_reload():
    f = file('baz.py', 'w')
    f.write("def foo(): return 42")
    f.close()
    b = l.LazyImport('baz')
    #import baz as b
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
    assert b.bar()==0x42
    os.remove('baz.py')
