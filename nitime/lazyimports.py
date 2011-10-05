""" This module provides lazy import functionality to improve the import
performance of nitime.

A generic LazyImport class is implemented which takes the module name as a
parameter, and acts as a proxy for that module, importing it only when the
module is used, but effectively acting as the module in every other way
(including inside IPython with respect to introspection and tab completion)
with the *exception* of reload().

Commonly used nitime lazy imports are also defined here, so they can be reused
throughout nitime.

>>> from nitime.lazyimports import mlab # lazy import for matplotlib.mlab

"""
import sys
import types

disable_lazy_imports = False

class LazyImport(types.ModuleType):
    """
    This class takes the module name as a parameter, and acts as a proxy for
    that module, importing it only when the module is used, but effectively
    acting as the module in every other way (including inside IPython with
    respect to introspection and tab completion) with the *exception* of
    reload().

    >>> mlab = LazyImport('matplotlib.mlab')

    No import happens on the above line, until we do something like call an
    mlab method or try to do tab completion or introspection on mlab in IPython.
    
    >>> mlab
    <module 'matplotlib.mlab' will be lazily loaded>
    
    Now the LazyImport will do an actual import, and call the dist function of
    the imported module.
    
    >>> mlab.dist(1969,2011)
    42.0
    >>> mlab
    <module 'matplotlib.mlab' from '.../site-packages/matplotlib/mlab.pyc'>
    """
    def __getattribute__(self,x):
        # This method will be called only once
        name = object.__getattribute__(self,'__name__')
        __import__(name)
        # if name above has package.foo.bar, package is returned, the docs
        # recommend that in order to get back the full thing, that we import
        # and then lookup the full name is sys.modules
        module = sys.modules[name]
        # Now that we've done the import, cutout the middleman and make self
        # into a regular module
        class LoadedLazyImport(types.ModuleType):
            __getattribute__ = module.__getattribute__
            __repr__ = module.__repr__
        object.__setattr__(self,'__class__', LoadedLazyImport)
        sys.modules[name] = self
        return module.__getattribute__(x)
    def __repr__(self):
        return "<module '%s' will be lazily loaded>" %\
                object.__getattribute__(self,'__name__')

if disable_lazy_imports:
    LazyImport = lambda x: __import__(x) and sys.modules[x]

# matplotlib.mlab
mlab = LazyImport('matplotlib.mlab')

# scipy
scipy = LazyImport('scipy')
stats = LazyImport('scipy.stats')
linalg = LazyImport('scipy.linalg')
signal = LazyImport('scipy.signal')
signaltools = LazyImport('scipy.signal.signaltools')
fftpack = LazyImport('scipy.fftpack')
interpolate = LazyImport('scipy.interpolate')
distributions = LazyImport('scipy.stats.distributions')

# numpy.testing
noseclasses = LazyImport('numpy.testing.noseclasses')
