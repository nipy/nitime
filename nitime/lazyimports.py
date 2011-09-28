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
class LazyImport(object):
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
    def __init__(self, modname):
        self.__lazyname__= modname
    def __getattribute__(self,x):
        # This method will be called only once
        name = object.__getattribute__(self,'__lazyname__')
        module =__import__(name, fromlist=name.split('.'))
        # Now that we've done the import, cutout the middleman
        class LoadedLazyImport(object): 
            __getattribute__ = module.__getattribute__
            __repr__ = module.__repr__
        object.__setattr__(self,'__class__', LoadedLazyImport)
        return module.__getattribute__(x)
    def __repr__(self):
        return "<module '%s' will be lazily loaded>" %\
                object.__getattribute__(self,'__lazyname__')

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
