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
import nitime.descriptors as desc

class LazyImport(object):
    """
    This class takes the module name as a parameter, and acts as a proxy for
    that module, importing it only when the module is used, but effectively
    acting as the module in every other way (including inside IPython with
    respect to introspection and tab completion) with the *exception* of
    reload().

    >>> mlab = LazyImport('matplotlib.mlab')

    No import happens on the above line, until we do something like call an
    mlab method, or cause mlab to represent itself in some manner, or try to
    do tab completion on mlab in IPython. For example, now the LazyImport will
    do an actual import, and call the __repr__ on the imported module.

    >>> mlab
    <module 'matplotlib.mlab' from '.../site-packages/matplotlib/mlab.pyc'>
    """
    def __init__(self, modname):
        self.__lazyname__= modname
    @desc.auto_attr # one-time property
    def __lazyimported__(self):
        name = object.__getattribute__(self,'__lazyname__')
        return __import__(name, fromlist=name.split('.'))
    def __getattribute__(self,x):
        return object.__getattribute__(self,'__lazyimported__').__getattribute__(x)
    def __repr__(self):
        return object.__getattribute__(self,'__lazyimported__').__repr__()

mlab=LazyImport('matplotlib.mlab')
