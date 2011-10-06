""" This module provides lazy import functionality to improve the import
performance of nitime.

A generic LazyImport class is implemented which takes the module name as a
parameter, and acts as a proxy for that module, importing it only when the
module is used, but effectively acting as the module in every other way
(including inside IPython with respect to introspection and tab completion)
with the *exception* of reload() - reloading a LazyImport has no effect.

Commonly used nitime lazy imports are also defined in lazy.py, so they can be
reused throughout nitime.
"""
import sys
import types

# This flag only has affect on this module's import, and if it is set to True,
# LazyImports are performed immediately. Note: this flag is currently here
# only for debugging purposes and must be set directly in the source code,
# since nitime.lazy imports this module, and nitime.lazy is used throughout
# nitime, importing nitime will import this module.
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
        # This method will be called only once, since we'll change
        # self.__class__ to LoadedLazyImport, and __getattribute__ will point
        # to module.__getattribute__
        name = object.__getattribute__(self,'__name__')
        __import__(name)
        # if name above is 'package.foo.bar', package is returned, the docs
        # recommend that in order to get back the full thing, that we import
        # and then lookup the full name is sys.modules, see:
        # http://docs.python.org/library/functions.html#__import__
        module = sys.modules[name]
        # Now that we've done the import, cutout the middleman and make self
        # act as the imported module
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
