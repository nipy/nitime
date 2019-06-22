
from inspect import getfullargspec

from nitime import descriptors as desc


class BaseAnalyzer(desc.ResetMixin):
    """
    Analyzer that implements the default data flow.

    All analyzers inherit from this class at least have to
    * implement a __init__ function to set parameters
    * define the 'output' property

    """

    @desc.setattr_on_read
    def parameterlist(self):
        plist = getfullargspec(self.__init__).args
        plist.remove('self')
        plist.remove('input')
        return plist

    @property
    def parameters(self):
        return dict([(p,
                    getattr(self, p, 'MISSING')) for p in self.parameterlist])

    def __init__(self, input=None):
        self.input = input

    def set_input(self, input):
        """Set the input of the analyzer, if you want to reuse the analyzer
        with a different input than the original """

        self.reset()
        self.input = input

    def __repr__(self):
        params = ', '.join(['%s=%r' % (p, getattr(self, p, 'MISSING'))
                            for p in self.parameterlist])

        return '%s(%s)' % (self.__class__.__name__, params)
