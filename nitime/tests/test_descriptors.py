"""Tests for the descriptors module.
"""
#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

import numpy.testing as nt

from nitime import descriptors as desc

#-----------------------------------------------------------------------------
# Support classes and functions
#-----------------------------------------------------------------------------


class A(desc.ResetMixin):
    @desc.auto_attr
    def y(self):
        return self.x / 2.0

    @desc.auto_attr
    def z(self):
        return self.x / 3.0

    def __init__(self, x=1.0):
        self.x = x

#-----------------------------------------------------------------------------
# Test functions
#-----------------------------------------------------------------------------


def test():
    a = A(10)
    nt.assert_('y' not in a.__dict__)
    nt.assert_(a.y == 5)
    nt.assert_('y' in a.__dict__)
    a.x = 20
    nt.assert_(a.y == 5)
    # Call reset and no error should be raised even though z was never accessed
    a.reset()
    nt.assert_(a.y == 10)
