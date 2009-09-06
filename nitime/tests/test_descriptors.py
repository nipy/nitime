"""Tests for the descriptors module.
"""
#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

import nose.tools as nt

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
    yield (nt.assert_false, 'y' in a.__dict__)
    yield (nt.assert_equals, a.y, 5)
    yield (nt.assert_true, 'y' in a.__dict__)
    a.x = 20
    yield (nt.assert_equals, a.y, 5)
    # Call reset and no error should be raised even though z was never accessed
    a.reset()
    yield (nt.assert_equals, a.y, 10)
