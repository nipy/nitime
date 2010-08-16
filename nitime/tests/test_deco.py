"""Tests for our own test utilities.

This file can be run as a script, and it will call unittest.main().  We must
check that it works with unittest as well as with nose...
"""

from decotest import as_unittest, ParametricTestCase, parametric

@as_unittest
def trivial():
    """A trivial test"""
    pass

# Some examples of parametric tests.

def is_smaller(i,j):
    assert i<j,"%s !< %s" % (i,j)

class Tester(ParametricTestCase):

    def test_parametric(self):
        yield is_smaller(3, 4)
        x, y = 1, 2
        yield is_smaller(x, y)

@parametric
def test_par_standalone():
    yield is_smaller(3, 4)
    x, y = 1, 2
    yield is_smaller(x, y)


def test_par_nose():
    yield (is_smaller,3, 4)
    x, y = 2, 3
    yield (is_smaller,x, y)


if __name__ == '__main__':
    import unittest
    unittest.main()
