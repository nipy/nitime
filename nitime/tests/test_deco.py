"""Tests for IPyhton's test support utilities.

This file can be run as a script, and it will call unittest.main().  We must
check that it works with unittest as well as with nose...
"""

from decotest import (as_unittest, ipdoctest, ipdocstring,
                        ParametricTestCase, parametric)

@as_unittest
def trivial():
    """A trivial test"""
    pass

## @ipdoctest
## def simple_dt():
##     """
##     >>> print 1+1
##     2
##     """

## @ipdoctest
## def ipdt_flush():
##     """
## In [20]: print 1
## 1

## In [26]: for i in range(10):
##    ....:     print i,
##    ....:     
##    ....:     
## 0 1 2 3 4 5 6 7 8 9

## In [27]: 3+4
## Out[27]: 7
## """

    
## @ipdoctest
## def ipdt_indented_test():
##     """
##     In [20]: print 1
##     1

##     In [26]: for i in range(10):
##        ....:     print i,
##        ....:     
##        ....:     
##     0 1 2 3 4 5 6 7 8 9

##     In [27]: 3+4
##     Out[27]: 7
##     """


## class Foo(object):
##     """For methods, the normal decorator doesn't work.

##     But rewriting the docstring with ip2py does, *but only if using nose
##     --with-doctest*.  Do we want to have that as a dependency?
##     """

##     @ipdocstring
##     def ipdt_method(self):
##         """
##     In [20]: print 1
##     3

##     In [26]: for i in range(10):
##        ....:     print i,
##        ....:     
##        ....:     
##     0 1 2 3 4 5 6 7 8 9

##     In [27]: 3+4
##     Out[27]: 7
##     """

##     def normaldt_method(self):
##         """
##         >>> print 1+1
##         2
##         """
    
## #-----------------------------------------------------------------------------
## # Broken tests - comment out decorators to see them, these are useful for
## # debugging and understanding how certain things work.
## #-----------------------------------------------------------------------------

## #@as_unittest
## def broken():
##     x, y = 1, 0
##     x/y


## #@ipdoctest
## def broken_dt():
##     """
##     #>>> print 1+1
##     2

##     #>>> print 1+1
##     3

##     #>>> print 1+2
##     4
##     """


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
