"""Utilities to facilitate the writing of tests for nitime.
"""
#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

# Stdlib
import sys

# Third-party
import nose
from nose.core import TestProgram

#-----------------------------------------------------------------------------
# Functions and classes
#-----------------------------------------------------------------------------

def test(doctests=False):
    """Run the nitime test suite using nose.
    """
    # We construct our own argv manually, so we must set argv[0] ourselves
    argv = [ 'nosetests',
             # Name the package to actually test, in this case nitime
             'nitime',
             
             # extra info in tracebacks
             '--detailed-errors',

             # We add --exe because of setuptools' imbecility (it blindly does
             # chmod +x on ALL files).  Nose does the right thing and it tries
             # to avoid executables, setuptools unfortunately forces our hand
             # here.  This has been discussed on the distutils list and the
             # setuptools devs refuse to fix this problem!
             '--exe',
             ]

    if doctests:
        argv.append('--with-doctest')

    # Now nose can run
    TestProgram(argv=argv, exit=False)


# Tell nose that the test() function itself isn't a test, otherwise we get a
# recursive loop inside nose.
test.__test__ = False
