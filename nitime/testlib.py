"""Utilities to facilitate the writing of tests for nitime.
"""
#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

# Third-party
import numpy as np

#-----------------------------------------------------------------------------
# Functions and classes
#-----------------------------------------------------------------------------


def test(doctests=False):
    """

    Run the nitime test suite using nose.

    """
    # Import this internally, so that nose doesn't get pulled into sys.modules,
    # unless you are really running the test-suite.
    from nose.core import TestProgram

    #Make sure that you only change the print options during the testing
    #of nitime and don't affect the user session after that:
    opt_dict = np.get_printoptions()
    np.set_printoptions(precision=4)
    # We construct our own argv manually, so we must set argv[0] ourselves
    argv = ['nosetests',
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
    try:
        TestProgram(argv=argv)#, exit=False)
    finally:
        np.set_printoptions(**opt_dict)

# Tell nose that the test() function itself isn't a test, otherwise we get a
# recursive loop inside nose.
test.__test__ = False
