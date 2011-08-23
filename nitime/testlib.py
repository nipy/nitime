"""Utilities to facilitate the writing of tests for nitime.
"""
#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

# Third-party
import numpy as np
from numpy.testing.noseclasses import NumpyTestProgram

#-----------------------------------------------------------------------------
# Functions and classes
#-----------------------------------------------------------------------------


def test(doctests=True, first_package_wins=True, extra_argv=None):
    """

    Run the nitime test suite using nose.

    Parameters
    ----------

    doctests: bool, optional
       Whether to run the doctests. Defaults to True

    first_package_wins: bool, optional
       Don't evict packages from sys.module, if detecting another package with
       the same name in some other location(nosetests default behavior is to do
       that).
       
    extra_argv: string, list or tuple, optional
       Additional argument (string) or arguments (list or tuple of strings) to
       be passed to nose when running the tests.

    """
    # Import this internally, so that nose doesn't get pulled into sys.modules,
    # unless you are really running the test-suite.
    from nose.core import TestProgram

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

    # If someone wants to add some other argv
    if extra_argv is not None:
        if isinstance(extra_argv, list) or isinstance(extra_argv, list):
            for this in extra_argv: argv.append(this)
        else:
            argv.append(extra_argv)

    if first_package_wins:
        argv.append('--first-package-wins')
            
    if doctests:
        argv.append('--with-doctest')
        
    # Now nose can run
    return NumpyTestProgram(argv=argv, exit=False)

# Tell nose that the test() function itself isn't a test, otherwise we get a
# recursive loop inside nose.
test.__test__ = False
