"""Utilities to facilitate the writing of tests for nitime.
"""
#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------
# Functions and classes
#-----------------------------------------------------------------------------

def import_nose():
    """
    Import nose only when needed.
    """
    fine_nose = True
    minimum_nose_version = (0,10,0)
    try:
        import nose
        from nose.tools import raises
    except ImportError:
        fine_nose = False
    else:
        if nose.__versioninfo__ < minimum_nose_version:
            fine_nose = False

    if not fine_nose:
        msg = 'Need nose >= %d.%d.%d for tests - see ' \
              'http://somethingaboutorange.com/mrl/projects/nose' % \
              minimum_nose_version

        raise ImportError(msg)

    return nose

def fpw_opt_str():
    """
    Return first-package-wins option string for this version of nose

    Versions of nose prior to 1.1.0 needed ``=True`` for ``first-package-wins``,
    versions after won't accept it.

    changeset: 816:c344a4552d76
    http://code.google.com/p/python-nose/issues/detail?id=293

    Returns
    -------
    fpw_str : str
    Either '--first-package-wins' or '--first-package-wins=True' depending
    on the nose version we are running.
    """
    # protect nose import to provide comprehensible error if missing
    nose = import_nose()
    config = nose.config.Config()
    fpw_str = '--first-package-wins'
    opt_parser = config.getParser('')
    opt_def = opt_parser.get_option('--first-package-wins')
    if opt_def is None:
        raise RuntimeError('Nose does not accept "first-package-wins"'
                           ' - is this an old nose version?')
    if opt_def.takes_value(): # the =True variant
        fpw_str += '=True'
    return fpw_str


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
    from numpy.testing import noseclasses
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
        argv.append(fpw_opt_str())
            
    if doctests:
        argv.append('--with-doctest')
    plugins = [noseclasses.KnownFailure()]
    # Now nose can run
    return noseclasses.NumpyTestProgram(argv=argv, exit=False,
            addplugins=plugins).result

# Tell nose that the test() function itself isn't a test, otherwise we get a
# recursive loop inside nose.
test.__test__ = False
