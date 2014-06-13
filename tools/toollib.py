"""Various utilities common to IPython release and maintenance tools.
"""
# Library imports
import os
import sys
import compileall

from subprocess import Popen, PIPE, CalledProcessError, check_call

from distutils.dir_util import remove_tree

# Useful shorthands
pjoin = os.path.join
cd = os.chdir

# Utility functions

#-----------------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------------
def sh(cmd):
    """Execute command in a subshell, return status code."""
    return check_call(cmd, shell=True)


def compile_tree():
    """Compile all Python files below current directory."""
    vstr = '.'.join(map(str, sys.version_info[:2]))
    ca = compileall.__file__
    stat = os.system('python %s .' % ca)
    if stat:
        msg = '*** ERROR: Some Python files in tree do NOT compile! ***\n'
        msg += 'See messages above for the actual file that produced it.\n'
        raise SystemExit(msg)
