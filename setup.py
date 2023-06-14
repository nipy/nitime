#!/usr/bin/env python
"""Setup file for the Python nitime package."""
from setuptools import setup

try:
    from setuptools import Extension
    from Cython.Build import cythonize
    from numpy import get_include

    # add Cython extensions to the setup options
    exts = [
        Extension(
            'nitime._utils',
            ['nitime/_utils.pyx'],
            include_dirs=[get_include()],
        )
    ]
    opts = {'ext_modules': cythonize(exts, language_level='3')}
except ImportError:
    # no loop for you!
    opts = {}

# Now call the actual setup function
if __name__ == '__main__':
    setup(**opts)
