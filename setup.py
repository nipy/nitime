#!/usr/bin/env python
"""Setup file for the Python nitime package."""

import os
import sys

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

from setuptools import find_packages
from distutils.core import setup

# Get version and release info, which is all stored in nitime/version.py
ver_file = os.path.join('nitime', 'version.py')
with open(ver_file) as f:
    exec(f.read())

PACKAGES = find_packages()


opts = dict(name=NAME,
            maintainer=MAINTAINER,
            maintainer_email=MAINTAINER_EMAIL,
            description=DESCRIPTION,
            long_description=LONG_DESCRIPTION,
            url=URL,
            download_url=DOWNLOAD_URL,
            license=LICENSE,
            classifiers=CLASSIFIERS,
            author=AUTHOR,
            author_email=AUTHOR_EMAIL,
            platforms=PLATFORMS,
            version=VERSION,
            packages=PACKAGES,
            package_data=PACKAGE_DATA,
            install_requires=REQUIRES,
            requires=REQUIRES,
            )

try:
    from distutils.extension import Extension
    from Cython.Distutils import build_ext as build_pyx_ext
    from numpy import get_include
    # add Cython extensions to the setup options
    exts = [Extension('nitime._utils', ['nitime/_utils.pyx'],
                      include_dirs=[get_include()])]
    opts['cmdclass'] = dict(build_ext=build_pyx_ext)
    opts['ext_modules'] = exts
except ImportError:
    # no loop for you!
    pass

# Now call the actual setup function
if __name__ == '__main__':
    setup(**opts)
