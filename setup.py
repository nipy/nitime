#!/usr/bin/env python
"""Setup file for the Python nitime package.

This file only contains cython components.
See pyproject.toml for the remaining configuration.
"""
import platform
import sys

from Cython.Build import cythonize
from numpy import get_include
from setuptools import setup, Extension
from wheel.bdist_wheel import bdist_wheel

# add Cython extensions to the setup options


# https://github.com/joerick/python-abi3-package-sample/blob/main/setup.py
class bdist_wheel_abi3(bdist_wheel):  # noqa: D101
    def get_tag(self):  # noqa: D102
        python, abi, plat = super().get_tag()

        if python.startswith("cp"):
            return "cp311", "abi3", plat

        return python, abi, plat


macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
ext_kwargs = {}
setup_kwargs = {}
if sys.version_info.minor >= 11 and platform.python_implementation() == "CPython":
    # Can create an abi3 wheel (typed memoryviews first available in 3.11)!
    macros.append(("Py_LIMITED_API", "0x030B0000"))
    ext_kwargs["py_limited_api"] = True
    setup_kwargs["cmdclass"] = {"bdist_wheel": bdist_wheel_abi3}


exts = [
    Extension(
        'nitime._utils',
        ['nitime/_utils.pyx'],
        include_dirs=[get_include()],
        define_macros=macros,
        **ext_kwargs,
    )
]
opts = {'ext_modules': cythonize(exts, language_level='3'), **setup_kwargs}

# Now call the actual setup function
if __name__ == '__main__':
    setup(**opts)
