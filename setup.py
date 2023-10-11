#!/usr/bin/env python
"""Setup file for the Python nitime package.

This file only contains cython components.
See pyproject.toml for the remaining configuration.
"""
from setuptools import setup, Extension
from wheel.bdist_wheel import bdist_wheel
from Cython.Build import cythonize
from numpy import get_include


class bdist_wheel_abi3(bdist_wheel):
    def get_tag(self):
        python, abi, plat = super().get_tag()

        # We support back to cp38
        if python.startswith('cp3'):
            python, abi = 'cp38', 'abi3'

        return python, abi, plat

setup(
    ext_modules=[
        Extension(
            'nitime._utils',
            ['nitime/_utils.pyx'],
            include_dirs=[get_include()],
            define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
            py_limited_api=True,
        )
    ],
    cmdclass={'bdist_wheel': bdist_wheel_abi3},
)
