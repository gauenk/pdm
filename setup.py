#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
from setuptools import setup, find_packages
# from distutils.core import setup
import os
import stat
import shutil
import platform
import sys
import site
import glob


# -- file paths --
long_description="""Code for publication"""
setup(
    name='pdm',
    version='100.100.100',
    description='Palette Image-to-Image Diffusion Augmented to Videos',
    long_description=long_description,
    url='https://github.com/gauenk/pdm',
    author='Kent Gauen',
    author_email='gauenk@purdue.edu',
    license='MIT',
    keywords='neural network',
    install_requires=[],
    package_dir={"": "lib"},
    packages=find_packages("lib"),
)
