#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 12:18:53 2018

@author: Sven serneels, Ponalytics
"""

from setuptools import setup, find_packages
import re
import sys
import os

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),"./src")
if SRC_DIR not in sys.path:
    sys.path.insert(0,SRC_DIR)
from mvskew import __version__, __author__, __license__

readme_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'README.md')
try:
    from m2r import parse_from_file
    readme = parse_from_file(readme_file)
except ImportError:
    # m2r may not be installed in user environment
    with open(readme_file) as f:
        readme = f.read()

setup(
    name="mvskew",
    version=__version__,
    author=__author__,
    author_email="svenserneels@gmail.com",
    description="Multivariate skew distribution samplers",
    long_description=readme,
    long_description_content_type='text/markdown',
    url="https://github.com/SvenSerneels/mvskew",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages('src'),  # include all packages under src
    package_dir={'':'src'},   # tell distutils packages are under src
    include_package_data = True,
    install_requires=[
        'numpy>=1.5.0',
        'scipy>=0.8.0'
    ]
)

