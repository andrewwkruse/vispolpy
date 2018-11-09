from setuptools import setup, find_packages
import sys
import os.path

import numpy as np
import colorspacious as cs

DESC = ("Implementation of polarization visualizaton method defined by Kruse et al 2018")

import codecs
# LONG_DESC = codecs.open("README.rst", encoding="utf-8").read()

# defines __version__
exec(open("vispol/version.py").read())

setup(
    name="vispol",
    version=__version__,
    description=DESC,
    author="Andrew W Kruse",
    author_email="a.kruse@student.adfa.edu.au",
    url="",
    license="MIT",
    classifiers =
        ["Intended Audience :: Science/Research",
         "Programming Language :: Python :: 2",
         "Programming Language :: Python :: 3"
        ],
    packages=find_packages(),
    install_requires=["numpy", "colorspacious", "scipy", "matplotlib"],
)
