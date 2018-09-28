from setuptools import setup, find_packages
import sys
import os.path

import numpy as np
import colorspacious as cs

DESC = ("A powerful, accurate, and easy-to-use Python library for "
        "doing colorspace conversions")

import codecs
# LONG_DESC = codecs.open("README.rst", encoding="utf-8").read()

# defines __version__
exec(open("vispol/version.py").read())

setup(
    name="vispol",
    version=__version__,
    description=DESC,
    # long_description=LONG_DESC,
    author="Andrew W Kruse",
    author_email="a.kruse@student.adfa.edu.au",
    url="",
    license="",
    classifiers =
      [ "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        ],
    packages=find_packages(),
    install_requires=["numpy", "colorspacious", "scipy", "matplotlib"],
)
