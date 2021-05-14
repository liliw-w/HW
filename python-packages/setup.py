# setup file to compile C++ library

from setuptools import setup
import sys, os
import setuptools

this_dir = os.path.dirname(os.path.realpath(__file__))

__version__ = '0.0.0'


setup(
    name='mypack',
    version=__version__,
    author='Brad Nelson',
    author_email='bradnelson@uchicago.edu',
    description='a basic example package',
    url="https://github.com/caam37830/python-packages",
    python_requires='>=3.6',
    packages=['mypack'],
    zip_safe=True,

)
