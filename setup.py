# -*- coding: utf-8 -*-

import io
import os
import re

from setuptools import find_packages
from setuptools import setup


# obtain version string from __init__.py
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'bpmodels', '__init__.py'), 'r') as f:
    init_py = f.read()
version = re.search('__version__ = "(.*)"', init_py).groups()[0]

# obtain long description from README and CHANGES
with io.open(os.path.join(here, 'README.md'), 'r', encoding='utf-8') as f:
    README = f.read()

# setup
setup(
    name='bpmodels',
    version=version,
    description='BrainPy-Models: An example package accompany with BrainPy.',
    long_description=README,
    author='Pku-Nip-Lab',
    author_email='adaduo@outlook.com',
    packages=find_packages(),
    python_requires='>=3.5',
    install_requires=[
        'Brain.Py',
    ],
    url='https://github.com/PKU-NIP-Lab/BrainPy-Models',
    keywords='computational neuroscience',
    classifiers=[
          'Intended Audience :: Science/Research',
          'Natural Language :: English',
          'Operating System :: OS Independent',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Bio-Informatics'
      ]
)
