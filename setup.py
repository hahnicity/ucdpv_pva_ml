#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


setup(name='b2caim1',
      version="1.0",
      packages=find_packages(),
      install_requires=[
          'elm',
          'imbalanced-learn',
          'ipython',
          'matplotlib',
          'numpy',
          'pandas',
          'prettytable',
          'scipy',
          'scikit-learn',
          'tabulate',
      ],
      entry_points={
          'console_scripts': [
          ]
      },
      include_package_data=True,
      )
