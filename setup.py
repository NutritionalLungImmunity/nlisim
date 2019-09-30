#!/usr/bin/env python

from setuptools import setup

setup(
    name='simulation',
    packages=['simulation'],
    install_requires=[
        'click',
        'numpy',
        'scipy',
        'tqdm'
    ]
)
