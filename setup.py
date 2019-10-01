#!/usr/bin/env python
from setuptools import setup

setup(
    name='simulation',
    packages=['simulation'],
    install_requires=[
        'click',
        'numpy',
        'scipy'
    ],
    entry_points={
        'console_scripts': [
            'simulation = simulation.cli:main'
        ]
    }
)
