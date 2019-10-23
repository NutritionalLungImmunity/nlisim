#!/usr/bin/env python
from setuptools import find_packages, setup

setup(
    name='simulation',
    packages=find_packages(exclude=['test', 'test.*']),
    install_requires=[
        'attrs',
        'click',
        'h5py',
        'matplotlib',
        'numpy'
    ],
    entry_points={
        'console_scripts': [
            'simulation = simulation.cli:main'
        ]
    }
)
