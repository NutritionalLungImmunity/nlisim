#!/usr/bin/env python
from setuptools import find_packages, setup

setup(
    name='simulation',
    packages=find_packages(exclude=['test', 'test.*']),
    install_requires=[
        'click',
        'matplotlib',
        'numpy'
    ],
    entry_points={
        'console_scripts': [
            'simulation = simulation.cli:main'
        ]
    }
)
