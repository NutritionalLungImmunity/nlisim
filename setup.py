#!/usr/bin/env python
from setuptools import find_packages, setup

setup(
    name='nlisim',
    packages=find_packages(exclude=['test', 'test.*']),
    install_requires=[
        'attrs',
        'click',
        'click-pathlib',
        'h5py',
        'matplotlib',
        'numpy',
        'scipy',
        'tqdm',
        'vtk',
    ],
    entry_points={'console_scripts': ['nlisim = nlisim.cli:main']},
)
