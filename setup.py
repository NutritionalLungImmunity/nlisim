#!/usr/bin/env python
import os

from setuptools import find_packages, setup


def prerelease_local_scheme(version):
    """Return local scheme version unless building on master in Gitlab.

    This function returns the local scheme version number
    (e.g. 0.0.0.dev<N>+g<HASH>) unless building on Gitlab for a
    pre-release in which case it ignores the hash and produces a
    PEP440 compliant pre-release version number (e.g. 0.0.0.dev<N>).
    """
    from setuptools_scm.version import get_local_node_and_date

    if 'CIRCLE_BRANCH' in os.environ and os.environ['CIRCLE_BRANCH'] == 'master':
        return ''
    else:
        return get_local_node_and_date(version)


setup(
    name='nlisim',
    packages=find_packages(exclude=['test', 'test.*']),
    package_data={'nlisim.modules': ['geometry.hdf5']},
    python_requires='>=3.6',
    install_requires=[
        'attrs',
        'click',
        'click-pathlib',
        'dataclasses;python_version<"3.8"',
        'h5py',
        'importlib-metadata;python_version<"3.8"',
        'matplotlib',
        'numpy',
        'scipy',
        'tqdm',
        'vtk',
    ],
    entry_points={'console_scripts': ['nlisim = nlisim.cli:main']},
    setup_requires=['setuptools_scm'],
    use_scm_version={'local_scheme': prerelease_local_scheme},
)
