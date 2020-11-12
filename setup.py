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

    if 'CI_COMMIT_REF_NAME' in os.environ and os.environ['CI_COMMIT_REF_NAME'] == 'master':
        return ''
    else:
        return get_local_node_and_date(version)


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
    setup_requires=['setuptools_scm'],
    use_scm_version={'local_scheme': prerelease_local_scheme},
)
