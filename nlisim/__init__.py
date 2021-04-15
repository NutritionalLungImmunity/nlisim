"""
A multiscale simulation framework of *Aspergillus fumigatus*.

.. include:: ../README.md
"""

try:
    from importlib.metadata import version  # type: ignore
except ImportError:
    from importlib_metadata import version  # type: ignore


__version__ = version('nlisim')
