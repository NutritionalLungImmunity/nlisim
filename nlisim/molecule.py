from enum import Enum, unique
from typing import Dict, List

import attr
from h5py import Group
import numpy as np

from nlisim.grid import RectangularGrid
from nlisim.state import State, get_class_path


@unique
class MoleculeTypes(Enum):
    """a enum class for the molecule type."""

    iron = 0
    tf = 1
    tfbi = 2
    tafc = 3
    tafcbi = 4
    ros = 5
    lactoferrin = 6
    lactoferrinbi = 7
    il6 = 8
    tnfa = 9
    il8 = 10
    il10 = 11
    hepcidin = 12
    tgfb = 13
    mcp1 = 14
    mip2 = 15
    mip1b = 16
    m_cyto = 17
    n_cyto = 18


@attr.s(kw_only=True, frozen=True, repr=False)
class MoleculeGrid(object):
    """A class contains a list of grids for each molecule type."""

    grid: RectangularGrid = attr.ib()
    _concentrations = attr.ib()
    _sources = attr.ib()
    _diffusivity: Dict[str, int] = {}
    _molecule_type: List[str] = attr.ib(factory=list)

    @_concentrations.default
    def __set_default_concentrations(self):
        return np.empty(0)

    @_sources.default
    def __set_default_sources(self):
        return np.empty(0)

    def __attrs_post_init__(self):
        grid = self.grid
        # noinspection PyTypeChecker
        object.__setattr__(
            self,
            '_concentrations',
            grid.allocate_variable(
                dtype={
                    'names': [molecule.name for molecule in MoleculeTypes],
                    'formats': ['f4'] * len(MoleculeTypes),
                }
            ),
        )

        # noinspection PyTypeChecker
        object.__setattr__(
            self,
            '_sources',
            grid.allocate_variable(
                dtype={
                    'names': [molecule.name for molecule in MoleculeTypes],
                    'formats': ['f4'] * len(MoleculeTypes),
                }
            ),
        )

    @property
    def diffusivity(self):
        return self._diffusivity

    @property
    def concentrations(self):
        return self._concentrations[[name for name in self._molecule_type]].view(np.recarray)

    @property
    def sources(self):
        return self._sources[[name for name in self._molecule_type]].view(np.recarray)

    @property
    def types(self):
        return self._molecule_type

    def __getitem__(self, index: str) -> np.ndarray:
        if not isinstance(index, str):
            raise TypeError('Expected an str index representing the type of molecule')
        if index not in self._molecule_type:
            raise KeyError(f'Molecule {index} is not declared or is knocked out')
        return self._concentrations[index]

    def append_molecule_type(self, molecule: str):
        if molecule not in self._concentrations.dtype.names:
            raise KeyError(f'Molecule {molecule} is not declared')
        self._molecule_type.append(molecule)

    def set_diffusivity(self, molecule: str, diffusivity: int):
        self._diffusivity[molecule] = diffusivity

    def incr(self):
        concentrations = self._concentrations
        sources = self._sources
        for name in self._molecule_type:
            concentrations[name] += sources[name]

    def shape(self):
        return self._concentrations.shape

    def save(self, group: Group, name: str, metadata: dict) -> Group:
        """Save the molecule grid.

        Save the list of grid as a new composite data structure inside
        an HDF5 group.
        """
        concentrations = self._concentrations
        sources = self._sources

        composite_group = group.create_group(name)

        composite_group.attrs['type'] = 'MoleculeGrid'
        composite_group.attrs['class'] = get_class_path(self)
        composite_group.attrs['molecule_type'] = self.types
        composite_group.create_dataset(name='concentrations', data=concentrations)
        composite_group.create_dataset(name='sources', data=sources)

        return composite_group

    @classmethod
    def load(cls, global_state: State, group: Group, name: str, metadata: dict) -> 'MoleculeGrid':
        """Load a molecule grid object."""
        composite_dataset = group[name]

        # TODO: load back the molecule types

        concentrations = composite_dataset['concentrations'][:]
        sources = composite_dataset['sources'][:]
        molecule_type = composite_dataset.attrs['molecule_type']

        return cls(
            grid=global_state.grid,
            concentrations=concentrations,
            sources=sources,
            molecule_type=molecule_type,
        )
