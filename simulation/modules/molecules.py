from enum import Enum, unique
import json

import attr
import numpy as np

from simulation.module import Module, ModuleState
from simulation.modules.geometry import GeometryState, TissueTypes
from simulation.state import grid_variable, State
from simulation.validation import ValidationError


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


@attr.s(kw_only=True, repr=False)
class MoleculesState(ModuleState):
    concentration = grid_variable(
        dtype={
            'names': [molecule.name for molecule in MoleculeTypes],
            'formats': ['f4'] * len(MoleculeTypes),
        }
    )

    # we may want to identify sources of molecules in the tissue e.g. blood
    # that preferentially increase at a time step
    # source = grid_variable()

    @concentration.validator
    def _validate_concentration(self, attribute: attr.Attribute, value: np.ndarray) -> None:
        for name in value.dtype.names:
            if not np.greater_equal(value[name], 0).all():
                raise ValidationError('concentration must >= 0')

    def __repr__(self):
        return f'MoleculesState(iron)'


class Molecules(Module):
    name = 'molecules'
    StateClass = MoleculesState

    def initialize(self, state: State):
        molecules: MoleculesState = state.molecules
        geometry: GeometryState = state.geometry

        # check if the geometry array is empty
        if not np.any(geometry.lung_tissue):
            raise RuntimeError('geometry molecule has to be initialized first')

        molecules_config = self.config.get('molecules')
        json_config = json.loads(molecules_config)

        for molecule in json_config:
            name = molecule['name']
            init_val = molecule['init_val']
            init_loc = molecule['init_loc']
            if name not in [e.name for e in MoleculeTypes]:
                raise TypeError(f'Molecule {name} is not implemented yet')

            elif init_loc not in [e.name for e in TissueTypes]:
                raise TypeError(f'Cannot find lung tissue type {init_loc}')

            molecules.concentration[name][
                np.where(geometry.lung_tissue == TissueTypes[init_loc].value)
            ] = init_val

        molecules.concentration = np.rec.array(molecules.concentration)

        return state

    def advance(self, state: State, previous_time: float):
        """Advance the state by a single time step."""
        molecules: MoleculesState = state.molecules
        # grid: RectangularGrid = state.grid

        tafc = molecules.concentration.tafc
        self.diffuse(tafc)
        self.degrade(tafc)
        # with open('testfile.txt', 'w') as outfile:
        #     for data_slice in tafc:
        #         np.savetxt(outfile, data_slice, fmt='%-7.2f')

        return state

    @classmethod
    def diffuse(cls, molecule: np.ndarray):
        # TODO These 2 functions should be implemented for all moleculess
        # the rest of the behavior (uptake, secretion, etc.) should be
        # handled in the cell specific module.
        return

    @classmethod
    def degrade(cls, molecule: np.ndarray):
        # TODO These 2 functions should be implemented for all moleculess
        # the rest of the behavior (uptake, secretion, etc.) should be
        # handled in the cell specific module.
        return
