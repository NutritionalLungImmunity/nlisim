from enum import Enum

import attr
import numpy as np

from simulation.module import Module, ModuleState
from simulation.state import grid_variable, State

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
    concentration = grid_variable(dtype = {
                    'names':[molecule.name for molecule in MoleculeTypes],
                    'formats':['f4'] * len(MoleculeTypes)
                })

    # we may want to identify sources of molecules in the tissue e.g. blood
    #  that preferentially increase at a time step
    # source = grid_variable()

    def __repr__(self):
        return f'MoleculesState(iron)'


class Molecules(Module):
    name = 'molecules'
    '''
    defaults = {
        'iron': '',
        'tf': '',
        'tfbi': '',
        'tafc': '',
        'tafcbi': '',
        'ros': '',
        'lactoferrin': '',
        'lactoferrinbi': '',
        'il6': '',
        'tnfa': '',
        'il8': '',
        'il10': '',
        'hepcidin': '',
        'tgfb': '',
        'mcp1': '',
        'mip2': '',
        'mip1b': '',
    }
    '''

    StateClass = MoleculesState

    def initialize(self, state: State):
        molecules: MoleculesState = state.molecules
        geometry: GeometryState = state.geometry

        # check if the geometry array is empty
        if not np.any(geometry.lung_tissue):
            raise RuntimeError('geometry molecule has to be initialized first')

        molecules.concentration = np.rec.array(molecules.concentration)

        iron_init_val = self.config.getfloat('iron_init_concentration')

        # TODO initialize in a user/geometry specific way
        molecules.concentration.iron[:] = iron_init_val

        # molecules.source[:] = 0

        return state

    def advance(self, state: State, previous_time: float):
        """Advance the state by a single time step."""
        molecules: MoleculesState = state.molecules

        iron = molecules.concentration.iron

        Molecules.diffuse(iron)
        Molecules.degrade(iron)

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
