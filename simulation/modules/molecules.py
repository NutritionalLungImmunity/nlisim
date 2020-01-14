import attr
import numpy as np

from simulation.module import Module, ModuleState
from simulation.state import grid_variable, State


@attr.s(kw_only=True, repr=False)
class MoleculesState(ModuleState):
    iron = grid_variable(np.float)
    tf = grid_variable(np.float)
    tfbi = grid_variable(np.float)
    tafc = grid_variable(np.float)
    tafcbi = grid_variable(np.float)
    ros = grid_variable(np.float)
    lactoferrin = grid_variable(np.float)
    lactoferrinbi = grid_variable(np.float)
    il6 = grid_variable(np.float)
    tnfa = grid_variable(np.float)
    il8 = grid_variable(np.float)
    il10 = grid_variable(np.float)
    hepcidin = grid_variable(np.float)
    tgfb = grid_variable(np.float)
    mcp1 = grid_variable(np.float)
    mip2 = grid_variable(np.float)
    mip1b = grid_variable(np.float)

    # we may want to identify sources of molecules in the tissue e.g. blood
    #  that preferentially increase at a time step
    # source = grid_variable()

    def __repr__(self):
        return f'MoleculesState(iron)'


class Molecules(Module):
    name = 'molecules'
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

    StateClass = MoleculesState

    def initialize(self, state: State):
        molecules: MoleculesState = state.molecules

        iron_init_val = self.config.getfloat('iron_init_concentration')

        # TODO initialize in a user/geometry specific way
        molecules.iron[:] = iron_init_val
        # molecules.source[:] = 0

        return state

    def advance(self, state: State, previous_time: float):
        """Advance the state by a single time step."""
        molecules: MoleculesState = state.molecules

        iron = molecules.iron

        diffuse(iron)
        degrade(iron)

        return state


def diffuse(molecule):
    # TODO These 2 functions should be implemented for all moleculess
    # the rest of the behavior (uptake, secretion, etc.) should be
    # handled in the cell specific module.
    return


def degrade(molecule):
    # TODO These 2 functions should be implemented for all moleculess
    # the rest of the behavior (uptake, secretion, etc.) should be
    # handled in the cell specific module.
    return
