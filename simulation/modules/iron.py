from math import floor

import attr
import numpy as np

from simulation.module import Module, ModuleState
from simulation.state import grid_variable, State


@attr.s(kw_only=True, repr=False)
class IronState(ModuleState):
    concentration = grid_variable(np.float)
    # we may want to identify sources of iron in the tissue e.g. blood
    #  that preferentially increase at a time step
    # source = grid_variable()

    def __repr__(self):
        return f'IronState(concentration)'


class Iron(Module):
    name = 'iron'
    defaults = {}
    StateClass = IronState

    def initialize(self, state: State):
        iron: IronState = state.iron

        init_val = self.config.getfloat('init_concentration')

        # TODO initialize in a user/geometry specific way
        iron.concentration[:] = init_val
        # iron.source[:] = 0

        return state

    def advance(self, state: State, previous_time: float):
        """Advance the state by a single time step."""
        iron: IronState = state.iron

        concentration = iron.concentration

        diffuse(self, concentration)
        degrade(self, concentration)

        return state


def diffuse(self, concentration):
    # TODO These 2 functions should be implemented for all molecules
    # the rest of the behavior (uptake, secretion, etc.) should be
    # handled in the cell specific module.
    return


def degrade(self, concentration):
    # TODO These 2 functions should be implemented for all molecules
    # the rest of the behavior (uptake, secretion, etc.) should be
    # handled in the cell specific module.
    return
