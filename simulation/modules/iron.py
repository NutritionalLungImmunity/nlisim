from math import floor

import attr
import numpy as np

from simulation.module import Module, ModuleState
from simulation.state import grid_variable, State


@attr.s(kw_only=True, repr=False)
class IronState(ModuleState):
    concentration = grid_variable()
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

        iron.concentration[:] = init_val
        # iron.source[:] = 0

        return state

    def advance(self, state: State, previous_time: float):
        """Advance the state by a single time step."""
        iron: IronState = state.iron

        concentration = iron.concentration

        #diffuse(concentration)
        #degrade(concentration)

        return state
