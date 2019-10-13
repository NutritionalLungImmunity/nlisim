import attr
import numpy as np

from simulation.module import Module, ModuleState
from simulation.state import grid_variable, State
from simulation.validator import ValidationError


@attr.s(kw_only=True)
class AdvectionState(ModuleState):
    diffusivity: float = attr.ib(default=0)
    concentration = grid_variable()
    wind_x = grid_variable()
    wind_y = grid_variable()
    source = grid_variable()

    @diffusivity.validator
    def _validate_diffusivity(self, attribute: attr.Attribute, value: float) -> None:
        if not np.isfinite(value) or value < 0:
            raise ValidationError('Invalid diffusivity')


class Advection(Module):
    name = 'advection'
    defaults = {
        'diffusivity': '0.05',
        'wind_x': '0.1',
        'wind_y': '-0.2',
        'px': '0.5',
        'py': '0.5',
        'value': '1.0'
    }
    StateClass = AdvectionState

    def initialize(self, state: State):
        advection: AdvectionState = state.advection

        advection.diffusivity = self.config.getfloat('diffusivity')
        advection.wind_x[:] = self.config.getfloat('wind_x')
        advection.wind_y[:] = self.config.getfloat('wind_y')
        advection.concentration[:] = 0
        advection.source[:] = 0
        return state

    def advance(self, state: State, time: float):
        return state
