"""
Solves the 2D advection-diffusion equation:

    ∂T
    -- = ∇ ⋅ (d ∇T) - ∇ ⋅ (wT) + S
    ∂t

With homogeneous diffusivity and incompressible flow this becomes:

    ∂T
    -- =  d ∆T - w ⋅ ∇T + S
    ∂t

The terms in the RHS of this equation expand to (where w = (u, v)):

1. diffusion
        d * T_xx + d * T_yy

2. advection
        - u * T_x - v * T_y

3. source
        S

Here, we solve this using Euler's method:

T(t + dt, x, y) <- T(t, x, y) + ( diffusion + advection + source ) * dt
"""
from math import floor

import attr
import numpy as np

from simulation.module import Module, ModuleState
from simulation.modules.advection.differences import gradient, laplacian
from simulation.state import grid_variable, State
from simulation.validation import ValidationError


@attr.s(kw_only=True, repr=False)
class AdvectionState(ModuleState):
    diffusivity: float = attr.ib(default=0)
    concentration = grid_variable()
    wind_x = grid_variable()
    wind_y = grid_variable()
    wind_z = grid_variable()
    source = grid_variable()

    @diffusivity.validator
    def _validate_diffusivity(self, attribute: attr.Attribute, value: float) -> None:
        raise ValidationError('test')
        if not np.isfinite(value) or value < 0:
            raise ValidationError('Invalid diffusivity')

    def __repr__(self):
        return f'AdvectionState(concentration, wind_x, wind_y, source)'


class Advection(Module):
    name = 'advection'
    defaults = {
        'diffusivity': '0.05',
        'wind_x': '0.1',
        'wind_y': '-0.2',
        'wind_z': '0.0',
        'px': '0.5',
        'py': '0.5',
        'pz': '0.5',
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

        px = self.config.getfloat('px')
        py = self.config.getfloat('py')
        pz = self.config.getfloat('pz')
        value = self.config.getfloat('value')

        ix = floor(px * (state.grid.shape[2] - 1))
        iy = floor(py * (state.grid.shape[1] - 1))
        iz = floor(pz * (state.grid.shape[0] - 1))
        advection.source[iz, iy, ix] = value

        return state

    def advance(self, state: State, previous_time: float):
        """Advance the state by a single time step."""
        advection: AdvectionState = state.advection

        concentration = advection.concentration
        delta_time = state.time - previous_time

        # diffusion
        dq = advection.diffusivity * laplacian(state, concentration)

        # advection
        dq -= advection.wind_x * gradient(state, concentration, 2)
        dq -= advection.wind_y * gradient(state, concentration, 1)
        dq -= advection.wind_z * gradient(state, concentration, 0)

        # source
        dq += advection.source

        # mutate the original concentration value
        concentration[:, :] += dq * delta_time
        return state
