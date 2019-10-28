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
class GeometryState(ModuleState):
    geometry_grid = grid_variable()

    @geometry_grid.validator
    def _validate_geometry_grid(self, attribute: attr.Attribute, value: np.ndarray) -> None:
        if not issubclass(value.dtype.type, int) or (value >= 0).all(): 
            raise ValidationError(f'Invalid geometry')

    def __repr__(self):
        return f'GeometryState(geometry_grid)'


class Geometry(Module):
    name = 'geometry'
    defaults = {
        'geometry_grid': '0'
    }
    StateClass = GeometryState

    def initialize(self, state: State):
        geometry: GeometryState = state.geometry

        g = np.load('geometry.npy')

        if (g.shape != state.grid.shape):
            raise ValidationError(f'imported geometry doesn\'t match the shape of primary grid')

        for z in g.shape[0]:
            for y in g.shape[1]:
                for x in g.shape[2]:
                    geometry.geometry_grid[z, y, x] = g[z][y][x]

        return state

    def advance(self, state: State, previous_time: float):
        """Advance the state by a single time step."""
        # do nothing since the geoemtry is constant
        return state
