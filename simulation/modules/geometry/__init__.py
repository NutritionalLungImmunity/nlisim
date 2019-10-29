import attr
import numpy as np
import h5py
import os

from simulation.module import Module, ModuleState
from simulation.modules.advection.differences import gradient, laplacian
from simulation.state import grid_variable, State
from simulation.validation import ValidationError


@attr.s(kw_only=True, repr=False)
class GeometryState(ModuleState):
    geometry_grid = grid_variable(np.dtype('int'))

    @geometry_grid.validator
    def _validate_geometry_grid(self, attribute: attr.Attribute, value: np.ndarray) -> None:
        if not (value >= 0).all():
            raise ValidationError(f'not >= 0')

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

        #g = np.load('./simulation/modules/geometry/geometry.npy')
        with h5py.File(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'geometry.hdf5'), 'r') as f:
            g = f['geometry'][:]

        if (g.shape != state.grid.shape):
            raise ValidationError(f'imported geometry doesn\'t match the shape of primary grid')

        geometry.geometry_grid = np.copy(g)

        return state

    def advance(self, state: State, previous_time: float):
        """Advance the state by a single time step."""
        # do nothing since the geoemtry is constant
        return state
