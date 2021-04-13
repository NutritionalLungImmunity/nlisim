from enum import IntEnum
from pathlib import Path

import attr
import h5py
import numpy as np

from nlisim.module import ModuleModel, ModuleState
from nlisim.state import grid_variable, State
from nlisim.validation import ValidationError


# I am not quite sure if we should put the definition of the lung tissue types here
class TissueType(IntEnum):
    AIR = 0
    BLOOD = 1
    OTHER = 2
    EPITHELIUM = 3
    SURFACTANT = 4  # unused 1/28/2021
    PORE = 5  # unused 1/28/2021

    @classmethod
    def validate(cls, value: np.ndarray):
        return np.logical_and(value >= 0, value <= 5).all() and \
               np.issubclass_(value.dtype.type, np.integer)


@attr.s(kw_only=True, repr=False)
class GeometryState(ModuleState):
    lung_tissue: np.ndarray = grid_variable(np.dtype('int'))
    voxel_volume: float
    space_volume: float
    time_step_size: float

    @lung_tissue.validator
    def _validate_lung_tissue(self, attribute: attr.Attribute, value: np.ndarray) -> None:
        if not TissueType.validate(value):
            raise ValidationError('input illegal')

    def __repr__(self):
        return 'GeometryState(lung_tissue)'


class Geometry(ModuleModel):
    name = 'geometry'
    StateClass = GeometryState

    def initialize(self, state: State):
        geometry: GeometryState = state.geometry

        geometry.voxel_volume = self.config.getfloat('voxel_volume')
        geometry.space_volume = geometry.voxel_volume * np.product(geometry.global_state.grid.shape)

        # The geometry data file is included next to this one
        path = Path(__file__).parent / 'geometry.hdf5'
        try:
            with h5py.File(path, 'r') as f:
                if f['geometry'][:].shape != state.grid.shape:
                    raise ValidationError("shape doesn\'t match")
                geometry.lung_tissue[:] = f['geometry'][:]
        except Exception:
            print(f'Error loading geometry file at {path}.')
            raise

        return state
