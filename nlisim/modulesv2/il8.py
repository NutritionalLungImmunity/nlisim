import math

import attr
import numpy as np

from nlisim.coordinates import Voxel
from nlisim.grid import RectangularGrid
from nlisim.module import ModuleState
from nlisim.modulesv2.geometry import GeometryState
from nlisim.modulesv2.macrophage import MacrophageState
from nlisim.modulesv2.molecule import MoleculeModel
from nlisim.modulesv2.molecules import MoleculesState
from nlisim.modulesv2.neutrophil import NeutrophilState
from nlisim.modulesv2.phagocyte import PhagocyteStatus
from nlisim.random import rg
from nlisim.state import State
from nlisim.util import activation_function, turnover_rate


def molecule_grid_factory(self: 'IL8State') -> np.ndarray:
    return np.zeros(shape=self.global_state.grid.shape, dtype=float)


@attr.s(kw_only=True, repr=False)
class IL8State(ModuleState):
    grid: np.ndarray = attr.ib(default=attr.Factory(molecule_grid_factory, takes_self=True))
    half_life: float
    half_life_multiplier: float
    macrophage_secretion_rate: float
    neutrophil_secretion_rate: float
    epithelial_secretion_rate: float
    macrophage_secretion_rate_unit_t: float
    neutrophil_secretion_rate_unit_t: float
    epithelial_secretion_rate_unit_t: float
    k_d: float


class IL8(MoleculeModel):
    """IL8"""

    name = 'il8'
    StateClass = IL8State

    def initialize(self, state: State) -> State:
        il8: IL8State = state.il8
        geometry: GeometryState = state.geometry
        voxel_volume = geometry.voxel_volume

        # config file values
        il8.half_life = self.config.getfloat('half_life')
        il8.macrophage_secretion_rate = self.config.getfloat('macrophage_secretion_rate')
        il8.neutrophil_secretion_rate = self.config.getfloat('neutrophil_secretion_rate')
        il8.epithelial_secretion_rate = self.config.getfloat('epithelial_secretion_rate')
        il8.k_d = self.config.getfloat('k_d')

        # computed values
        il8.half_life_multiplier = 1 + math.log(0.5) / (il8.half_life / state.simulation.time_step_size)
        # time unit conversions
        il8.macrophage_secretion_rate_unit_t = il8.macrophage_secretion_rate * 60 * state.simulation.time_step_size
        il8.neutrophil_secretion_rate_unit_t = il8.neutrophil_secretion_rate * 60 * state.simulation.time_step_size
        il8.epithelial_secretion_rate_unit_t = il8.epithelial_secretion_rate * 60 * state.simulation.time_step_size

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        il8: IL8State = state.il8
        molecules: MoleculesState = state.molecules
        macrophage: MacrophageState = state.macrophage
        neutrophil: NeutrophilState = state.neutrophil
        geometry: GeometryState = state.geometry
        grid: RectangularGrid = state.grid

        # IL8 activates neutrophils
        for neutrophil_cell_index in neutrophil.cells.alive():
            neutrophil_cell = macrophage.cells[neutrophil_cell_index]
            if neutrophil_cell['status'] in {PhagocyteStatus.RESTING or PhagocyteStatus.ACTIVE}:
                neutrophil_cell_voxel: Voxel = grid.get_voxel(neutrophil_cell['point'])
                if activation_function(x=il8.grid[tuple(neutrophil_cell_voxel)],
                                       kd=il8.k_d,
                                       h=state.simulation.time_step_size / 60,
                                       volume=geometry.voxel_volume) < rg():
                    neutrophil_cell['status'] = PhagocyteStatus.ACTIVE
                    neutrophil_cell['iteration'] = 0

        # Degrade IL8
        il8.grid *= il8.half_life_multiplier
        il8.grid *= turnover_rate(x_mol=np.ones(shape=il8.grid.shape, dtype=np.float64),
                                  x_system_mol=0.0,
                                  turnover_rate=molecules.turnover_rate,
                                  rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t)

        return state
