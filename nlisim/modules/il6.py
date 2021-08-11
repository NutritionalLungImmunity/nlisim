import math
from typing import Any, Dict

import attr
import numpy as np

from nlisim.coordinates import Voxel
from nlisim.grid import RectangularGrid
from nlisim.module import ModuleState
from nlisim.modules.molecules import MoleculeModel, MoleculesState
from nlisim.state import State
from nlisim.util import turnover_rate


def molecule_grid_factory(self: 'IL6State') -> np.ndarray:
    return np.zeros(shape=self.global_state.grid.shape, dtype=float)


@attr.s(kw_only=True, repr=False)
class IL6State(ModuleState):
    grid: np.ndarray = attr.ib(default=attr.Factory(molecule_grid_factory, takes_self=True))
    half_life: float
    half_life_multiplier: float
    macrophage_secretion_rate: float
    neutrophil_secretion_rate: float
    pneumocyte_secretion_rate: float
    macrophage_secretion_rate_unit_t: float
    neutrophil_secretion_rate_unit_t: float
    pneumocyte_secretion_rate_unit_t: float
    k_d: float


class IL6(MoleculeModel):
    """IL6"""

    name = 'il6'
    StateClass = IL6State

    def initialize(self, state: State) -> State:
        il6: IL6State = state.il6

        # config file values
        il6.half_life = self.config.getfloat('half_life')
        il6.macrophage_secretion_rate = self.config.getfloat('macrophage_secretion_rate')
        il6.neutrophil_secretion_rate = self.config.getfloat('neutrophil_secretion_rate')
        il6.pneumocyte_secretion_rate = self.config.getfloat('pneumocyte_secretion_rate')
        il6.k_d = self.config.getfloat('k_d')

        # computed values
        il6.half_life_multiplier = 1 + math.log(0.5) / (il6.half_life / self.time_step)
        # time unit conversions
        il6.macrophage_secretion_rate_unit_t = il6.macrophage_secretion_rate * 60 * self.time_step
        il6.neutrophil_secretion_rate_unit_t = il6.neutrophil_secretion_rate * 60 * self.time_step
        il6.pneumocyte_secretion_rate_unit_t = il6.pneumocyte_secretion_rate * 60 * self.time_step

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        from nlisim.modules.macrophage import MacrophageState
        from nlisim.modules.neutrophil import NeutrophilState
        from nlisim.modules.phagocyte import PhagocyteStatus

        il6: IL6State = state.il6
        molecules: MoleculesState = state.molecules
        macrophage: MacrophageState = state.macrophage
        neutrophil: NeutrophilState = state.neutrophil
        grid: RectangularGrid = state.grid

        # active Macrophages secrete il6
        for macrophage_cell_index in macrophage.cells.alive():
            macrophage_cell = macrophage.cells[macrophage_cell_index]
            if macrophage_cell['status'] == PhagocyteStatus.ACTIVE:
                macrophage_cell_voxel: Voxel = grid.get_voxel(macrophage_cell['point'])
                il6.grid[tuple(macrophage_cell_voxel)] += il6.macrophage_secretion_rate_unit_t

        # active Neutrophils secrete il6
        for neutrophil_cell_index in neutrophil.cells.alive():
            neutrophil_cell = neutrophil.cells[neutrophil_cell_index]
            if neutrophil_cell['status'] == PhagocyteStatus.ACTIVE:
                neutrophil_cell_voxel: Voxel = grid.get_voxel(neutrophil_cell['point'])
                il6.grid[tuple(neutrophil_cell_voxel)] += il6.neutrophil_secretion_rate_unit_t

        # TODO: were pneumocytes also going to secrete IL6?

        # Degrade IL6
        il6.grid *= il6.half_life_multiplier
        il6.grid *= turnover_rate(
            x=np.ones(shape=il6.grid.shape, dtype=np.float64),
            x_system=0.0,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )

        # Diffusion of IL6
        self.diffuse(il6.grid, state)

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        il6: IL6State = state.il6
        voxel_volume = state.voxel_volume

        return {
            'concentration': float(np.mean(il6.grid) / voxel_volume),
        }

    def visualization_data(self, state: State):
        il6: IL6State = state.il6
        return 'molecule', il6.grid
