import math
from typing import Any, Dict

import attr
import numpy as np

from nlisim.coordinates import Voxel
from nlisim.grid import RectangularGrid
from nlisim.module import ModuleState
from nlisim.modules.molecules import MoleculeModel, MoleculesState
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
    pneumocyte_secretion_rate: float
    macrophage_secretion_rate_unit_t: float
    neutrophil_secretion_rate_unit_t: float
    pneumocyte_secretion_rate_unit_t: float
    k_d: float


class IL8(MoleculeModel):
    """IL8"""

    name = 'il8'
    StateClass = IL8State

    def initialize(self, state: State) -> State:
        il8: IL8State = state.il8

        # config file values
        il8.half_life = self.config.getfloat('half_life')
        il8.macrophage_secretion_rate = self.config.getfloat('macrophage_secretion_rate')
        il8.neutrophil_secretion_rate = self.config.getfloat('neutrophil_secretion_rate')
        il8.pneumocyte_secretion_rate = self.config.getfloat('pneumocyte_secretion_rate')
        il8.k_d = self.config.getfloat('k_d')

        # computed values
        il8.half_life_multiplier = 1 + math.log(0.5) / (il8.half_life / self.time_step)
        # time unit conversions
        il8.macrophage_secretion_rate_unit_t = il8.macrophage_secretion_rate * 60 * self.time_step
        il8.neutrophil_secretion_rate_unit_t = il8.neutrophil_secretion_rate * 60 * self.time_step
        il8.pneumocyte_secretion_rate_unit_t = il8.pneumocyte_secretion_rate * 60 * self.time_step

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        from nlisim.modules.neutrophil import NeutrophilCellData, NeutrophilState
        from nlisim.modules.phagocyte import PhagocyteStatus

        il8: IL8State = state.il8
        molecules: MoleculesState = state.molecules
        neutrophil: NeutrophilState = state.neutrophil
        voxel_volume: float = state.voxel_volume
        grid: RectangularGrid = state.grid

        # IL8 activates neutrophils
        for neutrophil_cell_index in neutrophil.cells.alive():
            neutrophil_cell: NeutrophilCellData = neutrophil.cells[neutrophil_cell_index]
            if neutrophil_cell['status'] in {PhagocyteStatus.RESTING or PhagocyteStatus.ACTIVE}:
                neutrophil_cell_voxel: Voxel = grid.get_voxel(neutrophil_cell['point'])
                if (
                    activation_function(
                        x=il8.grid[tuple(neutrophil_cell_voxel)],
                        kd=il8.k_d,
                        h=self.time_step / 60,
                        volume=voxel_volume,
                        b=1,
                    )
                    > rg.uniform()
                ):
                    neutrophil_cell['status'] = PhagocyteStatus.ACTIVE
                    neutrophil_cell['status_iteration'] = 0

        # TODO: were macrophages and pneumocytes also going to secrete IL8?

        # Degrade IL8
        il8.grid *= il8.half_life_multiplier
        il8.grid *= turnover_rate(
            x=np.ones(shape=il8.grid.shape, dtype=np.float64),
            x_system=0.0,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )

        # Diffusion of IL8
        self.diffuse(il8.grid, state)

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        il8: IL8State = state.il8
        voxel_volume = state.voxel_volume

        return {
            'concentration': float(np.mean(il8.grid) / voxel_volume),
        }

    def visualization_data(self, state: State):
        il8: IL8State = state.il8
        return 'molecule', il8.grid
