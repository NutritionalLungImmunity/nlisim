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


def molecule_grid_factory(self: 'MIP2State') -> np.ndarray:
    return np.zeros(shape=self.global_state.grid.shape, dtype=float)


@attr.s(kw_only=True, repr=False)
class MIP2State(ModuleState):
    grid: np.ndarray = attr.ib(default=attr.Factory(molecule_grid_factory, takes_self=True))
    half_life: float
    half_life_multiplier: float
    macrophage_secretion_rate: float
    neutrophil_secretion_rate: float
    pneumocyte_secretion_rate: float
    macrophage_secretion_rate_unit_t: float
    pneumocyte_secretion_rate_unit_t: float
    neutrophil_secretion_rate_unit_t: float
    k_d: float


class MIP2(MoleculeModel):
    """MIP2"""

    name = 'mip2'
    StateClass = MIP2State

    def initialize(self, state: State) -> State:
        mip2: MIP2State = state.mip2

        # config file values
        mip2.half_life = self.config.getfloat('half_life')
        mip2.macrophage_secretion_rate = self.config.getfloat('macrophage_secretion_rate')
        mip2.neutrophil_secretion_rate = self.config.getfloat('neutrophil_secretion_rate')
        mip2.pneumocyte_secretion_rate = self.config.getfloat('pneumocyte_secretion_rate')
        mip2.k_d = self.config.getfloat('k_d')

        # computed values
        mip2.half_life_multiplier = 1 + math.log(0.5) / (mip2.half_life / self.time_step)
        # time unit conversions
        mip2.macrophage_secretion_rate_unit_t = mip2.macrophage_secretion_rate * 60 * self.time_step
        mip2.neutrophil_secretion_rate_unit_t = mip2.neutrophil_secretion_rate * 60 * self.time_step
        mip2.pneumocyte_secretion_rate_unit_t = mip2.pneumocyte_secretion_rate * 60 * self.time_step

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        from nlisim.modules.macrophage import MacrophageCellData, MacrophageState
        from nlisim.modules.neutrophil import NeutrophilCellData, NeutrophilState
        from nlisim.modules.phagocyte import PhagocyteStatus
        from nlisim.modules.pneumocyte import PneumocyteCellData, PneumocyteState

        mip2: MIP2State = state.mip2
        molecules: MoleculesState = state.molecules
        neutrophil: NeutrophilState = state.neutrophil
        pneumocyte: PneumocyteState = state.pneumocyte
        macrophage: MacrophageState = state.macrophage
        grid: RectangularGrid = state.grid
        voxel_volume = state.voxel_volume

        # interact with neutrophils
        neutrophil_activation: np.ndarray = activation_function(
            x=mip2.grid, kd=mip2.k_d, h=self.time_step / 60, volume=voxel_volume, b=1
        )
        for neutrophil_cell_index in neutrophil.cells.alive():
            neutrophil_cell: NeutrophilCellData = neutrophil.cells[neutrophil_cell_index]
            neutrophil_cell_voxel: Voxel = grid.get_voxel(neutrophil_cell['point'])

            if (
                neutrophil_cell['status'] == PhagocyteStatus.RESTING
                and neutrophil_activation[tuple(neutrophil_cell_voxel)] > rg.uniform()
            ):
                neutrophil_cell['status'] = PhagocyteStatus.ACTIVATING
                neutrophil_cell['status_iteration'] = 0  # TODO: had 'todo: set_status' in orig
            elif neutrophil_cell['tnfa']:
                mip2.grid[tuple(neutrophil_cell_voxel)] += mip2.neutrophil_secretion_rate_unit_t
                if neutrophil_activation[tuple(neutrophil_cell_voxel)] > rg.uniform():
                    neutrophil_cell['status_iteration'] = 0

        # interact with pneumocytes
        for pneumocyte_cell_index in pneumocyte.cells.alive():
            pneumocyte_cell: PneumocyteCellData = pneumocyte.cells[pneumocyte_cell_index]

            if pneumocyte_cell['tnfa']:
                pneumocyte_cell_voxel: Voxel = grid.get_voxel(pneumocyte_cell['point'])
                mip2.grid[tuple(pneumocyte_cell_voxel)] += mip2.pneumocyte_secretion_rate_unit_t

        # interact with macrophages
        for macrophage_cell_index in macrophage.cells.alive():
            macrophage_cell: MacrophageCellData = macrophage.cells[macrophage_cell_index]

            if macrophage_cell['tnfa']:
                macrophage_cell_voxel: Voxel = grid.get_voxel(macrophage_cell['point'])
                mip2.grid[tuple(macrophage_cell_voxel)] += mip2.macrophage_secretion_rate_unit_t

        # Degrade MIP2
        mip2.grid *= mip2.half_life_multiplier
        mip2.grid *= turnover_rate(
            x=np.array(1.0, dtype=np.float64),
            x_system=0.0,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )

        # Diffusion of MIP2
        self.diffuse(mip2.grid, state)

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        mip2: MIP2State = state.mip1b
        voxel_volume = state.voxel_volume

        return {
            'concentration': float(np.mean(mip2.grid) / voxel_volume),
        }

    def visualization_data(self, state: State):
        mip2: MIP2State = state.mip2
        return 'molecule', mip2.grid
