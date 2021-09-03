from typing import Any, Dict

import attr
import numpy as np

from nlisim.coordinates import Voxel
from nlisim.diffusion import apply_diffusion
from nlisim.grid import RectangularGrid
from nlisim.module import ModuleModel, ModuleState
from nlisim.modules.molecules import MoleculesState
from nlisim.state import State
from nlisim.util import turnover_rate


def molecule_grid_factory(self: 'MIP1BState') -> np.ndarray:
    return np.zeros(shape=self.global_state.grid.shape, dtype=float)


@attr.s(kw_only=True, repr=False)
class MIP1BState(ModuleState):
    grid: np.ndarray = attr.ib(
        default=attr.Factory(molecule_grid_factory, takes_self=True)
    )  # units: atto-mols
    half_life: float
    half_life_multiplier: float  # units: proportion
    macrophage_secretion_rate: float  # units: atto-mol/(cell*h)
    pneumocyte_secretion_rate: float  # units: atto-mol/(cell*h)
    macrophage_secretion_rate_unit_t: float  # units: atto-mol/(cell*step)
    pneumocyte_secretion_rate_unit_t: float  # units: atto-mol/(cell*step)
    k_d: float  # units: aM


class MIP1B(ModuleModel):
    """MIP1B"""

    name = 'mip1b'
    StateClass = MIP1BState

    def initialize(self, state: State) -> State:
        mip1b: MIP1BState = state.mip1b

        # config file values
        mip1b.half_life = self.config.getfloat('half_life')
        mip1b.macrophage_secretion_rate = self.config.getfloat(
            'macrophage_secretion_rate'
        )  # units: atto-mol/(cell*h)
        mip1b.pneumocyte_secretion_rate = self.config.getfloat(
            'pneumocyte_secretion_rate'
        )  # units: atto-mol/(cell*h)
        mip1b.k_d = self.config.getfloat('k_d')  # units: aM

        # computed values
        mip1b.half_life_multiplier = 0.5 ** (
            self.time_step / mip1b.half_life
        )  # units in exponent: (min/step) / min -> 1/step
        # time unit conversions
        # units: (atto-mol * cell^-1 * h^-1 * (min * step^-1) / (min * hour^-1)
        #        = atto-mol * cell^-1 * step^-1
        mip1b.macrophage_secretion_rate_unit_t = mip1b.macrophage_secretion_rate * (
            self.time_step / 60
        )
        mip1b.pneumocyte_secretion_rate_unit_t = mip1b.pneumocyte_secretion_rate * (
            self.time_step / 60
        )

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        from nlisim.modules.macrophage import MacrophageCellData, MacrophageState
        from nlisim.modules.pneumocyte import PneumocyteCellData, PneumocyteState

        mip1b: MIP1BState = state.mip1b
        molecules: MoleculesState = state.molecules
        pneumocyte: PneumocyteState = state.pneumocyte
        macrophage: MacrophageState = state.macrophage
        grid: RectangularGrid = state.grid

        # interact with pneumocytes
        for pneumocyte_cell_index in pneumocyte.cells.alive():
            pneumocyte_cell: PneumocyteCellData = pneumocyte.cells[pneumocyte_cell_index]

            if pneumocyte_cell['tnfa']:
                pneumocyte_cell_voxel: Voxel = grid.get_voxel(pneumocyte_cell['point'])
                mip1b.grid[tuple(pneumocyte_cell_voxel)] += mip1b.pneumocyte_secretion_rate_unit_t

        # interact with macrophages
        for macrophage_cell_index in macrophage.cells.alive():
            macrophage_cell: MacrophageCellData = macrophage.cells[macrophage_cell_index]

            if macrophage_cell['tnfa']:
                macrophage_cell_voxel: Voxel = grid.get_voxel(macrophage_cell['point'])
                mip1b.grid[tuple(macrophage_cell_voxel)] += mip1b.macrophage_secretion_rate_unit_t

        # Degrade MIP1B
        mip1b.grid *= mip1b.half_life_multiplier
        mip1b.grid *= turnover_rate(
            x=np.array(1.0, dtype=np.float64),
            x_system=0.0,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )

        # Diffusion of MIP1b
        mip1b.grid[:] = apply_diffusion(
            variable=mip1b.grid,
            laplacian=molecules.laplacian,
            diffusivity=molecules.diffusion_constant,
            dt=self.time_step,
        )

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        from nlisim.util import TissueType

        mip1b: MIP1BState = state.mip1b
        voxel_volume = state.voxel_volume
        mask = state.lung_tissue != TissueType.AIR

        return {
            'concentration (nM)': float(np.mean(mip1b.grid[mask]) / voxel_volume / 1e9),
        }

    def visualization_data(self, state: State):
        mip1b: MIP1BState = state.mip1b
        return 'molecule', mip1b.grid
