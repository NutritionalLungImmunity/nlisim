from typing import Any, Dict

import attr
import numpy as np

from nlisim.coordinates import Voxel
from nlisim.diffusion import apply_diffusion
from nlisim.grid import RectangularGrid
from nlisim.module import ModuleModel, ModuleState
from nlisim.modules.molecules import MoleculesState
from nlisim.random import rg
from nlisim.state import State
from nlisim.util import activation_function, turnover_rate


def molecule_grid_factory(self: 'MIP2State') -> np.ndarray:
    return np.zeros(shape=self.global_state.grid.shape, dtype=float)


@attr.s(kw_only=True, repr=False)
class MIP2State(ModuleState):
    grid: np.ndarray = attr.ib(
        default=attr.Factory(molecule_grid_factory, takes_self=True)
    )  # units: atto-mol
    half_life: float
    half_life_multiplier: float  # units: proportion
    macrophage_secretion_rate: float  # units: atto-mol * cell^-1 * h^-1
    neutrophil_secretion_rate: float  # units: atto-mol * cell^-1 * h^-1
    pneumocyte_secretion_rate: float  # units: atto-mol * cell^-1 * h^-1
    macrophage_secretion_rate_unit_t: float  # units: atto-mol * cell^-1 * step^-1
    pneumocyte_secretion_rate_unit_t: float  # units: atto-mol * cell^-1 * step^-1
    neutrophil_secretion_rate_unit_t: float  # units: atto-mol * cell^-1 * step^-1
    k_d: float  # aM


class MIP2(ModuleModel):
    """MIP2"""

    name = 'mip2'
    StateClass = MIP2State

    def initialize(self, state: State) -> State:
        mip2: MIP2State = state.mip2

        # config file values
        mip2.half_life = self.config.getfloat('half_life')
        mip2.macrophage_secretion_rate = self.config.getfloat(
            'macrophage_secretion_rate'
        )  # units: atto-mol * cell^-1 * h^-1
        mip2.neutrophil_secretion_rate = self.config.getfloat(
            'neutrophil_secretion_rate'
        )  # units: atto-mol * cell^-1 * h^-1
        mip2.pneumocyte_secretion_rate = self.config.getfloat(
            'pneumocyte_secretion_rate'
        )  # units: atto-mol * cell^-1 * h^-1
        mip2.k_d = self.config.getfloat('k_d')  # units: atto-mol * cell^-1 * h^-1

        # computed values
        mip2.half_life_multiplier = 0.5 ** (
            self.time_step / mip2.half_life
        )  # units in exponent: (min/step) / min -> 1/step
        # time unit conversions.
        # units: (atto-mol * cell^-1 * h^-1 * (min * step^-1) / (min * hour^-1)
        #        = atto-mol * cell^-1 * step^-1
        mip2.macrophage_secretion_rate_unit_t = mip2.macrophage_secretion_rate * (
            self.time_step / 60
        )
        mip2.neutrophil_secretion_rate_unit_t = mip2.neutrophil_secretion_rate * (
            self.time_step / 60
        )
        mip2.pneumocyte_secretion_rate_unit_t = mip2.pneumocyte_secretion_rate * (
            self.time_step / 60
        )

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
            x=mip2.grid,
            k_d=mip2.k_d,
            h=self.time_step / 60,  # units: (min/step) / (min/hour)
            volume=voxel_volume,
            b=1,
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
        mip2.grid[:] = apply_diffusion(
            variable=mip2.grid,
            laplacian=molecules.laplacian,
            diffusivity=molecules.diffusion_constant,
            dt=self.time_step,
        )

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        from nlisim.util import TissueType

        mip2: MIP2State = state.mip2
        voxel_volume = state.voxel_volume
        mask = state.lung_tissue != TissueType.AIR

        return {
            'concentration (nM)': float(np.mean(mip2.grid[mask]) / voxel_volume / 1e9),
        }

    def visualization_data(self, state: State):
        mip2: MIP2State = state.mip2
        return 'molecule', mip2.grid
