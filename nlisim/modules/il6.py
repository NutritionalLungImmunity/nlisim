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


def molecule_grid_factory(self: 'IL6State') -> np.ndarray:
    return np.zeros(shape=self.global_state.grid.shape, dtype=float)


@attr.s(kw_only=True, repr=False)
class IL6State(ModuleState):
    grid: np.ndarray = attr.ib(
        default=attr.Factory(molecule_grid_factory, takes_self=True)
    )  # units: atto-mol
    half_life: float  # units: min
    half_life_multiplier: float  # units: proportion
    macrophage_secretion_rate: float  # units: atto-mol/(cell*h)
    neutrophil_secretion_rate: float  # units: atto-mol/(cell*h)
    pneumocyte_secretion_rate: float  # units: atto-mol/(cell*h)
    macrophage_secretion_rate_unit_t: float  # units: atto-mol/(cell*step)
    neutrophil_secretion_rate_unit_t: float  # units: atto-mol/(cell*step)
    pneumocyte_secretion_rate_unit_t: float  # units: atto-mol/(cell*step)
    k_d: float  # units: aM


class IL6(ModuleModel):
    """IL6"""

    name = 'il6'
    StateClass = IL6State

    def initialize(self, state: State) -> State:
        il6: IL6State = state.il6

        # config file values
        il6.half_life = self.config.getfloat('half_life')
        il6.macrophage_secretion_rate = self.config.getfloat(
            'macrophage_secretion_rate'
        )  # units: atto-mol/(cell*h)
        il6.neutrophil_secretion_rate = self.config.getfloat(
            'neutrophil_secretion_rate'
        )  # units: atto-mol/(cell*h)
        il6.pneumocyte_secretion_rate = self.config.getfloat(
            'pneumocyte_secretion_rate'
        )  # units: atto-mol/(cell*h)
        il6.k_d = self.config.getfloat('k_d')  # units: atto-mol

        # computed values
        # units: %/step + %/min * (min/step) -> %/step
        il6.half_life_multiplier = 0.5 ** (
            self.time_step / il6.half_life
        )  # units in exponent: (min/step) / min -> 1/step
        # time unit conversions
        # units: (atto-mol * cell^-1 * h^-1 * (min * step^-1) / (min * hour^-1)
        #        = atto-mol * cell^-1 * step^-1
        il6.macrophage_secretion_rate_unit_t = il6.macrophage_secretion_rate * (self.time_step / 60)
        il6.neutrophil_secretion_rate_unit_t = il6.neutrophil_secretion_rate * (self.time_step / 60)
        il6.pneumocyte_secretion_rate_unit_t = il6.pneumocyte_secretion_rate * (self.time_step / 60)

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        from nlisim.modules.macrophage import MacrophageState
        from nlisim.modules.neutrophil import NeutrophilState
        from nlisim.modules.phagocyte import PhagocyteStatus
        from nlisim.modules.pneumocyte import PneumocyteState

        il6: IL6State = state.il6
        molecules: MoleculesState = state.molecules
        macrophage: MacrophageState = state.macrophage
        neutrophil: NeutrophilState = state.neutrophil
        pneumocyte: PneumocyteState = state.pneumocyte
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

        # active Pneumocytes secrete il6
        for pneumocyte_cell_index in pneumocyte.cells.alive():
            pneumocyte_cell = pneumocyte.cells[pneumocyte_cell_index]
            if pneumocyte_cell['status'] == PhagocyteStatus.ACTIVE:
                pneumocyte_cell_voxel: Voxel = grid.get_voxel(pneumocyte_cell['point'])
                il6.grid[tuple(pneumocyte_cell_voxel)] += il6.pneumocyte_secretion_rate_unit_t

        # Degrade IL6
        il6.grid *= il6.half_life_multiplier
        il6.grid *= turnover_rate(
            x=np.ones(shape=il6.grid.shape, dtype=np.float64),
            x_system=0.0,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )

        # Diffusion of IL6
        il6.grid[:] = apply_diffusion(
            variable=il6.grid,
            laplacian=molecules.laplacian,
            diffusivity=molecules.diffusion_constant,
            dt=self.time_step,
        )

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        from nlisim.util import TissueType

        il6: IL6State = state.il6
        voxel_volume = state.voxel_volume
        mask = state.lung_tissue != TissueType.AIR

        return {
            'concentration (nM)': float(np.mean(il6.grid[mask]) / voxel_volume / 1e9),
        }

    def visualization_data(self, state: State):
        il6: IL6State = state.il6
        return 'molecule', il6.grid
