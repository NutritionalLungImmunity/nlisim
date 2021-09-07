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


def molecule_grid_factory(self: 'IL8State') -> np.ndarray:
    return np.zeros(shape=self.global_state.grid.shape, dtype=float)


@attr.s(kw_only=True, repr=False)
class IL8State(ModuleState):
    grid: np.ndarray = attr.ib(default=attr.Factory(molecule_grid_factory, takes_self=True))
    half_life: float  # units: min
    half_life_multiplier: float  # units: proportion
    macrophage_secretion_rate: float  # units: atto-mol * cell^-1 * h^-1
    neutrophil_secretion_rate: float  # units: atto-mol * cell^-1 * h^-1
    pneumocyte_secretion_rate: float  # units: atto-mol * cell^-1 * h^-1
    macrophage_secretion_rate_unit_t: float  # units: atto-mol * cell^-1 * step^-1
    neutrophil_secretion_rate_unit_t: float  # units: atto-mol * cell^-1 * step^-1
    pneumocyte_secretion_rate_unit_t: float  # units: atto-mol * cell^-1 * step^-1
    k_d: float  # aM


class IL8(ModuleModel):
    """IL8"""

    name = 'il8'
    StateClass = IL8State

    def initialize(self, state: State) -> State:
        il8: IL8State = state.il8

        # config file values
        il8.half_life = self.config.getfloat('half_life')  # units: min
        il8.macrophage_secretion_rate = self.config.getfloat(
            'macrophage_secretion_rate'
        )  # units: atto-mol * cell^-1 * h^-1
        il8.neutrophil_secretion_rate = self.config.getfloat(
            'neutrophil_secretion_rate'
        )  # units: atto-mol * cell^-1 * h^-1
        il8.pneumocyte_secretion_rate = self.config.getfloat(
            'pneumocyte_secretion_rate'
        )  # units: atto-mol * cell^-1 * h^-1
        il8.k_d = self.config.getfloat('k_d')

        # computed values
        il8.half_life_multiplier = 0.5 ** (
            1 * self.time_step / il8.half_life
        )  # units: step * (min/step) / min -> 1
        # time unit conversions
        # units: (atto-mol * cell^-1 * h^-1 * (min * step^-1) / (min * hour^-1)
        #        = atto-mol * cell^-1 * step^-1
        il8.macrophage_secretion_rate_unit_t = il8.macrophage_secretion_rate * (self.time_step / 60)
        il8.neutrophil_secretion_rate_unit_t = il8.neutrophil_secretion_rate * (self.time_step / 60)
        il8.pneumocyte_secretion_rate_unit_t = il8.pneumocyte_secretion_rate * (self.time_step / 60)

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
                        k_d=il8.k_d,
                        h=self.time_step / 60,  # units: (min/step) / (min/hour)
                        volume=voxel_volume,
                        b=1,
                    )
                    > rg.uniform()
                ):
                    neutrophil_cell['status'] = PhagocyteStatus.ACTIVE
                    neutrophil_cell['status_iteration'] = 0

        # Degrade IL8
        il8.grid *= il8.half_life_multiplier
        il8.grid *= turnover_rate(
            x=np.ones(shape=il8.grid.shape, dtype=np.float64),
            x_system=0.0,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )

        # Diffusion of IL8
        il8.grid[:] = apply_diffusion(
            variable=il8.grid,
            laplacian=molecules.laplacian,
            diffusivity=molecules.diffusion_constant,
            dt=self.time_step,
        )

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        il8: IL8State = state.il8
        voxel_volume = state.voxel_volume

        return {
            'concentration (nM)': float(np.mean(il8.grid) / voxel_volume / 1e9),
        }

    def visualization_data(self, state: State):
        il8: IL8State = state.il8
        return 'molecule', il8.grid
