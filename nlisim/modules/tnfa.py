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


def molecule_grid_factory(self: 'TNFaState') -> np.ndarray:
    return np.zeros(shape=self.global_state.grid.shape, dtype=float)


@attr.s(kw_only=True, repr=False)
class TNFaState(ModuleState):
    grid: np.ndarray = attr.ib(
        default=attr.Factory(molecule_grid_factory, takes_self=True)
    )  # units: atto-mol
    half_life: float  # units: min
    half_life_multiplier: float  # units: proportion
    macrophage_secretion_rate: float  # units: atto-mol/(cell*h)
    neutrophil_secretion_rate: float  # units: atto-mol/(cell*h)
    epithelial_secretion_rate: float  # units: atto-mol/(cell*h)
    macrophage_secretion_rate_unit_t: float  # units: atto-mol/(cell*step)
    neutrophil_secretion_rate_unit_t: float  # units: atto-mol/(cell*step)
    epithelial_secretion_rate_unit_t: float  # units: atto-mol/(cell*step)
    k_d: float  # aM


class TNFa(ModuleModel):
    name = 'tnfa'
    StateClass = TNFaState

    def initialize(self, state: State) -> State:
        tnfa: TNFaState = state.tnfa

        # config file values
        tnfa.half_life = self.config.getfloat('half_life')  # units: min
        tnfa.macrophage_secretion_rate = self.config.getfloat(
            'macrophage_secretion_rate'
        )  # units: atto-mol/(cell*h)
        tnfa.neutrophil_secretion_rate = self.config.getfloat(
            'neutrophil_secretion_rate'
        )  # units: atto-mol/(cell*h)
        tnfa.epithelial_secretion_rate = self.config.getfloat(
            'epithelial_secretion_rate'
        )  # units: atto-mol/(cell*h)
        tnfa.k_d = self.config.getfloat('k_d')  # units: aM

        # computed values
        tnfa.half_life_multiplier = 0.5 ** (
            self.time_step / tnfa.half_life
        )  # units: (min/step) / min -> 1/step
        # time unit conversions
        # units: (atto-mol * cell^-1 * h^-1 * (min * step^-1) / (min * hour^-1)
        #        = atto-mol * cell^-1 * step^-1
        tnfa.macrophage_secretion_rate_unit_t = tnfa.macrophage_secretion_rate * (
            self.time_step / 60
        )
        tnfa.neutrophil_secretion_rate_unit_t = tnfa.neutrophil_secretion_rate * (
            self.time_step / 60
        )
        tnfa.epithelial_secretion_rate_unit_t = tnfa.epithelial_secretion_rate * (
            self.time_step / 60
        )

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        from nlisim.modules.macrophage import MacrophageCellData, MacrophageState
        from nlisim.modules.neutrophil import NeutrophilCellData, NeutrophilState
        from nlisim.modules.phagocyte import PhagocyteStatus

        tnfa: TNFaState = state.tnfa
        molecules: MoleculesState = state.molecules
        macrophage: MacrophageState = state.macrophage
        neutrophil: NeutrophilState = state.neutrophil
        voxel_volume: float = state.voxel_volume
        grid: RectangularGrid = state.grid

        for macrophage_cell_index in macrophage.cells.alive():
            macrophage_cell: MacrophageCellData = macrophage.cells[macrophage_cell_index]
            macrophage_cell_voxel: Voxel = grid.get_voxel(macrophage_cell['point'])

            if macrophage_cell['status'] == PhagocyteStatus.ACTIVE:
                tnfa.grid[tuple(macrophage_cell_voxel)] += tnfa.macrophage_secretion_rate_unit_t

            if macrophage_cell['status'] in {PhagocyteStatus.RESTING, PhagocyteStatus.ACTIVE}:
                if (
                    activation_function(
                        x=tnfa.grid[tuple(macrophage_cell_voxel)],
                        k_d=tnfa.k_d,
                        h=self.time_step / 60,  # units: (min/step) / (min/hour)
                        volume=voxel_volume,
                        b=1,
                    )
                    > rg.uniform()
                ):
                    if macrophage_cell['status'] == PhagocyteStatus.RESTING:
                        macrophage_cell['status'] = PhagocyteStatus.ACTIVATING
                    else:
                        macrophage_cell['status'] = PhagocyteStatus.ACTIVE
                    # Note: multiple activations will reset the 'clock'
                    macrophage_cell['status_iteration'] = 0
                    macrophage_cell['tnfa'] = True

        for neutrophil_cell_index in neutrophil.cells.alive():
            neutrophil_cell: NeutrophilCellData = neutrophil.cells[neutrophil_cell_index]
            neutrophil_cell_voxel: Voxel = grid.get_voxel(neutrophil_cell['point'])

            if neutrophil_cell['status'] == PhagocyteStatus.ACTIVE:
                tnfa.grid[tuple(neutrophil_cell_voxel)] += tnfa.neutrophil_secretion_rate_unit_t

            if neutrophil_cell['status'] in {PhagocyteStatus.RESTING, PhagocyteStatus.ACTIVE}:
                if (
                    activation_function(
                        x=tnfa.grid[tuple(neutrophil_cell_voxel)],
                        k_d=tnfa.k_d,
                        h=self.time_step / 60,  # units: (min/step) / (min/hour)
                        volume=voxel_volume,
                        b=1,
                    )
                    > rg.uniform()
                ):
                    if neutrophil_cell['status'] == PhagocyteStatus.RESTING:
                        neutrophil_cell['status'] = PhagocyteStatus.ACTIVATING
                    else:
                        neutrophil_cell['status'] = PhagocyteStatus.ACTIVE
                    # Note: multiple activations will reset the 'clock'
                    neutrophil_cell['status_iteration'] = 0
                    neutrophil_cell['tnfa'] = True

        # Degrade TNFa
        tnfa.grid *= tnfa.half_life_multiplier
        tnfa.grid *= turnover_rate(
            x=np.array(1.0, dtype=np.float64),
            x_system=0.0,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )

        # Diffusion of TNFa
        tnfa.grid[:] = apply_diffusion(
            variable=tnfa.grid,
            laplacian=molecules.laplacian,
            diffusivity=molecules.diffusion_constant,
            dt=self.time_step,
        )

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        from nlisim.util import TissueType

        tnfa: TNFaState = state.tnfa
        voxel_volume = state.voxel_volume
        mask = state.lung_tissue != TissueType.AIR

        return {
            'concentration (nM)': float(np.mean(tnfa.grid[mask]) / voxel_volume / 1e9),
        }

    def visualization_data(self, state: State):
        tnfa: TNFaState = state.tnfa
        return 'molecule', tnfa.grid
