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


def molecule_grid_factory(self: 'TGFBState') -> np.ndarray:
    return np.zeros(shape=self.global_state.grid.shape, dtype=float)


@attr.s(kw_only=True, repr=False)
class TGFBState(ModuleState):
    grid: np.ndarray = attr.ib(
        default=attr.Factory(molecule_grid_factory, takes_self=True)
    )  # units: atto-mols
    half_life: float  # units: min
    half_life_multiplier: float  # units: proportion
    macrophage_secretion_rate: float  # units: atto-mol * cell^-1 * h^-1
    macrophage_secretion_rate_unit_t: float  # units: atto-mol * cell^-1 * step^-1
    k_d: float  # aM


class TGFB(ModuleModel):
    """TGFB"""

    name = 'tgfb'
    StateClass = TGFBState

    def initialize(self, state: State) -> State:
        tgfb: TGFBState = state.tgfb

        # config file values
        tgfb.half_life = self.config.getfloat('half_life')  # units: min
        tgfb.macrophage_secretion_rate = self.config.getfloat(
            'macrophage_secretion_rate'
        )  # units: atto-mol * cell^-1 * h^-1
        tgfb.k_d = self.config.getfloat('k_d')  # units: aM

        # computed values
        tgfb.half_life_multiplier = 0.5 ** (
            self.time_step / tgfb.half_life
        )  # units in exponent: (min/step) / min -> 1/step
        # time unit conversions
        tgfb.macrophage_secretion_rate_unit_t = tgfb.macrophage_secretion_rate * (
            self.time_step / 60
        )  # units: atto-mol/(cell*h) * (min/step) / (min/hour)

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        from nlisim.modules.macrophage import MacrophageCellData, MacrophageState
        from nlisim.modules.phagocyte import PhagocyteStatus

        tgfb: TGFBState = state.tgfb
        molecules: MoleculesState = state.molecules
        macrophage: MacrophageState = state.macrophage
        voxel_volume: float = state.voxel_volume
        grid: RectangularGrid = state.grid

        for macrophage_cell_index in macrophage.cells.alive():
            macrophage_cell: MacrophageCellData = macrophage.cells[macrophage_cell_index]
            macrophage_cell_voxel: Voxel = grid.get_voxel(macrophage_cell['point'])

            if macrophage_cell['status'] == PhagocyteStatus.INACTIVE:
                tgfb.grid[tuple(macrophage_cell_voxel)] += tgfb.macrophage_secretion_rate_unit_t
                if (
                    activation_function(
                        x=tgfb.grid[tuple(macrophage_cell_voxel)],
                        k_d=tgfb.k_d,
                        h=self.time_step / 60,  # units: (min/step) / (min/hour)
                        volume=voxel_volume,
                        b=1,
                    )
                    > rg.uniform()
                ):
                    macrophage_cell['status_iteration'] = 0

            elif macrophage_cell['status'] not in {
                PhagocyteStatus.APOPTOTIC,
                PhagocyteStatus.NECROTIC,
                PhagocyteStatus.DEAD,
            }:
                if (
                    activation_function(
                        x=tgfb.grid[tuple(macrophage_cell_voxel)],
                        k_d=tgfb.k_d,
                        h=self.time_step / 60,  # units: (min/step) / (min/hour)
                        volume=voxel_volume,
                        b=1,
                    )
                    > rg.uniform()
                ):
                    macrophage_cell['status'] = PhagocyteStatus.INACTIVATING
                    macrophage_cell[
                        'status_iteration'
                    ] = 0  # Previously, was no reset of the status iteration

        # Degrade TGFB
        tgfb.grid *= tgfb.half_life_multiplier
        tgfb.grid *= turnover_rate(
            x=np.array(1.0, dtype=np.float64),
            x_system=0.0,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )

        # Diffusion of TGFB
        tgfb.grid[:] = apply_diffusion(
            variable=tgfb.grid,
            laplacian=molecules.laplacian,
            diffusivity=molecules.diffusion_constant,
            dt=self.time_step,
        )

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        tgfb: TGFBState = state.tgfb
        voxel_volume = state.voxel_volume

        return {
            'concentration (nM)': float(np.mean(tgfb.grid) / voxel_volume / 1e9),
        }

    def visualization_data(self, state: State):
        tgfb: TGFBState = state.tgfb
        return 'molecule', tgfb.grid
