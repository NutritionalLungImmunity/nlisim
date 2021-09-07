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


def molecule_grid_factory(self: 'IL10State') -> np.ndarray:
    return np.zeros(shape=self.global_state.grid.shape, dtype=float)


@attr.s(kw_only=True, repr=False)
class IL10State(ModuleState):
    grid: np.ndarray = attr.ib(
        default=attr.Factory(molecule_grid_factory, takes_self=True)
    )  # units: aM
    half_life: float  # units: min
    half_life_multiplier: float  # units: proportion
    macrophage_secretion_rate: float  # units: atto-mol * cell^-1 * h^-1
    macrophage_secretion_rate_unit_t: float  # units: atto-mol * cell^-1 * h^-1
    k_d: float  # units: aM


class IL10(ModuleModel):
    """IL10"""

    name = 'il10'
    StateClass = IL10State

    def initialize(self, state: State) -> State:
        il10: IL10State = state.il10

        # config file values
        il10.half_life = self.config.getfloat('half_life')  # units: min
        il10.macrophage_secretion_rate = self.config.getfloat(
            'macrophage_secretion_rate'
        )  # units: atto-mol * cell^-1 * h^-1
        il10.k_d = self.config.getfloat('k_d')  # units: aM

        # computed values
        il10.half_life_multiplier = 0.5 ** (
            self.time_step / il10.half_life
        )  # units in exponent: (min/step) / min -> 1/step
        # time unit conversions
        il10.macrophage_secretion_rate_unit_t = il10.macrophage_secretion_rate * (
            self.time_step / 60
        )  # units: atto-mol * cell^-1 * h^-1 * (min/step) / (min/hour)

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        from nlisim.modules.macrophage import MacrophageCellData, MacrophageState
        from nlisim.modules.phagocyte import PhagocyteState, PhagocyteStatus

        il10: IL10State = state.il10
        macrophage: MacrophageState = state.macrophage
        molecules: MoleculesState = state.molecules
        voxel_volume: float = state.voxel_volume
        grid: RectangularGrid = state.grid

        # active Macrophages secrete il10 and non-dead macrophages can become inactivated by il10
        for macrophage_cell_index in macrophage.cells.alive():
            macrophage_cell: MacrophageCellData = macrophage.cells[macrophage_cell_index]
            macrophage_cell_voxel: Voxel = grid.get_voxel(macrophage_cell['point'])

            if (
                macrophage_cell['status'] == PhagocyteStatus.ACTIVE
                and macrophage_cell['state'] == PhagocyteState.INTERACTING
            ):
                il10.grid[tuple(macrophage_cell_voxel)] += il10.macrophage_secretion_rate_unit_t

            if macrophage_cell['status'] not in {
                PhagocyteStatus.DEAD,
                PhagocyteStatus.APOPTOTIC,
                PhagocyteStatus.NECROTIC,
            } and (
                activation_function(
                    x=il10.grid[tuple(macrophage_cell_voxel)],
                    k_d=il10.k_d,
                    h=self.time_step / 60,  # units: (min/step) / (min/hour)
                    volume=voxel_volume,
                    b=1,
                )
                > rg.uniform()
            ):
                # inactive cells stay inactive, others become inactivating
                if macrophage_cell['status'] != PhagocyteStatus.INACTIVE:
                    macrophage_cell['status'] = PhagocyteStatus.INACTIVATING
                macrophage_cell['status_iteration'] = 0

        # Degrade IL10
        il10.grid *= il10.half_life_multiplier
        il10.grid *= turnover_rate(
            x=np.ones(shape=il10.grid.shape, dtype=np.float64),
            x_system=0.0,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )

        # Diffusion of IL10
        il10.grid[:] = apply_diffusion(
            variable=il10.grid,
            laplacian=molecules.laplacian,
            diffusivity=molecules.diffusion_constant,
            dt=self.time_step,
        )

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        from nlisim.util import TissueType

        il10: IL10State = state.il10
        voxel_volume = state.voxel_volume
        mask = state.lung_tissue != TissueType.AIR

        return {
            'concentration (nM)': float(np.mean(il10.grid[mask]) / voxel_volume / 1e9),
        }

    def visualization_data(self, state: State):
        il10: IL10State = state.il10
        return 'molecule', il10.grid
