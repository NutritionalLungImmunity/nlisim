from typing import Any, Dict

import attr
from attr import attrib, attrs
import numpy as np

from nlisim.diffusion import apply_diffusion
from nlisim.module import ModuleModel, ModuleState
from nlisim.modules.molecules import MoleculesState
from nlisim.state import State
from nlisim.util import michaelian_kinetics, turnover_rate


def molecule_grid_factory(self: 'HemopexinState') -> np.ndarray:
    return np.zeros(shape=self.global_state.grid.shape, dtype=float)


@attrs(kw_only=True, repr=False)
class HemopexinState(ModuleState):
    grid: np.ndarray = attrib(
        default=attr.Factory(molecule_grid_factory, takes_self=True)
    )  # units: atto-mol
    k_m: float  # units: aM
    k_cat: float  # units: XXX
    half_life: float  # units: min
    half_life_multiplier: float  # units: proportion
    system_concentration: float  # units: aM
    system_amount_per_voxel: float  # units: atto-mol


class Hemopexin(ModuleModel):
    """Hemopexin"""

    name = 'hemopexin'
    StateClass = HemopexinState

    def initialize(self, state: State) -> State:
        from nlisim.util import TissueType

        hemopexin: HemopexinState = state.hemopexin
        voxel_volume: float = state.voxel_volume  # units: L

        # config file values
        hemopexin.k_m = self.config.getfloat('k_m')
        hemopexin.k_cat = self.config.getfloat('k_cat')
        hemopexin.system_concentration = self.config.getfloat('system_concentration')

        # computed values
        hemopexin.system_amount_per_voxel = hemopexin.system_concentration * voxel_volume
        hemopexin.half_life_multiplier = 0.5 ** (
            self.time_step / hemopexin.half_life
        )  # units in exponent: (min/step) / min -> 1/step

        # initialize grid
        hemopexin.grid[state.lung_tissue != TissueType.AIR] = hemopexin.system_amount_per_voxel

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        from nlisim.modules.hemoglobin import HemoglobinState

        hemopexin: HemopexinState = state.hemopexin
        hemoglobin: HemoglobinState = state.hemoglobin
        molecules: MoleculesState = state.molecules
        voxel_volume: float = state.voxel_volume  # units: L

        # Hemopexin / Hemoglobin reaction
        reacted_quantity = michaelian_kinetics(
            substrate=hemopexin.grid,
            enzyme=hemoglobin.grid,
            k_m=hemopexin.k_m,
            h=self.time_step / 60,  # units: (min/step) / (min/hour)
            k_cat=hemopexin.k_cat,
            voxel_volume=voxel_volume,
        )
        reacted_quantity = np.min([reacted_quantity, hemopexin.grid, hemoglobin.grid], axis=0)
        hemopexin.grid[:] = np.maximum(0.0, hemopexin.grid - reacted_quantity)
        hemoglobin.grid[:] = np.maximum(0.0, hemoglobin.grid - reacted_quantity)

        # Degrade Hemopexin
        hemopexin.grid *= hemopexin.half_life_multiplier
        hemopexin.grid *= turnover_rate(
            x=hemopexin.grid,
            x_system=hemopexin.system_amount_per_voxel,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )

        # Diffusion of Hemolysin
        hemopexin.grid[:] = apply_diffusion(
            variable=hemopexin.grid,
            laplacian=molecules.laplacian,
            diffusivity=molecules.diffusion_constant,
            dt=self.time_step,
        )

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        from nlisim.util import TissueType

        hemopexin: HemopexinState = state.hemopexin
        voxel_volume = state.voxel_volume
        mask = state.lung_tissue != TissueType.AIR

        return {
            'concentration (nM)': float(np.mean(hemopexin.grid[mask]) / voxel_volume / 1e9),
        }

    def visualization_data(self, state: State):
        hemopexin: HemopexinState = state.hemopexin
        return 'molecule', hemopexin.grid
