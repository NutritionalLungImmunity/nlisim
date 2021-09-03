from typing import Any, Dict

import attr
import numpy as np

from nlisim.diffusion import apply_diffusion
from nlisim.module import ModuleModel, ModuleState
from nlisim.modules.molecules import MoleculesState
from nlisim.state import State
from nlisim.util import michaelian_kinetics, turnover_rate


def molecule_grid_factory(self: 'AntiTNFaState') -> np.ndarray:
    return np.zeros(shape=self.global_state.grid.shape, dtype=float)


@attr.s(kw_only=True, repr=False)
class AntiTNFaState(ModuleState):
    grid: np.ndarray = attr.ib(
        default=attr.Factory(molecule_grid_factory, takes_self=True)
    )  # units: atto-mol
    half_life: float  # units: min
    half_life_multiplier: float  # units: proportion
    react_time_unit: float  # units: hour/step
    k_m: float  # units: aM
    system_concentration: float  # units: aM
    system_amount_per_voxel: float  # units: atto-mol


class AntiTNFa(ModuleModel):
    name = 'antitnfa'
    StateClass = AntiTNFaState

    def initialize(self, state: State) -> State:
        from nlisim.util import TissueType

        anti_tnf_a: AntiTNFaState = state.antitnfa
        voxel_volume = state.voxel_volume
        lung_tissue = state.lung_tissue

        # config file values
        anti_tnf_a.half_life = self.config.getfloat('half_life')  # units: min
        anti_tnf_a.react_time_unit = self.config.getfloat(
            'react_time_unit'
        )  # units: sec TODO: understand this
        anti_tnf_a.k_m = self.config.getfloat('k_m')  # units: aM
        anti_tnf_a.system_concentration = self.config.getfloat('system_concentration')  # units: aM

        # computed values
        anti_tnf_a.system_amount_per_voxel = (
            anti_tnf_a.system_concentration * voxel_volume
        )  # units: aM * L = atto-mol
        anti_tnf_a.half_life_multiplier = 0.5 ** (
            self.time_step / anti_tnf_a.half_life
        )  # units in exponent: (min/step) / min -> 1/step

        # initialize concentration field
        anti_tnf_a.grid[lung_tissue != TissueType.AIR] = anti_tnf_a.system_amount_per_voxel

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advances the state by a single time step."""
        from nlisim.modules.tnfa import TNFaState

        anti_tnf_a: AntiTNFaState = state.antitnfa
        molecules: MoleculesState = state.molecules
        voxel_volume = state.voxel_volume
        tnf_a: TNFaState = state.tnfa

        # AntiTNFa / TNFa reaction
        reacted_quantity = michaelian_kinetics(
            substrate=anti_tnf_a.grid,
            enzyme=tnf_a.grid,
            k_m=anti_tnf_a.k_m,
            h=anti_tnf_a.react_time_unit,  # TODO: understand why units are seconds here
            k_cat=1.0,  # default
            voxel_volume=voxel_volume,
        )
        reacted_quantity = np.min([reacted_quantity, anti_tnf_a.grid, tnf_a.grid], axis=0)
        anti_tnf_a.grid[:] = np.maximum(0.0, anti_tnf_a.grid - reacted_quantity)
        tnf_a.grid[:] = np.maximum(0.0, tnf_a.grid - reacted_quantity)

        # Degradation of AntiTNFa
        anti_tnf_a.system_amount_per_voxel *= anti_tnf_a.half_life_multiplier
        anti_tnf_a.grid *= turnover_rate(
            x=anti_tnf_a.grid,
            x_system=anti_tnf_a.system_amount_per_voxel,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )

        # Diffusion of AntiTNFa
        anti_tnf_a.grid[:] = apply_diffusion(
            variable=anti_tnf_a.grid,
            laplacian=molecules.laplacian,
            diffusivity=molecules.diffusion_constant,
            dt=self.time_step,
        )

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        from nlisim.util import TissueType

        anti_tnf_a: AntiTNFaState = state.antitnfa
        voxel_volume = state.voxel_volume
        mask = state.lung_tissue != TissueType.AIR

        return {
            'concentration (nM)': float(np.mean(anti_tnf_a.grid[mask]) / voxel_volume / 1e9),
        }

    def visualization_data(self, state: State):
        anti_tnf_a: AntiTNFaState = state.antitnfa
        return 'molecule', anti_tnf_a.grid
