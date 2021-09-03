from typing import Any, Dict

import attr
from attr import attrib, attrs
import numpy as np

from nlisim.diffusion import apply_diffusion
from nlisim.module import ModuleModel, ModuleState
from nlisim.modules.molecules import MoleculesState
from nlisim.state import State
from nlisim.util import michaelian_kinetics, turnover_rate


def molecule_grid_factory(self: 'EstBState') -> np.ndarray:
    return np.zeros(shape=self.global_state.grid.shape, dtype=float)


@attrs(kw_only=True, repr=False)
class EstBState(ModuleState):
    grid: np.ndarray = attrib(
        default=attr.Factory(molecule_grid_factory, takes_self=True)
    )  # units: atto-mols
    iron_buffer: np.ndarray = attrib(default=attr.Factory(molecule_grid_factory, takes_self=True))
    half_life: float  # units: min
    half_life_multiplier: float  # units: proportion
    k_m: float  # units: aM
    k_cat: float
    system_concentration: float
    system_amount_per_voxel: float


class EstB(ModuleModel):
    """Esterase B"""

    name = 'estb'
    StateClass = EstBState

    def initialize(self, state: State) -> State:
        from nlisim.util import TissueType

        estb: EstBState = state.estb
        voxel_volume = state.voxel_volume
        lung_tissue = state.lung_tissue

        # config file values
        estb.half_life = self.config.getfloat('half_life')
        estb.k_m = self.config.getfloat('k_m')
        estb.k_cat = self.config.getfloat('k_cat')
        estb.system_concentration = self.config.getfloat('system_concentration')

        # computed values
        estb.half_life_multiplier = 0.5 ** (
            self.time_step / estb.half_life
        )  # units: (min/step) / min -> 1/step
        estb.system_amount_per_voxel = estb.system_concentration * voxel_volume

        # initialize concentration field
        estb.grid[lung_tissue != TissueType.AIR] = estb.system_amount_per_voxel

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        from nlisim.modules.iron import IronState
        from nlisim.modules.tafc import TAFCState

        estb: EstBState = state.estb
        iron: IronState = state.iron
        tafc: TAFCState = state.tafc
        molecules: MoleculesState = state.molecules
        voxel_volume = state.voxel_volume

        # contribute our iron buffer to the iron pool
        iron.grid += estb.iron_buffer
        estb.iron_buffer[:] = 0.0

        # interact with TAFC
        v1 = michaelian_kinetics(
            substrate=tafc.grid["TAFC"],
            enzyme=estb.grid,
            k_m=estb.k_m,
            k_cat=estb.k_cat,
            h=self.time_step / 60,  # units: (min/step) / (min/hour)
            voxel_volume=voxel_volume,
        )
        v2 = michaelian_kinetics(
            substrate=tafc.grid["TAFCBI"],
            enzyme=estb.grid,
            k_m=estb.k_m,
            k_cat=estb.k_cat,
            h=self.time_step / 60,  # units: (min/step) / (min/hour)
            voxel_volume=voxel_volume,
        )
        tafc.grid["TAFC"] -= v1
        tafc.grid["TAFCBI"] -= v2
        estb.iron_buffer += v2  # set equal to zero previously

        # Degrade EstB
        estb.grid *= estb.half_life_multiplier
        estb.grid *= turnover_rate(
            x=estb.grid,
            x_system=estb.system_amount_per_voxel,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )

        # Diffusion of EstB
        estb.grid[:] = apply_diffusion(
            variable=estb.grid,
            laplacian=molecules.laplacian,
            diffusivity=molecules.diffusion_constant,
            dt=self.time_step,
        )

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        estb: EstBState = state.estb
        voxel_volume = state.voxel_volume

        return {
            'concentration (nM)': float(np.mean(estb.grid) / voxel_volume / 1e9),
        }

    def visualization_data(self, state: State):
        estb: EstBState = state.estb
        return 'molecule', estb.grid
