import math

import attr
import numpy as np

from nlisim.module import ModuleState
from nlisim.modulesv2.geometry import GeometryState
from nlisim.modulesv2.molecule import MoleculeModel
from nlisim.modulesv2.molecules import MoleculesState
from nlisim.modulesv2.tnfa import TNFaState
from nlisim.state import State
from nlisim.util import turnover_rate


def molecule_grid_factory(self: 'AntiTNFaState') -> np.ndarray:
    return np.zeros(shape=self.global_state.grid.shape, dtype=float)


@attr.s(kw_only=True, repr=False)
class AntiTNFaState(ModuleState):
    grid: np.ndarray = attr.ib(default=attr.Factory(molecule_grid_factory, takes_self=True))
    half_life: float
    half_life_multiplier: float
    react_time_unit: float
    k_m: float
    system_concentration: float
    system_amount_per_voxel: float
    turnover_rate: float


class AntiTNFa(MoleculeModel):
    name = 'antitnfa'
    StateClass = AntiTNFaState

    def initialize(self, state: State) -> State:
        anti_tnf_a: AntiTNFaState = state.antitnfa
        geometry: GeometryState = state.geometry
        voxel_volume = geometry.voxel_volume

        # config file values
        anti_tnf_a.half_life = self.config.getfloat('half_life')
        anti_tnf_a.react_time_unit = self.config.getfloat('react_time_unit')
        anti_tnf_a.k_m = self.config.getfloat('k_m')
        anti_tnf_a.system_concentration = self.config.getfloat('system_concentration')

        # computed values
        anti_tnf_a.system_amount_per_voxel = anti_tnf_a.system_concentration * voxel_volume
        anti_tnf_a.half_life_multiplier = 1 + math.log(0.5) / (anti_tnf_a.half_life / state.simulation.time_step_size)

        # initialize concentration field
        anti_tnf_a.grid = anti_tnf_a.system_amount_per_voxel

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advances the state by a single time step."""
        anti_tnf_a: AntiTNFaState = state.antitnfa
        molecules: MoleculesState = state.molecules
        geometry: GeometryState = state.geometry
        voxel_volume = geometry.voxel_volume
        tnf_a: TNFaState = state.tnfa

        # AntiTNFa / TNFa reaction
        reacted_quantity = self.michaelian_kinetics(substrate=anti_tnf_a.grid,
                                                    enzyme=tnf_a.grid,
                                                    km=anti_tnf_a.k_m,
                                                    h=anti_tnf_a.react_time_unit,
                                                    voxel_volume=voxel_volume)
        reacted_quantity = np.min([reacted_quantity, anti_tnf_a.grid, tnf_a.grid], axis=0)
        anti_tnf_a.grid = np.maximum(0.0, anti_tnf_a.grid - reacted_quantity)
        tnf_a.grid = np.maximum(0.0, tnf_a.grid - reacted_quantity)

        # Degradation of AntiTNFa
        anti_tnf_a.system_amount_per_voxel *= anti_tnf_a.half_life_multiplier
        anti_tnf_a.grid *= turnover_rate(x_mol=anti_tnf_a.grid,
                                         x_system_mol=anti_tnf_a.system_amount_per_voxel,
                                         turnover_rate=molecules.turnover_rate,
                                         rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t)

        # Diffusion of AntiTNFa
        self.diffuse(anti_tnf_a.grid, molecules.diffusion_constant_timestep)

        return state
