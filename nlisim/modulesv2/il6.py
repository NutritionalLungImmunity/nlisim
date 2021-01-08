import math

import attr
import numpy as np

from nlisim.module import ModuleState
from nlisim.modulesv2.geometry import GeometryState
from nlisim.modulesv2.molecules import MoleculesState
from nlisim.modulesv2.molecule import MoleculeModel
from nlisim.state import State


def molecule_grid_factory(self: 'IL6State') -> np.ndarray:
    return np.zeros(shape=self.global_state.grid.shape, dtype=float)


@attr.s(kw_only=True, repr=False)
class IL6State(ModuleState):
    grid: np.ndarray = attr.ib(default=attr.Factory(molecule_grid_factory, takes_self=True))
    half_life: float
    half_life_multiplier: float
    macrophage_secretion_rate: float
    neutrophil_secretion_rate: float
    epithelial_secretion_rate: float
    macrophage_secretion_rate_unit_t: float
    neutrophil_secretion_rate_unit_t: float
    epithelial_secretion_rate_unit_t: float
    k_d: float


class IL6(MoleculeModel):
    """IL6"""

    name = 'il6'
    StateClass = IL6State

    def initialize(self, state: State) -> State:
        il6: IL6State = state.il6
        geometry: GeometryState = state.geometry
        voxel_volume = geometry.voxel_volume

        # config file values
        il6.half_life = self.config.getfloat('half_life')
        il6.macrophage_secretion_rate = self.config.getfloat('macrophage_secretion_rate')
        il6.neutrophil_secretion_rate = self.config.getfloat('neutrophil_secretion_rate')
        il6.epithelial_secretion_rate = self.config.getfloat('epithelial_secretion_rate')
        il6.k_d = self.config.getfloat('k_d')

        # computed values
        il6.half_life_multiplier = 1 + math.log(0.5) / (il6.half_life / state.simulation.time_step_size)
        # time unit conversions
        il6.macrophage_secretion_rate_unit_t = il6.macrophage_secretion_rate * 60 * state.simulation.time_step_size
        il6.neutrophil_secretion_rate_unit_t = il6.neutrophil_secretion_rate * 60 * state.simulation.time_step_size
        il6.epithelial_secretion_rate_unit_t = il6.epithelial_secretion_rate * 60 * state.simulation.time_step_size

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        il6: IL6State = state.il6
        molecules: MoleculesState = state.molecules

        # TODO: move to cell
        # elif itype is Macrophage:
        #     if interactable.status == Phagocyte.ACTIVE:  # and interactable.state == Neutrophil.INTERACTING:
        #         self.inc(Constants.MA_IL6_QTTY, 0)
        #     return True

        # TODO: move to cell
        # elif itype is Neutrophil:
        #     if interactable.status == Phagocyte.ACTIVE:  # and interactable.state == Neutrophil.INTERACTING:
        #         self.inc(Constants.N_IL6_QTTY, 0)
        #     return True

        # Degrade IL6
        il6.grid *= il6.half_life_multiplier
        il6.grid *= self.turnover_rate(x_mol=np.ones(shape=il6.grid.shape, dtype=np.float),
                                       x_system_mol=0.0,
                                       turnover_rate=molecules.turnover_rate,
                                       rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t)

        return state
