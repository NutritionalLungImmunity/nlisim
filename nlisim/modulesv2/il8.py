import math

import attr
import numpy as np

from nlisim.module import ModuleState
from nlisim.modulesv2.geometry import GeometryState
from nlisim.modulesv2.molecules import MoleculesState
from nlisim.modulesv2.molecule import MoleculeModel
from nlisim.state import State


def molecule_grid_factory(self: 'IL8State') -> np.ndarray:
    return np.zeros(shape=self.global_state.grid.shape, dtype=float)


@attr.s(kw_only=True, repr=False)
class IL8State(ModuleState):
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


class IL8(MoleculeModel):
    """IL8"""

    name = 'il8'
    StateClass = IL8State

    def initialize(self, state: State) -> State:
        il8: IL8State = state.il8
        geometry: GeometryState = state.geometry
        voxel_volume = geometry.voxel_volume

        # config file values
        il8.half_life = self.config.getfloat('half_life')
        il8.macrophage_secretion_rate = self.config.getfloat('macrophage_secretion_rate')
        il8.neutrophil_secretion_rate = self.config.getfloat('neutrophil_secretion_rate')
        il8.epithelial_secretion_rate = self.config.getfloat('epithelial_secretion_rate')
        il8.k_d = self.config.getfloat('k_d')

        # computed values
        il8.half_life_multiplier = 1 + math.log(0.5) / (il8.half_life / state.simulation.time_step_size)
        # time unit conversions
        il8.macrophage_secretion_rate_unit_t = il8.macrophage_secretion_rate * 60 * state.simulation.time_step_size
        il8.neutrophil_secretion_rate_unit_t = il8.neutrophil_secretion_rate * 60 * state.simulation.time_step_size
        il8.epithelial_secretion_rate_unit_t = il8.epithelial_secretion_rate * 60 * state.simulation.time_step_size

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        il8: IL8State = state.il8
        molecules: MoleculesState = state.molecules

        # TODO: move to cell
        # elif itype is Macrophage or itype is Neutrophil:
        #     if interactable.tnfa:  # interactable.status == Phagocyte.ACTIVE and interactable.state == Neutrophil.INTERACTING:
        #         self.inc(0, 0)
        #     if (interactable.status == Phagocyte.RESTING or interactable.status == Phagocyte.ACTIVE) and type(
        #             interactable) is Neutrophil:
        #         if Util.activation_function(self.get(0), Constants.Kd_IL8, Constants.STD_UNIT_T) > random():
        #             interactable.status = Phagocyte.ACTIVE
        #             interactable.iteration = 0
        #     return True

        # Degrade IL8
        il8.grid *= il8.half_life_multiplier
        il8.grid *= self.turnover_rate(x_mol=np.ones(shape=il8.grid.shape, dtype=np.float),
                                       x_system_mol=0.0,
                                       turnover_rate=molecules.turnover_rate,
                                       rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t)

        return state
