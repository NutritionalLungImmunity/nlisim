import math

import attr
import numpy as np

from nlisim.module import ModuleState
from nlisim.modulesv2.geometry import GeometryState
from nlisim.modulesv2.molecules import MoleculesState
from nlisim.modulesv2.molecule import MoleculeModel
from nlisim.state import State


def molecule_grid_factory(self: 'IL10State') -> np.ndarray:
    return np.zeros(shape=self.global_state.grid.shape, dtype=float)


@attr.s(kw_only=True, repr=False)
class IL10State(ModuleState):
    grid: np.ndarray = attr.ib(default=attr.Factory(molecule_grid_factory, takes_self=True))
    half_life: float
    half_life_multiplier: float
    macrophage_secretion_rate: float
    macrophage_secretion_rate_unit_t: float
    k_d: float


class IL10(MoleculeModel):
    """IL10"""

    name = 'il10'
    StateClass = IL10State

    def initialize(self, state: State) -> State:
        il10: IL10State = state.il10
        geometry: GeometryState = state.geometry
        voxel_volume = geometry.voxel_volume

        # config file values
        il10.half_life = self.config.getfloat('half_life')
        il10.macrophage_secretion_rate = self.config.getfloat('macrophage_secretion_rate')
        il10.k_d = self.config.getfloat('k_d')

        # computed values
        il10.half_life_multiplier = 1 + math.log(0.5) / (il10.half_life / state.simulation.time_step_size)
        # time unit conversions
        il10.macrophage_secretion_rate_unit_t = il10.macrophage_secretion_rate * 60 * state.simulation.time_step_size

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        il10: IL10State = state.il10
        molecules: MoleculesState = state.molecules

        # TODO: move to cell
        # elif itype is Macrophage:  # or type(interactable) is Neutrophil:
        #     if interactable.status == Phagocyte.ACTIVE and interactable.state == Neutrophil.INTERACTING:
        #         self.inc(Constants.MA_IL10_QTTY, 0)
        #     if interactable.status != Phagocyte.DEAD and interactable.status != Phagocyte.APOPTOTIC and interactable.status != Phagocyte.NECROTIC:
        #         if Util.activation_function(self.get(0), Constants.Kd_IL10, Constants.STD_UNIT_T) > random():
        #             interactable.status = Phagocyte.INACTIVATING if interactable.status != Phagocyte.INACTIVE else Phagocyte.INACTIVE
        #             interactable.interation = 0
        #     return True
        
        # Degrade IL10
        il10.grid *= il10.half_life_multiplier
        il10.grid *= self.turnover_rate(x_mol=np.ones(shape=il10.grid.shape, dtype=np.float),
                                        x_system_mol=0.0,
                                        turnover_rate=molecules.turnover_rate,
                                        rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t)

        return state
