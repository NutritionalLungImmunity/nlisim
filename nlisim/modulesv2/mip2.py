import math

import attr
import numpy as np

from nlisim.module import ModuleState
from nlisim.modulesv2.geometry import GeometryState
from nlisim.modulesv2.molecules import MoleculesState
from nlisim.modulesv2.molecule import MoleculeModel
from nlisim.state import State


def molecule_grid_factory(self: 'MIP2State') -> np.ndarray:
    return np.zeros(shape=self.global_state.grid.shape, dtype=float)


@attr.s(kw_only=True, repr=False)
class MIP2State(ModuleState):
    grid: np.ndarray = attr.ib(default=attr.Factory(molecule_grid_factory, takes_self=True))
    half_life: float
    half_life_multiplier: float
    macrophage_secretion_rate: float
    epithelial_secretion_rate: float
    macrophage_secretion_rate_unit_t: float
    epithelial_secretion_rate_unit_t: float
    k_d: float


class MIP2(MoleculeModel):
    """MIP2"""

    name = 'mip2'
    StateClass = MIP2State

    def initialize(self, state: State) -> State:
        mip2: MIP2State = state.mip2
        geometry: GeometryState = state.geometry
        voxel_volume = geometry.voxel_volume

        # config file values
        mip2.half_life = self.config.getfloat('half_life')
        mip2.macrophage_secretion_rate = self.config.getfloat('macrophage_secretion_rate')
        mip2.neutrophil_secretion_rate = self.config.getfloat('neutrophil_secretion_rate')
        mip2.epithelial_secretion_rate = self.config.getfloat('epithelial_secretion_rate')
        mip2.k_d = self.config.getfloat('k_d')

        # computed values
        mip2.half_life_multiplier = 1 + math.log(0.5) / (mip2.half_life / state.simulation.time_step_size)
        # time unit conversions
        mip2.macrophage_secretion_rate_unit_t = mip2.macrophage_secretion_rate * 60 * state.simulation.time_step_size
        mip2.neutrophil_secretion_rate_unit_t = mip2.neutrophil_secretion_rate * 60 * state.simulation.time_step_size
        mip2.epithelial_secretion_rate_unit_t = mip2.epithelial_secretion_rate * 60 * state.simulation.time_step_size

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        mip2: MIP2State = state.mip2
        molecules: MoleculesState = state.molecules

        # TODO: move to cell
        # elif itype is Neutrophil:
        #     if interactable.status == Phagocyte.RESTING:
        #         if Util.activation_function(self.get(0), Constants.Kd_MIP2, Constants.STD_UNIT_T) > random():
        #             interactable.status = Phagocyte.ACTIVATING
        #     elif interactable.tnfa:  # interactable.status == Phagocyte.ACTIVE and interactable.state == Neutrophil.INTERACTING:
        #         self.inc(Constants.N_MIP2_QTTY, 0)
        #         if Util.activation_function(self.get(0), Constants.Kd_MIP2, Constants.STD_UNIT_T) > random():
        #             interactable.interaction = 0
        #     # if Util.activation_function(self.values[0], Constants.Kd_MIP2) > random():
        #     #    self.pdec(0.5)
        #     return True

        # TODO: move to cell
        # elif itype is Pneumocyte:
        #     if interactable.tnfa:  # interactable.status == Phagocyte.ACTIVE:
        #         self.inc(Constants.P_MIP2_QTTY, 0)
        #     return True

        # TODO: move to cell
        # elif itype is Macrophage:
        #     if interactable.tnfa:  # interactable.status == Phagocyte.ACTIVE:# and interactable.state == Neutrophil.INTERACTING:
        #         self.inc(Constants.MA_MIP2_QTTY, 0)
        #     return True

        # Degrade MIP2
        mip2.grid *= mip2.half_life_multiplier
        mip2.grid *= self.turnover_rate(x_mol=np.array(1.0, dtype=np.float64),
                                       x_system_mol=0.0,
                                       turnover_rate=molecules.turnover_rate,
                                       rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t)

        return state
