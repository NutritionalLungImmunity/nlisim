import math

import attr
import numpy as np

from nlisim.module import ModuleState
from nlisim.modulesv2.molecules import MoleculesState
from nlisim.modulesv2.molecule import MoleculeModel
from nlisim.state import State


def molecule_grid_factory(self: 'TNFaState') -> np.ndarray:
    return np.zeros(shape=self.global_state.grid.shape, dtype=float)


@attr.s(kw_only=True, repr=False)
class TNFaState(ModuleState):
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


class TNFa(MoleculeModel):
    name = 'tnfa'
    StateClass = TNFaState

    def initialize(self, state: State) -> State:
        tnfa: TNFaState = state.antitnfa

        # config file values
        tnfa.half_life = self.config.getfloat('half_life')
        tnfa.macrophage_secretion_rate = self.config.getfloat('macrophage_secretion_rate')
        tnfa.neutrophil_secretion_rate = self.config.getfloat('neutrophil_secretion_rate')
        tnfa.epithelial_secretion_rate = self.config.getfloat('epithelial_secretion_rate')
        tnfa.k_d = self.config.getfloat('k_d')

        # computed values
        tnfa.half_life_multiplier = 1 + math.log(0.5) / (tnfa.half_life / state.simulation.time_step_size)
        # time unit conversions
        tnfa.macrophage_secretion_rate_unit_t = tnfa.macrophage_secretion_rate * 60 * state.simulation.time_step_size
        tnfa.neutrophil_secretion_rate_unit_t = tnfa.neutrophil_secretion_rate * 60 * state.simulation.time_step_size
        tnfa.epithelial_secretion_rate_unit_t = tnfa.epithelial_secretion_rate * 60 * state.simulation.time_step_size

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        tnfa: TNFaState = state.antitnfa
        molecules: MoleculesState = state.molecules

        # Degrade TNFa
        tnfa.grid *= tnfa.half_life_multiplier
        tnfa.grid *= self.turnover_rate(x_mol=1,
                                        x_system_mol=0,
                                        turnover_rate=molecules.turnover_rate,
                                        rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t)

        # TODO: move these interactions to the cells
        # elif itype is Macrophage:
        #     if interactable.status == Phagocyte.ACTIVE:  # and interactable.state == Neutrophil.INTERACTING:
        #         self.inc(Constants.MA_TNF_QTTY, 0)
        #     if interactable.status == Phagocyte.RESTING or interactable.status == Phagocyte.ACTIVE:
        #         if Util.activation_function(self.get(0), Constants.Kd_TNF, Constants.STD_UNIT_T) > random():
        #             interactable.status = Phagocyte.ACTIVATING \
        #                 if interactable.status == Phagocyte.RESTING else Phagocyte.ACTIVE
        #             interactable.iteration = 0
        #             interactable.tnfa = True
        #     return True

        # TODO: move these interactions to the cells
        # elif itype is Neutrophil:
        #     if interactable.status == Phagocyte.ACTIVE:  # and interactable.state == Neutrophil.INTERACTING:
        #         self.inc(Constants.N_TNF_QTTY, 0)
        #     if interactable.status == Phagocyte.RESTING or interactable.status == Phagocyte.ACTIVE:
        #         if Util.activation_function(self.get(0), Constants.Kd_TNF, Constants.STD_UNIT_T) > random():
        #             interactable.status = Phagocyte.ACTIVATING \
        #                 if interactable.status == Phagocyte.RESTING else Phagocyte.ACTIVE
        #             interactable.iteration = 0
        #             interactable.tnfa = True
        #     return True

        # Degrade TNFA
        tnfa.grid *= tnfa.half_life_multiplier
        tnfa.grid *= self.turnover_rate(x_mol=np.array(1.0, dtype=np.float),
                                        x_system_mol=0.0,
                                        turnover_rate=molecules.turnover_rate,
                                        rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t)

        return state
