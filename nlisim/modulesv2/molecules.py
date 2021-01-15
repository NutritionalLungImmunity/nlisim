import math

import attr

from nlisim.module import ModuleModel, ModuleState
from nlisim.state import State


@attr.s(kw_only=True, repr=False)
class MoleculesState(ModuleState):
    turnover_rate: float
    cyt_bind_t: float
    rel_cyt_bind_unit_t: float
    turnover_rate: float

class Molecules(ModuleModel):
    name = 'molecules'
    StateClass = MoleculesState

    def initialize(self, state: State):
        molecules: MoleculesState = state.molecules

        molecules.cyt_bind_t = self.config.getfloat('cyt_bind_t')
        molecules.turnover_rate = self.config.getfloat('turnover_rate')

        # Computed values
        molecules.rel_cyt_bind_unit_t = state.simulation.time_step_size / molecules.cyt_bind_t
        # TODO: original comments as below. Is the param 0.2? i.e. ...math.log(1+0.2)... Yes, 20% per hour
        # 0.2 # 10.1124/jpet.118.250134 (approx) 0.2/h CHANGE!!!!
        molecules.turnover_rate = 1 - math.log(1.2) / int(30 / state.simulation.time_step_size)

        return state

    def advance(self, state: State, previous_time: float):
        """Advance the state by a single time step."""
        return state
