import math

import attr
import numpy as np

from nlisim.module import ModuleState
from nlisim.modulesv2.geometry import GeometryState
from nlisim.modulesv2.molecules import MoleculesState
from nlisim.modulesv2.molecule import MoleculeModel
from nlisim.state import State


def molecule_grid_factory(self: 'MIP1BState') -> np.ndarray:
    return np.zeros(shape=self.global_state.grid.shape, dtype=float)


@attr.s(kw_only=True, repr=False)
class MIP1BState(ModuleState):
    grid: np.ndarray = attr.ib(default=attr.Factory(molecule_grid_factory, takes_self=True))
    half_life: float
    half_life_multiplier: float
    macrophage_secretion_rate: float
    epithelial_secretion_rate: float
    macrophage_secretion_rate_unit_t: float
    epithelial_secretion_rate_unit_t: float
    k_d: float


class MIP1B(MoleculeModel):
    """MIP1B"""

    name = 'mip1b'
    StateClass = MIP1BState

    def initialize(self, state: State) -> State:
        mip1b: MIP1BState = state.mip1b
        geometry: GeometryState = state.geometry
        voxel_volume = geometry.voxel_volume

        # config file values
        mip1b.half_life = self.config.getfloat('half_life')
        mip1b.macrophage_secretion_rate = self.config.getfloat('macrophage_secretion_rate')
        mip1b.epithelial_secretion_rate = self.config.getfloat('epithelial_secretion_rate')
        mip1b.k_d = self.config.getfloat('k_d')

        # computed values
        mip1b.half_life_multiplier = 1 + math.log(0.5) / (mip1b.half_life / state.simulation.time_step_size)
        # time unit conversions
        mip1b.macrophage_secretion_rate_unit_t = mip1b.macrophage_secretion_rate * 60 * state.simulation.time_step_size
        mip1b.epithelial_secretion_rate_unit_t = mip1b.epithelial_secretion_rate * 60 * state.simulation.time_step_size

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        mip1b: MIP1BState = state.mip1b
        molecules: MoleculesState = state.molecules

        # TODO: move to cell
        # elif itype is Pneumocyte:
        #     if interactable.tnfa:  # interactable.status == Phagocyte.ACTIVE:
        #         self.inc(Constants.P_MIP1B_QTTY, 0)
        #     return True

        # TODO: move to cell
        # elif itype is Macrophage:
        #     if interactable.tnfa:  # interactable.status == Phagocyte.ACTIVE:# and interactable.state == Neutrophil.INTERACTING:
        #         self.inc(Constants.MA_MIP1B_QTTY, 0)
        #     # if Util.activation_function(self.values[0], Constants.Kd_MIP1B) > random():
        #     #    self.pdec(0.5)
        #     return True

        # Degrade MIP1B
        mip1b.grid *= mip1b.half_life_multiplier
        mip1b.grid *= self.turnover_rate(x_mol=np.array(1.0, dtype=np.float),
                                         x_system_mol=0.0,
                                         turnover_rate=molecules.turnover_rate,
                                         rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t)

        return state
