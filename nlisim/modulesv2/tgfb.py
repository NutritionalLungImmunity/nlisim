import math

import attr
import numpy as np

from nlisim.module import ModuleState
from nlisim.modulesv2.geometry import GeometryState
from nlisim.modulesv2.molecules import MoleculesState
from nlisim.modulesv2.molecule import MoleculeModel
from nlisim.state import State


def molecule_grid_factory(self: 'TGFBState') -> np.ndarray:
    return np.zeros(shape=self.global_state.grid.shape, dtype=float)


@attr.s(kw_only=True, repr=False)
class TGFBState(ModuleState):
    grid: np.ndarray = attr.ib(default=attr.Factory(molecule_grid_factory, takes_self=True))
    half_life: float
    half_life_multiplier: float
    macrophage_secretion_rate: float
    macrophage_secretion_rate_unit_t: float
    k_d: float


class TGFB(MoleculeModel):
    """TGFB"""

    name = 'tgfb'
    StateClass = TGFBState

    def initialize(self, state: State) -> State:
        tgfb: TGFBState = state.tgfb
        geometry: GeometryState = state.geometry
        voxel_volume = geometry.voxel_volume

        # config file values
        tgfb.half_life = self.config.getfloat('half_life')
        tgfb.macrophage_secretion_rate = self.config.getfloat('macrophage_secretion_rate')
        tgfb.k_d = self.config.getfloat('k_d')

        # computed values
        tgfb.half_life_multiplier = 1 + math.log(0.5) / (tgfb.half_life / state.simulation.time_step_size)
        # time unit conversions
        tgfb.macrophage_secretion_rate_unit_t = tgfb.macrophage_secretion_rate * 60 * state.simulation.time_step_size

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        tgfb: TGFBState = state.tgfb
        molecules: MoleculesState = state.molecules

        # TODO: move to cell
        # elif itype is Macrophage:
        #     if interactable.status == Phagocyte.INACTIVE:
        #         self.inc(Constants.MA_TGF_QTTY, 0)
        #         if Util.activation_function(self.get(0), Constants.Kd_TGF, Constants.STD_UNIT_T) > random():
        #             interactable.iteration = 0
        #     elif interactable.status != Phagocyte.APOPTOTIC and interactable.status != Phagocyte.NECROTIC and interactable.status != Phagocyte.DEAD:
        #         if Util.activation_function(self.get(0), Constants.Kd_TGF, Constants.STD_UNIT_T) > random():
        #             interactable.status = Phagocyte.INACTIVATING
        #     return True

        # Degrade TGFB
        tgfb.grid *= tgfb.half_life_multiplier
        tgfb.grid *= self.turnover_rate(x_mol=np.array(1.0, dtype=np.float64),
                                        x_system_mol=0.0,
                                        turnover_rate=molecules.turnover_rate,
                                        rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t)

        return state
