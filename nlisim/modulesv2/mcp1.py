import math

import attr
import numpy as np

from nlisim.module import ModuleState
from nlisim.modulesv2.geometry import GeometryState
from nlisim.modulesv2.molecules import MoleculesState
from nlisim.modulesv2.molecule import MoleculeModel
from nlisim.state import State


def molecule_grid_factory(self: 'MCP1State') -> np.ndarray:
    return np.zeros(shape=self.global_state.grid.shape, dtype=float)


@attr.s(kw_only=True, repr=False)
class MCP1State(ModuleState):
    grid: np.ndarray = attr.ib(default=attr.Factory(molecule_grid_factory, takes_self=True))
    half_life: float
    half_life_multiplier: float
    macrophage_secretion_rate: float
    epithelial_secretion_rate: float
    macrophage_secretion_rate_unit_t: float
    epithelial_secretion_rate_unit_t: float
    k_d: float


class MCP1(MoleculeModel):
    """MCP1"""

    name = 'mcp1'
    StateClass = MCP1State

    def initialize(self, state: State) -> State:
        mcp1: MCP1State = state.mcp1
        geometry: GeometryState = state.geometry
        voxel_volume = geometry.voxel_volume

        # config file values
        mcp1.half_life = self.config.getfloat('half_life')
        mcp1.macrophage_secretion_rate = self.config.getfloat('macrophage_secretion_rate')
        mcp1.epithelial_secretion_rate = self.config.getfloat('epithelial_secretion_rate')
        mcp1.k_d = self.config.getfloat('k_d')

        # computed values
        mcp1.half_life_multiplier = 1 + math.log(0.5) / (mcp1.half_life / state.simulation.time_step_size)
        # time unit conversions
        mcp1.macrophage_secretion_rate_unit_t = mcp1.macrophage_secretion_rate * 60 * state.simulation.time_step_size
        mcp1.epithelial_secretion_rate_unit_t = mcp1.epithelial_secretion_rate * 60 * state.simulation.time_step_size

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        mcp1: MCP1State = state.mcp1
        molecules: MoleculesState = state.molecules

        # TODO: move to cell
        # elif itype is Pneumocyte:
        #     if interactable.tnfa:  # interactable.status == Phagocyte.ACTIVE:
        #         self.inc(Constants.P_MCP1_QTTY, 0)
        #     return True

        # TODO: move to cell
        # elif itype is Macrophage:
        #     if interactable.tnfa:  # interactable.status == Phagocyte.ACTIVE and interactable.state == Neutrophil.INTERACTING:
        #         self.inc(Constants.MA_MCP1_QTTY, 0)
        #     return True

        # Degrade MCP1
        mcp1.grid *= mcp1.half_life_multiplier
        mcp1.grid *= self.turnover_rate(x_mol=np.array(1.0, dtype=np.float),
                                        x_system_mol=0.0,
                                        turnover_rate=molecules.turnover_rate,
                                        rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t)

        return state
