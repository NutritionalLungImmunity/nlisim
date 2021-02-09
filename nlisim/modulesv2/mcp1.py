import math

import attr
import numpy as np

from nlisim.coordinates import Voxel
from nlisim.grid import RectangularGrid
from nlisim.module import ModuleState
from nlisim.modulesv2.molecule import MoleculeModel
from nlisim.modulesv2.molecules import MoleculesState
from nlisim.state import State
from nlisim.util import turnover_rate


def molecule_grid_factory(self: 'MCP1State') -> np.ndarray:
    return np.zeros(shape=self.global_state.grid.shape, dtype=float)


@attr.s(kw_only=True, repr=False)
class MCP1State(ModuleState):
    grid: np.ndarray = attr.ib(default=attr.Factory(molecule_grid_factory, takes_self=True))
    half_life: float
    half_life_multiplier: float
    macrophage_secretion_rate: float
    pneumocyte_secretion_rate: float
    macrophage_secretion_rate_unit_t: float
    pneumocyte_secretion_rate_unit_t: float
    k_d: float


class MCP1(MoleculeModel):
    """MCP1"""

    name = 'mcp1'
    StateClass = MCP1State

    def initialize(self, state: State) -> State:
        mcp1: MCP1State = state.mcp1

        # config file values
        mcp1.half_life = self.config.getfloat('half_life')
        mcp1.macrophage_secretion_rate = self.config.getfloat('macrophage_secretion_rate')
        mcp1.pneumocyte_secretion_rate = self.config.getfloat('pneumocyte_secretion_rate')
        mcp1.k_d = self.config.getfloat('k_d')

        # computed values
        mcp1.half_life_multiplier = 1 + math.log(0.5) / (mcp1.half_life / state.simulation.time_step_size)
        # time unit conversions
        mcp1.macrophage_secretion_rate_unit_t = mcp1.macrophage_secretion_rate * 60 * state.simulation.time_step_size
        mcp1.pneumocyte_secretion_rate_unit_t = mcp1.pneumocyte_secretion_rate * 60 * state.simulation.time_step_size

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        from nlisim.modulesv2.macrophage import MacrophageCellData, MacrophageState
        from nlisim.modulesv2.pneumocyte import PneumocyteCellData, PneumocyteState

        mcp1: MCP1State = state.mcp1
        molecules: MoleculesState = state.molecules
        pneumocyte: PneumocyteState = state.pneumocyte
        macrophage: MacrophageState = state.macrophage
        grid: RectangularGrid = state.grid

        # interact with pneumocytes
        for pneumocyte_cell_index in pneumocyte.cells.alive():
            pneumocyte_cell: PneumocyteCellData = pneumocyte.cells[pneumocyte_cell_index]

            if pneumocyte_cell['tnfa']:
                pneumocyte_cell_voxel: Voxel = grid.get_voxel(pneumocyte_cell['point'])
                mcp1.grid[tuple(pneumocyte_cell_voxel)] += mcp1.pneumocyte_secretion_rate_unit_t

        # interact with macrophages
        for macrophage_cell_index in macrophage.cells.alive():
            macrophage_cell: MacrophageCellData = macrophage.cells[macrophage_cell_index]

            if macrophage_cell['tnfa']:
                macrophage_cell_voxel: Voxel = grid.get_voxel(macrophage_cell['point'])
                mcp1.grid[tuple(macrophage_cell_voxel)] += mcp1.macrophage_secretion_rate_unit_t

        # Degrade MCP1
        mcp1.grid *= mcp1.half_life_multiplier
        mcp1.grid *= turnover_rate(x_mol=np.array(1.0, dtype=np.float64),
                                   x_system_mol=0.0,
                                   base_turnover_rate=molecules.turnover_rate,
                                   rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t)

        return state
