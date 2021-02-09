import math

import attr
import numpy as np

from nlisim.coordinates import Voxel
from nlisim.grid import RectangularGrid
from nlisim.module import ModuleState
from nlisim.modulesv2.geometry import GeometryState
from nlisim.modulesv2.molecules import MoleculeModel, MoleculesState
from nlisim.random import rg
from nlisim.state import State
from nlisim.util import activation_function, turnover_rate


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

        # config file values
        il10.half_life = self.config.getfloat('half_life')
        il10.macrophage_secretion_rate = self.config.getfloat('macrophage_secretion_rate')
        il10.k_d = self.config.getfloat('k_d')

        # computed values
        il10.half_life_multiplier = 1 + math.log(0.5) / (il10.half_life / self.time_step)
        # time unit conversions
        il10.macrophage_secretion_rate_unit_t = il10.macrophage_secretion_rate * 60 * self.time_step

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        from nlisim.modulesv2.phagocyte import PhagocyteStatus
        from nlisim.modulesv2.macrophage import MacrophageState

        il10: IL10State = state.il10
        macrophage: MacrophageState = state.macrophage
        molecules: MoleculesState = state.molecules
        geometry: GeometryState = state.geometry
        grid: RectangularGrid = state.grid

        # active Macrophages secrete il10 and non-dead macrophages can become inactivated by il10
        for macrophage_cell in macrophage.cells:
            macrophage_cell_voxel: Voxel = grid.get_voxel(macrophage_cell['point'])

            if macrophage_cell['status'] in PhagocyteStatus.ACTIVE:
                il10.grid[tuple(macrophage_cell_voxel)] += il10.macrophage_secretion_rate_unit_t

            if macrophage_cell['status'] not in {PhagocyteStatus.DEAD,
                                                 PhagocyteStatus.APOPTOTIC,
                                                 PhagocyteStatus.NECROTIC}:
                if activation_function(x=il10.grid[tuple(macrophage_cell_voxel)],
                                       kd=il10.k_d,
                                       h=self.time_step / 60,
                                       volume=geometry.voxel_volume) < rg():
                    if macrophage_cell['status'] != PhagocyteStatus.INACTIVE:
                        macrophage_cell['status'] = PhagocyteStatus.INACTIVATING
                    macrophage_cell['status_iteration'] = 0  # TODO: ask about this, why is it reset each time?

        # Degrade IL10
        il10.grid *= il10.half_life_multiplier
        il10.grid *= turnover_rate(x_mol=np.ones(shape=il10.grid.shape, dtype=np.float64),
                                   x_system_mol=0.0,
                                   base_turnover_rate=molecules.turnover_rate,
                                   rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t)

        return state
