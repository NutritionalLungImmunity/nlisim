from typing import Any, Dict

import attr
import numpy as np

from nlisim.coordinates import Voxel
from nlisim.grid import RectangularGrid
from nlisim.module import ModuleState
from nlisim.modules.molecules import MoleculeModel
from nlisim.state import State


def molecule_grid_factory(self: 'IronState') -> np.ndarray:
    return np.zeros(shape=self.global_state.grid.shape, dtype=float)


@attr.s(kw_only=True, repr=False)
class IronState(ModuleState):
    grid: np.ndarray = attr.ib(default=attr.Factory(molecule_grid_factory, takes_self=True))


class Iron(MoleculeModel):
    """Iron"""

    name = 'iron'
    StateClass = IronState

    def initialize(self, state: State) -> State:
        # iron: IronState = state.iron
        # geometry: GeometryState = state.geometry
        # voxel_volume = geometry.voxel_volume

        # config file values

        # computed values

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        from nlisim.modules.macrophage import MacrophageState
        from nlisim.modules.phagocyte import PhagocyteStatus

        iron: IronState = state.iron
        macrophage: MacrophageState = state.macrophage
        grid: RectangularGrid = state.grid

        # dead macrophages contribute their iron to the environment
        for macrophage_cell in macrophage.cells:
            if macrophage_cell['status'] in {
                PhagocyteStatus.NECROTIC,
                PhagocyteStatus.APOPTOTIC,
                PhagocyteStatus.DEAD,
            }:
                macrophage_cell_voxel: Voxel = grid.get_voxel(macrophage_cell['point'])
                iron.grid[tuple(macrophage_cell_voxel)] += macrophage_cell['iron_pool']
                macrophage_cell['iron_pool'] = 0.0

        # Degrade Iron
        # turnover done by liver, if at all (2/4/2021: not currently)

        # iron does not diffuse

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        iron: IronState = state.iron
        voxel_volume = state.voxel_volume

        return {
            'concentration': float(np.mean(iron.grid) / voxel_volume),
        }

    def visualization_data(self, state: State):
        iron: IronState = state.iron
        return 'molecule', iron.grid
