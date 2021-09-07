from typing import Any, Dict

import attr
from attr import attrib, attrs
import numpy as np

from nlisim.coordinates import Voxel
from nlisim.module import ModuleModel, ModuleState
from nlisim.random import rg
from nlisim.state import State
from nlisim.util import activation_function


def molecule_grid_factory(self: 'HepcidinState') -> np.ndarray:
    return np.zeros(shape=self.global_state.grid.shape, dtype=float)


@attrs(kw_only=True, repr=False)
class HepcidinState(ModuleState):
    grid: np.ndarray = attrib(
        default=attr.Factory(molecule_grid_factory, takes_self=True)
    )  # units: atto-mol
    k_d: float  # units: aM


class Hepcidin(ModuleModel):
    """Hepcidin"""

    name = 'hepcidin'
    StateClass = HepcidinState

    def initialize(self, state: State) -> State:
        hepcidin: HepcidinState = state.hepcidin

        # config file values
        hepcidin.k_d = self.config.getfloat('k_d')  # aM

        # computed values (none)

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        from nlisim.modules.macrophage import MacrophageState

        hepcidin: HepcidinState = state.hepcidin
        macrophage: MacrophageState = state.macrophage
        voxel_volume: float = state.voxel_volume

        # interaction with macrophages
        activated_voxels = zip(
            *np.where(
                activation_function(
                    x=hepcidin.grid,
                    k_d=hepcidin.k_d,
                    h=self.time_step / 60,  # units: (min/step) / (min/hour)
                    volume=voxel_volume,
                    b=1,
                )
                > rg.random(size=hepcidin.grid.shape)
            )
        )
        for z, y, x in activated_voxels:
            for macrophage_cell_index in macrophage.cells.get_cells_in_voxel(Voxel(x=x, y=y, z=z)):
                macrophage_cell = macrophage.cells[macrophage_cell_index]
                macrophage_cell['fpn'] = False
                macrophage_cell['fpn_iteration'] = 0

        # Degrading Hepcidin is done by the "liver"

        # hepcidin does not diffuse

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        from nlisim.util import TissueType

        hepcidin: HepcidinState = state.hepcidin
        voxel_volume = state.voxel_volume
        mask = state.lung_tissue != TissueType.AIR

        return {
            'concentration (nM)': float(np.mean(hepcidin.grid[mask]) / voxel_volume / 1e9),
        }

    def visualization_data(self, state: State):
        hepcidin: HepcidinState = state.hepcidin
        return 'molecule', hepcidin.grid
