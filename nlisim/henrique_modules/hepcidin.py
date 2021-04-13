import attr
from attr import attrib, attrs
import numpy as np

from nlisim.coordinates import Voxel
from nlisim.module import ModuleState
from nlisim.henrique_modules.geometry import GeometryState
from nlisim.henrique_modules.molecules import MoleculeModel
from nlisim.state import State
from nlisim.util import activation_function


def molecule_grid_factory(self: 'HepcidinState') -> np.ndarray:
    return np.zeros(shape=self.global_state.grid.shape, dtype=float)


@attrs(kw_only=True, repr=False)
class HepcidinState(ModuleState):
    grid: np.ndarray = attrib(default=attr.Factory(molecule_grid_factory, takes_self=True))
    kd_hep: float


class Hepcidin(MoleculeModel):
    """Hepcidin"""

    name = 'hepcidin'
    StateClass = HepcidinState

    def initialize(self, state: State) -> State:
        hepcidin: HepcidinState = state.hepcidin

        # config file values
        hepcidin.kd_hep = self.config.getfloat('kd_hep')

        # computed values (none)

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        from nlisim.henrique_modules.macrophage import MacrophageState

        hepcidin: HepcidinState = state.hepcidin
        macrophage: MacrophageState = state.macrophage
        geometry: GeometryState = state.geometry

        # interaction with macrophages
        activated_voxels = \
            zip(*np.where(activation_function(x=hepcidin.grid,
                                              kd=hepcidin.kd_hep,
                                              h=self.time_step / 60,
                                              volume=geometry.voxel_volume) >
                          np.random.random(hepcidin.grid.shape)))
        for z, y, x in activated_voxels:
            for macrophage_cell_index in macrophage.cells.get_cells_in_voxel(Voxel(x=x, y=y, z=z)):
                macrophage_cell = macrophage.cells[macrophage_cell_index]
                macrophage_cell['fpn'] = False
                macrophage_cell['fpn_iteration'] = 0

        # Degrading Hepcidin is done by the "liver"

        # hepcidin does not diffuse

        return state
