import attr
import numpy as np

from nlisim.module import ModuleState
from nlisim.modulesv2.geometry import GeometryState
from nlisim.modulesv2.molecules import MoleculesState
from nlisim.modulesv2.molecule import MoleculeModel
from nlisim.state import State


def molecule_grid_factory(self: 'HepcidinState') -> np.ndarray:
    return np.zeros(shape=self.global_state.grid.shape, dtype=float)


@attr.s(kw_only=True, repr=False)
class HepcidinState(ModuleState):
    grid: np.ndarray = attr.ib(default=attr.Factory(molecule_grid_factory, takes_self=True))
    hepcidin_qtty: float


class Hepcidin(MoleculeModel):
    """Hepcidin"""

    name = 'hepcidin'
    StateClass = HepcidinState

    def initialize(self, state: State) -> State:
        hepcidin: HepcidinState = state.hepcidin
        geometry: GeometryState = state.geometry
        voxel_volume = geometry.voxel_volume

        # config file values
        # TODO: ? where did this come from hepcidin.hepcidin_qtty = self.config.getfloat('hepcidin_qtty')

        # computed values (none)

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        hepcidin: HepcidinState = state.hepcidin
        molecules: MoleculesState = state.molecules

        # TODO: move to cell
        # elif itype is Afumigatus:
        #     if interactable.status == Afumigatus.HYPHAE:
        #         self.inc(Constants.HEPCIDIN_QTTY)
        #     return True

        # Degrading Hepcidin is done by the "liver"

        return state
