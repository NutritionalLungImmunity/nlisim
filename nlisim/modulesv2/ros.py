import attr
import numpy as np

from nlisim.module import ModuleState
from nlisim.modulesv2.geometry import GeometryState
from nlisim.modulesv2.molecules import MoleculesState
from nlisim.modulesv2.molecule import MoleculeModel
from nlisim.state import State


def molecule_grid_factory(self: 'ROSState') -> np.ndarray:
    return np.zeros(shape=self.global_state.grid.shape, dtype=float)


@attr.s(kw_only=True, repr=False)
class ROSState(ModuleState):
    grid: np.ndarray = attr.ib(default=attr.Factory(molecule_grid_factory, takes_self=True))


class ROS(MoleculeModel):
    """Reactive Oxygen Species"""

    name = 'ros'
    StateClass = ROSState

    def initialize(self, state: State) -> State:
        ros: ROSState = state.ros
        geometry: GeometryState = state.geometry
        voxel_volume = geometry.voxel_volume

        # config file values

        # computed values

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        ros: ROSState = state.ros
        molecules: MoleculesState = state.molecules

        # TODO: move to cell
        # elif type(interactable) is Afumigatus:
        #     # if interactable.state == Afumigatus.FREE and not (
        #     #         interactable.status == Afumigatus.DYING or interactable.status == Afumigatus.DEAD):
        #     #     if Util.hillProbability(self.get()) > random():
        #     #         interactable.status = Afumigatus.DYING
        #     return True

        # TODO: move to cell
        # elif type(interactable) is Macrophage:
        #     if interactable.status == Macrophage.ACTIVE:
        #         self.inc(0)
        #     return True

        # Degrade ROS (does not degrade) TODO: verify

        return state
