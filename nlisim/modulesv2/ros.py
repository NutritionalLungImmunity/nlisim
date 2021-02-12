import attr
import numpy as np

from nlisim.module import ModuleState
from nlisim.modulesv2.molecules import MoleculeModel, MoleculesState
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
        # ros: ROSState = state.ros
        # geometry: GeometryState = state.geometry
        # voxel_volume = geometry.voxel_volume

        # config file values

        # computed values

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        ros: ROSState = state.ros
        molecules: MoleculesState = state.molecules

        # TODO: code below adds zero, omitting until we have a non-trivial model

        # elif type(interactable) is Macrophage:
        #     assert isinstance(interactable, Macrophage)
        #     macrophage: Macrophage = interactable
        #
        #     if macrophage.status == Macrophage.ACTIVE:
        #         self.increase(0, x, y, z)
        #     return True

        # Degrade ROS (does not degrade) (obsolete, will be reintroduced later)

        # Diffusion of ros
        self.diffuse(ros.grid, molecules.diffusion_constant_timestep)

        return state
