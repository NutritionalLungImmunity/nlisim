from typing import Any, Dict

import attr
import numpy as np

from nlisim.diffusion import apply_diffusion
from nlisim.module import ModuleModel, ModuleState
from nlisim.state import State


def molecule_grid_factory(self: 'ROSState') -> np.ndarray:
    return np.zeros(shape=self.global_state.grid.shape, dtype=float)


@attr.s(kw_only=True, repr=False)
class ROSState(ModuleState):
    grid: np.ndarray = attr.ib(
        default=attr.Factory(molecule_grid_factory, takes_self=True)
    )  # units: atto-mol


class ROS(ModuleModel):
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
        from nlisim.modules.molecules import MoleculesState

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
        ros.grid[:] = apply_diffusion(
            variable=ros.grid,
            laplacian=molecules.laplacian,
            diffusivity=molecules.diffusion_constant,
            dt=self.time_step,
        )

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        ros: ROSState = state.ros
        voxel_volume = state.voxel_volume

        return {
            'concentration (nM)': float(np.mean(ros.grid) / voxel_volume / 1e9),
        }

    def visualization_data(self, state: State):
        ros: ROSState = state.ros
        return 'molecule', ros.grid
