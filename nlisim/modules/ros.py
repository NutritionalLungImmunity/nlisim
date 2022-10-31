from typing import Any, Dict

import attr
import numpy as np
from scipy.sparse import csr_matrix

from nlisim.diffusion import (
    apply_mesh_diffusion_crank_nicholson,
    assemble_mesh_laplacian_crank_nicholson,
)
from nlisim.grid import TetrahedralMesh
from nlisim.module import ModuleModel, ModuleState
from nlisim.state import State
from nlisim.util import logger


def molecule_point_field_factory(self: 'ROSState') -> np.ndarray:
    return self.global_state.mesh.allocate_point_variable(dtype=np.float64)


@attr.s(kw_only=True, repr=False)
class ROSState(ModuleState):
    field: np.ndarray = attr.ib(
        default=attr.Factory(molecule_point_field_factory, takes_self=True)
    )  # units: atto-M
    diffusion_constant: float  # units: µm^2/min
    cn_a: csr_matrix  # `A` matrix for Crank-Nicholson
    cn_b: csr_matrix  # `B` matrix for Crank-Nicholson


class ROS(ModuleModel):
    """Reactive Oxygen Species"""

    name = 'ros'
    StateClass = ROSState

    def initialize(self, state: State) -> State:
        logger.info("Initializing " + self.name)
        ros: ROSState = state.ros
        mesh: TetrahedralMesh = state.mesh

        # config file values

        # computed values

        # matrices for diffusion
        cn_a, cn_b = assemble_mesh_laplacian_crank_nicholson(
            laplacian=mesh.laplacian, diffusivity=ros.diffusion_constant, dt=self.time_step
        )
        ros.cn_a = cn_a
        ros.cn_b = cn_b

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        logger.info("Advancing " + self.name + f" from t={previous_time}")

        ros: ROSState = state.ros

        assert np.alltrue(ros.field >= 0.0)

        # From Henrique's code: commented region below adds zero, omitting until we have
        #   a non-trivial model

        # elif type(interactable) is Macrophage:
        #     assert isinstance(interactable, Macrophage)
        #     macrophage: Macrophage = interactable
        #
        #     if macrophage.status == Macrophage.ACTIVE:
        #         self.increase(0, x, y, z)
        #     return True

        # Degrade ROS (does not degrade) (obsolete, will be reintroduced later)

        # Diffusion of ros
        logger.info(f"diffusing {self.name}")
        apply_mesh_diffusion_crank_nicholson(
            variable=ros.field,
            cn_a=ros.cn_a,
            cn_b=ros.cn_b,
        )

        assert np.alltrue(ros.field >= 0.0)

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        ros: ROSState = state.ros
        mesh: TetrahedralMesh = state.mesh

        return {
            'concentration (nM)': float(
                mesh.integrate_point_function(ros.field) / 1e9 / mesh.total_volume
            ),
        }

    def visualization_data(self, state: State):
        ros: ROSState = state.ros
        return 'molecule', ros.field
