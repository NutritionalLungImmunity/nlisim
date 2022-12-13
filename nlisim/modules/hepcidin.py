from typing import Any, Dict

import attr
from attr import attrib, attrs
import numpy as np

from nlisim.grid import TetrahedralMesh
from nlisim.module import ModuleModel, ModuleState
from nlisim.modules.macrophage import MacrophageCellData
from nlisim.random import rg
from nlisim.state import State
from nlisim.util import activation_function, logger


def molecule_point_field_factory(self: 'HepcidinState') -> np.ndarray:
    return self.global_state.mesh.allocate_point_variable(dtype=np.float64)


@attrs(kw_only=True, repr=False)
class HepcidinState(ModuleState):
    field: np.ndarray = attrib(
        default=attr.Factory(molecule_point_field_factory, takes_self=True)
    )  # units: atto-mol
    k_d: float  # units: aM
    # diffusion_constant: float  # units: µm^2/min
    # cn_a: csr_matrix  # `A` matrix for Crank-Nicholson -> Hepcidin does not diffuse
    # cn_b: csr_matrix  # `B` matrix for Crank-Nicholson -> Hepcidin does not diffuse


class Hepcidin(ModuleModel):
    """Hepcidin"""

    name = 'hepcidin'
    StateClass = HepcidinState

    def initialize(self, state: State) -> State:
        logger.info("Initializing " + self.name)
        hepcidin: HepcidinState = state.hepcidin
        # mesh: TetrahedralMesh = state.mesh

        # config file values
        hepcidin.k_d = self.config.getfloat('k_d')  # aM
        # hepcidin.diffusion_constant = \
        #     self.config.getfloat('diffusion_constant')  # units: µm^2/min

        # computed values (none)

        # matrices for diffusion -> Hepcidin does not diffuse
        # cn_a, cn_b = new_assemble_mesh_laplacian_crank_nicholson(
        #     laplacian=mesh.laplacian, diffusivity=hepcidin.diffusion_constant, dt=self.time_step
        # )
        # hepcidin.cn_a = cn_a
        # hepcidin.cn_b = cn_b

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        logger.info("Advancing " + self.name + f" from t={previous_time}")

        from nlisim.modules.macrophage import MacrophageState

        hepcidin: HepcidinState = state.hepcidin
        macrophage: MacrophageState = state.macrophage
        mesh: TetrahedralMesh = state.mesh

        assert np.alltrue(hepcidin.field >= 0.0)

        # interaction with macrophages
        live_macrophage_cells: MacrophageCellData = macrophage.cells.cell_data[
            macrophage.cells.alive()
        ]
        hepcidin_concentration_at_macrophages = mesh.evaluate_point_function(
            point_function=hepcidin.field,
            element_index=macrophage.cells.cell_data[macrophage.cells.alive()]['element_index'],
            point=live_macrophage_cells['point'],
        )
        activation_mask = activation_function(
            x=hepcidin_concentration_at_macrophages,
            k_d=hepcidin.k_d,
            h=self.time_step / 60,  # units: (min/step) / (min/hour),
            volume=1,  # already a concentration
            b=1,
        ) > rg.random(hepcidin_concentration_at_macrophages.shape)
        live_macrophage_cells[activation_mask]['fpn'] = False
        live_macrophage_cells[activation_mask]['fpn_iteration'] = 0

        # # python for-loop version of the same:
        # live_macrophage_indices = macrophage.cells.alive()
        # for macrophage_index in live_macrophage_indices:
        #     macrophage_cell: MacrophageCellData = macrophage.cells[macrophage_index]
        #     hepcidin_concentration_at_macrophage = mesh.evaluate_point_function(
        #         point_function=hepcidin.field,
        #         element_index=macrophage.cells.element_index[macrophage_index],
        #         point=macrophage_cell['point'],
        #     )
        #     if (
        #         activation_function(
        #             x=hepcidin_concentration_at_macrophage,
        #             k_d=hepcidin.k_d,
        #             h=self.time_step / 60,  # units: (min/step) / (min/hour),
        #             volume=1,  # already a concentration
        #             b=1,
        #         )
        #         > rg.random()
        #     ):
        #         macrophage_cell['fpn'] = False
        #         macrophage_cell['fpn_iteration'] = 0

        # Degrading Hepcidin is done by the "liver"

        # hepcidin does not diffuse

        assert np.alltrue(hepcidin.field >= 0.0)

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        hepcidin: HepcidinState = state.hepcidin
        mesh: TetrahedralMesh = state.mesh

        return {
            'concentration (nM)': float(
                mesh.integrate_point_function(hepcidin.field) / 1e9 / mesh.total_volume
            ),
        }

    def visualization_data(self, state: State):
        hepcidin: HepcidinState = state.hepcidin
        return 'molecule', hepcidin.field
