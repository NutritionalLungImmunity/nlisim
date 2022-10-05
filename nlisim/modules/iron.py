import logging
from typing import Any, Dict

# noinspection PyPackageRequirements
import attr

# noinspection PyPackageRequirements
import numpy as np

from nlisim.grid import TetrahedralMesh
from nlisim.module import ModuleModel, ModuleState
from nlisim.modules.macrophage import MacrophageCellData
from nlisim.state import State
from nlisim.util import secrete_in_element


def molecule_point_field_factory(self: 'IronState') -> np.ndarray:
    return self.global_state.mesh.allocate_point_variable(dtype=np.float64)


@attr.s(kw_only=True, repr=False)
class IronState(ModuleState):
    field: np.ndarray = attr.ib(
        default=attr.Factory(molecule_point_field_factory, takes_self=True)
    )  # units: atto-M


class Iron(ModuleModel):
    """Iron"""

    name = 'iron'
    StateClass = IronState

    def initialize(self, state: State) -> State:
        logging.getLogger('nlisim').debug("Initializing " + self.name)
        # iron: IronState = state.iron
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
        mesh: TetrahedralMesh = state.mesh

        assert np.alltrue(iron.field >= 0.0)

        # dead macrophages contribute their iron to the environment
        for macrophage_cell_index in macrophage.cells.alive():
            macrophage_cell: MacrophageCellData = macrophage.cells[macrophage_cell_index]

            if macrophage_cell['status'] in {
                PhagocyteStatus.NECROTIC,
                PhagocyteStatus.APOPTOTIC,
                PhagocyteStatus.DEAD,
            }:
                internal_iron = macrophage_cell['iron_pool']
                macrophage_cell['iron_pool'] = 0.0
                secrete_in_element(
                    mesh=mesh,
                    point_field=iron.field,
                    element_index=macrophage_cell['element_index'],
                    point=macrophage_cell['point'],
                    amount=internal_iron,
                )

                # Degrade Iron
        # turnover done by liver, if at all (2/4/2021: not currently)

        # iron does not diffuse

        assert np.alltrue(iron.field >= 0.0)

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        iron: IronState = state.iron
        mesh: TetrahedralMesh = state.mesh

        return {
            'concentration (nM)': float(
                mesh.integrate_point_function(iron.field) / 1e9 / mesh.total_volume
            ),
        }

    def visualization_data(self, state: State):
        iron: IronState = state.iron
        return 'molecule', iron.field
