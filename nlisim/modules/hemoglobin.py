import logging
from typing import Any, Dict

# noinspection PyPackageRequirements
import attr

# noinspection PyPackageRequirements
from attr import attrib, attrs

# noinspection PyPackageRequirements
import numpy as np

# noinspection PyPackageRequirements
from scipy.sparse import csr_matrix

from nlisim.diffusion import (
    apply_mesh_diffusion_crank_nicholson,
    assemble_mesh_laplacian_crank_nicholson,
)
from nlisim.grid import TetrahedralMesh
from nlisim.module import ModuleModel, ModuleState
from nlisim.modules.molecules import MoleculesState
from nlisim.state import State
from nlisim.util import turnover_rate, uptake_in_element


def molecule_point_field_factory(self: 'HemoglobinState') -> np.ndarray:
    return self.global_state.mesh.allocate_point_variable(dtype=np.float64)


@attrs(kw_only=True, repr=False)
class HemoglobinState(ModuleState):
    field: np.ndarray = attrib(
        default=attr.Factory(molecule_point_field_factory, takes_self=True)
    )  # units: atto-M
    uptake_rate: float
    ma_heme_import_rate: float
    diffusion_constant: float  # units: µm^2/min
    cn_a: csr_matrix  # `A` matrix for Crank-Nicholson
    cn_b: csr_matrix  # `B` matrix for Crank-Nicholson
    dofs: Any  # degrees of freedom in mesh


class Hemoglobin(ModuleModel):
    """Hemoglobin"""

    name = 'hemoglobin'
    StateClass = HemoglobinState

    def initialize(self, state: State) -> State:
        logging.getLogger('nlisim').debug("Initializing " + self.name)
        hemoglobin: HemoglobinState = state.hemoglobin

        # config file values
        hemoglobin.uptake_rate = self.config.getfloat('uptake_rate')
        hemoglobin.ma_heme_import_rate = self.config.getfloat('ma_heme_import_rate')
        hemoglobin.diffusion_constant = self.config.getfloat(
            'diffusion_constant'
        )  # units: µm^2/min

        # computed values (none)

        # matrices for diffusion
        cn_a, cn_b, dofs = assemble_mesh_laplacian_crank_nicholson(
            state=state, diffusivity=hemoglobin.diffusion_constant, dt=self.time_step
        )
        hemoglobin.cn_a = cn_a
        hemoglobin.cn_b = cn_b
        hemoglobin.dofs = dofs

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        from nlisim.modules.afumigatus import AfumigatusCellStatus, AfumigatusState

        hemoglobin: HemoglobinState = state.hemoglobin
        molecules: MoleculesState = state.molecules
        afumigatus: AfumigatusState = state.afumigatus
        mesh: TetrahedralMesh = state.mesh

        # afumigatus uptakes iron from hemoglobin

        # find the cells that will take up iron
        live_afumigatus = afumigatus.cells.alive()
        iron_uptaking_afumigatus_indices = live_afumigatus[
            np.logical_or(
                afumigatus.cells.cell_data[live_afumigatus]['status']
                == AfumigatusCellStatus.HYPHAE,
                afumigatus.cells.cell_data[live_afumigatus]['status']
                == AfumigatusCellStatus.GERM_TUBE,
            )
        ]
        # you could have too many aspergillus in an element, vying for just a little hemoglobin

        # The snippet below is kind of clever, if I do say so myself, but it needs some explanation.
        # by example:
        # >>> arr = np.array([0, 0, 1, 1, 1, 2, 10])
        # >>> np.bincount(arr)
        # array([2, 3, 1, 0, 0, 0, 0, 0, 0, 0, 1])
        # >>> np.bincount(arr)[arr]
        # array([2, 2, 3, 3, 3, 1, 1])
        afumigatus_elements = np.array(afumigatus.cells[:]['element_index'])[
            iron_uptaking_afumigatus_indices
        ]
        afumigatus_count_in_same_element = np.bincount(afumigatus_elements)[afumigatus_elements]

        available_hemoglobin = mesh.integrate_point_function_single_element(
            element_index=afumigatus_elements, point_function=hemoglobin.field
        )
        hemoglobin_uptake = np.where(
            hemoglobin.uptake_rate * afumigatus_count_in_same_element
            <= 1.0,  # uptakes less than 100%
            available_hemoglobin * hemoglobin.uptake_rate,  # uptake as normal
            available_hemoglobin / afumigatus_count_in_same_element,  # divide evenly
        )

        afumigatus_cells_taking_iron = afumigatus.cells.cell_data[iron_uptaking_afumigatus_indices]
        afumigatus_cells_taking_iron['iron_pool'] += (
            4 * hemoglobin_uptake * mesh.point_dual_volumes[afumigatus_elements]
        )  # units: atto-M * L = atto-mols
        uptake_in_element(
            mesh=mesh,
            point_field=hemoglobin.field,
            element_index=afumigatus_elements,
            point=afumigatus_cells_taking_iron['point'],
            amount=hemoglobin_uptake,
        )

        # Degrade Hemoglobin
        hemoglobin.field *= turnover_rate(
            x=hemoglobin.field,
            x_system=0.0,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )

        # Diffusion of Hemoglobin
        hemoglobin.field[:] = apply_mesh_diffusion_crank_nicholson(
            variable=hemoglobin.field,
            cn_a=hemoglobin.cn_a,
            cn_b=hemoglobin.cn_b,
            dofs=hemoglobin.dofs,
        )

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        hemoglobin: HemoglobinState = state.hemoglobin
        mesh: TetrahedralMesh = state.mesh

        return {
            'concentration (nM)': float(
                mesh.integrate_point_function(hemoglobin.field) / 1e9 / mesh.total_volume
            ),
        }

    def visualization_data(self, state: State):
        hemoglobin: HemoglobinState = state.hemoglobin
        return 'molecule', hemoglobin.field
