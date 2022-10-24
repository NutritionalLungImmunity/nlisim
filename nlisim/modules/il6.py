from typing import Any, Dict

from attr import Factory, attrib, attrs
import numpy as np
from scipy.sparse import csr_matrix

from nlisim.diffusion import (
    apply_mesh_diffusion_crank_nicholson,
    assemble_mesh_laplacian_crank_nicholson,
)
from nlisim.grid import TetrahedralMesh
from nlisim.module import ModuleModel, ModuleState
from nlisim.modules.molecules import MoleculesState
from nlisim.state import State
from nlisim.util import logger, secrete_in_element, turnover


def molecule_point_field_factory(self: 'IL6State') -> np.ndarray:
    return self.global_state.mesh.allocate_point_variable(dtype=np.float64)


@attrs(kw_only=True, repr=False)
class IL6State(ModuleState):
    field: np.ndarray = attrib(
        default=Factory(molecule_point_field_factory, takes_self=True)
    )  # units: atto-M
    half_life: float  # units: min
    half_life_multiplier: float  # units: proportion
    macrophage_secretion_rate: float  # units: atto-mol/(cell*h)
    neutrophil_secretion_rate: float  # units: atto-mol/(cell*h)
    pneumocyte_secretion_rate: float  # units: atto-mol/(cell*h)
    macrophage_secretion_rate_unit_t: float  # units: atto-mol/(cell*step)
    neutrophil_secretion_rate_unit_t: float  # units: atto-mol/(cell*step)
    pneumocyte_secretion_rate_unit_t: float  # units: atto-mol/(cell*step)
    k_d: float  # units: atto-M
    diffusion_constant: float  # units: µm^2/min
    cn_a: csr_matrix  # `A` matrix for Crank-Nicholson
    cn_b: csr_matrix  # `B` matrix for Crank-Nicholson
    dofs: Any  # degrees of freedom in mesh


class IL6(ModuleModel):
    """IL6"""

    name = 'il6'
    StateClass = IL6State

    def initialize(self, state: State) -> State:
        logger.info("Initializing " + self.name)
        il6: IL6State = state.il6

        # config file values
        il6.half_life = self.config.getfloat('half_life')
        il6.macrophage_secretion_rate = self.config.getfloat(
            'macrophage_secretion_rate'
        )  # units: atto-mol/(cell*h)
        il6.neutrophil_secretion_rate = self.config.getfloat(
            'neutrophil_secretion_rate'
        )  # units: atto-mol/(cell*h)
        il6.pneumocyte_secretion_rate = self.config.getfloat(
            'pneumocyte_secretion_rate'
        )  # units: atto-mol/(cell*h)
        il6.k_d = self.config.getfloat('k_d')  # units: atto-mol
        il6.diffusion_constant = self.config.getfloat('diffusion_constant')  # units: µm^2/min

        # computed values
        il6.half_life_multiplier = 0.5 ** (
            self.time_step / il6.half_life
        )  # units in exponent: (min/step) / min -> 1/step

        # time unit conversions
        # units: (atto-mol * cell^-1 * h^-1 * (min * step^-1) / (min * hour^-1)
        #        = atto-mol * cell^-1 * step^-1
        il6.macrophage_secretion_rate_unit_t = il6.macrophage_secretion_rate * (self.time_step / 60)
        il6.neutrophil_secretion_rate_unit_t = il6.neutrophil_secretion_rate * (self.time_step / 60)
        il6.pneumocyte_secretion_rate_unit_t = il6.pneumocyte_secretion_rate * (self.time_step / 60)

        # matrices for diffusion
        cn_a, cn_b, dofs = assemble_mesh_laplacian_crank_nicholson(
            state=state, diffusivity=il6.diffusion_constant, dt=self.time_step
        )
        il6.cn_a = cn_a
        il6.cn_b = cn_b
        il6.dofs = dofs

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        logger.info("Advancing " + self.name + f" from t={previous_time}")

        from nlisim.modules.macrophage import MacrophageState
        from nlisim.modules.neutrophil import NeutrophilState
        from nlisim.modules.phagocyte import PhagocyteStatus
        from nlisim.modules.pneumocyte import PneumocyteState

        il6: IL6State = state.il6
        macrophage: MacrophageState = state.macrophage
        neutrophil: NeutrophilState = state.neutrophil
        pneumocyte: PneumocyteState = state.pneumocyte
        molecules: MoleculesState = state.molecules
        mesh: TetrahedralMesh = state.mesh

        assert np.alltrue(il6.field >= 0.0)

        # active Macrophages secrete il6
        for macrophage_cell_index in macrophage.cells.alive():
            macrophage_cell = macrophage.cells[macrophage_cell_index]
            if macrophage_cell['status'] == PhagocyteStatus.ACTIVE:
                secrete_in_element(
                    mesh=mesh,
                    point_field=il6.field,
                    element_index=macrophage_cell['element_index'],
                    point=macrophage_cell['point'],
                    amount=il6.macrophage_secretion_rate_unit_t,
                )

        # active Neutrophils secrete il6
        for neutrophil_cell_index in neutrophil.cells.alive():
            neutrophil_cell = neutrophil.cells[neutrophil_cell_index]
            if neutrophil_cell['status'] == PhagocyteStatus.ACTIVE:
                secrete_in_element(
                    mesh=mesh,
                    point_field=il6.field,
                    element_index=neutrophil_cell['element_index'],
                    point=neutrophil_cell['point'],
                    amount=il6.neutrophil_secretion_rate_unit_t,
                )

        # active Pneumocytes secrete il6
        for pneumocyte_cell_index in pneumocyte.cells.alive():
            pneumocyte_cell = pneumocyte.cells[pneumocyte_cell_index]
            if pneumocyte_cell['status'] == PhagocyteStatus.ACTIVE:
                secrete_in_element(
                    mesh=mesh,
                    point_field=il6.field,
                    element_index=pneumocyte_cell['element_index'],
                    point=pneumocyte_cell['point'],
                    amount=il6.pneumocyte_secretion_rate_unit_t,
                )

        # Degrade IL6
        il6.field *= il6.half_life_multiplier
        turnover(
            field=il6.field,
            system_concentration=0.0,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )

        # Diffusion of IL6
        il6.field[:] = apply_mesh_diffusion_crank_nicholson(
            variable=il6.field,
            cn_a=il6.cn_a,
            cn_b=il6.cn_b,
            dofs=il6.dofs,
        )

        assert np.alltrue(il6.field >= 0.0)

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        il6: IL6State = state.il6
        mesh: TetrahedralMesh = state.mesh

        return {
            'concentration (nM)': float(
                mesh.integrate_point_function(il6.field) / 1e9 / mesh.total_volume
            ),
        }

    def visualization_data(self, state: State):
        il6: IL6State = state.il6
        return 'molecule', il6.field
