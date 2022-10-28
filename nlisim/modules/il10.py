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
from nlisim.modules.molecules import MoleculesState
from nlisim.random import rg
from nlisim.state import State
from nlisim.util import activation_function, logger, secrete_in_element, turnover


def molecule_point_field_factory(self: 'IL10State') -> np.ndarray:
    return self.global_state.mesh.allocate_point_variable(dtype=np.float64)


@attr.s(kw_only=True, repr=False)
class IL10State(ModuleState):
    field: np.ndarray = attr.ib(
        default=attr.Factory(molecule_point_field_factory, takes_self=True)
    )  # units: atto-M
    half_life: float  # units: min
    half_life_multiplier: float  # units: proportion
    macrophage_secretion_rate: float  # units: atto-mol * cell^-1 * h^-1
    macrophage_secretion_rate_unit_t: float  # units: atto-mol * cell^-1 * h^-1
    k_d: float  # units: aM
    diffusion_constant: float  # units: µm^2/min
    cn_a: csr_matrix  # `A` matrix for Crank-Nicholson
    cn_b: csr_matrix  # `B` matrix for Crank-Nicholson
    dofs: Any  # degrees of freedom in mesh


class IL10(ModuleModel):
    """IL10"""

    name = 'il10'
    StateClass = IL10State

    def initialize(self, state: State) -> State:
        logger.info("Initializing " + self.name)
        il10: IL10State = state.il10

        # config file values
        il10.half_life = self.config.getfloat('half_life')  # units: min
        il10.macrophage_secretion_rate = self.config.getfloat(
            'macrophage_secretion_rate'
        )  # units: atto-mol * cell^-1 * h^-1
        il10.k_d = self.config.getfloat('k_d')  # units: aM
        il10.diffusion_constant = self.config.getfloat('diffusion_constant')  # units: µm^2/min

        # computed values
        il10.half_life_multiplier = 0.5 ** (
            self.time_step / il10.half_life
        )  # units in exponent: (min/step) / min -> 1/step
        logger.info(f"Computed {il10.half_life_multiplier=}")
        # time unit conversions
        il10.macrophage_secretion_rate_unit_t = il10.macrophage_secretion_rate * (
            self.time_step / 60
        )  # units: atto-mol * cell^-1 * h^-1 * (min/step) / (min/hour)
        logger.info(f"Computed {il10.macrophage_secretion_rate_unit_t=}")

        # matrices for diffusion
        cn_a, cn_b, dofs = assemble_mesh_laplacian_crank_nicholson(
            state=state, diffusivity=il10.diffusion_constant, dt=self.time_step
        )
        il10.cn_a = cn_a
        il10.cn_b = cn_b
        il10.dofs = dofs

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        logger.info("Advancing " + self.name + f" from t={previous_time}")

        from nlisim.modules.macrophage import MacrophageCellData, MacrophageState
        from nlisim.modules.phagocyte import PhagocyteState, PhagocyteStatus

        il10: IL10State = state.il10
        macrophage: MacrophageState = state.macrophage
        molecules: MoleculesState = state.molecules
        mesh: TetrahedralMesh = state.mesh

        assert np.alltrue(il10.field >= 0.0)

        # active Macrophages secrete il10 and non-dead macrophages can become inactivated by il10
        for macrophage_cell_index in macrophage.cells.alive():
            macrophage_cell: MacrophageCellData = macrophage.cells[macrophage_cell_index]

            if (
                macrophage_cell['status'] == PhagocyteStatus.ACTIVE
                and macrophage_cell['state'] == PhagocyteState.INTERACTING
            ):
                secrete_in_element(
                    mesh=mesh,
                    point_field=il10.field,
                    element_index=macrophage_cell['element_index'],
                    point=macrophage_cell['point'],
                    amount=il10.macrophage_secretion_rate_unit_t,
                )

            if macrophage_cell['status'] not in {
                PhagocyteStatus.DEAD,
                PhagocyteStatus.APOPTOTIC,
                PhagocyteStatus.NECROTIC,
            }:
                il10_concentration_at_macrophage = mesh.evaluate_point_function(
                    point_function=il10.field,
                    element_index=macrophage_cell['element_index'],
                    point=macrophage_cell['point'],
                )
                if (
                    activation_function(
                        x=il10_concentration_at_macrophage,
                        k_d=il10.k_d,
                        h=self.time_step / 60,  # units: (min/step) / (min/hour)
                        volume=1,  # already a concentration
                        b=1,
                    )
                    > rg.uniform()
                ):
                    # inactive cells stay inactive, others become inactivating
                    if macrophage_cell['status'] != PhagocyteStatus.INACTIVE:
                        macrophage_cell['status'] = PhagocyteStatus.INACTIVATING
                    macrophage_cell['status_iteration'] = 0

        # Degrade IL10
        il10.field *= il10.half_life_multiplier
        turnover(
            field=il10.field,
            system_concentration=0.0,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )

        # Diffusion of IL10
        il10.field[:] = apply_mesh_diffusion_crank_nicholson(
            variable=il10.field,
            cn_a=il10.cn_a,
            cn_b=il10.cn_b,
            dofs=il10.dofs,
        )

        assert np.alltrue(il10.field >= 0.0)

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        il10: IL10State = state.il10
        mesh: TetrahedralMesh = state.mesh

        return {
            'concentration (nM)': float(
                mesh.integrate_point_function(il10.field) / 1e9 / mesh.total_volume
            ),
        }

    def visualization_data(self, state: State):
        il10: IL10State = state.il10
        return 'molecule', il10.field
