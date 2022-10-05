import logging
from typing import Any, Dict

# noinspection PyPackageRequirements
import attr

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
from nlisim.random import rg
from nlisim.state import State
from nlisim.util import activation_function, secrete_in_element, turnover_rate


def molecule_point_field_factory(self: 'TNFaState') -> np.ndarray:
    return self.global_state.mesh.allocate_point_variable(dtype=np.float64)


@attr.s(kw_only=True, repr=False)
class TNFaState(ModuleState):
    field: np.ndarray = attr.ib(
        default=attr.Factory(molecule_point_field_factory, takes_self=True)
    )  # units: atto-mol
    half_life: float  # units: min
    half_life_multiplier: float  # units: proportion
    macrophage_secretion_rate: float  # units: atto-mol/(cell*h)
    neutrophil_secretion_rate: float  # units: atto-mol/(cell*h)
    epithelial_secretion_rate: float  # units: atto-mol/(cell*h)
    macrophage_secretion_rate_unit_t: float  # units: atto-mol/(cell*step)
    neutrophil_secretion_rate_unit_t: float  # units: atto-mol/(cell*step)
    epithelial_secretion_rate_unit_t: float  # units: atto-mol/(cell*step)
    k_d: float  # aM
    turnover_rate: float
    diffusion_constant: float  # units: µm^2/min
    cn_a: csr_matrix  # `A` matrix for Crank-Nicholson
    cn_b: csr_matrix  # `B` matrix for Crank-Nicholson
    dofs: Any  # degrees of freedom in mesh


class TNFa(ModuleModel):
    name = 'tnfa'
    StateClass = TNFaState

    def initialize(self, state: State) -> State:
        logging.getLogger('nlisim').debug("Initializing " + self.name)
        tnfa: TNFaState = state.tnfa
        molecules: MoleculesState = state.molecules

        # config file values
        tnfa.half_life = self.config.getfloat('half_life')  # units: min
        tnfa.macrophage_secretion_rate = self.config.getfloat(
            'macrophage_secretion_rate'
        )  # units: atto-mol/(cell*h)
        tnfa.neutrophil_secretion_rate = self.config.getfloat(
            'neutrophil_secretion_rate'
        )  # units: atto-mol/(cell*h)
        tnfa.epithelial_secretion_rate = self.config.getfloat(
            'epithelial_secretion_rate'
        )  # units: atto-mol/(cell*h)
        tnfa.k_d = self.config.getfloat('k_d')  # units: aM
        tnfa.diffusion_constant = self.config.getfloat('diffusion_constant')  # units: µm^2/min

        # computed values
        tnfa.turnover_rate = turnover_rate(
            x=1.0,
            x_system=0.0,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )
        tnfa.half_life_multiplier = 0.5 ** (
            self.time_step / tnfa.half_life
        )  # units: (min/step) / min -> 1/step
        # time unit conversions
        # units: (atto-mol * cell^-1 * h^-1 * (min * step^-1) / (min * hour^-1)
        #        = atto-mol * cell^-1 * step^-1
        tnfa.macrophage_secretion_rate_unit_t = tnfa.macrophage_secretion_rate * (
            self.time_step / 60
        )
        tnfa.neutrophil_secretion_rate_unit_t = tnfa.neutrophil_secretion_rate * (
            self.time_step / 60
        )
        tnfa.epithelial_secretion_rate_unit_t = tnfa.epithelial_secretion_rate * (
            self.time_step / 60
        )

        # matrices for diffusion
        cn_a, cn_b, dofs = assemble_mesh_laplacian_crank_nicholson(
            state=state, diffusivity=tnfa.diffusion_constant, dt=self.time_step
        )
        tnfa.cn_a = cn_a
        tnfa.cn_b = cn_b
        tnfa.dofs = dofs

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        from nlisim.modules.macrophage import MacrophageCellData, MacrophageState
        from nlisim.modules.neutrophil import NeutrophilCellData, NeutrophilState
        from nlisim.modules.phagocyte import PhagocyteStatus

        tnfa: TNFaState = state.tnfa
        macrophage: MacrophageState = state.macrophage
        neutrophil: NeutrophilState = state.neutrophil
        mesh: TetrahedralMesh = state.mesh

        assert np.alltrue(tnfa.field >= 0.0)

        for macrophage_cell_index in macrophage.cells.alive():
            macrophage_cell: MacrophageCellData = macrophage.cells[macrophage_cell_index]
            macrophage_cell_element: int = macrophage_cell['element_index']

            if macrophage_cell['status'] == PhagocyteStatus.ACTIVE:
                secrete_in_element(
                    mesh=mesh,
                    point_field=tnfa.field,
                    element_index=macrophage_cell_element,
                    point=macrophage_cell['point'],
                    amount=tnfa.macrophage_secretion_rate_unit_t,
                )

            if macrophage_cell['status'] in {PhagocyteStatus.RESTING, PhagocyteStatus.ACTIVE}:
                if (
                    activation_function(
                        x=mesh.evaluate_point_function(
                            point_function=tnfa.field,
                            point=macrophage_cell['point'],
                            element_index=macrophage_cell_element,
                        ),
                        k_d=tnfa.k_d,
                        h=self.time_step / 60,  # units: (min/step) / (min/hour)
                        volume=1.0,  # already a concentration
                        b=1,
                    )
                    > rg.uniform()
                ):
                    if macrophage_cell['status'] == PhagocyteStatus.RESTING:
                        macrophage_cell['status'] = PhagocyteStatus.ACTIVATING
                    else:
                        macrophage_cell['status'] = PhagocyteStatus.ACTIVE
                    # Note: multiple activations will reset the 'clock'
                    macrophage_cell['status_iteration'] = 0
                    macrophage_cell['tnfa'] = True

        for neutrophil_cell_index in neutrophil.cells.alive():
            neutrophil_cell: NeutrophilCellData = neutrophil.cells[neutrophil_cell_index]
            neutrophil_cell_element: int = neutrophil_cell['element_index']

            if neutrophil_cell['status'] == PhagocyteStatus.ACTIVE:
                secrete_in_element(
                    mesh=mesh,
                    point_field=tnfa.field,
                    element_index=neutrophil_cell_element,
                    point=neutrophil_cell['point'],
                    amount=tnfa.neutrophil_secretion_rate_unit_t,
                )

            if neutrophil_cell['status'] in {PhagocyteStatus.RESTING, PhagocyteStatus.ACTIVE}:
                if (
                    activation_function(
                        x=tnfa.field[neutrophil_cell_element],
                        k_d=tnfa.k_d,
                        h=self.time_step / 60,  # units: (min/step) / (min/hour)
                        volume=1.0,  # already a concentration
                        b=1,
                    )
                    > rg.uniform()
                ):
                    if neutrophil_cell['status'] == PhagocyteStatus.RESTING:
                        neutrophil_cell['status'] = PhagocyteStatus.ACTIVATING
                    else:
                        neutrophil_cell['status'] = PhagocyteStatus.ACTIVE
                    # Note: multiple activations will reset the 'clock'
                    neutrophil_cell['status_iteration'] = 0
                    neutrophil_cell['tnfa'] = True

        # Degrade TNFa
        tnfa.field *= tnfa.half_life_multiplier * tnfa.turnover_rate

        # Diffusion of TNFa
        tnfa.field[:] = apply_mesh_diffusion_crank_nicholson(
            variable=tnfa.field,
            cn_a=tnfa.cn_a,
            cn_b=tnfa.cn_b,
            dofs=tnfa.dofs,
        )

        assert np.alltrue(tnfa.field >= 0.0)

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        tnfa: TNFaState = state.tnfa
        mesh: TetrahedralMesh = state.mesh

        return {
            'concentration (nM)': float(
                mesh.integrate_point_function(tnfa.field) / 1e9 / mesh.total_volume
            ),
        }

    def visualization_data(self, state: State):
        tnfa: TNFaState = state.tnfa
        return 'molecule', tnfa.field
