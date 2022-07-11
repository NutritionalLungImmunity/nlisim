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
from nlisim.util import activation_function, turnover_rate


def molecule_point_field_factory(self: 'IL8State') -> np.ndarray:
    return self.global_state.mesh.allocate_point_variable(dtype=np.float64)


@attr.s(kw_only=True, repr=False)
class IL8State(ModuleState):
    field: np.ndarray = attr.ib(
        default=attr.Factory(molecule_point_field_factory, takes_self=True)
    )  # units: atto-M
    half_life: float  # units: min
    half_life_multiplier: float  # units: proportion
    macrophage_secretion_rate: float  # units: atto-mol * cell^-1 * h^-1
    neutrophil_secretion_rate: float  # units: atto-mol * cell^-1 * h^-1
    pneumocyte_secretion_rate: float  # units: atto-mol * cell^-1 * h^-1
    macrophage_secretion_rate_unit_t: float  # units: atto-mol * cell^-1 * step^-1
    neutrophil_secretion_rate_unit_t: float  # units: atto-mol * cell^-1 * step^-1
    pneumocyte_secretion_rate_unit_t: float  # units: atto-mol * cell^-1 * step^-1
    k_d: float  # aM
    diffusion_constant: float  # units: µm^2/min
    cn_a: csr_matrix  # `A` matrix for Crank-Nicholson
    cn_b: csr_matrix  # `B` matrix for Crank-Nicholson
    dofs: Any  # degrees of freedom in mesh


class IL8(ModuleModel):
    """IL8"""

    name = 'il8'
    StateClass = IL8State

    def initialize(self, state: State) -> State:
        logging.getLogger('nlisim').debug("Initializing " + self.name)
        il8: IL8State = state.il8

        # config file values
        il8.half_life = self.config.getfloat('half_life')  # units: min
        il8.macrophage_secretion_rate = self.config.getfloat(
            'macrophage_secretion_rate'
        )  # units: atto-mol * cell^-1 * h^-1
        il8.neutrophil_secretion_rate = self.config.getfloat(
            'neutrophil_secretion_rate'
        )  # units: atto-mol * cell^-1 * h^-1
        il8.pneumocyte_secretion_rate = self.config.getfloat(
            'pneumocyte_secretion_rate'
        )  # units: atto-mol * cell^-1 * h^-1
        il8.k_d = self.config.getfloat('k_d')  # units: atto-mol
        il8.diffusion_constant = self.config.getfloat('diffusion_constant')  # units: µm^2/min

        # computed values
        il8.half_life_multiplier = 0.5 ** (
            1 * self.time_step / il8.half_life
        )  # units: step * (min/step) / min -> 1
        # time unit conversions
        # units: (atto-mol * cell^-1 * h^-1 * (min * step^-1) / (min * hour^-1)
        #        = atto-mol * cell^-1 * step^-1
        il8.macrophage_secretion_rate_unit_t = il8.macrophage_secretion_rate * (self.time_step / 60)
        il8.neutrophil_secretion_rate_unit_t = il8.neutrophil_secretion_rate * (self.time_step / 60)
        il8.pneumocyte_secretion_rate_unit_t = il8.pneumocyte_secretion_rate * (self.time_step / 60)

        # matrices for diffusion
        cn_a, cn_b, dofs = assemble_mesh_laplacian_crank_nicholson(
            state=state, diffusivity=il8.diffusion_constant, dt=self.time_step
        )
        il8.cn_a = cn_a
        il8.cn_b = cn_b
        il8.dofs = dofs

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        from nlisim.modules.neutrophil import NeutrophilCellData, NeutrophilState
        from nlisim.modules.phagocyte import PhagocyteStatus

        il8: IL8State = state.il8
        molecules: MoleculesState = state.molecules
        neutrophil: NeutrophilState = state.neutrophil
        mesh: TetrahedralMesh = state.mesh

        # IL8 activates neutrophils
        for neutrophil_cell_index in neutrophil.cells.alive():
            neutrophil_cell: NeutrophilCellData = neutrophil.cells[neutrophil_cell_index]
            if neutrophil_cell['status'] in {PhagocyteStatus.RESTING or PhagocyteStatus.ACTIVE}:
                il8_concentration_at_neutrophil = mesh.evaluate_point_function(
                    point_function=il8.field,
                    element_index=neutrophil_cell['element_index'],
                    point=neutrophil_cell['point'],
                )
                if (
                    activation_function(
                        x=il8_concentration_at_neutrophil,
                        k_d=il8.k_d,
                        h=self.time_step / 60,  # units: (min/step) / (min/hour)
                        volume=1,  # already a concentration
                        b=1,
                    )
                    > rg.uniform()
                ):
                    neutrophil_cell['status'] = PhagocyteStatus.ACTIVE
                    neutrophil_cell['status_iteration'] = 0

        # Degrade IL8
        il8.field *= il8.half_life_multiplier
        il8.field *= turnover_rate(
            x=np.ones(shape=il8.field.shape, dtype=np.float64),
            x_system=0.0,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )

        # Diffusion of IL8
        il8.field[:] = apply_mesh_diffusion_crank_nicholson(
            variable=il8.field,
            cn_a=il8.cn_a,
            cn_b=il8.cn_b,
            dofs=il8.dofs,
        )

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        il8: IL8State = state.il8
        mesh: TetrahedralMesh = state.mesh

        return {
            'concentration (nM)': float(
                mesh.integrate_point_function(il8.field) / 1e9 / mesh.total_volume
            ),
        }

    def visualization_data(self, state: State):
        il8: IL8State = state.il8
        return 'molecule', il8.field
