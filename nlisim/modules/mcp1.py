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
from nlisim.state import State
from nlisim.util import secrete_in_element, turnover_rate


def molecule_point_field_factory(self: 'MCP1State') -> np.ndarray:
    return self.global_state.mesh.allocate_point_variable(dtype=np.float64)


@attr.s(kw_only=True, repr=False)
class MCP1State(ModuleState):
    field: np.ndarray = attr.ib(
        default=attr.Factory(molecule_point_field_factory, takes_self=True)
    )  # units: atto-mols
    half_life: float  # units: min
    half_life_multiplier: float  # units: proportion
    macrophage_secretion_rate: float  # units: atto-mol * cell^-1 * h^-1
    pneumocyte_secretion_rate: float  # units: atto-mol * cell^-1 * h^-1
    macrophage_secretion_rate_unit_t: float  # units: atto-mol * cell^-1 * step^-1
    pneumocyte_secretion_rate_unit_t: float  # units: atto-mol * cell^-1 * step^-1
    k_d: float  # units: aM
    turnover_rate: float
    diffusion_constant: float  # units: µm^2/min
    cn_a: csr_matrix  # `A` matrix for Crank-Nicholson
    cn_b: csr_matrix  # `B` matrix for Crank-Nicholson
    dofs: Any  # degrees of freedom in mesh


class MCP1(ModuleModel):
    """MCP1"""

    name = 'mcp1'
    StateClass = MCP1State

    def initialize(self, state: State) -> State:
        mcp1: MCP1State = state.mcp1
        molecules: MoleculesState = state.molecules

        # config file values
        mcp1.half_life = self.config.getfloat('half_life')  # units: min
        mcp1.macrophage_secretion_rate = self.config.getfloat(
            'macrophage_secretion_rate'
        )  # units: atto-mol * cell^-1 * h^-1
        mcp1.pneumocyte_secretion_rate = self.config.getfloat(
            'pneumocyte_secretion_rate'
        )  # units: atto-mol * cell^-1 * h^-1
        mcp1.k_d = self.config.getfloat('k_d')  # units: aM
        mcp1.diffusion_constant = self.config.getfloat('diffusion_constant')  # units: µm^2/min

        # computed values
        mcp1.turnover_rate = turnover_rate(
            x=1.0,
            x_system=0.0,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )
        mcp1.half_life_multiplier = 0.5 ** (
            self.time_step / mcp1.half_life
        )  # units in exponent: (min/step) / min -> 1/step
        # time unit conversions
        # units: (atto-mol * cell^-1 * h^-1 * (min * step^-1) / (min * hour^-1)
        #        = atto-mol * cell^-1 * step^-1
        mcp1.macrophage_secretion_rate_unit_t = mcp1.macrophage_secretion_rate * (
            self.time_step / 60
        )
        mcp1.pneumocyte_secretion_rate_unit_t = mcp1.pneumocyte_secretion_rate * (
            self.time_step / 60
        )

        # matrices for diffusion
        cn_a, cn_b, dofs = assemble_mesh_laplacian_crank_nicholson(
            state=state, diffusivity=mcp1.diffusion_constant, dt=self.time_step
        )
        mcp1.cn_a = cn_a
        mcp1.cn_b = cn_b
        mcp1.dofs = dofs

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        from nlisim.modules.macrophage import MacrophageCellData, MacrophageState
        from nlisim.modules.pneumocyte import PneumocyteCellData, PneumocyteState

        mcp1: MCP1State = state.mcp1
        pneumocyte: PneumocyteState = state.pneumocyte
        macrophage: MacrophageState = state.macrophage
        mesh: TetrahedralMesh = state.mesh

        # interact with pneumocytes
        for pneumocyte_cell_index in pneumocyte.cells.alive():
            pneumocyte_cell: PneumocyteCellData = pneumocyte.cells[pneumocyte_cell_index]

            if pneumocyte_cell['tnfa']:
                secrete_in_element(
                    mesh=mesh,
                    point_field=mcp1.field,
                    element_index=pneumocyte.cells.element_index[pneumocyte_cell_index],
                    point=pneumocyte_cell['point'],
                    amount=mcp1.pneumocyte_secretion_rate_unit_t,
                )

        # interact with macrophages
        for macrophage_cell_index in macrophage.cells.alive():
            macrophage_cell: MacrophageCellData = macrophage.cells[macrophage_cell_index]

            if macrophage_cell['tnfa']:
                secrete_in_element(
                    mesh=mesh,
                    point_field=mcp1.field,
                    element_index=macrophage.cells.element_index[macrophage_cell_index],
                    point=macrophage_cell['point'],
                    amount=mcp1.macrophage_secretion_rate_unit_t,
                )

        # Degrade MCP1
        mcp1.field *= mcp1.half_life_multiplier * mcp1.turnover_rate

        # Diffusion of MCP1
        mcp1.field[:] = apply_mesh_diffusion_crank_nicholson(
            variable=mcp1.field,
            cn_a=mcp1.cn_a,
            cn_b=mcp1.cn_b,
            dofs=mcp1.dofs,
        )

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        mcp1: MCP1State = state.mcp1
        mesh: TetrahedralMesh = state.mesh

        return {
            'concentration (nM)': float(
                mesh.integrate_point_function(mcp1.field) / 1e9 / mesh.total_volume
            ),
        }

    def visualization_data(self, state: State):
        mcp1: MCP1State = state.mcp1
        return 'molecule', mcp1.field
