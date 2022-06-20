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
from nlisim.state import State
from nlisim.util import secrete_in_element, turnover_rate


def molecule_point_field_factory(self: 'MIP1BState') -> np.ndarray:
    return self.global_state.mesh.allocate_point_variable(dtype=np.float64)


@attr.s(kw_only=True, repr=False)
class MIP1BState(ModuleState):
    field: np.ndarray = attr.ib(
        default=attr.Factory(molecule_point_field_factory, takes_self=True)
    )  # units: atto-M
    half_life: float  # units: minutes
    half_life_multiplier: float  # units: proportion
    macrophage_secretion_rate: float  # units: atto-mol/(cell*h)
    pneumocyte_secretion_rate: float  # units: atto-mol/(cell*h)
    macrophage_secretion_rate_unit_t: float  # units: atto-mol/(cell*step)
    pneumocyte_secretion_rate_unit_t: float  # units: atto-mol/(cell*step)
    k_d: float  # units: aM
    turnover_rate: float
    diffusion_constant: float  # units: µm^2/min
    cn_a: csr_matrix  # `A` matrix for Crank-Nicholson
    cn_b: csr_matrix  # `B` matrix for Crank-Nicholson
    dofs: Any  # degrees of freedom in mesh


class MIP1B(ModuleModel):
    """MIP1B"""

    name = 'mip1b'
    StateClass = MIP1BState

    def initialize(self, state: State) -> State:
        logging.getLogger('nlisim').debug("Initializing " + self.name)
        mip1b: MIP1BState = state.mip1b
        molecules: MoleculesState = state.molecules

        # config file values
        mip1b.half_life = self.config.getfloat('half_life')
        mip1b.macrophage_secretion_rate = self.config.getfloat(
            'macrophage_secretion_rate'
        )  # units: atto-mol/(cell*h)
        mip1b.pneumocyte_secretion_rate = self.config.getfloat(
            'pneumocyte_secretion_rate'
        )  # units: atto-mol/(cell*h)
        mip1b.k_d = self.config.getfloat('k_d')  # units: aM
        mip1b.diffusion_constant = self.config.getfloat('diffusion_constant')  # units: µm^2/min

        # computed values
        mip1b.turnover_rate = turnover_rate(
            x=1.0,
            x_system=0.0,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )
        mip1b.half_life_multiplier = 0.5 ** (
            self.time_step / mip1b.half_life
        )  # units in exponent: (min/step) / min -> 1/step
        # time unit conversions
        # units: (atto-mol * cell^-1 * h^-1 * (min * step^-1) / (min * hour^-1)
        #        = atto-mol * cell^-1 * step^-1
        mip1b.macrophage_secretion_rate_unit_t = mip1b.macrophage_secretion_rate * (
            self.time_step / 60
        )
        mip1b.pneumocyte_secretion_rate_unit_t = mip1b.pneumocyte_secretion_rate * (
            self.time_step / 60
        )

        # matrices for diffusion
        cn_a, cn_b, dofs = assemble_mesh_laplacian_crank_nicholson(
            state=state, diffusivity=mip1b.diffusion_constant, dt=self.time_step
        )
        mip1b.cn_a = cn_a
        mip1b.cn_b = cn_b
        mip1b.dofs = dofs

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        from nlisim.modules.macrophage import MacrophageCellData, MacrophageState
        from nlisim.modules.pneumocyte import PneumocyteCellData, PneumocyteState

        mip1b: MIP1BState = state.mip1b
        pneumocyte: PneumocyteState = state.pneumocyte
        macrophage: MacrophageState = state.macrophage
        mesh: TetrahedralMesh = state.mesh

        # interact with pneumocytes
        for pneumocyte_cell_index in pneumocyte.cells.alive():
            pneumocyte_cell: PneumocyteCellData = pneumocyte.cells[pneumocyte_cell_index]

            if pneumocyte_cell['tnfa']:
                secrete_in_element(
                    mesh=mesh,
                    point_field=mip1b.field,
                    element_index=pneumocyte.cells.element_index[pneumocyte_cell_index],
                    point=pneumocyte_cell['point'],
                    amount=mip1b.pneumocyte_secretion_rate_unit_t,
                )

        # interact with macrophages
        for macrophage_cell_index in macrophage.cells.alive():
            macrophage_cell: MacrophageCellData = macrophage.cells[macrophage_cell_index]

            if macrophage_cell['tnfa']:
                secrete_in_element(
                    mesh=mesh,
                    point_field=mip1b.field,
                    element_index=macrophage.cells.element_index[macrophage_cell_index],
                    point=macrophage_cell['point'],
                    amount=mip1b.macrophage_secretion_rate_unit_t,
                )

        # Degrade MIP1B
        mip1b.field *= mip1b.half_life_multiplier * mip1b.turnover_rate

        # Diffusion of MIP1b
        mip1b.field[:] = apply_mesh_diffusion_crank_nicholson(
            variable=mip1b.field,
            cn_a=mip1b.cn_a,
            cn_b=mip1b.cn_b,
            dofs=mip1b.dofs,
        )

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        mip1b: MIP1BState = state.mip1b
        mesh: TetrahedralMesh = state.mesh

        return {
            'concentration (nM)': float(
                mesh.integrate_point_function(mip1b.field) / 1e9 / mesh.total_volume
            ),
        }

    def visualization_data(self, state: State):
        mip1b: MIP1BState = state.mip1b
        return 'molecule', mip1b.field
