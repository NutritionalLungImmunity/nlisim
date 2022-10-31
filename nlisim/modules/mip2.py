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


def molecule_point_field_factory(self: 'MIP2State') -> np.ndarray:
    return self.global_state.mesh.allocate_point_variable(dtype=np.float64)


@attr.s(kw_only=True, repr=False)
class MIP2State(ModuleState):
    field: np.ndarray = attr.ib(
        default=attr.Factory(molecule_point_field_factory, takes_self=True)
    )  # units: atto-M
    half_life: float
    half_life_multiplier: float  # units: proportion
    macrophage_secretion_rate: float  # units: atto-mol * cell^-1 * h^-1
    neutrophil_secretion_rate: float  # units: atto-mol * cell^-1 * h^-1
    pneumocyte_secretion_rate: float  # units: atto-mol * cell^-1 * h^-1
    macrophage_secretion_rate_unit_t: float  # units: atto-mol * cell^-1 * step^-1
    pneumocyte_secretion_rate_unit_t: float  # units: atto-mol * cell^-1 * step^-1
    neutrophil_secretion_rate_unit_t: float  # units: atto-mol * cell^-1 * step^-1
    k_d: float  # aM
    diffusion_constant: float  # units: µm^2/min
    cn_a: csr_matrix  # `A` matrix for Crank-Nicholson
    cn_b: csr_matrix  # `B` matrix for Crank-Nicholson


class MIP2(ModuleModel):
    """MIP2"""

    name = 'mip2'
    StateClass = MIP2State

    def initialize(self, state: State) -> State:
        logger.info("Initializing " + self.name)
        mip2: MIP2State = state.mip2
        mesh: TetrahedralMesh = state.mesh

        # config file values
        mip2.half_life = self.config.getfloat('half_life')
        mip2.macrophage_secretion_rate = self.config.getfloat(
            'macrophage_secretion_rate'
        )  # units: atto-mol * cell^-1 * h^-1
        mip2.neutrophil_secretion_rate = self.config.getfloat(
            'neutrophil_secretion_rate'
        )  # units: atto-mol * cell^-1 * h^-1
        mip2.pneumocyte_secretion_rate = self.config.getfloat(
            'pneumocyte_secretion_rate'
        )  # units: atto-mol * cell^-1 * h^-1
        mip2.k_d = self.config.getfloat('k_d')  # units: atto-mol * cell^-1 * h^-1
        mip2.diffusion_constant = self.config.getfloat('diffusion_constant')  # units: µm^2/min

        # computed values
        mip2.half_life_multiplier = 0.5 ** (
            self.time_step / mip2.half_life
        )  # units in exponent: (min/step) / min -> 1/step
        logger.info(f"Computed {mip2.half_life_multiplier=}")
        # time unit conversions.
        # units: (atto-mol * cell^-1 * h^-1 * (min * step^-1) / (min * hour^-1)
        #        = atto-mol * cell^-1 * step^-1
        mip2.macrophage_secretion_rate_unit_t = mip2.macrophage_secretion_rate * (
            self.time_step / 60
        )  # units: atto-mol * cell^-1 * step^-1
        logger.info(f"Computed {mip2.macrophage_secretion_rate_unit_t=}")
        mip2.neutrophil_secretion_rate_unit_t = mip2.neutrophil_secretion_rate * (
            self.time_step / 60
        )  # units: atto-mol * cell^-1 * step^-1
        logger.info(f"Computed {mip2.neutrophil_secretion_rate_unit_t=}")
        mip2.pneumocyte_secretion_rate_unit_t = mip2.pneumocyte_secretion_rate * (
            self.time_step / 60
        )  # units: atto-mol * cell^-1 * step^-1
        logger.info(f"Computed {mip2.pneumocyte_secretion_rate_unit_t=}")

        # matrices for diffusion
        cn_a, cn_b = assemble_mesh_laplacian_crank_nicholson(
            laplacian=mesh.laplacian, diffusivity=mip2.diffusion_constant, dt=self.time_step
        )
        mip2.cn_a = cn_a
        mip2.cn_b = cn_b

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        logger.info("Advancing " + self.name + f" from t={previous_time}")

        from nlisim.modules.macrophage import MacrophageCellData, MacrophageState
        from nlisim.modules.neutrophil import NeutrophilCellData, NeutrophilState
        from nlisim.modules.phagocyte import PhagocyteStatus
        from nlisim.modules.pneumocyte import PneumocyteCellData, PneumocyteState

        mip2: MIP2State = state.mip2
        neutrophil: NeutrophilState = state.neutrophil
        pneumocyte: PneumocyteState = state.pneumocyte
        macrophage: MacrophageState = state.macrophage
        molecules: MoleculesState = state.molecules
        mesh: TetrahedralMesh = state.mesh

        assert np.alltrue(mip2.field >= 0.0)

        # interact with neutrophils
        neutrophil_activation: np.ndarray = activation_function(
            x=mip2.field,
            k_d=mip2.k_d,
            h=self.time_step / 60,  # units: (min/step) / (min/hour)
            volume=1.0,  # already a concentration
            b=1,
        )
        for neutrophil_cell_index in neutrophil.cells.alive():
            neutrophil_cell: NeutrophilCellData = neutrophil.cells[neutrophil_cell_index]
            neutrophil_cell_element: int = neutrophil_cell['element_index']

            if (
                neutrophil_cell['status'] == PhagocyteStatus.RESTING
                and neutrophil_activation[neutrophil_cell_element] > rg.uniform()
            ):
                neutrophil_cell['status'] = PhagocyteStatus.ACTIVATING
                neutrophil_cell['status_iteration'] = 0
            elif neutrophil_cell['tnfa']:
                secrete_in_element(
                    mesh=mesh,
                    point_field=mip2.field,
                    element_index=neutrophil_cell_element,
                    point=neutrophil_cell["point"],
                    amount=mip2.neutrophil_secretion_rate_unit_t,
                )
                if neutrophil_activation[neutrophil_cell_element] > rg.uniform():
                    neutrophil_cell['status_iteration'] = 0

        # interact with pneumocytes
        for pneumocyte_cell_index in pneumocyte.cells.alive():
            pneumocyte_cell: PneumocyteCellData = pneumocyte.cells[pneumocyte_cell_index]

            if pneumocyte_cell['tnfa']:
                secrete_in_element(
                    mesh=mesh,
                    point_field=mip2.field,
                    element_index=pneumocyte_cell['element_index'],
                    point=pneumocyte_cell['point'],
                    amount=mip2.pneumocyte_secretion_rate_unit_t,
                )

        # interact with macrophages
        for macrophage_cell_index in macrophage.cells.alive():
            macrophage_cell: MacrophageCellData = macrophage.cells[macrophage_cell_index]

            if macrophage_cell['tnfa']:
                secrete_in_element(
                    mesh=mesh,
                    point_field=mip2.field,
                    element_index=macrophage_cell['element_index'],
                    point=macrophage_cell['point'],
                    amount=mip2.macrophage_secretion_rate_unit_t,
                )

        # Degrade MIP2
        mip2.field *= mip2.half_life_multiplier
        turnover(
            field=mip2.field,
            system_concentration=0.0,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )

        # Diffusion of MIP2
        logger.info(f"diffusing {self.name}")
        apply_mesh_diffusion_crank_nicholson(
            variable=mip2.field,
            cn_a=mip2.cn_a,
            cn_b=mip2.cn_b,
        )

        assert np.alltrue(mip2.field >= 0.0)

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        mip2: MIP2State = state.mip2
        mesh: TetrahedralMesh = state.mesh

        return {
            'concentration (nM)': float(
                mesh.integrate_point_function(mip2.field) / 1e9 / mesh.total_volume
            ),
        }

    def visualization_data(self, state: State):
        mip2: MIP2State = state.mip2
        return 'molecule', mip2.field
