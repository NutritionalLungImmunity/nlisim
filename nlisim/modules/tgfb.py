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
from nlisim.util import activation_function, logger, turnover


def molecule_point_field_factory(self: 'TGFBState') -> np.ndarray:
    return self.global_state.mesh.allocate_point_variable(dtype=np.float64)


@attr.s(kw_only=True, repr=False)
class TGFBState(ModuleState):
    field: np.ndarray = attr.ib(
        default=attr.Factory(molecule_point_field_factory, takes_self=True)
    )  # units: atto-mols
    half_life: float  # units: min
    half_life_multiplier: float  # units: proportion
    macrophage_secretion_rate: float  # units: atto-mol * cell^-1 * h^-1
    macrophage_secretion_rate_unit_t: float  # units: atto-mol * cell^-1 * step^-1
    k_d: float  # aM
    diffusion_constant: float  # units: µm^2/min
    cn_a: csr_matrix  # `A` matrix for Crank-Nicholson
    cn_b: csr_matrix  # `B` matrix for Crank-Nicholson
    dofs: Any  # degrees of freedom in mesh


class TGFB(ModuleModel):
    """TGFB"""

    name = 'tgfb'
    StateClass = TGFBState

    def initialize(self, state: State) -> State:
        logger.info("Initializing " + self.name)
        tgfb: TGFBState = state.tgfb

        # config file values
        tgfb.half_life = self.config.getfloat('half_life')  # units: min
        tgfb.macrophage_secretion_rate = self.config.getfloat(
            'macrophage_secretion_rate'
        )  # units: atto-mol * cell^-1 * h^-1
        tgfb.k_d = self.config.getfloat('k_d')  # units: aM
        tgfb.diffusion_constant = self.config.getfloat('diffusion_constant')  # units: µm^2/min

        # computed values
        tgfb.half_life_multiplier = 0.5 ** (
            self.time_step / tgfb.half_life
        )  # units in exponent: (min/step) / min -> 1/step
        logger.info(f"Computed {tgfb.half_life_multiplier=}")
        # time unit conversions
        tgfb.macrophage_secretion_rate_unit_t = tgfb.macrophage_secretion_rate * (
            self.time_step / 60
        )  # units: atto-mol/(cell*h) * (min/step) / (min/hour)
        logger.info(f"Computed {tgfb.macrophage_secretion_rate_unit_t=}")

        # matrices for diffusion
        cn_a, cn_b, dofs = assemble_mesh_laplacian_crank_nicholson(
            state=state, diffusivity=tgfb.diffusion_constant, dt=self.time_step
        )
        tgfb.cn_a = cn_a
        tgfb.cn_b = cn_b
        tgfb.dofs = dofs

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        logger.info("Advancing " + self.name + f" from t={previous_time}")

        from nlisim.modules.macrophage import MacrophageCellData, MacrophageState
        from nlisim.modules.phagocyte import PhagocyteStatus

        tgfb: TGFBState = state.tgfb
        macrophage: MacrophageState = state.macrophage
        molecules: MoleculesState = state.molecules
        mesh: TetrahedralMesh = state.mesh

        assert np.alltrue(tgfb.field >= 0.0)

        for macrophage_cell_index in macrophage.cells.alive():
            macrophage_cell: MacrophageCellData = macrophage.cells[macrophage_cell_index]
            macrophage_cell_element: int = macrophage_cell['element_index']

            if macrophage_cell['status'] == PhagocyteStatus.INACTIVE:
                tgfb.field[macrophage_cell_element] += tgfb.macrophage_secretion_rate_unit_t
                if (
                    activation_function(
                        x=mesh.evaluate_point_function(
                            point_function=tgfb.field,
                            element_index=macrophage_cell_element,
                            point=macrophage_cell['point'],
                        ),
                        k_d=tgfb.k_d,
                        h=self.time_step / 60,  # units: (min/step) / (min/hour)
                        volume=1.0,  # already a concentration
                        b=1,
                    )
                    > rg.uniform()
                ):
                    macrophage_cell['status_iteration'] = 0

            elif macrophage_cell['status'] not in {
                PhagocyteStatus.APOPTOTIC,
                PhagocyteStatus.NECROTIC,
                PhagocyteStatus.DEAD,
            }:
                if (
                    activation_function(
                        x=mesh.evaluate_point_function(
                            point_function=tgfb.field,
                            element_index=macrophage_cell_element,
                            point=macrophage_cell['point'],
                        ),
                        k_d=tgfb.k_d,
                        h=self.time_step / 60,  # units: (min/step) / (min/hour)
                        volume=1.0,  # already a concentration
                        b=1,
                    )
                    > rg.uniform()
                ):
                    macrophage_cell['status'] = PhagocyteStatus.INACTIVATING
                    macrophage_cell[
                        'status_iteration'
                    ] = 0  # Previously, was no reset of the status iteration

        # Degrade TGFB
        tgfb.field *= tgfb.half_life_multiplier
        turnover(
            field=tgfb.field,
            system_concentration=0.0,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )

        # Diffusion of TGFB
        tgfb.field[:] = apply_mesh_diffusion_crank_nicholson(
            variable=tgfb.field,
            cn_a=tgfb.cn_a,
            cn_b=tgfb.cn_b,
            dofs=tgfb.dofs,
        )

        assert np.alltrue(tgfb.field >= 0.0)

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        tgfb: TGFBState = state.tgfb
        mesh: TetrahedralMesh = state.mesh

        return {
            'concentration (nM)': float(
                mesh.integrate_point_function(tgfb.field) / 1e9 / mesh.total_volume
            ),
        }

    def visualization_data(self, state: State):
        tgfb: TGFBState = state.tgfb
        return 'molecule', tgfb.field
