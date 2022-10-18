import logging
from typing import Any, Dict

import attr
from attr import attrib, attrs
import numpy as np
from scipy.sparse import csr_matrix

from nlisim.diffusion import (
    apply_mesh_diffusion_crank_nicholson,
    assemble_mesh_laplacian_crank_nicholson,
)
from nlisim.grid import TetrahedralMesh
from nlisim.module import ModuleModel, ModuleState
from nlisim.modules.afumigatus import AfumigatusCellData
from nlisim.modules.molecules import MoleculesState
from nlisim.state import State
from nlisim.util import secrete_in_element, turnover


def molecule_point_field_factory(self: 'HemolysinState') -> np.ndarray:
    return self.global_state.mesh.allocate_point_variable(dtype=np.float64)


@attrs(kw_only=True, repr=False)
class HemolysinState(ModuleState):
    field: np.ndarray = attrib(default=attr.Factory(molecule_point_field_factory, takes_self=True))
    hemolysin_qtty: float
    diffusion_constant: float  # units: µm^2/min
    cn_a: csr_matrix  # `A` matrix for Crank-Nicholson
    cn_b: csr_matrix  # `B` matrix for Crank-Nicholson
    dofs: Any  # degrees of freedom in mesh


class Hemolysin(ModuleModel):
    """Hemolysin"""

    name = 'hemolysin'
    StateClass = HemolysinState

    def initialize(self, state: State) -> State:
        logging.info("Initializing " + self.name)
        hemolysin: HemolysinState = state.hemolysin

        # config file values
        hemolysin.hemolysin_qtty = self.config.getfloat('hemolysin_qtty')
        # constant from setting rate of secretion rate to 1
        hemolysin.diffusion_constant = self.config.getfloat('diffusion_constant')  # units: µm^2/min

        # computed values (none)

        # matrices for diffusion
        cn_a, cn_b, dofs = assemble_mesh_laplacian_crank_nicholson(
            state=state, diffusivity=hemolysin.diffusion_constant, dt=self.time_step
        )
        hemolysin.cn_a = cn_a
        hemolysin.cn_b = cn_b
        hemolysin.dofs = dofs

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        logging.info("Advancing " + self.name + f" from t={previous_time}")

        from nlisim.modules.afumigatus import AfumigatusCellStatus, AfumigatusState

        hemolysin: HemolysinState = state.hemolysin
        molecules: MoleculesState = state.molecules
        afumigatus: AfumigatusState = state.afumigatus
        mesh: TetrahedralMesh = state.mesh

        assert np.alltrue(hemolysin.field >= 0.0)

        # fungus releases hemolysin

        # find the fungal cells that release hemolysin
        live_afumigatus = afumigatus.cells.alive()
        hemolysin_releasing_afumigatus = live_afumigatus[
            afumigatus.cells.cell_data[live_afumigatus]['status'] == AfumigatusCellStatus.HYPHAE
        ]

        for afumigatus_cell_index in hemolysin_releasing_afumigatus:
            afumigatus_cell: AfumigatusCellData = afumigatus.cells[afumigatus_cell_index]
            secrete_in_element(
                mesh=mesh,
                point_field=hemolysin.field,
                element_index=afumigatus_cell['element_index'],
                point=afumigatus_cell['point'],
                amount=hemolysin.hemolysin_qtty,
            )

        # Degrade Hemolysin
        turnover(
            field=hemolysin.field,
            system_concentration=0.0,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )

        # Diffusion of Hemolysin
        hemolysin.field[:] = apply_mesh_diffusion_crank_nicholson(
            variable=hemolysin.field,
            cn_a=hemolysin.cn_a,
            cn_b=hemolysin.cn_b,
            dofs=hemolysin.dofs,
        )

        assert np.alltrue(hemolysin.field >= 0.0)

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        hemolysin: HemolysinState = state.hemolysin
        mesh: TetrahedralMesh = state.mesh

        return {
            'concentration (nM)': float(
                mesh.integrate_point_function(hemolysin.field) / 1e9 / mesh.total_volume
            ),
        }

    def visualization_data(self, state: State):
        hemolysin: HemolysinState = state.hemolysin
        return 'molecule', hemolysin.field
