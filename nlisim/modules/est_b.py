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
from nlisim.modules.molecules import MoleculesState
from nlisim.state import State
from nlisim.util import michaelian_kinetics, turnover_rate


def molecule_point_field_factory(self: 'EstBState') -> np.ndarray:
    return self.global_state.mesh.allocate_point_variable(dtype=np.float64)


@attrs(kw_only=True, repr=False)
class EstBState(ModuleState):
    field: np.ndarray = attrib(
        default=attr.Factory(molecule_point_field_factory, takes_self=True)
    )  # units: atto-M
    iron_buffer: np.ndarray = attrib(
        default=attr.Factory(molecule_point_field_factory, takes_self=True)
    )  # units: atto-M
    half_life: float  # units: min
    half_life_multiplier: float  # units: proportion
    k_m: float  # units: aM
    k_cat: float
    system_concentration: float
    diffusion_constant: float  # units: µm^2/min
    cn_a: csr_matrix  # `A` matrix for Crank-Nicholson
    cn_b: csr_matrix  # `B` matrix for Crank-Nicholson
    dofs: Any  # degrees of freedom in mesh


class EstB(ModuleModel):
    """Esterase B"""

    name = 'estb'
    StateClass = EstBState

    def initialize(self, state: State) -> State:
        estb: EstBState = state.estb

        # config file values
        estb.half_life = self.config.getfloat('half_life')
        estb.k_m = self.config.getfloat('k_m')
        estb.k_cat = self.config.getfloat('k_cat')
        estb.system_concentration = self.config.getfloat('system_concentration')
        estb.diffusion_constant = self.config.getfloat('diffusion_constant')  # units: µm^2/min

        # computed values
        estb.half_life_multiplier = 0.5 ** (
            self.time_step / estb.half_life
        )  # units: (min/step) / min -> 1/step

        # initialize concentration field
        estb.field = estb.system_concentration

        # matrices for diffusion
        cn_a, cn_b, dofs = assemble_mesh_laplacian_crank_nicholson(
            state=state, diffusivity=estb.diffusion_constant, dt=self.time_step
        )
        estb.cn_a = cn_a
        estb.cn_b = cn_b
        estb.dofs = dofs

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        from nlisim.modules.iron import IronState
        from nlisim.modules.tafc import TAFCState

        estb: EstBState = state.estb
        iron: IronState = state.iron
        tafc: TAFCState = state.tafc
        molecules: MoleculesState = state.molecules
        mesh: TetrahedralMesh = state.mesh

        # contribute our iron buffer to the iron pool
        iron.grid += estb.iron_buffer
        estb.iron_buffer[:] = 0.0

        # interact with TAFC
        v1 = michaelian_kinetics(
            substrate=tafc.field["TAFC"],
            enzyme=estb.field,
            k_m=estb.k_m,
            k_cat=estb.k_cat,
            h=self.time_step / 60,  # units: (min/step) / (min/hour)
            volume=mesh.point_dual_volumes,
        )
        v2 = michaelian_kinetics(
            substrate=tafc.field["TAFCBI"],
            enzyme=estb.field,
            k_m=estb.k_m,
            k_cat=estb.k_cat,
            h=self.time_step / 60,  # units: (min/step) / (min/hour)
            volume=mesh.point_dual_volumes,
        )
        tafc.grid["TAFC"] -= v1
        tafc.grid["TAFCBI"] -= v2
        estb.iron_buffer += v2  # set equal to zero previously

        # Degrade EstB
        estb.field *= estb.half_life_multiplier
        estb.field *= turnover_rate(
            x=estb.field,
            x_system=estb.system_concentration,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )

        # Diffusion of EstB
        estb.field[:] = apply_mesh_diffusion_crank_nicholson(
            variable=estb.field,
            cn_a=estb.cn_a,
            cn_b=estb.cn_b,
            dofs=estb.dofs,
        )

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        estb: EstBState = state.estb
        mesh: TetrahedralMesh = state.mesh

        return {
            'concentration (nM)': float(mesh.integrate_point_function(estb.field) / 1e9),
        }

    def visualization_data(self, state: State):
        estb: EstBState = state.estb
        return 'molecule', estb.field
