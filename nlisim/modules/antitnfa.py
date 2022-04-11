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
from nlisim.state import State
from nlisim.util import michaelian_kinetics, turnover_rate


def molecule_point_field_factory(self: 'AntiTNFaState') -> np.ndarray:
    return self.global_state.mesh.allocate_point_variable(dtype=np.float64)


@attr.s(kw_only=True, repr=False)
class AntiTNFaState(ModuleState):
    field: np.ndarray = attr.ib(
        default=attr.Factory(molecule_point_field_factory, takes_self=True)
    )  # units: atto-M
    half_life: float  # units: min
    half_life_multiplier: float  # units: proportion
    react_time_unit: float  # units: hour/step
    k_m: float  # units: aM
    system_concentration: float  # units: aM
    diffusion_constant: float  # units: µm^2/min
    cn_a: csr_matrix  # `A` matrix for Crank-Nicholson
    cn_b: csr_matrix  # `B` matrix for Crank-Nicholson
    dofs: Any  # degrees of freedom in mesh


class AntiTNFa(ModuleModel):
    name = 'antitnfa'
    StateClass = AntiTNFaState

    def initialize(self, state: State) -> State:
        anti_tnf_a: AntiTNFaState = state.antitnfa
        mesh: TetrahedralMesh = state.mesh

        # config file values
        anti_tnf_a.half_life = self.config.getfloat('half_life')  # units: min
        anti_tnf_a.react_time_unit = self.config.getfloat(
            'react_time_unit'
        )  # units: sec TODO: understand this
        anti_tnf_a.k_m = self.config.getfloat('k_m')  # units: aM
        anti_tnf_a.system_concentration = self.config.getfloat('system_concentration')  # units: aM
        anti_tnf_a.diffusion_constant = self.config.getfloat(
            'diffusion_constant'
        )  # units: µm^2/min

        # computed values
        anti_tnf_a.half_life_multiplier = 0.5 ** (
            self.time_step / anti_tnf_a.half_life
        )  # units in exponent: (min/step) / min -> 1/step

        # initialize concentration field TODO: tissue vs. air
        anti_tnf_a.field = anti_tnf_a.system_concentration

        # matrices for diffusion
        cn_a, cn_b, dofs = assemble_mesh_laplacian_crank_nicholson(
            state=state, diffusivity=anti_tnf_a.diffusion_constant, dt=self.time_step
        )
        anti_tnf_a.cn_a = cn_a
        anti_tnf_a.cn_b = cn_b
        anti_tnf_a.dofs = dofs

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advances the state by a single time step."""
        from nlisim.modules.tnfa import TNFaState

        anti_tnf_a: AntiTNFaState = state.antitnfa
        molecules: MoleculesState = state.molecules
        tnf_a: TNFaState = state.tnfa
        mesh: TetrahedralMesh = state.mesh

        # AntiTNFa / TNFa reaction
        reacted_quantity = michaelian_kinetics(
            substrate=anti_tnf_a.field,
            enzyme=tnf_a.mesh,
            k_m=anti_tnf_a.k_m,
            h=anti_tnf_a.react_time_unit,  # TODO: understand why units are seconds here
            k_cat=1.0,  # default TODO use k_cat to reparameterize into hours
            volume=mesh.point_dual_volumes,
        )
        reacted_quantity = np.min([reacted_quantity, anti_tnf_a.field, tnf_a.mesh], axis=0)
        anti_tnf_a.field[:] = np.maximum(0.0, anti_tnf_a.field - reacted_quantity)
        tnf_a.mesh[:] = np.maximum(0.0, tnf_a.mesh - reacted_quantity)

        # Degradation of AntiTNFa
        anti_tnf_a.system_concentration *= (
            anti_tnf_a.half_life_multiplier
        )  # TODO: document meaning of this
        anti_tnf_a.field *= turnover_rate(
            x=anti_tnf_a.field,
            x_system=anti_tnf_a.system_concentration,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )

        # Diffusion of AntiTNFa
        anti_tnf_a.field[:] = apply_mesh_diffusion_crank_nicholson(
            variable=anti_tnf_a.field,
            cn_a=anti_tnf_a.cn_a,
            cn_b=anti_tnf_a.cn_b,
            dofs=anti_tnf_a.dofs,
        )

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        anti_tnf_a: AntiTNFaState = state.antitnfa
        mesh: TetrahedralMesh = state.mesh

        return {
            'concentration (nM)': float(
                mesh.integrate_point_function(anti_tnf_a.field) / 1e9 / mesh.total_volume
            ),
        }

    def visualization_data(self, state: State):
        anti_tnf_a: AntiTNFaState = state.antitnfa
        return 'molecule', anti_tnf_a.field
