import logging
from typing import Any, Dict

# noinspection PyPackageRequirements
import attr

# noinspection PyPackageRequirements
from attr import attrib, attrs

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
from nlisim.util import michaelian_kinetics, turnover_rate


def molecule_point_field_factory(self: 'HemopexinState') -> np.ndarray:
    return self.global_state.mesh.allocate_point_variable(dtype=np.float64)


@attrs(kw_only=True, repr=False)
class HemopexinState(ModuleState):
    field: np.ndarray = attrib(
        default=attr.Factory(molecule_point_field_factory, takes_self=True)
    )  # units: atto-M
    k_m: float  # units: aM
    k_cat: float  # units: XXX
    half_life: float  # units: min
    half_life_multiplier: float  # units: proportion
    system_concentration: float  # units: aM
    diffusion_constant: float  # units: µm^2/min
    cn_a: csr_matrix  # `A` matrix for Crank-Nicholson
    cn_b: csr_matrix  # `B` matrix for Crank-Nicholson
    dofs: Any  # degrees of freedom in mesh


class Hemopexin(ModuleModel):
    """Hemopexin"""

    name = 'hemopexin'
    StateClass = HemopexinState

    def initialize(self, state: State) -> State:
        logging.getLogger('nlisim').debug("Initializing " + self.name)
        hemopexin: HemopexinState = state.hemopexin

        # config file values
        hemopexin.k_m = self.config.getfloat('k_m')
        hemopexin.k_cat = self.config.getfloat('k_cat')
        hemopexin.system_concentration = self.config.getfloat('system_concentration')
        hemopexin.diffusion_constant = self.config.getfloat('diffusion_constant')  # units: µm^2/min

        # computed values
        hemopexin.half_life_multiplier = 0.5 ** (
            self.time_step / hemopexin.half_life
        )  # units in exponent: (min/step) / min -> 1/step

        # initialize mesh TODO: tissue vs. air
        hemopexin.field = hemopexin.system_concentration

        # matrices for diffusion
        cn_a, cn_b, dofs = assemble_mesh_laplacian_crank_nicholson(
            state=state, diffusivity=hemopexin.diffusion_constant, dt=self.time_step
        )
        hemopexin.cn_a = cn_a
        hemopexin.cn_b = cn_b
        hemopexin.dofs = dofs

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        from nlisim.modules.hemoglobin import HemoglobinState

        hemopexin: HemopexinState = state.hemopexin
        hemoglobin: HemoglobinState = state.hemoglobin
        molecules: MoleculesState = state.molecules
        mesh: TetrahedralMesh = state.mesh

        # Hemopexin / Hemoglobin reaction
        reacted_quantity = michaelian_kinetics(
            substrate=hemopexin.field,
            enzyme=hemoglobin.field,
            k_m=hemopexin.k_m,
            h=self.time_step / 60,  # units: (min/step) / (min/hour)
            k_cat=hemopexin.k_cat,
            volume=mesh.point_dual_volumes,
        )
        reacted_quantity = np.min([reacted_quantity, hemopexin.field, hemoglobin.field], axis=0)
        hemopexin.field[:] = np.maximum(0.0, hemopexin.field - reacted_quantity)
        hemoglobin.field[:] = np.maximum(0.0, hemoglobin.field - reacted_quantity)

        # Degrade Hemopexin
        hemopexin.field *= hemopexin.half_life_multiplier
        hemopexin.field *= turnover_rate(
            x=hemopexin.field,
            x_system=hemopexin.system_concentration,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )

        # Diffusion of Hemolysin
        hemopexin.field[:] = apply_mesh_diffusion_crank_nicholson(
            variable=hemopexin.field,
            cn_a=hemopexin.cn_a,
            cn_b=hemopexin.cn_b,
            dofs=hemopexin.dofs,
        )

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        hemopexin: HemopexinState = state.hemopexin
        mesh: TetrahedralMesh = state.mesh

        return {
            'concentration (nM)': float(
                mesh.integrate_point_function(hemopexin.field) / 1e9 / mesh.total_volume
            ),
        }

    def visualization_data(self, state: State):
        hemopexin: HemopexinState = state.hemopexin
        return 'molecule', hemopexin.field
