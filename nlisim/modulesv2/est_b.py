import math

import attr
from attr import attrib, attrs
import numpy as np

from nlisim.module import ModuleState
from nlisim.modulesv2.geometry import GeometryState
from nlisim.modulesv2.molecule import MoleculeModel
from nlisim.modulesv2.molecules import MoleculesState
from nlisim.state import State
from nlisim.util import turnover_rate


def molecule_grid_factory(self: 'EstBState') -> np.ndarray:
    return np.zeros(shape=self.global_state.grid.shape, dtype=float)


@attrs(kw_only=True, repr=False)
class EstBState(ModuleState):
    grid: np.ndarray = attrib(default=attr.Factory(molecule_grid_factory, takes_self=True))
    iron_buffer: np.ndarray = attrib(default=attr.Factory(molecule_grid_factory, takes_self=True))
    half_life: float
    half_life_multiplier: float
    k_m: float
    kcat: float
    system_concentration: float
    system_amount_per_voxel: float


class EstB(MoleculeModel):
    """Esterase B"""

    name = 'estb'
    StateClass = EstBState

    def initialize(self, state: State) -> State:
        estb: EstBState = state.estb
        geometry: GeometryState = state.geometry
        voxel_volume = geometry.voxel_volume

        # config file values
        estb.half_life = self.config.getfloat('half_life')
        estb.km = self.config.getfloat('km')
        estb.kcat = self.config.getfloat('kcat')
        estb.system_concentration = self.config.getfloat('system_concentration')

        # computed values
        estb.half_life_multiplier = 1 + math.log(0.5) / (estb.half_life / state.simulation.time_step_size)
        estb.system_amount_per_voxel = estb.system_concentration * voxel_volume

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        from nlisim.modulesv2.iron import IronState
        from nlisim.modulesv2.tafc import TAFCState

        estb: EstBState = state.estb
        iron: IronState = state.iron
        tafc: TAFCState = state.tafc
        molecules: MoleculesState = state.molecules
        geometry: GeometryState = state.geometry
        voxel_volume = geometry.voxel_volume

        # contribute our iron buffer to the iron pool
        iron.grid += estb.iron_buffer
        estb.iron_buffer.fill(0.0)

        # interact with TAFC
        v1 = self.michaelian_kinetics(substrate=tafc.grid["TAFC"],
                                      enzyme=estb.grid,
                                      km=estb.k_m,
                                      k_cat=estb.kcat,
                                      h=state.simulation.time_step_size / 60,
                                      voxel_volume=voxel_volume)
        v2 = self.michaelian_kinetics(substrate=tafc.grid["TAFCBI"],
                                      enzyme=estb.grid,
                                      km=estb.k_m,
                                      k_cat=estb.kcat,
                                      h=state.simulation.time_step_size / 60,
                                      voxel_volume=voxel_volume)
        tafc.grid["TAFC"] -= v1
        tafc.grid["TAFCBI"] -= v2
        estb.iron_buffer += v2  # set equal to zero previously

        # Degrade EstB
        estb.grid *= estb.half_life_multiplier
        estb.grid *= turnover_rate(x_mol=estb.grid,
                                   x_system_mol=estb.system_amount_per_voxel,
                                   base_turnover_rate=molecules.turnover_rate,
                                   rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t)

        # Diffusion of EstB
        self.diffuse(estb.grid, molecules.diffusion_constant_timestep)

        return state
