import math

import attr
import numpy as np

from nlisim.coordinates import Voxel
from nlisim.grid import RectangularGrid
from nlisim.module import ModuleState
from nlisim.henrique_modules.geometry import GeometryState
from nlisim.henrique_modules.molecules import MoleculeModel, MoleculesState
from nlisim.random import rg
from nlisim.state import State
from nlisim.util import activation_function, turnover_rate


def molecule_grid_factory(self: 'TNFaState') -> np.ndarray:
    return np.zeros(shape=self.global_state.grid.shape, dtype=float)


@attr.s(kw_only=True, repr=False)
class TNFaState(ModuleState):
    grid: np.ndarray = attr.ib(default=attr.Factory(molecule_grid_factory, takes_self=True))
    half_life: float
    half_life_multiplier: float
    macrophage_secretion_rate: float
    neutrophil_secretion_rate: float
    epithelial_secretion_rate: float
    macrophage_secretion_rate_unit_t: float
    neutrophil_secretion_rate_unit_t: float
    epithelial_secretion_rate_unit_t: float
    k_d: float


class TNFa(MoleculeModel):
    name = 'tnfa'
    StateClass = TNFaState

    def initialize(self, state: State) -> State:
        tnfa: TNFaState = state.antitnfa

        # config file values
        tnfa.half_life = self.config.getfloat('half_life')
        tnfa.macrophage_secretion_rate = self.config.getfloat('macrophage_secretion_rate')
        tnfa.neutrophil_secretion_rate = self.config.getfloat('neutrophil_secretion_rate')
        tnfa.epithelial_secretion_rate = self.config.getfloat('epithelial_secretion_rate')
        tnfa.k_d = self.config.getfloat('k_d')

        # computed values
        tnfa.half_life_multiplier = 1 + math.log(0.5) / (tnfa.half_life / self.time_step)
        # time unit conversions
        tnfa.macrophage_secretion_rate_unit_t = tnfa.macrophage_secretion_rate * 60 * self.time_step
        tnfa.neutrophil_secretion_rate_unit_t = tnfa.neutrophil_secretion_rate * 60 * self.time_step
        tnfa.epithelial_secretion_rate_unit_t = tnfa.epithelial_secretion_rate * 60 * self.time_step

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        from nlisim.henrique_modules.macrophage import MacrophageCellData, MacrophageState
        from nlisim.henrique_modules.neutrophil import NeutrophilCellData, NeutrophilState
        from nlisim.henrique_modules.phagocyte import PhagocyteStatus

        tnfa: TNFaState = state.antitnfa
        molecules: MoleculesState = state.molecules
        macrophage: MacrophageState = state.macrophage
        neutrophil: NeutrophilState = state.neutrophil
        geometry: GeometryState = state.geometry
        grid: RectangularGrid = state.grid

        for macrophage_cell_index in macrophage.cells.alive():
            macrophage_cell: MacrophageCellData = macrophage.cells[macrophage_cell_index]
            macrophage_cell_voxel: Voxel = grid.get_voxel(macrophage_cell['point'])

            if macrophage_cell['status'] == PhagocyteStatus.ACTIVE:
                tnfa.grid[tuple(macrophage_cell_voxel)] += tnfa.macrophage_secretion_rate_unit_t

            if macrophage_cell['status'] in {PhagocyteStatus.RESTING, PhagocyteStatus.ACTIVE}:
                if activation_function(x=tnfa.grid[tuple(macrophage_cell_voxel)],
                                       kd=tnfa.k_d,
                                       h=self.time_step / 60,
                                       volume=geometry.voxel_volume) > rg.uniform():
                    if macrophage_cell['status'] == PhagocyteStatus.RESTING:
                        macrophage_cell['status'] = PhagocyteStatus.ACTIVATING
                    else:
                        macrophage_cell['status'] = PhagocyteStatus.ACTIVE
                    # Note: multiple activations will reset the 'clock'
                    macrophage_cell['status_iteration'] = 0
                    macrophage_cell['tnfa'] = True

        for neutrophil_cell_index in neutrophil.cells.alive():
            neutrophil_cell: NeutrophilCellData = neutrophil.cells[neutrophil_cell_index]
            neutrophil_cell_voxel: Voxel = grid.get_voxel(neutrophil_cell['point'])

            if neutrophil_cell['status'] == PhagocyteStatus.ACTIVE:
                tnfa.grid[tuple(neutrophil_cell_voxel)] += tnfa.neutrophil_secretion_rate_unit_t

            if neutrophil_cell['status'] in {PhagocyteStatus.RESTING, PhagocyteStatus.ACTIVE}:
                if activation_function(x=tnfa.grid[tuple(neutrophil_cell_voxel)],
                                       kd=tnfa.k_d,
                                       h=self.time_step / 60,
                                       volume=geometry.voxel_volume) > rg.uniform():
                    if neutrophil_cell['status'] == PhagocyteStatus.RESTING:
                        neutrophil_cell['status'] = PhagocyteStatus.ACTIVATING
                    else:
                        neutrophil_cell['status'] = PhagocyteStatus.ACTIVE
                    # Note: multiple activations will reset the 'clock'
                    neutrophil_cell['status_iteration'] = 0
                    neutrophil_cell['tnfa'] = True

        # Degrade TNFa
        tnfa.grid *= tnfa.half_life_multiplier
        tnfa.grid *= turnover_rate(x_mol=np.array(1.0, dtype=np.float64),
                                   x_system_mol=0.0,
                                   base_turnover_rate=molecules.turnover_rate,
                                   rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t)

        # Diffusion of TNFa
        self.diffuse(tnfa.grid, molecules.diffusion_constant_timestep)

        return state
