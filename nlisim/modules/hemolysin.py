from typing import Any, Dict

import attr
from attr import attrib, attrs
import numpy as np

from nlisim.coordinates import Voxel
from nlisim.diffusion import apply_diffusion
from nlisim.grid import RectangularGrid
from nlisim.module import ModuleModel, ModuleState
from nlisim.modules.molecules import MoleculesState
from nlisim.state import State
from nlisim.util import turnover_rate


def molecule_grid_factory(self: 'HemolysinState') -> np.ndarray:
    return np.zeros(shape=self.global_state.grid.shape, dtype=float)


@attrs(kw_only=True, repr=False)
class HemolysinState(ModuleState):
    grid: np.ndarray = attrib(default=attr.Factory(molecule_grid_factory, takes_self=True))
    hemolysin_qtty: float


class Hemolysin(ModuleModel):
    """Hemolysin"""

    name = 'hemolysin'
    StateClass = HemolysinState

    def initialize(self, state: State) -> State:
        hemolysin: HemolysinState = state.hemolysin

        # config file values
        hemolysin.hemolysin_qtty = self.config.getfloat('hemolysin_qtty')
        # constant from setting rate of secretion rate to 1

        # computed values (none)

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        from nlisim.modules.afumigatus import (
            AfumigatusCellData,
            AfumigatusCellStatus,
            AfumigatusState,
        )

        hemolysin: HemolysinState = state.hemolysin
        molecules: MoleculesState = state.molecules
        afumigatus: AfumigatusState = state.afumigatus
        grid: RectangularGrid = state.grid

        # fungus releases hemolysin
        for afumigatus_cell_index in afumigatus.cells.alive():
            afumigatus_cell: AfumigatusCellData = afumigatus.cells[afumigatus_cell_index]
            if afumigatus_cell['status'] == AfumigatusCellStatus.HYPHAE:
                afumigatus_cell_voxel: Voxel = grid.get_voxel(afumigatus_cell['point'])
                hemolysin.grid[tuple(afumigatus_cell_voxel)] += hemolysin.hemolysin_qtty

        # Degrade Hemolysin
        hemolysin.grid *= turnover_rate(
            x=hemolysin.grid,
            x_system=0.0,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )

        # Diffusion of Hemolysin
        hemolysin.grid[:] = apply_diffusion(
            variable=hemolysin.grid,
            laplacian=molecules.laplacian,
            diffusivity=molecules.diffusion_constant,
            dt=self.time_step,
        )

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        from nlisim.util import TissueType

        hemolysin: HemolysinState = state.hemolysin
        voxel_volume = state.voxel_volume
        mask = state.lung_tissue != TissueType.AIR

        return {
            'concentration (nM)': float(np.mean(hemolysin.grid[mask]) / voxel_volume / 1e9),
        }

    def visualization_data(self, state: State):
        hemolysin: HemolysinState = state.hemolysin
        return 'molecule', hemolysin.grid
