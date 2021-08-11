from typing import Any, Dict

import attr
from attr import attrib, attrs
import numpy as np

from nlisim.coordinates import Voxel
from nlisim.grid import RectangularGrid
from nlisim.module import ModuleState
from nlisim.modules.molecules import MoleculeModel, MoleculesState
from nlisim.state import State
from nlisim.util import turnover_rate


def molecule_grid_factory(self: 'HemolysinState') -> np.ndarray:
    return np.zeros(shape=self.global_state.grid.shape, dtype=float)


@attrs(kw_only=True, repr=False)
class HemolysinState(ModuleState):
    grid: np.ndarray = attrib(default=attr.Factory(molecule_grid_factory, takes_self=True))
    hemolysin_qtty: float


class Hemolysin(MoleculeModel):
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
        self.diffuse(hemolysin.grid, state)

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        hemolysin: HemolysinState = state.hemolysin
        voxel_volume = state.voxel_volume

        return {
            'concentration': float(np.mean(hemolysin.grid) / voxel_volume),
        }

    def visualization_data(self, state: State):
        hemolysin: HemolysinState = state.hemolysin
        return 'molecule', hemolysin.grid
