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


def molecule_grid_factory(self: 'HemoglobinState') -> np.ndarray:
    return np.zeros(shape=self.global_state.grid.shape, dtype=float)


@attrs(kw_only=True, repr=False)
class HemoglobinState(ModuleState):
    grid: np.ndarray = attrib(default=attr.Factory(molecule_grid_factory, takes_self=True))
    uptake_rate: float
    ma_heme_import_rate: float


class Hemoglobin(MoleculeModel):
    """Hemoglobin"""

    name = 'hemoglobin'
    StateClass = HemoglobinState

    def initialize(self, state: State) -> State:
        hemoglobin: HemoglobinState = state.hemoglobin

        # config file values
        hemoglobin.uptake_rate = self.config.getfloat('uptake_rate')
        hemoglobin.ma_heme_import_rate = self.config.getfloat('ma_heme_import_rate')

        # computed values (none)

        return state

    def advance(self, state: State, previous_time: float) -> State:
        """Advance the state by a single time step."""
        from nlisim.modules.afumigatus import (
            AfumigatusCellData,
            AfumigatusCellStatus,
            AfumigatusState,
        )

        hemoglobin: HemoglobinState = state.hemoglobin
        molecules: MoleculesState = state.molecules
        afumigatus: AfumigatusState = state.afumigatus
        grid: RectangularGrid = state.grid

        # afumigatus uptakes iron from hemoglobin
        for afumigatus_cell_index in afumigatus.cells.alive():
            afumigatus_cell: AfumigatusCellData = afumigatus.cells[afumigatus_cell_index]
            if afumigatus_cell['status'] in {
                AfumigatusCellStatus.HYPHAE,
                AfumigatusCellStatus.GERM_TUBE,
            }:
                afumigatus_cell_voxel: Voxel = grid.get_voxel(afumigatus_cell['point'])
                fungal_absorbed_hemoglobin = (
                    hemoglobin.uptake_rate * hemoglobin.grid[tuple(afumigatus_cell_voxel)]
                )
                hemoglobin.grid[tuple(afumigatus_cell_voxel)] -= fungal_absorbed_hemoglobin
                afumigatus_cell['iron_pool'] += 4 * fungal_absorbed_hemoglobin

        # Degrade Hemoglobin
        hemoglobin.grid *= turnover_rate(
            x=hemoglobin.grid,
            x_system=0.0,
            base_turnover_rate=molecules.turnover_rate,
            rel_cyt_bind_unit_t=molecules.rel_cyt_bind_unit_t,
        )

        # Diffusion of Hemoglobin
        self.diffuse(hemoglobin.grid, state)

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        hemoglobin: HemoglobinState = state.hemoglobin
        voxel_volume = state.voxel_volume

        return {
            'concentration': float(np.mean(hemoglobin.grid) / voxel_volume),
        }

    def visualization_data(self, state: State):
        hemoglobin: HemoglobinState = state.hemoglobin
        return 'molecule', hemoglobin.grid
