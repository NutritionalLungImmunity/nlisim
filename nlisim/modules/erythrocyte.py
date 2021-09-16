import math
from typing import Any, Dict

import attr
from attr import attrib, attrs
import numpy as np

from nlisim.coordinates import Voxel
from nlisim.grid import RectangularGrid
from nlisim.module import ModuleModel, ModuleState
from nlisim.modules.afumigatus import AfumigatusCellStatus, AfumigatusState
from nlisim.modules.hemoglobin import HemoglobinState
from nlisim.modules.hemolysin import HemolysinState
from nlisim.modules.macrophage import MacrophageState
from nlisim.modules.molecules import MoleculesState
from nlisim.state import State
from nlisim.util import TissueType, activation_function


# note: treating these a bit more like molecules than cells.
# hence the adaptation of molecule_grid_factory
def cell_grid_factory(self: 'ErythrocyteState') -> np.ndarray:
    return np.zeros(
        shape=self.global_state.grid.shape,
        dtype=[('count', np.int64), ('hemoglobin', np.float64), ('hemorrhage', bool)],
    )


@attrs(kw_only=True)
class ErythrocyteState(ModuleState):
    cells: np.ndarray = attrib(default=attr.Factory(cell_grid_factory, takes_self=True))
    kd_hemo: float
    init_erythrocyte_level: int  # units: count
    max_erythrocyte_voxel: int  # units: count
    hemoglobin_quantity: float  # units: atto-mols
    pr_macrophage_phagocytize_erythrocyte_param: float
    pr_macrophage_phagocytize_erythrocyte: float  # units: probability


class ErythrocyteModel(ModuleModel):
    name = 'erythrocyte'
    StateClass = ErythrocyteState

    def initialize(self, state: State):
        erythrocyte: ErythrocyteState = state.erythrocyte
        voxel_volume = state.voxel_volume
        lung_tissue = state.lung_tissue
        time_step_size: float = self.time_step

        erythrocyte.kd_hemo = self.config.getfloat('kd_hemo')
        erythrocyte.init_erythrocyte_level = self.config.getint(
            'init_erythrocyte_level'
        )  # units: count
        erythrocyte.max_erythrocyte_voxel = self.config.getint(
            'max_erythrocyte_voxel'
        )  # units: count
        erythrocyte.hemoglobin_quantity = self.config.getfloat(
            'hemoglobin_concentration'
        )  # units: atto-mols
        erythrocyte.pr_macrophage_phagocytize_erythrocyte_param = self.config.getfloat(
            'pr_macrophage_phagocytize_erythrocyte_param'
        )

        # initialize cells
        # TODO: discuss
        erythrocyte.cells[lung_tissue == TissueType.BLOOD] = erythrocyte.init_erythrocyte_level
        erythrocyte.pr_macrophage_phagocytize_erythrocyte = -math.expm1(
            -time_step_size
            / 60
            / voxel_volume
            / erythrocyte.pr_macrophage_phagocytize_erythrocyte_param
        )

        return state

    def advance(self, state: State, previous_time: float):
        erythrocyte: ErythrocyteState = state.erythrocyte
        molecules: MoleculesState = state.molecules
        hemoglobin: HemoglobinState = state.hemoglobin
        hemolysin: HemolysinState = state.hemolysin
        macrophage: MacrophageState = state.macrophage
        afumigatus: AfumigatusState = state.afumigatus
        grid: RectangularGrid = state.grid
        voxel_volume: float = state.voxel_volume

        shape = erythrocyte.cells['count'].shape

        # erythrocytes replenish themselves
        avg_number_of_new_erythrocytes = (1 - molecules.turnover_rate) * (
            1 - erythrocyte.cells['count'] / erythrocyte.max_erythrocyte_voxel
        )
        mask = avg_number_of_new_erythrocytes > 0
        erythrocyte.cells['count'][mask] += np.random.poisson(
            avg_number_of_new_erythrocytes[mask], avg_number_of_new_erythrocytes[mask].shape
        )

        # ---------- interactions

        # uptake hemoglobin
        erythrocyte.cells['hemoglobin'] += hemoglobin.grid
        hemoglobin.grid.fill(0.0)

        # interact with hemolysin. pop goes the blood cell
        # TODO: avg? variable name improvement?
        avg_lysed_erythrocytes = erythrocyte.cells['count'] * activation_function(
            x=hemolysin.grid,
            k_d=erythrocyte.kd_hemo,
            h=self.time_step / 60,  # units: (min/step) / (min/hour)
            volume=voxel_volume,
            b=1,
        )
        number_lysed = np.minimum(
            np.random.poisson(avg_lysed_erythrocytes, shape), erythrocyte.cells['count']
        )
        erythrocyte.cells['hemoglobin'] += number_lysed * erythrocyte.hemoglobin_quantity
        erythrocyte.cells['count'] -= number_lysed

        # interact with Macrophage
        erythrocytes_to_hemorrhage = erythrocyte.cells['hemorrhage'] * np.random.poisson(
            erythrocyte.pr_macrophage_phagocytize_erythrocyte * erythrocyte.cells['count'], shape
        )

        for z, y, x in zip(*np.where(erythrocytes_to_hemorrhage > 0)):
            local_macrophages = macrophage.cells.get_cells_in_voxel(Voxel(x=x, y=y, z=z))
            num_local_macrophages = len(local_macrophages)
            for macrophage_index in local_macrophages:
                macrophage_cell = macrophage.cells[macrophage_index]
                if macrophage_cell['dead']:
                    continue
                macrophage_cell['iron_pool'] += (
                    4  # number of iron atoms in hemoglobin
                    * erythrocyte.hemoglobin_quantity
                    * erythrocytes_to_hemorrhage[z, y, x]
                    / num_local_macrophages
                )
        erythrocyte.cells['count'] -= erythrocytes_to_hemorrhage

        # interact with fungus
        for fungal_cell_index in afumigatus.cells.alive():
            fungal_cell = afumigatus.cells[fungal_cell_index]
            if fungal_cell['status'] == AfumigatusCellStatus.HYPHAE:
                fungal_voxel: Voxel = grid.get_voxel(fungal_cell['point'])
                erythrocyte.cells['hemorrhage'][tuple(fungal_voxel)] = True

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        erythrocyte: ErythrocyteState = state.erythrocyte
        # voxel_volume = state.voxel_volume

        return {
            'count': int(np.sum(erythrocyte.cells['count'])),
        }

    def visualization_data(self, state: State):
        erythrocyte: ErythrocyteState = state.erythrocyte
        return 'molecule', erythrocyte.cells
