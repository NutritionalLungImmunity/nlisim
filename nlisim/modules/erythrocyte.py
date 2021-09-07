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
    max_erythrocyte_voxel: int
    hemoglobin_concentration: float
    pr_ma_phag_eryt: float


class ErythrocyteModel(ModuleModel):
    name = 'erythrocyte'
    StateClass = ErythrocyteState

    def initialize(self, state: State):
        erythrocyte: ErythrocyteState = state.erythrocyte
        voxel_volume = state.voxel_volume
        lung_tissue = state.lung_tissue
        time_step_size: float = self.time_step

        erythrocyte.kd_hemo = self.config.getfloat('kd_hemo')
        erythrocyte.max_erythrocyte_voxel = self.config.getint('max_erythrocyte_voxel')
        erythrocyte.hemoglobin_concentration = self.config.getfloat('hemoglobin_concentration')

        # initialize cells
        # TODO: discuss
        erythrocyte.cells[lung_tissue == TissueType.BLOOD] = self.config.getfloat(
            'init_erythrocyte_level'
        )
        rel_n_hyphae_int_unit_t = time_step_size / 60  # per hour # TODO: not like this
        erythrocyte.pr_ma_phag_eryt = 1 - math.exp(
            -rel_n_hyphae_int_unit_t / (voxel_volume * self.config.getfloat('pr_ma_phag_eryt'))
        )  # TODO: -expm1?

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
        # TODO: avg? variable name improvement?
        avg = (1 - molecules.turnover_rate) * (
            1 - erythrocyte.cells['count'] / erythrocyte.max_erythrocyte_voxel
        )
        mask = avg > 0
        erythrocyte.cells['count'][mask] += np.random.poisson(avg[mask], avg[mask].shape)

        # ---------- interactions

        # uptake hemoglobin
        erythrocyte.cells['hemoglobin'] += hemoglobin.grid
        hemoglobin.grid.fill(0.0)

        # interact with hemolysin. pop goes the blood cell
        # TODO: avg? variable name improvement?
        avg = erythrocyte.cells['count'] * activation_function(
            x=hemolysin.grid,
            k_d=erythrocyte.kd_hemo,
            h=self.time_step / 60,  # units: (min/step) / (min/hour)
            volume=voxel_volume,
            b=1,
        )
        num = np.minimum(np.random.poisson(avg, shape), erythrocyte.cells['count'])
        erythrocyte.cells['hemoglobin'] += num * erythrocyte.hemoglobin_concentration
        erythrocyte.cells['count'] -= num

        # interact with Macrophage
        erythrocytes_to_hemorrhage = erythrocyte.cells['hemorrhage'] * np.random.poisson(
            erythrocyte.pr_ma_phag_eryt * erythrocyte.cells['count'], shape
        )
        # TODO: python for loop, possible performance issue
        zs, ys, xs = np.where(erythrocytes_to_hemorrhage > 0)
        for z, y, x in zip(zs, ys, xs):
            # TODO: make sure that these macrophages are alive!
            local_macrophages = macrophage.cells.get_cells_in_voxel(Voxel(x=x, y=y, z=z))
            num_local_macrophages = len(local_macrophages)
            for macrophage_index in local_macrophages:
                macrophage_cell = macrophage.cells[macrophage_index]
                # TODO: what's the 4 all about?
                macrophage_cell['iron_pool'] += (
                    4
                    * erythrocyte.hemoglobin_concentration
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
