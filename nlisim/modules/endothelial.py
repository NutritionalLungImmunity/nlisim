import math
from typing import Any, Dict, Tuple

import attr
from attr import attrib, attrs
import numpy as np

from nlisim.cell import CellData, CellFields, CellList
from nlisim.coordinates import Point, Voxel
from nlisim.grid import RectangularGrid
from nlisim.modules.phagocyte import (
    PhagocyteCellData,
    PhagocyteModel,
    PhagocyteModuleState,
    PhagocyteStatus,
)
from nlisim.random import rg
from nlisim.state import State
from nlisim.util import TissueType, activation_function


class EndothelialCellData(PhagocyteCellData):
    ENDOTHELIAL_FIELDS: CellFields = [
        ('status', np.uint8),
        ('status_iteration', np.uint),
        ('tnfa', bool),
    ]

    dtype = np.dtype(
        CellData.FIELDS + PhagocyteCellData.PHAGOCYTE_FIELDS + ENDOTHELIAL_FIELDS, align=True
    )  # type: ignore

    @classmethod
    def create_cell_tuple(
        cls,
        **kwargs,
    ) -> Tuple:
        initializer = {
            'status': kwargs.get('status', PhagocyteStatus.RESTING),
            'status_iteration': kwargs.get('status_iteration', 0),
            'tnfa': kwargs.get('tnfa', False),
        }

        # ensure that these come in the correct order
        return PhagocyteCellData.create_cell_tuple(**kwargs) + tuple(
            [initializer[key] for key, *_ in EndothelialCellData.ENDOTHELIAL_FIELDS]
        )


@attrs(kw_only=True, frozen=True, repr=False)
class EndothelialCellList(CellList):
    CellDataClass = EndothelialCellData


def cell_list_factory(self: 'EndothelialState') -> EndothelialCellList:
    return EndothelialCellList(grid=self.global_state.grid)


@attrs(kw_only=True)
class EndothelialState(PhagocyteModuleState):
    cells: EndothelialCellList = attrib(default=attr.Factory(cell_list_factory, takes_self=True))
    max_conidia: int  # units: conidia
    time_to_rest: float  # units: hours
    iter_to_rest: int  # units: steps
    time_to_change_state: float  # units: hours
    iter_to_change_state: int  # units: steps
    # p_il6_qtty: float  # units: mol * cell^-1 * h^-1
    # p_il8_qtty: float # units: mol * cell^-1 * h^-1
    p_tnf_qtty: float  # units: atto-mol * cell^-1 * h^-1
    pr_p_int: float  # units: probability
    pr_p_int_param: float


class Endothelial(PhagocyteModel):
    name = 'endothelial'
    StateClass = EndothelialState

    def initialize(self, state: State):
        endothelial: EndothelialState = state.endothelial
        voxel_volume: float = state.voxel_volume
        time_step_size: float = self.time_step
        lung_tissue: np.ndarray = state.lung_tissue

        endothelial.max_conidia = self.config.getint('max_conidia')  # units: conidia
        endothelial.time_to_rest = self.config.getint('time_to_rest')  # units: hours
        endothelial.time_to_change_state = self.config.getint(
            'time_to_change_state')  # units: hours
        endothelial.p_tnf_qtty = self.config.getfloat(
            'p_tnf_qtty'
        )  # units: atto-mol * cell^-1 * h^-1
        endothelial.pr_p_int_param = self.config.getfloat('pr_p_int_param')

        # computed values
        endothelial.iter_to_rest = int(
            endothelial.time_to_rest * (60 / self.time_step)
        )  # units: hours * (min/hour) / (min/step) = step
        endothelial.iter_to_change_state = int(
            endothelial.time_to_change_state * (60 / self.time_step)
        )  # units: hours * (min/hour) / (min/step) = step
        endothelial.pr_p_int = -math.expm1(
            -time_step_size / 60 / (voxel_volume * endothelial.pr_p_int_param)
        )  # units: probability

        # initialize cells, placing one per epithelial voxel
        dz_field: np.ndarray = state.grid.delta(axis=0)
        dy_field: np.ndarray = state.grid.delta(axis=1)
        dx_field: np.ndarray = state.grid.delta(axis=2)
        epithelial_voxels = list(zip(*np.where(lung_tissue == TissueType.EPITHELIUM)))
        rg.shuffle(epithelial_voxels)
        for vox_z, vox_y, vox_x in epithelial_voxels[: self.config.getint('count')]:
            # the x,y,z coordinates are in the centers of the grids
            z = state.grid.z[vox_z]
            y = state.grid.y[vox_y]
            x = state.grid.x[vox_x]
            dz = dz_field[vox_z, vox_y, vox_x]
            dy = dy_field[vox_z, vox_y, vox_x]
            dx = dx_field[vox_z, vox_y, vox_x]
            endothelial.cells.append(
                EndothelialCellData.create_cell(
                    point=Point(
                        x=x + rg.uniform(-dx / 2, dx / 2),
                        y=y + rg.uniform(-dy / 2, dy / 2),
                        z=z + rg.uniform(-dz / 2, dz / 2),
                    )
                )
            )

        return state

    def single_step_probabilistic_drift(
        self, state: State, cell: PhagocyteCellData, voxel: Voxel
    ) -> Point:
        # endothelials do not move
        pass

    def advance(self, state: State, previous_time: float):
        """Advance the state by a single time step."""
        from nlisim.modules.afumigatus import (
            AfumigatusCellData,
            AfumigatusCellStatus,
            AfumigatusState,
        )

        # from nlisim.modules.il6 import IL6State
        # from nlisim.modules.il8 import IL8State
        from nlisim.modules.tnfa import TNFaState

        endothelial: EndothelialState = state.endothelial
        afumigatus: AfumigatusState = state.afumigatus
        # il6: IL6State = getattr(state, 'il6', None)
        # il8: IL8State = getattr(state, 'il8', None)
        tnfa: TNFaState = state.tnfa
        grid: RectangularGrid = state.grid
        voxel_volume: float = state.voxel_volume

        for endothelial_cell_index in endothelial.cells.alive():
            endothelial_cell = endothelial.cells[endothelial_cell_index]
            endothelial_cell_voxel: Voxel = grid.get_voxel(endothelial_cell['point'])

            # self update
            if endothelial_cell['status'] == PhagocyteStatus.ACTIVE:
                if endothelial_cell['status_iteration'] >= endothelial.iter_to_rest:
                    endothelial_cell['status_iteration'] = 0
                    endothelial_cell['status'] = PhagocyteStatus.RESTING
                    endothelial_cell['tnfa'] = False
                else:
                    endothelial_cell['status_iteration'] += 1

            elif endothelial_cell['status'] == PhagocyteStatus.ACTIVATING:
                if endothelial_cell['status_iteration'] >= endothelial.iter_to_change_state:
                    endothelial_cell['status_iteration'] = 0
                    endothelial_cell['status'] = PhagocyteStatus.ACTIVE
                else:
                    endothelial_cell['status_iteration'] += 1

            # ----------- interactions

            # Activating from TNFa
            if endothelial_cell['status'] not in {
                PhagocyteStatus.APOPTOTIC,
                PhagocyteStatus.NECROTIC,
                PhagocyteStatus.DEAD,
            }:

                if (
                    activation_function(
                        x=tnfa.grid[tuple(endothelial_cell_voxel)],
                        k_d=tnfa.k_d,
                        h=self.time_step / 60,  # units: (min/step) / (min/hour)
                        volume=voxel_volume,
                        b=1,
                    )
                    < rg.uniform()
                ):
                    if endothelial_cell['status'] == PhagocyteStatus.ACTIVATING:
                        endothelial_cell['status'] = PhagocyteStatus.ACTIVE
                        endothelial_cell['status_iteration'] = 0
                    else:
                        endothelial_cell['status'] = PhagocyteStatus.ACTIVATING
                        endothelial_cell['status_iteration'] = 0

            # # secrete IL6
            # if il6 is not None and endothelial_cell['status'] == PhagocyteStatus.ACTIVE:
            #     il6.grid[tuple(endothelial_cell_voxel)] += endothelial.p_il6_qtty
            #
            # # secrete IL8
            # if il8 is not None and endothelial_cell['tnfa']:
            #     il8.grid[tuple(endothelial_cell_voxel)] += endothelial.p_il8_qtty

            # secrete TNFa
            #    tnfa.grid[tuple(endothelial_cell_voxel)] += endothelial.p_tnf_qtty

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        endothelial: EndothelialState = state.endothelial
        live_endothelials = endothelial.cells.alive()

        max_index = max(map(int, PhagocyteStatus))
        status_counts = np.bincount(
            np.fromiter(
                (
                    endothelial.cells[endothelial_cell_index]['status']
                    for endothelial_cell_index in live_endothelials
                ),
                dtype=np.uint8,
            ),
            minlength=max_index + 1,
        )

        # tnfa_active = int(
        #    np.sum(
        #       np.fromiter(
        #          (
        #             endothelial.cells[endothelial_cell_index]['tnfa']
        #            for endothelial_cell_index in live_endothelials
        #       ),
        #      dtype=bool,
        #        )
        #    )
        #  )

        return {
            'count': len(endothelial.cells.alive()),
            'inactive': int(status_counts[PhagocyteStatus.INACTIVE]),
            'inactivating': int(status_counts[PhagocyteStatus.INACTIVATING]),
            'resting': int(status_counts[PhagocyteStatus.RESTING]),
            'activating': int(status_counts[PhagocyteStatus.ACTIVATING]),
            'active': int(status_counts[PhagocyteStatus.ACTIVE]),
            'apoptotic': int(status_counts[PhagocyteStatus.APOPTOTIC]),
            'necrotic': int(status_counts[PhagocyteStatus.NECROTIC]),
            'interacting': int(status_counts[PhagocyteStatus.INTERACTING]),
            #'TNFa active': tnfa_active,
        }

    def visualization_data(self, state: State):
        return 'cells', state.endothelial.cells
