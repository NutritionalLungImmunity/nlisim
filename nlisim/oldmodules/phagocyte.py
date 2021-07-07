from enum import IntEnum
import math
import random
from typing import Tuple

import attr
import numpy as np

from nlisim.cell import CellData, CellList
from nlisim.coordinates import Point, Voxel
from nlisim.grid import RectangularGrid
from nlisim.random import rg
from nlisim.util import TissueType

MAX_PHAGOSOME_LENGTH = 100


class PhagocyteCellData(CellData):
    RECRUIT_RATE = 0.0
    LEAVE_RATE = 0.0
    CHEMOKINE_THRESHOLD = 0.0
    LEAVES_BOOL = True
    MAX_CONIDIA = MAX_PHAGOSOME_LENGTH

    class Status(IntEnum):
        INACTIVE = 0
        INACTIVATING = 1
        RESTING = 2
        ACTIVATING = 3
        ACTIVE = 4
        APOPTOTIC = 5
        NECROTIC = 6
        DEAD = 7

    class State(IntEnum):
        FREE = 0
        INTERACTING = 1

    PHAGOCYTE_FIELDS = [
        ('status', 'u1'),
        ('state', 'u1'),
        ('iron_pool', 'f8'),
        ('iteration', 'i4'),
        ('phagosome', (np.int32, (MAX_CONIDIA))),
    ]

    dtype = np.dtype(CellData.FIELDS + PHAGOCYTE_FIELDS, align=True)  # type: ignore

    @classmethod
    def create_cell_tuple(
        cls,
        *,
        iron_pool: float = 0,
        status: Status = Status.RESTING,
        state: State = State.FREE,
        **kwargs,
    ) -> Tuple:
        iteration = 0
        phagosome = np.empty(MAX_PHAGOSOME_LENGTH)
        phagosome.fill(-1)
        return CellData.create_cell_tuple(**kwargs) + (
            status,
            state,
            iron_pool,
            iteration,
            phagosome,
        )


@attr.s(kw_only=True, frozen=True, repr=False)
class PhagocyteCellList(CellList):
    CellDataClass = PhagocyteCellData

    def is_moveable(self, grid: RectangularGrid):
        cells = self.cell_data
        return self.alive(
            (cells['status'] == PhagocyteCellData.Status.RESTING)
            & cells.point_mask(cells['point'], grid)
        )

    def len_phagosome(self, index):
        cell = self[index]
        return len(np.argwhere(cell['phagosome'] != -1))

    def append_to_phagosome(self, index, pathogen_index, max_size):
        cell = self[index]
        index_to_append = PhagocyteCellList.len_phagosome(self, index)
        if index_to_append < MAX_PHAGOSOME_LENGTH and index_to_append < max_size:
            cell['phagosome'][index_to_append] = pathogen_index
            return True
        else:
            return False

    def remove_from_phagosome(self, index, pathogen_index):
        phagosome = self[index]['phagosome']
        if pathogen_index in phagosome:
            itemindex = np.argwhere(phagosome == pathogen_index)[0][0]
            size = PhagocyteCellList.len_phagosome(self, index)
            if itemindex == size - 1:
                # full phagosome
                phagosome[itemindex] = -1
                return True
            else:
                phagosome[itemindex:-1] = phagosome[itemindex + 1 :]
                phagosome[-1] = -1
                return True
        else:
            return False

    def clear_all_phagosome(self, index):
        self[index]['phagosome'].fill(-1)

    def recruit(self, rate, molecule, grid: RectangularGrid):
        # TODO - add recruitment
        # indices = np.argwhere(molecule_to_recruit >= threshold_value)
        # then for each index create a cell with prob 'rec_rate'
        return

    def remove(self, rate, molecule, grid: RectangularGrid):
        # TODO - add leaving
        # indices = np.argwhere(molecule_to_leave <= threshold_value)
        # then for each index kill a cell with prob 'leave_rate'
        return

    # move
    def chemotaxis(
        self,
        molecule,
        drift_lambda,
        drift_bias,
        tissue,
        grid: RectangularGrid,
    ):
        # 'molecule' = state.'molecule'.concentration
        # prob = 0-1 random number to determine which voxel is chosen to move

        # 1. Get cells that are alive
        for index in self.alive():
            prob = rg.random()

            # 2. Get voxel for each cell to get molecule in that voxel
            cell = self[index]
            vox = grid.get_voxel(cell['point'])

            # 3. Set prob for neighboring voxels
            p = []
            vox_list = []
            p_tot = 0.0
            i = -1

            # calculate individual probability
            for x in [0, 1, -1]:
                for y in [0, 1, -1]:
                    for z in [0, 1, -1]:
                        p.append(0.0)
                        vox_list.append([x, y, z])
                        i += 1
                        zk = vox.z + z
                        yj = vox.y + y
                        xi = vox.x + x
                        if grid.is_valid_voxel(Voxel(x=xi, y=yj, z=zk)):
                            if tissue[zk, yj, xi] in [
                                TissueType.SURFACTANT.value,
                                TissueType.BLOOD.value,
                                TissueType.EPITHELIUM.value,
                                TissueType.PORE.value,
                            ]:
                                p[i] = logistic(molecule[zk, yj, xi], drift_lambda, drift_bias)
                                p_tot += p[i]

            # scale to sum of probabilities
            if p_tot:
                for i in range(len(p)):
                    p[i] = p[i] / p_tot

            # chose vox from neighbors
            cum_p = 0.0
            for i in range(len(p)):
                cum_p += p[i]
                if prob <= cum_p:
                    cell['point'] = Point(
                        x=random.uniform(
                            grid.xv[vox.x + vox_list[i][0]], grid.xv[vox.x + vox_list[i][0] + 1]
                        ),
                        y=random.uniform(
                            grid.yv[vox.y + vox_list[i][1]], grid.yv[vox.y + vox_list[i][1] + 1]
                        ),
                        z=random.uniform(
                            grid.zv[vox.z + vox_list[i][2]], grid.zv[vox.z + vox_list[i][2] + 1]
                        ),
                    )
                    self.update_voxel_index([index])
                    break


def logistic(x, lamb, bias):
    return 1 - bias * math.exp(-((x / lamb) ** 2))
