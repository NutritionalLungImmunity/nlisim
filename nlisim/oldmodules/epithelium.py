from enum import IntEnum
import itertools
from random import shuffle
from typing import Any, Dict, Tuple

import attr
import numpy as np

from nlisim.cell import CellData, CellList
from nlisim.coordinates import Point, Voxel
from nlisim.grid import RectangularGrid
from nlisim.module import ModuleModel, ModuleState
from nlisim.oldmodules.fungus import FungusCellData, FungusCellList
from nlisim.random import rg
from nlisim.state import State
from nlisim.util import TissueType

MAX_PHAGOSOME_LENGTH = 100


class EpitheliumCellData(CellData):
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

    PHAGOCYTE_FIELDS = [
        ('status', 'u1'),
        ('iron_pool', 'f8'),
        ('iteration', 'i4'),
        ('phagosome', (np.int32, MAX_CONIDIA)),
    ]

    dtype = np.dtype(CellData.FIELDS + PHAGOCYTE_FIELDS, align=True)  # type: ignore

    @classmethod
    def create_cell_tuple(
        cls,
        *,
        iron_pool: float = 0,
        status: Status = Status.RESTING,
        **kwargs,
    ) -> Tuple:
        iteration = 0
        phagosome = np.empty(MAX_PHAGOSOME_LENGTH)
        phagosome.fill(-1)
        return CellData.create_cell_tuple(**kwargs) + (
            status,
            iron_pool,
            iteration,
            phagosome,
        )


@attr.s(kw_only=True, frozen=True, repr=False)
class EpitheliumCellList(CellList):
    CellDataClass = EpitheliumCellData

    def len_phagosome(self, index):
        cell = self[index]
        return len(np.argwhere(cell['phagosome'] != -1))

    def append_to_phagosome(self, index, pathogen_index, max_size):
        cell = self[index]
        index_to_append = EpitheliumCellList.len_phagosome(self, index)
        if (
            index_to_append < MAX_PHAGOSOME_LENGTH
            and index_to_append < max_size
            and pathogen_index not in cell['phagosome']
        ):
            cell['phagosome'][index_to_append] = pathogen_index
            return True
        else:
            return False

    def remove_from_phagosome(self, index, pathogen_index):
        phagosome = self[index]['phagosome']
        if pathogen_index in phagosome:
            itemindex = np.argwhere(phagosome == pathogen_index)[0][0]
            size = EpitheliumCellList.len_phagosome(self, index)
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

    def clear_all_phagosome(self, index, fungus: FungusCellList):
        for i in range(0, self.len_phagosome(index)):
            f_index = self[index]['phagosome'][i]
            fungus[f_index]['internalized'] = False
        self[index]['phagosome'].fill(-1)

    def internalize_conidia(self, e_det, max_spores, p_in, grid, spores: FungusCellList):
        for i in self.alive():
            cell = self[i]
            vox = grid.get_voxel(cell['point'])

            # Moore neighborhood, but order partially randomized. Closest to furthest order, but
            # the order of any set of points of equal distance is random
            neighborhood = list(itertools.product(tuple(range(-1 * e_det, e_det + 1)), repeat=3))
            shuffle(neighborhood)
            neighborhood = sorted(neighborhood, key=lambda v: v[0] ** 2 + v[1] ** 2 + v[2] ** 2)

            for dx, dy, dz in neighborhood:
                zi = vox.z + dz
                yj = vox.y + dy
                xk = vox.x + dx
                if grid.is_valid_voxel(Voxel(x=xk, y=yj, z=zi)):
                    index_arr = spores.get_cells_in_voxel(Voxel(x=xk, y=yj, z=zi))
                    for index in index_arr:
                        if (
                            spores[index]['form'] == FungusCellData.Form.CONIDIA
                            and not spores[index]['internalized']
                            and p_in > rg.random()
                        ):
                            spores[index]['internalized'] = True
                            if self.append_to_phagosome(i, index, max_spores):
                                spores[index]['mobile'] = False
                            else:
                                spores[index]['internalized'] = False

    def remove_dead_fungus(self, spores):
        for epi_index in self.alive():
            for ii in range(0, self.len_phagosome(epi_index)):
                index = self[epi_index]['phagosome'][ii]
                if spores[index]['dead']:
                    self.remove_from_phagosome(epi_index, index)

    def cytokine_update(self, s_det, h_det, cyto_rate, m_cyto, n_cyto, fungus, grid):
        for i in self.alive():
            vox = grid.get_voxel(self[i]['point'])

            # spores
            spore_count = 0

            # Moore neighborhood
            neighborhood = tuple(itertools.product(tuple(range(-1 * s_det, s_det + 1)), repeat=3))

            for dx, dy, dz in neighborhood:
                zi = vox.z + dz
                yj = vox.y + dy
                xk = vox.x + dx
                if grid.is_valid_voxel(Voxel(x=xk, y=yj, z=zi)):
                    index_arr = fungus.get_cells_in_voxel(Voxel(x=xk, y=yj, z=zi))
                    for index in index_arr:
                        if fungus[index]['form'] == FungusCellData.Form.CONIDIA and fungus[index][
                            'status'
                        ] in [
                            FungusCellData.Status.SWOLLEN,
                            FungusCellData.Status.GERMINATED,
                        ]:
                            spore_count += 1

            # hyphae_count
            hyphae_count = 0

            # Moore neighborhood
            neighborhood = tuple(itertools.product(tuple(range(-1 * h_det, h_det + 1)), repeat=3))

            for dx, dy, dz in neighborhood:
                zi = vox.z + dz
                yj = vox.y + dy
                xk = vox.x + dx
                if grid.is_valid_voxel(Voxel(x=xk, y=yj, z=zi)):
                    index_arr = fungus.get_cells_in_voxel(Voxel(x=xk, y=yj, z=zi))
                    for index in index_arr:
                        if fungus[index]['form'] == FungusCellData.Form.HYPHAE:
                            hyphae_count += 1

            m_cyto[vox.z, vox.y, vox.x] += cyto_rate * spore_count
            n_cyto[vox.z, vox.y, vox.x] += cyto_rate * (spore_count + hyphae_count)

    def damage(self, kill, t, health, fungus):
        for i in self.alive():
            cell = self[i]
            for ii in range(0, self.len_phagosome(i)):
                index = cell['phagosome'][ii]
                fungus[index]['health'] = fungus[index]['health'] - (health * (t / kill))

    def die_by_germination(self, spores):
        for index in self.alive():
            e_cell = self[index]
            for ii in range(0, self.len_phagosome(index)):
                spore_index = e_cell['phagosome'][ii]
                if spores[spore_index]['status'] == FungusCellData.Status.GERMINATED:
                    e_cell['dead'] = True
                    self.clear_all_phagosome(index, spores)
                    break


def cell_list_factory(self: 'EpitheliumState'):
    return EpitheliumCellList(grid=self.global_state.grid)


@attr.s(kw_only=True)
class EpitheliumState(ModuleState):
    cells: EpitheliumCellList = attr.ib(default=attr.Factory(cell_list_factory, takes_self=True))
    init_health: float
    e_kill: float
    cyto_rate: float
    s_det: int
    h_det: int
    time_e: float
    max_conidia_in_phag: int
    p_internalization: float


class Epithelium(ModuleModel):
    name = 'epithelium'

    StateClass = EpitheliumState

    def initialize(self, state: State):
        epithelium: EpitheliumState = state.epithelium
        grid: RectangularGrid = state.grid
        tissue = state.geometry.lung_tissue

        epithelium.init_health = self.config.getfloat('init_health')
        epithelium.e_kill = self.config.getfloat('e_kill')
        epithelium.cyto_rate = self.config.getfloat('cyto_rate')
        epithelium.s_det = self.config.getint('s_det')
        epithelium.h_det = self.config.getint('h_det')
        epithelium.time_e = self.config.getfloat('time_e')
        epithelium.max_conidia_in_phag = self.config.getint('max_conidia_in_phag')
        epithelium.cells = EpitheliumCellList(grid=grid)
        epithelium.p_internalization = self.config.getfloat('p_internalization')

        indices = np.argwhere(tissue == TissueType.EPITHELIUM.value)

        for i in range(0, len(indices)):
            x = grid.x[indices[i][2]]
            y = grid.y[indices[i][1]]
            z = grid.z[indices[i][0]]

            point = Point(x=x, y=y, z=z)

            epithelium.cells.append(
                EpitheliumCellData.create_cell(
                    point=point,
                )
            )

        return state

    def advance(self, state: State, previous_time: float):
        epi: EpitheliumState = state.epithelium
        cells = epi.cells

        grid: RectangularGrid = state.grid

        spores = state.fungus.cells
        health = state.fungus.health

        m_cyto = state.molecules.grid['m_cyto']
        n_cyto = state.molecules.grid['n_cyto']

        # internalize
        if len(spores.alive(spores.cell_data['form'] == FungusCellData.Form.CONIDIA)) > 0:
            cells.internalize_conidia(
                epi.s_det, epi.max_conidia_in_phag, epi.p_internalization, grid, spores
            )

        # remove killed spores from phagosome
        cells.remove_dead_fungus(spores)

        # produce cytokines
        cells.cytokine_update(epi.s_det, epi.h_det, epi.cyto_rate, m_cyto, n_cyto, spores, grid)

        # damage internalized spores
        cells.damage(epi.e_kill, epi.time_e, health, spores)

        # kill epithelium with germinated spore in its phagosome
        cells.die_by_germination(spores)

        return state

    def summary_stats(self, state: State) -> Dict[str, Any]:
        epi: EpitheliumState = state.epithelium

        return {
            'count': len(epi.cells.alive()),
        }

    def visualization_data(self, state: State) -> Tuple[str, Any]:
        return 'cells', state.epithelium.cells
