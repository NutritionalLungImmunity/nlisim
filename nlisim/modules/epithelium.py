from enum import IntEnum

import attr
import numpy as np

from nlisim.cell import CellData, CellList
from nlisim.coordinates import Point, Voxel
from nlisim.grid import RectangularGrid
from nlisim.module import Module, ModuleState
from nlisim.modules.fungus import FungusCellData, FungusCellList
from nlisim.modules.geometry import TissueTypes
from nlisim.random import rg
from nlisim.state import State


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
        ('phagosome', (np.int32, (MAX_CONIDIA))),
    ]

    dtype = np.dtype(CellData.FIELDS + PHAGOCYTE_FIELDS, align=True)  # type: ignore

    @classmethod
    def create_cell_tuple(
        cls,
        *,
        iron_pool: float = 0,
        status: Status = Status.RESTING,
        **kwargs,
    ) -> np.record:

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

            x_r = []
            y_r = []
            z_r = []

            if e_det == 0:
                index_arr = spores.get_cells_in_voxel(vox)
                for index in index_arr:
                    if (
                        spores[index]['form'] == FungusCellData.Form.CONIDIA
                        and not spores[index]['internalized']
                        and p_in > rg.random()
                    ):
                        spores[index]['internalized'] = True
                        val = self.append_to_phagosome(i, index, max_spores)
                        if val:
                            spores[index]['mobile'] = False
                        else:
                            spores[index]['internalized'] = False
            else:
                for num in range(0, e_det + 1):
                    x_r.append(num)
                    y_r.append(num)
                    z_r.append(num)

                for num in range(-1 * e_det, 0):
                    x_r.append(num)
                    y_r.append(num)
                    z_r.append(num)

                for x in x_r:
                    for y in y_r:
                        for z in z_r:
                            zk = vox.z + z
                            yj = vox.y + y
                            xi = vox.x + x
                            if grid.is_valid_voxel(Voxel(x=xi, y=yj, z=zk)):
                                index_arr = spores.get_cells_in_voxel(Voxel(x=xi, y=yj, z=zk))
                                for index in index_arr:
                                    if (
                                        spores[index]['form'] == FungusCellData.Form.CONIDIA
                                        and not spores[index]['internalized']
                                        and p_in > rg.random()
                                    ):
                                        spores[index]['internalized'] = True
                                        val = self.append_to_phagosome(i, index, max_spores)
                                        if val:
                                            spores[index]['mobile'] = False
                                        else:
                                            spores[index]['internalized'] = False

    def remove_dead_fungus(self, spores, grid):
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

            x_r = []
            y_r = []
            z_r = []

            if s_det == 0:
                index_arr = fungus.get_cells_in_voxel(vox)
                for index in index_arr:
                    if fungus[index]['form'] == FungusCellData.Form.CONIDIA and fungus[index][
                        'status'
                    ] in [FungusCellData.Status.SWOLLEN, FungusCellData.Status.GERMINATED]:
                        spore_count += 1

            else:
                for num in range(0, s_det + 1):
                    x_r.append(num)
                    y_r.append(num)
                    z_r.append(num)

                for num in range(-1 * s_det, 0):
                    x_r.append(num)
                    y_r.append(num)
                    z_r.append(num)

                for x in x_r:
                    for y in y_r:
                        for z in z_r:
                            zk = vox.z + z
                            yj = vox.y + y
                            xi = vox.x + x
                            if grid.is_valid_voxel(Voxel(x=xi, y=yj, z=zk)):
                                index_arr = fungus.get_cells_in_voxel(Voxel(x=xi, y=yj, z=zk))
                                for index in index_arr:
                                    if fungus[index][
                                        'form'
                                    ] == FungusCellData.Form.CONIDIA and fungus[index][
                                        'status'
                                    ] in [
                                        FungusCellData.Status.SWOLLEN,
                                        FungusCellData.Status.GERMINATED,
                                    ]:
                                        spore_count += 1

            # hyphae_count
            hyphae_count = 0

            x_r = []
            y_r = []
            z_r = []

            if h_det == 0:
                index_arr = fungus.get_cells_in_voxel(vox)
                for index in index_arr:
                    if fungus[index]['form'] == FungusCellData.Form.HYPHAE:
                        hyphae_count += 1

            else:
                for num in range(0, h_det + 1):
                    x_r.append(num)
                    y_r.append(num)
                    z_r.append(num)

                for num in range(-1 * h_det, 0):
                    x_r.append(num)
                    y_r.append(num)
                    z_r.append(num)

                for x in x_r:
                    for y in y_r:
                        for z in z_r:
                            zk = vox.z + z
                            yj = vox.y + y
                            xi = vox.x + x
                            if grid.is_valid_voxel(Voxel(x=xi, y=yj, z=zk)):
                                index_arr = fungus.get_cells_in_voxel(Voxel(x=xi, y=yj, z=zk))
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


class Epithelium(Module):
    name = 'epithelium'

    defaults = {
        'init_health': '100',
        'e_kill': '10',
        'cyto_rate': '5',
        's_det': '1',
        'h_det': '1',
        'time_e': '1',
        'max_conidia_in_phag': '50',
        'p_internalization': '0.3',
    }
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
        epithelium.time_e = self.config.getfloat('time_step')
        epithelium.max_conidia_in_phag = self.config.getint('max_conidia_in_phag')
        epithelium.cells = EpitheliumCellList(grid=grid)
        epithelium.p_internalization = self.config.getfloat('p_internalization')

        indices = np.argwhere(tissue == TissueTypes.EPITHELIUM.value)

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
        cells.remove_dead_fungus(spores, grid)

        # produce cytokines
        cells.cytokine_update(epi.s_det, epi.h_det, epi.cyto_rate, m_cyto, n_cyto, spores, grid)

        # damage internalized spores
        cells.damage(epi.e_kill, epi.time_e, health, spores)

        # kill epithelium with germinated spore in its phagosome
        cells.die_by_germination(spores)

        return state
