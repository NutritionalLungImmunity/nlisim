from enum import IntEnum

import attr
import numpy as np

from simulation.cell import CellData, CellList
from simulation.coordinates import Point, Voxel
from simulation.grid import RectangularGrid
from simulation.module import Module, ModuleState
from simulation.modules.fungus import FungusCellData
from simulation.modules.geometry import TissueTypes
from simulation.state import State


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
        cls, *, iron_pool: float = 0, status: Status = Status.RESTING, **kwargs,
    ) -> np.record:

        iteration = 0
        phagosome = np.empty(MAX_PHAGOSOME_LENGTH)
        phagosome.fill(-1)
        return CellData.create_cell_tuple(**kwargs) + (status, iron_pool, iteration, phagosome,)


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

    def clear_all_phagosome(self, index):
        self[index]['phagosome'].fill(-1)

    def internalize(self, max_conidia, spores, grid):
        # for every index where tissue type = 3 get the spores.
        # If the spores are internalized == True then add to epithelium at that voxel
        for epi_index in self.alive():
            vox = grid.get_voxel(self[epi_index]['point'])
            spore_indices = spores.get_cells_in_voxel(vox)

            for index in spore_indices:
                if spores[index]['internalized']:
                    val = self.append_to_phagosome(epi_index, index, max_conidia)
                    if val:
                        spores[index]['mobile'] = False
                    else:
                        spores[index]['internalized'] = False

    def remove_dead_fungus(self, spores, grid):
        for epi_index in self.alive():
            vox = grid.get_voxel(self[epi_index]['point'])
            spore_indices = spores.get_cells_in_voxel(vox)

            for index in spore_indices:
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
                    if(
                        fungus[index]['form'] == FungusCellData.Form.CONIDIA and
                        fungus[index]['status'] 
                        in [FungusCellData.Status.SWOLLEN, FungusCellData.Status.GERMINATED]
                    ):
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
                                    if(
                                        fungus[index]['form'] == FungusCellData.Form.CONIDIA and
                                        fungus[index]['status'] in [FungusCellData.Status.SWOLLEN, FungusCellData.Status.GERMINATED]
                                    ):
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
    max_conidia_in_phag: int


class Epithelium(Module):
    name = 'epithelium'

    defaults = {
        'init_health': '100',
        'e_kill': '10',
        'cyto_rate': '5',
        's_det': '1',
        'h_det': '1',
        'max_conidia_in_phag': '50',
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
        epithelium.max_conidia_in_phag = self.config.getint('max_conidia_in_phag')
        epithelium.cells = EpitheliumCellList(grid=grid)

        indices = np.argwhere(tissue == TissueTypes.EPITHELIUM.value)

        for i in range(0, len(indices)):
            x = grid.x[indices[i][2]]
            y = grid.y[indices[i][1]]
            z = grid.z[indices[i][0]]

            point = Point(x=x, y=y, z=z)

            epithelium.cells.append(EpitheliumCellData.create_cell(point=point,))

        return state

    def advance(self, state: State, previous_time: float):
        epi: EpitheliumState = state.epithelium
        cells = epi.cells

        grid: RectangularGrid = state.grid
        tissue = state.geometry.lung_tissue

        spores = state.fungus.cells

        m_cyto = state.molecules.grid['m_cyto']
        n_cyto = state.molecules.grid['n_cyto']


        # internalize spores. The logic for internalization flag is in fungus
        # add internalized == true spores to phagosome
        cells.internalize(epi.max_conidia_in_phag, spores, grid)
        
        # remove killed spores from phagosome
        cells.remove_dead_fungus(spores, grid)

        # produce cytokines
        cells.cytokine_update(epi.s_det, epi.h_det, epi.cyto_rate, m_cyto, n_cyto, spores, grid)
        
        damage(state, previous_time)
        die_by_germination(state)

        return state


def damage(state, time_step):
    epi: EpitheliumState = state.epithelium
    cells = epi.cells
    spores = state.fungus.cells

    for index in cells.alive():
        e_cell = cells[index]
        for spore_index in e_cell['phagosome']:
            spore = spores[spore_index]
            spore['health'] = spore['health'] - (epi.init_health * time_step / epi.e_kill)


def die_by_germination(state):
    epithelium: EpitheliumState = state.epithelium
    cells = epithelium.cells
    spores = state.fungus.cells

    for index in cells.alive():
        e_cell = cells[index]
        if cells.len_phagosome(index) > 0:
            for spore_index in e_cell['phagosome']:
                if spores[spore_index]['status'] == FungusCellData.Status.GERMINATED:
                    e_cell['dead'] = True
                    cells.clear_all_phagosome(index)
                    for spore_index_i in e_cell['phagosome']:
                        spores[spore_index_i]['internalized'] = False
                    break
