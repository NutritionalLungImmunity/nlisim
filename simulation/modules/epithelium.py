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

def cell_list_factory(self: 'EpitheliumState'):
    return EpitheliumCellList(grid=self.global_state.grid)


@attr.s(kw_only=True)
class EpitheliumState(ModuleState):
    cells: EpitheliumCellList = attr.ib(default=attr.Factory(cell_list_factory, takes_self=True))
    init_health: float
    e_kill: float
    cyto_rate: float
    s_det: float
    h_det: float
    max_conidia_in_phag: int


class Epithelium(Module):
    name = 'epithelium'

    defaults = {
        'init_health': '100',
        'e_kill': '10',
        'cyto_rate': '1.5',
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
        epithelium.s_det = self.config.getfloat('s_det')
        epithelium.h_det = self.config.getfloat('h_det')
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

        # internalize spores. The logic for internalization flag is in fungus
        # add internalized == true spores to phagosome
        cells.internalize(epi.max_conidia, spores, grid)
        
        remove_dead_fungus(state)
        cytokine_update(state)
        damage(state, previous_time)
        die_by_germination(state)

        return state


def remove_dead_fungus(state):
    epithelium: EpitheliumState = state.epithelium
    tissue = state.geometry.lung_tissue
    cells = epithelium.cells
    spores = state.fungus.cells

    for vox_index in np.argwhere(tissue == TissueTypes.EPITHELIUM.value):
        vox = Voxel(x=vox_index[2], y=vox_index[1], z=vox_index[0])

        epi_index = cells.get_cells_in_voxel(vox)[0]
        spore_index = spores.get_cells_in_voxel(vox)

        for index in spore_index:
            if spores[index]['dead']:
                cells.remove_from_phagosome(epi_index, index)


def cytokine_update(state):
    epithelium: EpitheliumState = state.epithelium
    tissue = state.geometry.lung_tissue
    cells = epithelium.cells
    fungus = state.fungus.cells
    m_cyto = state.molecules.grid['m_cyto']
    n_cyto = state.molecules.grid['n_cyto']

    for vox_index in np.argwhere(tissue == TissueTypes.EPITHELIUM.value):
        vox = Voxel(x=vox_index[2], y=vox_index[1], z=vox_index[0])

        spore_count = 0
        hyphae_count = 0

        # TODO get neighboring voxels within detection distance
        # eg. for vox in (vox.x + det_d, vox.y, vox.z, )
        epi_index = cells.get_cells_in_voxel(vox)[0]
        fungus_index = fungus.get_cells_in_voxel(vox)

        if not cells.cell_data[epi_index]['dead']:
            for index in fungus_index:
                f_cell = fungus[index]
                if f_cell['form'] == FungusCellData.Form.CONIDIA and (
                    f_cell['status']
                    in [FungusCellData.Status.SWOLLEN, FungusCellData.Status.GERMINATED]
                ):
                    spore_count += 1
                elif f_cell['form'] == FungusCellData.Form.HYPHAE:
                    hyphae_count += 1

            m_cyto[vox.z, vox.y, vox.x] = (
                m_cyto[vox.z, vox.y, vox.x] + epithelium.cyto_rate * spore_count
            )
            n_cyto[vox.z, vox.y, vox.x] = n_cyto[vox.z, vox.y, vox.x] + epithelium.cyto_rate * (
                spore_count + hyphae_count
            )


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
