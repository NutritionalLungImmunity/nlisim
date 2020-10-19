import attr
import numpy as np

from nlisim.cell import CellData, CellList
from nlisim.coordinates import Point, Voxel
from nlisim.grid import RectangularGrid
from nlisim.module import ModuleModel, ModuleState
from nlisim.modules.fungus import FungusCellData, FungusCellList
from nlisim.modules.geometry import TissueTypes
from nlisim.random import rg
from nlisim.state import State

MAX_CONIDIA = 100


class MacrophageCellData(CellData):

    MACROPHAGE_FIELDS = [
        ('iteration', 'i4'),
        ('phagosome', (np.int32, (MAX_CONIDIA))),
    ]

    dtype = np.dtype(CellData.FIELDS + MACROPHAGE_FIELDS, align=True)  # type: ignore

    @classmethod
    def create_cell_tuple(
        cls,
        **kwargs,
    ) -> np.record:

        iteration = 0
        phagosome = np.empty(MAX_CONIDIA)
        phagosome.fill(-1)
        return CellData.create_cell_tuple(**kwargs) + (
            iteration,
            phagosome,
        )


@attr.s(kw_only=True, frozen=True, repr=False)
class MacrophageCellList(CellList):
    CellDataClass = MacrophageCellData

    def len_phagosome(self, index):
        cell = self[index]
        return len(np.argwhere(cell['phagosome'] != -1))

    def append_to_phagosome(self, index, pathogen_index, max_size):
        cell = self[index]
        index_to_append = MacrophageCellList.len_phagosome(self, index)
        if (
            index_to_append < MAX_CONIDIA
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
            size = MacrophageCellList.len_phagosome(self, index)
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
            index = self[index]['phagosome'][i]
            fungus[index]['internalized'] = False
        self[index]['phagosome'].fill(-1)

    def recruit_new(self, rec_rate_ph, rec_r, p_rec_r, tissue, grid, cyto):
        num_reps = rec_rate_ph  # number of macrophages recruited per time step

        blood_index = np.argwhere(tissue == TissueTypes.BLOOD.value)
        blood_index = np.transpose(blood_index)
        mask = cyto[blood_index[2], blood_index[1], blood_index[0]] >= rec_r
        blood_index = np.transpose(blood_index)
        cyto_index = blood_index[mask]
        rg.shuffle(cyto_index)

        for _ in range(0, num_reps):
            if len(cyto_index) > 0:
                ii = rg.integers(len(cyto_index))
                point = Point(
                    x=grid.x[cyto_index[ii][2]],
                    y=grid.y[cyto_index[ii][1]],
                    z=grid.z[cyto_index[ii][0]],
                )

                if p_rec_r > rg.random():
                    self.append(MacrophageCellData.create_cell(point=point))

    def absorb_cytokines(self, m_abs, cyto, grid):
        for index in self.alive():
            vox = grid.get_voxel(self[index]['point'])
            x = vox.x
            y = vox.y
            z = vox.z
            cyto[z, y, x] = (1 - m_abs) * cyto[z, y, x]

    def produce_cytokines(self, m_det, m_n, grid, fungus: FungusCellList, cyto):
        for i in self.alive():
            vox = grid.get_voxel(self[i]['point'])

            hyphae_count = 0

            x_r = []
            y_r = []
            z_r = []

            if m_det == 0:
                index_arr = fungus.get_cells_in_voxel(vox)
                for index in index_arr:
                    if fungus[index]['form'] == FungusCellData.Form.HYPHAE:
                        hyphae_count += 1

            else:
                for num in range(0, m_det + 1):
                    x_r.append(num)
                    y_r.append(num)
                    z_r.append(num)

                for num in range(-1 * m_det, 0):
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

            cyto[vox.z, vox.y, vox.x] = cyto[vox.z, vox.y, vox.x] + m_n * hyphae_count

    def move(self, rec_r, grid, cyto, tissue, fungus: FungusCellList):
        for cell_index in self.alive():
            cell = self[cell_index]
            vox = grid.get_voxel(cell['point'])

            p = np.zeros(shape=27)
            vox_list = []
            i = -1

            for x in [0, 1, -1]:
                for y in [0, 1, -1]:
                    for z in [0, 1, -1]:
                        zk = vox.z + z
                        yj = vox.y + y
                        xi = vox.x + x
                        if (
                            grid.is_valid_voxel(Voxel(x=xi, y=yj, z=zk))
                            and tissue[zk, yj, xi] != TissueTypes.AIR.value
                        ):
                            vox_list.append([x, y, z])
                            i += 1
                            if cyto[zk, yj, xi] >= rec_r:
                                p[i] = cyto[zk, yj, xi]

            indices = np.argwhere(p != 0)
            num_vox_possible = len(indices)
            if num_vox_possible == 1:
                i = indices[0][0]
            elif num_vox_possible > 1:
                inds = np.argwhere(p == p[np.argmax(p)])
                rg.shuffle(inds)
                i = inds[0][0]
            else:
                i = rg.integers(len(vox_list))

            point = Point(
                x=grid.x[vox.x + vox_list[i][0]],
                y=grid.y[vox.y + vox_list[i][1]],
                z=grid.z[vox.z + vox_list[i][2]],
            )

            cell['point'] = point
            self.update_voxel_index([cell_index])

            for i in range(0, self.len_phagosome(cell_index)):
                f_index = cell['phagosome'][i]
                fungus[f_index]['point'] = point
                fungus.update_voxel_index([f_index])

    def internalize_conidia(self, m_det, max_spores, p_in, grid, fungus: FungusCellList):
        for i in self.alive():
            cell = self[i]
            vox = grid.get_voxel(cell['point'])

            x_r = []
            y_r = []
            z_r = []

            if m_det == 0:
                index_arr = fungus.get_cells_in_voxel(vox)
                for index in index_arr:
                    if (
                        fungus[index]['form'] == FungusCellData.Form.CONIDIA
                        and not fungus[index]['internalized']
                        and p_in > rg.random()
                    ):
                        fungus[index]['internalized'] = True
                        self.append_to_phagosome(i, index, max_spores)
            else:
                for num in range(0, m_det + 1):
                    x_r.append(num)
                    y_r.append(num)
                    z_r.append(num)

                for num in range(-1 * m_det, 0):
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
                                    if (
                                        fungus[index]['form'] == FungusCellData.Form.CONIDIA
                                        and not fungus[index]['internalized']
                                        and p_in > rg.random()
                                    ):
                                        fungus[index]['internalized'] = True
                                        self.append_to_phagosome(i, index, max_spores)

    def damage_conidia(self, kill, t, health, fungus):
        for i in self.alive():
            cell = self[i]
            for ii in range(0, self.len_phagosome(i)):
                index = cell['phagosome'][ii]
                fungus[index]['health'] = fungus[index]['health'] - (health * (t / kill))
                if fungus[index]['dead']:
                    self.remove_from_phagosome(i, index)

    def remove_if_sporeless(self, val):
        living = self.alive()
        living_len = len(living)
        num = int(val * living_len)
        if num == 0 and living_len > 0:
            num = 1
        for _ in range(num):
            r = rg.integers(living_len)
            self.cell_data[living[r]]['dead'] = True


def cell_list_factory(self: 'MacrophageState'):
    return MacrophageCellList(grid=self.global_state.grid)


@attr.s(kw_only=True)
class MacrophageState(ModuleState):
    cells: MacrophageCellList = attr.ib(default=attr.Factory(cell_list_factory, takes_self=True))
    rec_r: float
    p_rec_r: float
    m_abs: float
    m_n: float
    kill: float
    m_det: int
    rec_rate_ph: int
    time_m: float
    max_conidia_in_phag: int
    p_internalization: float
    rm: float


class Macrophage(ModuleModel):
    name = 'macrophage'
    StateClass = MacrophageState

    def initialize(self, state: State):
        macrophage: MacrophageState = state.macrophage
        grid: RectangularGrid = state.grid

        macrophage.rec_r = self.config.getfloat('rec_r')
        macrophage.p_rec_r = self.config.getfloat('p_rec_r')
        macrophage.m_abs = self.config.getfloat('m_abs')
        macrophage.m_n = self.config.getfloat('Mn')
        macrophage.kill = self.config.getfloat('kill')
        macrophage.m_det = self.config.getint('m_det')  # radius
        macrophage.rec_rate_ph = self.config.getint('rec_rate_ph')
        macrophage.time_m = self.config.getfloat('time_step')
        macrophage.max_conidia_in_phag = self.config.getint('max_conidia_in_phag')
        macrophage.rm = self.config.getfloat('rm')
        macrophage.p_internalization = self.config.getfloat('p_internalization')
        macrophage.cells = MacrophageCellList(grid=grid)

        return state

    def advance(self, state: State, previous_time: float):
        macrophage: MacrophageState = state.macrophage
        m_cells: MacrophageCellList = macrophage.cells
        tissue = state.geometry.lung_tissue
        grid = state.grid
        cyto = state.molecules.grid['m_cyto']
        n_cyto = state.molecules.grid['n_cyto']
        fungus: FungusCellList = state.fungus.cells
        health = state.fungus.health

        # recruit new
        m_cells.recruit_new(
            macrophage.rec_rate_ph, macrophage.rec_r, macrophage.p_rec_r, tissue, grid, cyto
        )

        # absorb cytokines
        m_cells.absorb_cytokines(macrophage.m_abs, cyto, grid)

        # produce cytokines
        m_cells.produce_cytokines(macrophage.m_det, macrophage.m_n, grid, fungus, n_cyto)

        # move
        m_cells.move(macrophage.rec_r, grid, cyto, tissue, fungus)

        # internalize
        m_cells.internalize_conidia(
            macrophage.m_det,
            macrophage.max_conidia_in_phag,
            macrophage.p_internalization,
            grid,
            fungus,
        )

        # damage conidia
        m_cells.damage_conidia(macrophage.kill, macrophage.time_m, health, fungus)

        if len(fungus.alive(fungus.cell_data['form'] == FungusCellData.Form.CONIDIA)) == 0:
            m_cells.remove_if_sporeless(macrophage.rm)

        return state
