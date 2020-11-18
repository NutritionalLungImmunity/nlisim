from enum import IntEnum

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


class NeutrophilCellData(CellData):
    class Status(IntEnum):
        NONGRANULATING = 0
        GRANULATING = 1

    NEUTROPHIL_FIELDS = [('status', 'u1'), ('iteration', 'i4'), ('granule_count', 'i4')]

    dtype = np.dtype(CellData.FIELDS + NEUTROPHIL_FIELDS, align=True)  # type: ignore

    @classmethod
    def create_cell_tuple(
        cls,
        *,
        status=Status.NONGRANULATING,
        granule_count=0,
        **kwargs,
    ) -> np.record:
        iteration = 0
        return CellData.create_cell_tuple(**kwargs) + (
            status,
            iteration,
            granule_count,
        )


@attr.s(kw_only=True, frozen=True, repr=False)
class NeutrophilCellList(CellList):
    CellDataClass = NeutrophilCellData

    def recruit_new(self, rec_rate_ph, rec_r, granule_count, neutropenic, time, grid, tissue, cyto):
        num_reps = 0
        if not neutropenic:
            num_reps = rec_rate_ph  # number of neutrophils recruited per time step
        elif neutropenic and time >= 48 and time <= 96:
            num_reps = int((time - 48) / 8) * 3

        if num_reps > 0:
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

                    status = NeutrophilCellData.Status.NONGRANULATING
                    gc = granule_count
                    self.append(
                        NeutrophilCellData.create_cell(point=point, status=status, granule_count=gc)
                    )

    def absorb_cytokines(self, n_absorb, cyto, grid):
        for index in self.alive():
            vox = grid.get_voxel(self[index]['point'])
            x = vox.x
            y = vox.y
            z = vox.z
            cyto[z, y, x] = (1 - n_absorb) * cyto[z, y, x]

    def produce_cytokines(self, n_det, n_n, grid, fungus: FungusCellList, cyto):
        for i in self.alive():
            vox = grid.get_voxel(self[i]['point'])

            hyphae_count = 0

            x_r = []
            y_r = []
            z_r = []

            if n_det == 0:
                index_arr = fungus.get_cells_in_voxel(vox)
                for index in index_arr:
                    if fungus[index]['form'] == FungusCellData.Form.HYPHAE:
                        hyphae_count += 1

            else:
                for num in range(0, n_det + 1):
                    x_r.append(num)
                    y_r.append(num)
                    z_r.append(num)

                for num in range(-1 * n_det, 0):
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

            cyto[vox.z, vox.y, vox.x] = cyto[vox.z, vox.y, vox.x] + (n_n * hyphae_count)

    def move(self, rec_r, grid, cyto, tissue):
        for cell_index in self.alive(
            self.cell_data['status'] == NeutrophilCellData.Status.NONGRANULATING
        ):
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

            cell['point'] = Point(
                x=grid.x[vox.x + vox_list[i][0]],
                y=grid.y[vox.y + vox_list[i][1]],
                z=grid.z[vox.z + vox_list[i][2]],
            )

            self.update_voxel_index([cell_index])

    def damage_hyphae(self, n_det, n_kill, time, health, grid, fungus: FungusCellList, iron):
        for i in self.alive(self.cell_data['granule_count'] > 0):
            cell = self[i]
            vox = grid.get_voxel(cell['point'])

            x_r = []
            y_r = []
            z_r = []

            if n_det == 0:
                index_arr = fungus.get_cells_in_voxel(vox)
                if len(index_arr) > 0:
                    iron[vox.z, vox.y, vox.x] = 0
                for index in index_arr:
                    if (
                        fungus[index]['form'] == FungusCellData.Form.HYPHAE
                        and cell['granule_count'] > 0
                    ):
                        fungus[index]['health'] -= health * (time / n_kill)
                        cell['granule_count'] -= 1
                        cell['status'] = NeutrophilCellData.Status.GRANULATING
                    elif cell['granule_count'] == 0:
                        cell['status'] = NeutrophilCellData.Status.NONGRANULATING
                        break
            else:
                for num in range(0, n_det + 1):
                    x_r.append(num)
                    y_r.append(num)
                    z_r.append(num)

                for num in range(-1 * n_det, 0):
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
                                if len(index_arr) > 0:
                                    iron[zk, yj, xi] = 0
                                for index in index_arr:
                                    if (
                                        fungus[index]['form'] == FungusCellData.Form.HYPHAE
                                        and cell['granule_count'] > 0
                                    ):
                                        fungus[index]['health'] -= health * (time / n_kill)
                                        cell['granule_count'] -= 1
                                        cell['status'] = NeutrophilCellData.Status.GRANULATING
                                    elif cell['granule_count'] == 0:
                                        cell['status'] = NeutrophilCellData.Status.NONGRANULATING
                                        break

    def update(self):
        for i in self.alive(self.cell_data['granule_count'] == 0):
            self[i]['status'] = NeutrophilCellData.Status.NONGRANULATING

    def age(self):
        self.cell_data['iteration'] += 1

    def kill_by_age(self, age_limit):
        for i in self.alive(self.cell_data['iteration'] > age_limit):
            self[i]['dead'] = True


def cell_list_factory(self: 'NeutrophilState'):
    return NeutrophilCellList(grid=self.global_state.grid)


@attr.s(kw_only=True)
class NeutrophilState(ModuleState):
    cells: NeutrophilCellList = attr.ib(default=attr.Factory(cell_list_factory, takes_self=True))
    neutropenic: bool
    rec_rate_ph: int
    rec_r: float
    n_absorb: float
    n_n: float
    n_det: int
    granule_count: int
    n_kill: float
    time_n: float
    age_limit: int


class Neutrophil(ModuleModel):
    name = 'neutrophil'

    StateClass = NeutrophilState

    def initialize(self, state: State):
        neutrophil: NeutrophilState = state.neutrophil
        grid: RectangularGrid = state.grid

        neutrophil.neutropenic = self.config.getboolean('neutropenic')
        neutrophil.rec_rate_ph = self.config.getint('rec_rate_ph')
        neutrophil.rec_r = self.config.getfloat('rec_r')
        neutrophil.n_absorb = self.config.getfloat('n_absorb')
        neutrophil.n_n = self.config.getfloat('Nn')
        neutrophil.n_det = self.config.getint('n_det')
        neutrophil.granule_count = self.config.getint('granule_count')
        neutrophil.n_kill = self.config.getfloat('n_kill')
        neutrophil.time_n = self.config.getfloat('time_step')
        neutrophil.age_limit = self.config.getint('age_limit')

        neutrophil.cells = NeutrophilCellList(grid=grid)

        return state

    def advance(self, state: State, previous_time: float):
        neutrophil: NeutrophilState = state.neutrophil
        n_cells = neutrophil.cells
        fungus = state.fungus.cells
        health = state.fungus.health

        tissue = state.geometry.lung_tissue
        grid = state.grid
        cyto = state.molecules.grid['n_cyto']
        iron = state.molecules.grid['iron']

        # recruit new
        n_cells.recruit_new(
            neutrophil.rec_rate_ph,
            neutrophil.rec_r,
            neutrophil.granule_count,
            neutrophil.neutropenic,
            previous_time,
            grid,
            tissue,
            cyto,
        )

        # absorb cytokines
        n_cells.absorb_cytokines(neutrophil.n_absorb, cyto, grid)

        # produce cytokines
        n_cells.produce_cytokines(neutrophil.n_det, neutrophil.n_n, grid, fungus, cyto)

        # move
        n_cells.move(neutrophil.rec_r, grid, cyto, tissue)

        n_cells.damage_hyphae(
            neutrophil.n_det, neutrophil.n_kill, neutrophil.time_n, health, grid, fungus, iron
        )

        # update granule == 0 status
        n_cells.update()

        n_cells.age()

        n_cells.kill_by_age(neutrophil.age_limit)

        return state
