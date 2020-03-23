from enum import IntEnum
import random

import attr
import numpy as np

from simulation.cell import CellData, CellList
from simulation.coordinates import Point, Voxel
from simulation.grid import RectangularGrid
from simulation.module import Module, ModuleState
from simulation.modules.geometry import TissueTypes
from simulation.modules.fungus import FungusCellData
from simulation.state import State


class NeutrophilCellData(CellData):

    class Status(IntEnum):
        RESTING = 0
        GRANULATING = 1

    NEUTROPHIL_FIELDS = [
        ('status', 'u1'),
        ('iteration', 'i4'),
        ('granule_count', 'i4')
    ]

    dtype = np.dtype(
        CellData.FIELDS + NEUTROPHIL_FIELDS, align=True
    )  # type: ignore

    @classmethod
    def create_cell_tuple(cls, status, granule_count, **kwargs,) -> np.record:
        status = status
        iteration = 0
        return CellData.create_cell_tuple(**kwargs) + (status,iteration,granule_count,)


@attr.s(kw_only=True, frozen=True, repr=False)
class NeutrophilCellList(CellList):
    CellDataClass = NeutrophilCellData


def cell_list_factory(self: 'NeutrophilState'):
    return NeutrophilCellList(grid=self.global_state.grid)


@attr.s(kw_only=True)
class NeutrophilState(ModuleState):
    cells: NeutrophilCellList = attr.ib(default=attr.Factory(cell_list_factory, takes_self=True))
    neutropenic: bool = False
    rec_rate_ph: int = 6
    n_absorb: float = 0.9
    Nn: float = 100
    n_det: float = 15


class Neutrophil(Module):
    name = 'neutrophil'
    defaults = {
        'cells': '',
        'neutropenic': 'False',
        'rec_rate_ph': '10',
    }
    StateClass = NeutrophilState

    def initialize(self, state: State):
        neutrophil: NeutrophilState = state.neutrophil
        grid: RectangularGrid = state.grid
        tissue = state.geometry.lung_tissue

        neutrophil.neutropenic = self.config.getboolean('neutropenic')
        neutrophil.rec_rate_ph = self.config.getint('rec_rate_ph')
        neutrophil.rec_r = self.config.getfloat('rec_r')
        neutrophil.n_absorb = self.config.getfloat('n_absorb')
        neutrophil.Nn = self.config.getfloat('Nn')
        neutrophil.n_det = self.config.getfloat('n_det')
        neutrophil.granule_count = self.config.getint('granule_count')
        neutrophil.n_kill = self.config.getfloat('n_kill')
        #NeutrophilCellData.LEAVE_RATE = self.config.getfloat('LEAVE_RATE')

        neutrophil.cells = NeutrophilCellList(grid=grid)

        return state

    def advance(self, state: State, previous_time: float):

        recruit_new(state, previous_time)
        absorb_cytokines(state)
        produce_cytokines(state)
        move(state)
        damage_hyphae(state, previous_time)

        return state


def recruit_new(state, time):
    neutrophil: NeutrophilState = state.neutrophil
    n_cells = neutrophil.cells
    tissue = state.geometry.lung_tissue
    grid = state.grid
    cyto = state.molecules.grid['n_cyto']

    num_reps = neutrophil.rec_rate_ph # number of neutrophils recruited per time step
    if (neutrophil.neutropenic and time >= 48 and time <= 96):
        num_reps = (time - 48) / 8
    
    blood_index = np.argwhere(tissue == TissueTypes.BLOOD.value)
    blood_index = np.transpose(blood_index)
    mask = cyto[blood_index[2], blood_index[1], blood_index[0]] >= neutrophil.rec_r
    blood_index = np.transpose(blood_index)
    cyto_index = blood_index[mask]
    np.random.shuffle(cyto_index)

    for i in range(0, num_reps):
        if(len(cyto_index) > 0):
            ii = random.randint(0, len(cyto_index) - 1)
            point = Point(
                x = grid.x[cyto_index[ii][2]], 
                y = grid.y[cyto_index[ii][1]], 
                z = grid.z[cyto_index[ii][0]])

            status = NeutrophilCellData.Status.RESTING
            gc = neutrophil.granule_count
            n_cells.append(NeutrophilCellData.create_cell(point=point, status=status, granule_count=gc))
            

def absorb_cytokines(state):
    neutrophil: NeutrophilState = state.neutrophil
    n_cells = neutrophil.cells
    cyto = state.molecules.grid['n_cyto']
    grid = state.grid

    for index in n_cells.alive():
        vox = grid.get_voxel(n_cells[index]['point'])
        x = vox.x
        y = vox.y
        z = vox.z
        cyto[z,y,x] = (1 - neutrophil.n_absorb) * cyto[z,y,x]

    return state


def produce_cytokines(state):
    neutrophil: NeutrophilState = state.neutrophil
    n_cells = neutrophil.cells
    fungus = state.fungus.cells

    tissue = state.geometry.lung_tissue
    grid = state.grid
    cyto = state.molecules.grid['n_cyto']

    for i in n_cells.alive():
        vox = grid.get_voxel(n_cells[i]['point'])

        hyphae_count = 0

        n_det = int(neutrophil.n_det / 2)
        x_r = []
        y_r = []
        z_r = []

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
                            if(fungus[index]['form'] == FungusCellData.Form.HYPHAE):
                                hyphae_count +=1

        cyto[vox.z, vox.y, vox.x] = cyto[vox.z, vox.y, vox.x] + neutrophil.Nn * hyphae_count

    return state


def move(state):
    neutrophil = state.neutrophil
    n_cells = neutrophil.cells
    fungus = state.fungus.cells

    tissue = state.geometry.lung_tissue
    grid = state.grid
    cyto = state.molecules.grid['n_cyto']

    for cell_index in n_cells.alive():
        cell = n_cells[cell_index]
        if (cell['status'] == NeutrophilCellData.Status.RESTING):
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
                        if(grid.is_valid_voxel(Voxel(x=xi, y=yj, z=zk)) and 
                            tissue[zk, yj, xi] != TissueTypes.AIR.value):
                            vox_list.append([x, y, z])
                            i += 1
                            if cyto[zk, yj, xi] >= neutrophil.rec_r:
                                p[i] = cyto[zk, yj, xi]


            indices = np.argwhere(p != 0)
            l = len(indices)
            if(l == 1):
                i = indices[0][0]
            elif(l > 1):
                inds = np.argwhere(p == p[np.argmax(p)])
                np.random.shuffle(inds)
                i = inds[0][0]
            else:
                i = random.randint(0,len(vox_list) - 1)

            cell['point'] = Point(
                x=grid.x[vox.x + vox_list[i][0]],
                y=grid.y[vox.y + vox_list[i][1]],
                z=grid.z[vox.z + vox_list[i][2]]
                )

            n_cells.update_voxel_index([cell_index])              

    return state


def damage_hyphae(state, time):
    neutrophil: NeutrophilState = state.neutrophil
    n_cells = neutrophil.cells
    fungus = state.fungus.cells

    tissue = state.geometry.lung_tissue
    grid = state.grid
    cyto = state.molecules.grid['n_cyto']
    iron = state.molecules.grid['iron']

    for i in n_cells.alive():
        cell = n_cells[i]
        vox = grid.get_voxel(cell['point'])

        n_det = int(neutrophil.n_det / 2)
        x_r = []
        y_r = []
        z_r = []

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
                        iron[vox.z, vox.y, vox.x] = 0
                        index_arr = fungus.get_cells_in_voxel(Voxel(x=xi, y=yj, z=zk))
                        for index in index_arr:
                            if(fungus[index]['form'] == FungusCellData.Form.HYPHAE):
                                fungus[index]['health'] -= time / neutrophil.n_kill
                                cell['granule_count'] -= 1
                                cell['status'] = NeutrophilCellData.Status.GRANULATING
                            elif(cell['granule_count'] <= 0):
                                cell['status'] = NeutrophilCellData.Status.RESTING
                                break

    return state
